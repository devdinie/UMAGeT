from asyncore import read
import os
import json
import settings

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from argparser import args

import nibabel as nib

class dataset_generator:
    
    def __init__(self, input_dim,
                 root_dir=settings.root_dir,
                 data_dir=settings.data_dir,
                 batch_size=settings.batch_size,
                 train_test_split=settings.train_test_split,
                 validate_test_split=settings.validate_test_split,
                 no_output_classes=settings.no_output_classes,
                 random_seed=settings.random_seed, shard=0):

        self.shard      = shard      # For Horovod, gives different shard per worker
        self.input_dim  = input_dim
        self.batch_size = batch_size
        self.random_seed= random_seed
        self.root_dir   = root_dir
        self.data_dir   = data_dir
        
        self.no_output_classes = no_output_classes
        self.train_test_split  = train_test_split
        self.validate_test_split = validate_test_split

        self.create_file_list()
        
        if not (args.mode == "test"):
            self.ds_train, self.ds_validate, self.ds_test = self.get_train_dataset()
        else:
            self.ds_test = self.get_test_dataset()

    #region create dictionary with dataset information
    def create_file_list(self):
         
        try:
            with open(os.path.join(self.data_dir, "dataset_dict.json"), "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            print("File {} doesn't exist. It should be located in the directory named 'data' ".format(json_filename))
           
        
        self.name            = experiment_data["name"]
        self.description     = experiment_data["description"]
        self.reference       = experiment_data["reference"]
        self.input_channels  = experiment_data["modality"]
        
        self.filenames = {}
        
        self.output_channels = experiment_data["labels"]
            
        if not (args.mode == "test"):
            self.no_files = experiment_data["numTraining"]
            for idx in range(self.no_files):
                self.filenames[idx] = [os.path.join(experiment_data["training"][idx]["image"]),
                                       os.path.join(experiment_data["training"][idx]["label"])]     
        else:
            self.no_files = experiment_data["numTesting"] 
            for idx in range(self.no_files):
                if settings.labels_available:
                    for idx in range(self.no_files):
                        self.filenames[idx] = [os.path.join(experiment_data["testing"][idx]["image"]),
                                               os.path.join(experiment_data["testing"][idx]["label"])] 
                else: 
                    self.filenames[idx] = [os.path.join(experiment_data["testing"][idx]["image"])]      
    #endregion create dictionary with dataset information
    
    #region function to read input images  
    def read_nifti_file(self, idx, itest=False):

        idx = idx.numpy()
        img_fname = self.filenames[idx][0]
        msk_fname = self.filenames[idx][1]
        
        img_file = sitk.ReadImage(img_fname, imageIO=settings.imgio_type)
        msk_file = sitk.ReadImage(msk_fname, imageIO=settings.imgio_type)
        
        img_arr = sitk.GetArrayFromImage(img_file)
        msk_arr = sitk.GetArrayFromImage(msk_file)
      
        img_arr = np.expand_dims(img_arr, -1)
        msk_arr = np.expand_dims(msk_arr, -1)

        return img_arr, msk_arr

    def get_train(self):
        # Return train dataset 
        return self.ds_train
 
    def get_test(self):
        # Return test dataset
        return self.ds_test
    
    def get_validate(self):
        # Return validation dataset
        return self.ds_validate

    #region get training dataset
    """
    # Get number of training data based on train_test_split
    """
    def get_train_dataset(self):
        
        self.no_train = int(self.no_files * self.train_test_split)
        self.no_validate = int((self.no_files - self.no_train)*self.validate_test_split)
        self.no_test  = int(self.no_files - (self.no_train+self.no_validate))
        
        ds = tf.data.Dataset.range(self.no_files).shuffle(self.no_files, self.random_seed)  # Shuffle the dataset
        
        ds_train = ds.take(self.no_train).shuffle(self.no_train, self.shard)  # Reshuffle based on shard
        
        ds_val_test = ds.skip(self.no_train)
        
        ds_test = ds_val_test.skip(self.no_validate)
        ds_validate = ds_val_test.take(self.no_validate)
        
        
        ds_train = ds_train.map(lambda x: tf.py_function(self.read_nifti_file, [x, True], [tf.float32, tf.float32]),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
                            
        ds_validate = ds_validate.map(lambda x: tf.py_function(self.read_nifti_file, [x, False], [tf.float32, tf.float32]),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        ds_test = ds_test.map(lambda x: tf.py_function(self.read_nifti_file, [x, False], [tf.float32, tf.float32]), 
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
              
        ds_train = ds_train.repeat()
        ds_train = ds_train.batch(self.batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        
        batch_size_val = 2
        ds_validate = ds_validate.batch(batch_size_val)
        ds_validate = ds_validate.prefetch(tf.data.experimental.AUTOTUNE)
         
        batch_size_test = 1
        ds_test = ds_test.batch(batch_size_test)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds_train, ds_validate, ds_test
    #endregion get training dataset
    
    def get_test_dataset(self):
        
        ds = tf.data.Dataset.range(self.no_files).shuffle(self.no_files, self.random_seed)
        ds_test = ds.take(self.no_files)
        
        ds_test = ds_test.map(lambda x: tf.py_function(self.read_nifti_file, [x, False], [tf.float32, tf.float32]), 
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        batch_size_test = 1
        ds_test = ds_test.batch(batch_size_test)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds_test
        
if __name__ == "__main__":
    # Load the dataset    
    data = dataset_generator(input_dim=settings.img_size, data_dir=settings.data_dir, batch_size=args.batch_size,
                             train_test_split=args.train_test_split, validate_test_split=args.validate_test_split,
                             number_output_classes=args.number_output_classes, random_seed=args.random_seed)
