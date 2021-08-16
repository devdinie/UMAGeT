import os
import settings

import scipy
import numpy   as np
import nibabel as nib

import tensorflow as tf

from scipy.ndimage     import rotate
from skimage.transform import resize
class DatasetGenerator:

    def __init__(self, input_dim,
                 root_dir=settings.ROOT_DIR,
                 data_path=settings.DATA_PATH,
                 batch_size=settings.BATCH_SIZE,
                 train_test_split=settings.TRAIN_TEST_SPLIT,
                 validate_test_split=settings.VALIDATE_TEST_SPLIT,
                 number_output_classes=settings.NUMBER_OUTPUT_CLASSES,
                 random_seed=settings.RANDOM_SEED,
                 augment=settings.AUGMENT,
                 shard=0):

        self.shard      = shard      # For Horovod, gives different shard per worker
        self.input_dim  = input_dim
        self.batch_size = batch_size
        self.random_seed= random_seed
        self.augment    = augment
        self.root_dir   = root_dir
        self.data_path  = data_path
        
        self.train_test_split      = train_test_split
        self.validate_test_split   = validate_test_split
        self.number_output_classes = number_output_classes

        self.create_file_list()

        self.ds_train, self.ds_val, self.ds_test = self.get_dataset()

    def create_file_list(self):
        """
        Get list of the files from the input data
        Split into training and testing sets.
        """
        import os
        import json

        json_filename = os.path.join(self.data_path, "dataset.json")

        try:
            with open(json_filename, "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            print("File {} doesn't exist. It should be located in the directory named 'data' ".format(json_filename))

        self.name            = experiment_data["name"]
        self.output_channels = experiment_data["labels"]
        self.release         = experiment_data["release"]
        self.license         = experiment_data["licence"]
        self.input_channels  = experiment_data["modality"]
        self.reference       = experiment_data["reference"]
        self.description     = experiment_data["description"]
        self.numFiles        = experiment_data["numTraining"]
        self.tensorImageSize = experiment_data["tensorImageSize"]

        """
        Create a dictionary of tuples with image filename and label filename
        """
        self.aug_flp_arr = np.random.choice([0, 1], self.numFiles, p=[0.6, 0.4])
        self.aug_rot_arr = np.random.choice([-7, -5, -2, 0, 2, 5, 7], self.numFiles, p=[0.03, 0.07, 0.10, 0.60, 0.10, 0.07, 0.03 ])

        self.filenames = {}
        for idx in range(self.numFiles):
            self.filenames[idx] = [os.path.join(experiment_data["training"][idx]["image"]),
                                   os.path.join(experiment_data["training"][idx]["label"])]

    def print_info(self):
        """
        Print the dataset information
        """

        print("="*30)
        print("Dataset name:        ", self.name)
        print("Dataset release:     ", self.release)
        print("Dataset license:     ", self.license)
        print("Dataset reference:   ", self.reference)
        print("Dataset description: ", self.description)
        print("Input channels:      ", self.input_channels)
        print("Output labels:       ", self.output_channels)
        print("Tensor image size:   ", self.tensorImageSize)
        print("="*30)

    def normalize_img(self, img): 
        """
        Normalize the image 
        """

        for channel in range(img.shape[-1]):
            img_temp = img[..., channel]
            img_temp = (img_temp - np.min(img_temp)) / (np.max(img_temp) - np.min(img_temp))
            img[..., channel] = img_temp

        return img

    def resize_img(self, img, msk, img_aff, msk_aff): 
        """
        resize the image to user defined dimensions
        """

        img = np.rint(resize(img,np.array(self.input_dim)))
        msk = np.rint(resize(msk,np.array(self.input_dim)))

        msk[msk < 0] = 0

        return img, msk

    def augment_data(self, img, msk, idx): # TO EDIT
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """

        # Flip (randomly) selected images and masks
        if self.aug_flp_arr[idx] == 1:
            img = nib.orientations.flip_axis(img,axis=0)
            msk = nib.orientations.flip_axis(msk,axis=0)

            #nib.save(nib.Nifti1Image(img,np.eye(4)), os.path.join(self.root_dir,"int_output_check/augment",os.path.basename(self.filenames[idx][0]).split(".")[0]+"_flp_"+str(idx)))
            #nib.save(nib.Nifti1Image(msk,np.eye(4)), os.path.join(self.root_dir,"int_output_check/augment",os.path.basename(self.filenames[idx][1]).split(".")[0]+"_flp_"+str(idx))) 

        # Rotate (randomly) selected images and masks
        if self.aug_rot_arr[idx] != 0:
            
            angle = self.aug_rot_arr[idx]

            if angle < 0:
                angle = 360+angle

            img = rotate(rotate(img,angle,reshape=False),angle,axes=tuple((1,0)),reshape=False)
            msk = np.rint(rotate(rotate(msk,angle,reshape=False),angle,axes=tuple((1,0)),reshape=False))

            msk[msk < 0] = 0
        
            #nib.save(nib.Nifti1Image(img,np.eye(4)), os.path.join(self.root_dir,"int_output_check/augment",os.path.basename(self.filenames[idx][0]).split(".")[0]+"_rot_"+str(idx)+"_"+str(angle)))
            #nib.save(nib.Nifti1Image(msk,np.eye(4)), os.path.join(self.root_dir,"int_output_check/augment",os.path.basename(self.filenames[idx][1]).split(".")[0]+"_rot_"+str(idx)+"_"+str(angle))) 
            
        return img, msk

    def read_nifti_file(self, idx, augment=False):
        """
        Read Nifti file
        """

        idx = idx.numpy()
        imgFile = self.filenames[idx][0]
        mskFile = self.filenames[idx][1]
        
        img_aff = nib.load(imgFile).affine
        msk_aff = nib.load(mskFile).affine

        img = np.array(nib.load(imgFile).dataobj)
        msk = np.array(nib.load(mskFile).dataobj)

        img = rotate(rotate(img,90),90,axes=tuple((1,0)))
        msk = rotate(rotate(msk,90),90,axes=tuple((1,0)))

        """
        "labels": {
             "0": "background",
             "1": "Hippocampus"}
        """

        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
         
        # Resize image
        img, msk = self.resize_img(img, msk, img_aff, msk_aff)
        
        # Normalize
        img = self.normalize_img(img)
        
        # Sanity check
        # print("##### NO OF FILES",self.numFiles,idx, self.aug_flp_arr[idx],self.aug_rot_arr[idx])

        if(self.aug_flp_arr[idx] or (self.aug_rot_arr[idx] != 0)):
            augment = self.augment
            
            if augment:
                img, msk = self.augment_data(img, msk, idx)


        #nib.save(nib.Nifti1Image(img,np.eye(4)), os.path.join(self.root_dir,"int_output_check",os.path.basename(imgFile).split(".")[0]+"_nrm"))
        #nib.save(nib.Nifti1Image(msk,np.eye(4)), os.path.join(self.root_dir,"int_output_check",os.path.basename(mskFile).split(".")[0]+"_nrm")) 

        return img, msk

    def plot_images(self, ds, slice_num=90):
        """
        Plot images from dataset
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 20))

        num_cols = 2

        msk_channel = 1
        img_channel = 0

        for img, msk in ds.take(1):
            bs = img.shape[0]

            for idx in range(bs):
                plt.subplot(bs, num_cols, idx*num_cols + 1)
                plt.imshow(img[idx, :, :, slice_num, img_channel], cmap="bone")
                plt.title("MRI", fontsize=18)
                plt.subplot(bs, num_cols, idx*num_cols + 2)
                plt.imshow(msk[idx, :, :, slice_num, msk_channel], cmap="bone")
                plt.title("Tumor", fontsize=18)

        plt.show()

        print("Mean pixel value of image = {}".format(
            np.mean(img[0, :, :, :, 0])))

    def display_train_images(self, slice_num=90):
        """
        Plots some training images
        """
        self.plot_images(self.ds_train, slice_num)

    def display_validation_images(self, slice_num=90):
        """
        Plots some validation images
        """
        self.plot_images(self.ds_val, slice_num)

    def display_test_images(self, slice_num=90):
        """
        Plots some test images
        """
        self.plot_images(self.ds_test, slice_num)

    def get_train(self):
        """
        Return train dataset
        """
        return self.ds_train

    def get_test(self):
        """
        Return test dataset
        """
        return self.ds_test

    def get_validate(self):
        """
        Return validation dataset
        """
        return self.ds_val

    def get_dataset(self):
        """
        Create a TensorFlow data loader
        """
        self.num_train = int(self.numFiles * self.train_test_split)
        numValTest     = self.numFiles - self.num_train

        ds = tf.data.Dataset.range(self.numFiles).shuffle(self.numFiles, self.random_seed)  # Shuffle the dataset

        """
        Horovod Sharding
        Here we are not actually dividing the dataset into shards
        but instead just reshuffling the training dataset for every
        shard. Then in the training loop we just go through the training
        dataset but the number of steps is divided by the number of shards.
        """
        ds_train     = ds.take(self.num_train).shuffle(self.num_train, self.shard)  # Reshuffle based on shard
        ds_val_test  = ds.skip(self.num_train)
        self.num_val = int(numValTest * self.validate_test_split)
        self.num_test= self.num_train - self.num_val
        
        ds_val  = ds_val_test.take(self.num_val)
        ds_test = ds_val_test.skip(self.num_val)

        ds_train = ds_train.map(lambda x: tf.py_function(self.read_nifti_file, [x, True], [tf.float32, tf.float32]), 
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
        ds_val   = ds_val.map(lambda x: tf.py_function(self.read_nifti_file, [x, False], [tf.float32, tf.float32]),
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                                                    
        ds_test  = ds_test.map(lambda x: tf.py_function(self.read_nifti_file, [x, False], [tf.float32, tf.float32]),
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        ds_train = ds_train.repeat()
        ds_train = ds_train.batch(self.batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        batch_size_val = 4
        ds_val = ds_val.batch(batch_size_val)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        batch_size_test = 1
        ds_test = ds_test.batch(batch_size_test)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_val, ds_test


if __name__ == "__main__":

    print("Load the data and plot a few examples")

    from argparser import args
    
    resize_dim = (args.tile_height, args.tile_width, args.tile_depth, args.number_input_channels)

    """
    Load the dataset
    """
    data = DatasetGenerator(crop_dim,
                                  data_path=args.data_path,
                                  batch_size=args.batch_size,
                                  train_test_split=args.train_test_split,
                                  validate_test_split=args.validate_test_split,
                                  number_output_classes=args.number_output_classes,
                                  random_seed=args.random_seed)

    data.print_info()