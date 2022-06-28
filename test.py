from genericpath import exists
import os
import time
import augment
import settings
import tensorflow as tf 

from argparser  import args
from tensorflow import keras as K
from preprocess import preprocess_data
from UMAGeT.eval_old.dataloader import dataset_generator
from createjson import create_json_file

#from model import dice_coef, soft_dice_coef, dice_loss, unet_3d

mode = args.mode
input_dim = settings.img_size

datainput_dir = os.path.join(settings.testdata_dir,"data_input")
 
augment.augment_data(data_dir=settings.testdata_dir,augtypes_in = None, output_dir=datainput_dir)

create_json_file(datainput_dir)

#region generating test dataset
data_net = dataset_generator(input_dim, data_dir=datainput_dir,
                            train_test_split=args.train_test_split, 
                            no_output_classes=settings.no_output_classes)
#endregion generating test dataset

#region preprocessing
preprocess_data(settings.testdata_dir)
#endregion preprocessing