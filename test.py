import os
import datetime
import settings

import tensorflow as tf 

from tensorflow  import keras as K
from argparser   import args
from prepdataset import prepdata
from dataloader  import DatasetGenerator
from model       import dice_coef, soft_dice_coef, dice_loss, unet_3d
from time        import perf_counter

testing_datapath = settings.TESTDATA_PATH

if os.path.exists(os.path.join(testing_datapath,"target_labels")):
    # region TESTDATA PREP
    print("- Preprocessing data ...")
    prepdata(data_path=testing_datapath)
    print("- Preprocessing complete.")
    # endregion TESTDATA PREP