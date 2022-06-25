# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
# Adapted from  intel https://github.com/IntelAI/unet

import os

#region define mode and main directory paths
"""
# Define directory paths with respect to current working directory
# Setting if data augmentation is required (in train mode only)
"""

mode = "train"
augment = True

root_dir = os.path.join(os.path.dirname(__file__),"..")
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/")
testdata_dir =  os.path.join(os.path.dirname(os.path.dirname(__file__)),"testData/")

augdata_dname = "data_aug" 
#endregion define mode and main directory paths

#region image related settings
img_size = (144,144,144)
imgio_type = "NiftiImageIO"
#endregion image related settings

#region data and model related settings
"""
# Names used to save the models
# Ratio to split dataset for training and testing (train_test_split)
# From the percentage of dataset for testing, ratio to split between
# validation and testing
"""
batch_size = 2
train_test_split = 0.6
validate_test_split = 0.5

loc_model_name = "net1_model_localize"
seg_model_name = "net2_model_segment"

epochs = 40
filters = 16
use_upsampling = True #Use upsampling instead of transposed convolution

no_input_classes = 1
no_output_classes = 1
#endregion data and model related settings


"""
ROOT_DIR       = os.path.join(os.path.dirname(__file__),"..")
DATA_PATH      = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/")
TESTDATA_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)),"testData/")

MODE    = "train"
AUGMENT = True

if MODE == "train":
    DATAPATH_INPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/data_input")

if MODE == "test":
    DATAPATH_INPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),"testData/data_input")

IMAGE_FILETYPE = "NiftiImageIO"

SAVED_MODEL1_NAME = "UNET1_LOCALIZE"
SAVED_MODEL2_NAME = "UNET2_SEGMENT"

EPOCHS     =40
BATCH_SIZE =2
TILE_HEIGHT=144
TILE_WIDTH =144
TILE_DEPTH =144

NUMBER_INPUT_CHANNELS=1
NUMBER_OUTPUT_CLASSES=1

TRAIN_TEST_SPLIT   =0.60
VALIDATE_TEST_SPLIT=0.50

PRINT_MODEL   =False
FILTERS       =16
USE_UPSAMPLING=False

RANDOM_SEED   =816
"""