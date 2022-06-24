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
#
import os

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
