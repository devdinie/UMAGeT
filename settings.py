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
#mode = "test"
augment = False
is_overwrite = True

visualize_training = True

root_dir = os.path.join(os.path.dirname(__file__),"..")
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/")
testdata_dir =  os.path.join(os.path.dirname(os.path.dirname(__file__)),"testdata/")
visualizations_dir =  os.path.join(os.path.dirname(os.path.dirname(__file__)),"visualizations")

labels_available = os.path.exists(os.path.join(data_dir,"target_labels"))
augdata_dname = "data_input" 
net1data_dname= "data_net1_loc"
net2data_dname= "data_net2_seg"
#endregion define mode and main directory paths

#region image related settings
img_size = (144,144,144)
imgio_type = "NiftiImageIO"
#endregion image related settings

#region data and model related settings
"""
# Names used to save the models
# Ratio to split dataset for training and testing (train_test_split)
# From the remaining test dataset, the ratio to split between
# validation and testing data (test dataset = no. files - train data)
# test dataset = {validation data, test data}
"""
random_seed= 816

batch_size = 2
train_test_split = 0.6
validate_test_split = 0.5

net1_loc_modelname = "net1_model_localize"
net2_seg_modelname = "net2_model_segment"

epochs = 40
filters = 16
use_upsampling = True #Use upsampling instead of transposed convolution

no_input_classes = 1
no_output_classes= 1
#endregion data and model related settings
