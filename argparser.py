#!/usr/bin/env python
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
import settings
import argparse

parser = argparse.ArgumentParser(description="3D U-Net model", 
                                 add_help=True, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--mode", default=settings.mode,
                    help="Mode to use the network in\n"
                         +"To train : --mode train\n"
                         +"To test  : --mode test")

parser.add_argument("--augment",
                    default=settings.augment,
                    help="Set to --augment False if augmentation of input data is not required\n"
                         +"Set to --augment True  if augmented data needs to be generated from input data\n"
                         +"True (as defined in settings) by default")

parser.add_argument("--network", default="all",
                    help="The network(s) used in the given mode\n"
                    +"Input: --network all -for both localization and segmentation\n" 
                    +"Input: --network 1   -for localization\n" 
                    +"       --network 2   -for segmentation\n")

parser.add_argument("--epochs", type=int, default=settings.epochs,
                    help="Number of epochs to train the network")

parser.add_argument("--output_classes", type=int,
                    default=settings.no_output_classes,
                    help="Number of output classes")

parser.add_argument("--train_test_split", type=float,
                    default=settings.train_test_split,
                    help="Split ratio between train and test data (value between 0 and 1)")

args = parser.parse_args()