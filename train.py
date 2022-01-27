import os
import datetime

import tensorflow as tf 

from argparser   import args
from prepdataset import prepdata
from dataloader  import DatasetGenerator
 
from tensorflow import keras as K

def test_intel_tensorflow():
    # Check if Intel version of TensorFlow is installed
    
    import tensorflow as tf
 
    print("Tensorflow version {}".format(tf.__version__))
    
    from tensorflow.python import _pywrap_util_port
    print("Intel-optimizations (DNNL) enabled:",_pywrap_util_port.IsMklEnabled())  
test_intel_tensorflow()

# region DATA PREP
print("- Preprocessing data ...")
prepdata(data_path=args.data_path)
print("- Preprocessing complete.")
# endregion DATA PREP

"""
#region DATA GENERATOR
input_dim = (args.tile_height, args.tile_width,args.tile_depth)
training_datapath = os.path.join(settings.DATAPATH_INPUT)
data1 = DatasetGenerator(input_dim, data_path=training_datapath, batch_size=args.batch_size,
                        train_test_split=args.train_test_split, validate_test_split=args.validate_test_split, 
                        number_output_classes=args.number_output_classes,random_seed=args.random_seed,augment=augment)
data1.print_info()
#endregion DATA GENERATOR
"""