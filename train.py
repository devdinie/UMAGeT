import os
import datetime

import tensorflow as tf 

from argparser   import args
from prepdataset import prepdata
 

from tensorflow import keras as K

def test_intel_tensorflow():
    # Check if Intel version of TensorFlow is installed
    
    import tensorflow as tf
 
    print("Tensorflow version {}".format(tf.__version__))
    
    from tensorflow.python.util import _pywrap_util_port
    print("Intel-optimizations (DNNL) enabled:",_pywrap_util_port.IsMklEnabled())  
test_intel_tensorflow()

# region DATA PREP
print("------------------ DATA PREP (IP): BEGIN  ------------------")
prepdata(data_path=args.data_path)
print("------------------ DATA PREP (IP):  DONE  ------------------")
# endregion DATA PREP