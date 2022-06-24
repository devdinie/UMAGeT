import settings
import tensorflow as tf
from prepare_data import prepdata

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install tensorflow version compatible with GPU")

data_path=settings.DATA_PATH

prepdata(data_path=data_path, augmentation=settings.AUGMENT)
