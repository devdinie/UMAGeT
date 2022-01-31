import os
import datetime
import settings

import tensorflow as tf 

from tensorflow  import keras as K
from argparser   import args
from prepdataset import prepdata
from dataloader  import DatasetGenerator
from model       import dice_coef, soft_dice_coef, dice_loss, unet_3d


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


#region DATA GENERATOR
print("- Starting data generator ...")
input_dim = (args.tile_height, args.tile_width,args.tile_depth)
#training_datapath = os.path.join(settings.DATAPATH_INPUT)
training_datapath = os.path.join(settings.DATA_PATH,"data_net1")
data_net1 = DatasetGenerator(input_dim, data_path=training_datapath, batch_size=args.batch_size,
                        train_test_split=args.train_test_split, validate_test_split=args.validate_test_split, 
                        number_output_classes=args.number_output_classes,random_seed=args.random_seed,augment=settings.AUGMENT)
data_net1.print_info()
print("- Data generator complete.")
#endregion DATA GENERATOR


# region NETWORK1: CREATE_MODEL
print("- Creating network 1 model ...")
input_dim = (args.tile_height, args.tile_width,args.tile_depth, args.number_input_channels)

model1 = unet_3d(input_dim=input_dim, filters=args.filters,number_output_classes=args.number_output_classes,
                use_upsampling=args.use_upsampling, concat_axis=-1, model_name=args.saved_model1_name)

local_opt = K.optimizers.Adam()
model1.compile(loss=dice_loss,metrics=[dice_coef, soft_dice_coef],optimizer=local_opt)
checkpoint = K.callbacks.ModelCheckpoint(args.saved_model1_name, verbose=1, save_best_only=True)

logs_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_logs  = K.callbacks.TensorBoard(log_dir=logs_dir)
callbacks= [checkpoint, tb_logs]
print("- Creating network 1 model complete")
# endregion NETWORK1: CREATE_MODEL

# region NETWORK1: TRAIN_MODEL
print("- Training network 1 model ...")
steps_per_epoch = data_net1.num_train // args.batch_size
model1.fit(data_net1.get_train(), epochs=args.epochs, steps_per_epoch=steps_per_epoch, validation_data=data_net1.get_validate(), callbacks=callbacks, verbose=1)
print("- Training network 1 complete ...")
# endregion NETWORK1: TRAIN_MODEL
