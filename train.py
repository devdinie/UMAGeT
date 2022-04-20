import os
import datetime
import settings

import tensorflow as tf 

from tensorflow   import keras as K
from argparser    import args
from prepare_data import prepdata
from dataloader   import DatasetGenerator
from model        import dice_coef, soft_dice_coef, dice_loss, unet_3d
from time         import perf_counter


def test_intel_tensorflow():
    # Check if Intel version of TensorFlow is installed
    
    import tensorflow as tf
 
    print("Tensorflow version {}".format(tf.__version__))
    
    from tensorflow.python import _pywrap_util_port
    print("Intel-optimizations (DNNL) enabled:",_pywrap_util_port.IsMklEnabled())  

test_intel_tensorflow()

#region Initialization 
if args.network == "1":
    network_dir = "data_net1"
    model_name  = args.saved_model1_name
    do_preprocessing = True
elif args.network == "2":
    network_dir = "data_net2" 
    model_name  = args.saved_model2_name
    do_preprocessing = False
else:
    network_dir = "data_net2" 
    model_name  = args.saved_model2_name
    do_preprocessing = False
    print("Network not defined. Set to --network 2 parameters by default.")

input_dim         = (args.tile_height, args.tile_width,args.tile_depth)
training_datapath = os.path.join(settings.DATA_PATH,network_dir)
#endregion Initialization

"""
# region DATA PREP
if do_preprocessing == True:
    print("- Preprocessing data ...")
    prepdata(data_path=args.data_path)
    print("- Preprocessing complete.")
# endregion DATA PREP
"""

#region DATA GENERATOR
print("- Starting data generator for network {} ...".format(args.network))

data_net = DatasetGenerator(input_dim, data_path=training_datapath, batch_size=args.batch_size,
                            train_test_split=args.train_test_split, validate_test_split=args.validate_test_split, 
                            number_output_classes=args.number_output_classes,random_seed=args.random_seed,
                            augment=settings.AUGMENT)
data_net.print_info()
print("- Data generator for network {} complete.".format(args.network))
#endregion DATA GENERATOR


#region NETWORK: CREATE_MODEL
print("- Creating network {} model ...".format(args.network))
input_dim = (args.tile_height, args.tile_width,args.tile_depth, args.number_input_channels)

model = unet_3d(input_dim=input_dim, filters=args.filters, number_output_classes=args.number_output_classes,
                use_upsampling=args.use_upsampling, concat_axis=-1, model_name=model_name)

local_opt = K.optimizers.Adam()
model.compile(loss=dice_loss,metrics=[dice_coef, soft_dice_coef],optimizer=local_opt)
checkpoint = K.callbacks.ModelCheckpoint(model_name, verbose=1, save_best_only=True)

if not os.path.exists(os.path.join("logs","logs_"+model_name)):
    os.mkdir(os.path.join("logs","logs_"+model_name))

logs_dir = os.path.join("logs","logs_"+model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_logs  = K.callbacks.TensorBoard(log_dir=logs_dir)
callbacks= [checkpoint, tb_logs]
print("- Creating network {} model complete".format(args.network))
#endregion NETWORK: CREATE_MODEL

#region NETWORK: TRAIN_MODEL
print("- Training network {} model ...".format(args.network))

start_traintime = perf_counter()
steps_per_epoch = data_net.num_train // args.batch_size

model.fit(data_net.get_train(), epochs=args.epochs, steps_per_epoch=steps_per_epoch, 
          validation_data=data_net.get_validate(), callbacks=callbacks, verbose=1)

end_traintime = perf_counter()
print("- Training network {} complete . Time taken {:.4f}".format(args.network, end_traintime-start_traintime))
#endregion NETWORK: TRAIN_MODEL

#region NETWORK: EVALUATE MODEL
local_opt = K.optimizers.Adam()
print("- Evaluating network {} model ...".format(args.network))
best_model = K.models.load_model(model_name, custom_objects={"dice_loss": dice_loss,
                                                             "dice_coef": dice_coef, 
                                                             "soft_dice_coef": soft_dice_coef})

best_model.compile(loss=dice_loss, metrics=[dice_coef, soft_dice_coef], optimizer=local_opt)
loss, dice_coef, soft_dice_coef = best_model.evaluate(data_net.get_test())
print("Average Dice Coefficient on testing dataset = {:.4f}".format(dice_coef))
print("- Evaluating network {} model complete".format(args.network))
#endregion NETWORK: EVALUATE MODEL

#region NETWORK: SAVE MODEL
print("- Saving network {} model ...".format(args.network))
best_model_name = model_name + "_final"
best_model.compile(loss="binary_crossentropy", metrics=["accuracy"],optimizer="adam")
K.models.save_model(best_model, best_model_name, include_optimizer=False)
print("- Saving network {} model complete.".format(args.network))
#endregion NETWORK: SAVE MODEL