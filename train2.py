
import os
import datetime

import tensorflow as tf  

from tensorflow import keras as K

from argparser   import args
from prepdataset import prepdata
from dataloader  import DatasetGenerator
from model       import dice_coef, soft_dice_coef, dice_loss, unet_3d


def test_intel_tensorflow():
    """
    Check if Intel version of TensorFlow is installed
    """
    import tensorflow as tf
 
    print("Tensorflow version {}".format(tf.__version__))
    
    from tensorflow.python import _pywrap_util_port
    print("Intel-optimizations (DNNL) enabled:",_pywrap_util_port.IsMklEnabled())
    
test_intel_tensorflow()

"""
# region DATA PREP
print("------------------ DATA PREP (IP): BEGIN  ------------------")
prepdata(data_path=args.data_path)
print("------------------ DATA PREP (IP):  DONE  ------------------")
print("*************************************************************")
# endregion DATA PREP
"""

# region NETWORK1

# region NETWORK1: DATA_GENERATOR
"""
print("------------------ DATA GENERATOR1:  BEGIN ------------------")
input_dim = (args.tile_height, args.tile_width,args.tile_depth)
augment = True
data1 = DatasetGenerator(input_dim, data_path=os.path.join(args.data_path,"data_net1"), batch_size=args.batch_size,
                        train_test_split=args.train_test_split, validate_test_split=args.validate_test_split, 
                        number_output_classes=args.number_output_classes,random_seed=args.random_seed,augment=augment)
data1.print_info()
print("------------------ DATA GENERATOR1:  DONE  ------------------")

print("*************************************************************")
"""
# endregion NETWORK1: DATA_GENERATOR



# region NETWORK1: CREATE_MODEL
"""
print("------------------ CREATING MODEL1: BEGIN  ------------------")
# Create tensorflow model
input_dim = (args.tile_height, args.tile_width,args.tile_depth, args.number_input_channels)

model1 = unet_3d(input_dim=input_dim, filters=args.filters,number_output_classes=args.number_output_classes,
                use_upsampling=args.use_upsampling, concat_axis=-1, model_name=args.saved_model1_name)

local_opt = K.optimizers.Adam()
model1.compile(loss=dice_loss,metrics=[dice_coef, soft_dice_coef],optimizer=local_opt)
checkpoint = K.callbacks.ModelCheckpoint(args.saved_model1_name, verbose=1, save_best_only=True)
print("Tensorflow model created")

logs_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_logs  = K.callbacks.TensorBoard(log_dir=logs_dir)
callbacks= [checkpoint, tb_logs]
print("------------------ CREATING MODEL1:  DONE  ------------------")

print("*************************************************************")
"""
# endregion NETWORK1: CREATE_MODEL


# region NETWORK1: TRAIN_MODEL
"""
print("------------------ TRAINING MODEL1: BEGIN  ------------------")
# Train the model
steps_per_epoch = data1.num_train // args.batch_size
model1.fit(data1.get_train(), epochs=args.epochs, steps_per_epoch=steps_per_epoch, validation_data=data1.get_validate(), callbacks=callbacks, verbose=1)
print("------------------ TRAINING MODEL1:  DONE  ------------------")

print("*************************************************************")
"""
# endregion NETWORK1: TRAIN_MODEL


# region NETWORK1: SAVE_EVALUATE_MODEL
"""
print("------------------ BEST MDL1 EVAL|SAVE:BEG ------------------")
best_model1 = K.models.load_model(args.saved_model1_name, 
                                 custom_objects={"dice_loss": dice_loss,
                                                 "dice_coef": dice_coef, 
                                                 "soft_dice_coef": soft_dice_coef})
best_model1.compile(loss=dice_loss,metrics=[dice_coef, soft_dice_coef],optimizer=local_opt)
print("Evaluating best model on the testing dataset.")
print("=============================================")
loss, dice_coef, soft_dice_coef = best_model1.evaluate(data1.get_test())
print("Average Dice Coefficient on testing dataset = {:.4f}".format(dice_coef))

final_model1_name = args.saved_model1_name + "_final"
best_model1.compile(loss="binary_crossentropy", metrics=["accuracy"],optimizer="adam")
K.models.save_model(best_model1, final_model1_name,include_optimizer=False)
print("------------------ BEST MDL1 EVAL|SAVE:DNE ------------------")

print("*************************************************************")
"""
#endregion NETWORK1: SAVE_EVALUATE_MODEL


# region NETWORK1: CONVERT_MODEL
"""
print("------------------ CONVERTING MODEL1: BEGIN  ------------------")
print("\n\nConvert the TensorFlow model to OpenVINO by running:\n")
print("source /opt/intel/openvino/bin/setupvars.sh")
print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
print("       --saved_model_dir {} \\".format(final_model1_name))
print("       --model_name {} \\".format(args.saved_model1_name))
print("       --batch 1  \\")
print("       --output_dir {} \\".format(os.path.join("openvino_models", "FP32")))
print("       --data_type FP32\n\n")
print("------------------ CONVERTING MODEL1:  DONE  ------------------")

# endregion NETWORK1: CONVERT_MODEL
"""
print("==================  NETWORK 1: COMPLETE.  ==================") 

# endregion NETWORK1



# region NETWORK2

# region NETWORK2: DATA_GENERATOR
print("------------------ DATA GENERATOR2:  BEGIN ------------------")
input_dim = (args.tile_height, args.tile_width,args.tile_depth)
augment = False
data2 = DatasetGenerator(input_dim, data_path=os.path.join(args.data_path,"data_net2"), batch_size=args.batch_size,
                        train_test_split=args.train_test_split, validate_test_split=args.validate_test_split, 
                        number_output_classes=args.number_output_classes,random_seed=args.random_seed,augment=augment)
data2.print_info()
print("------------------ DATA GENERATOR2:  DONE  ------------------")

print("*************************************************************")

# endregion NETWORK2: DATA_GENERATOR


# region NETWORK2: CREATE_MODEL
print("------------------ CREATING MODEL2: BEGIN  ------------------")
# Create tensorflow model
input_dim = (args.tile_height, args.tile_width,args.tile_depth, args.number_input_channels)

model2 = unet_3d(input_dim=input_dim, filters=args.filters,number_output_classes=args.number_output_classes,
                use_upsampling=args.use_upsampling, concat_axis=-1, model_name=args.saved_model2_name)

local_opt = K.optimizers.Adam()
model2.compile(loss=dice_loss,metrics=[dice_coef, soft_dice_coef],optimizer=local_opt)
checkpoint2 = K.callbacks.ModelCheckpoint(args.saved_model2_name, verbose=1, save_best_only=True)
print("Tensorflow model created")

logs_dir2 = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_logs2  = K.callbacks.TensorBoard(log_dir=logs_dir2)
callbacks2= [checkpoint2, tb_logs2]
print("------------------ CREATING MODEL2:  DONE  ------------------")

print("*************************************************************")

# endregion NETWORK2: CREATE_MODEL


# region NETWORK2: TRAIN_MODEL
print("------------------ TRAINING MODEL2: BEGIN  ------------------")
# Train the model
steps_per_epoch2 = data2.num_train // args.batch_size
model2.fit(data2.get_train(), epochs=args.epochs, steps_per_epoch=steps_per_epoch2, validation_data=data2.get_validate(), callbacks=callbacks2, verbose=1)
print("------------------ TRAINING MODEL2:  DONE  ------------------")

print("*************************************************************")

# endregion NETWORK2: TRAIN_MODEL


# region NETWORK2: SAVE_EVALUATE_MODEL
print("------------------ BEST MDL EVAL|SAVE2:BEG ------------------")
best_model2 = K.models.load_model(args.saved_model2_name, 
                                 custom_objects={"dice_loss": dice_loss,
                                                 "dice_coef": dice_coef, 
                                                 "soft_dice_coef": soft_dice_coef})
best_model2.compile(loss=dice_loss,metrics=[dice_coef, soft_dice_coef],optimizer=local_opt)
print("Evaluating best model on the testing dataset.")
print("=============================================")
loss, dice_coef, soft_dice_coef = best_model2.evaluate(data2.get_test())
print("Average Dice Coefficient on testing dataset = {:.4f}".format(dice_coef))

final_model2_name = args.saved_model2_name + "_final"
best_model2.compile(loss="binary_crossentropy", metrics=["accuracy"],optimizer="adam")
K.models.save_model(best_model2, final_model2_name,include_optimizer=False)
print("------------------ BEST MDL EVAL|SAVE2:DNE ------------------")

print("*************************************************************")

# endregion NETWORK2: SAVE_EVALUATE_MODEL


# region NETWORK2: CONVERT_MODEL
print("------------------ CONVERTING MODEL2: BEGIN  ------------------")
print("\n\nConvert the TensorFlow model to OpenVINO by running:\n")
print("source /opt/intel/openvino/bin/setupvars.sh")
print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
print("       --saved_model_dir {} \\".format(final_model2_name))
print("       --model_name {} \\".format(args.saved_model2_name))
print("       --batch 1  \\")
print("       --output_dir {} \\".format(os.path.join("openvino_models", "FP32")))
print("       --data_type FP32\n\n")
print("------------------ CONVERTING MODEL2:  DONE  ------------------")
# endregion NETWORK2: CONVERT_MODEL

print("==================  NETWORK 2: COMPLETE.  ==================")

# endregion NETWORK2