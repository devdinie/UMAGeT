import os
import time
import shutil
import tensorflow as tf 

import augment
import settings


from argparser  import args
from preprocess import preprocess_data
from createjson import create_json_file
from dataloader import dataset_generator

from tensorflow import keras as K

#from prepare_data import prepdata
#from dataloader   import DatasetGenerator
#from model        import dice_coef, soft_dice_coef, dice_loss, unet_3d

#region GPU and tensorflow info
"""
# Check if Intel version of TensorFlow is installed
# Get GPU information
"""
def test_intel_tensorflow():
    print("Tensorflow version {}".format(tf.__version__))  
    from tensorflow.python.util import _pywrap_util_port
    print("Intel-optimizations (DNNL) enabled:",_pywrap_util_port.IsMklEnabled())  

test_intel_tensorflow()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
	tf.config.experimental.set_memory_growth(device, True)
#endregion GPU and tensorflow info


def main():
    #region initialization
    """
    # Assign model name to load based on input arguments
    """ 

    if args.network == "1":
        net_dname  = "data_net1"
        model_name = settings.net1_loc_modelname
    elif args.network == "2":
        net_dname  = "data_net2"
        model_name = settings.net2_seg_modelname
    else:
        net_dname = "data_net2" 
        model_name= settings.net2_seg_modelname
        print("Network not given as input. Proceeding with default network --network 2.")

    input_dim = settings.img_size
    
    datainput_dir= os.path.join(settings.data_dir,settings.augdata_dname)
    datanet1_dir = os.path.join(settings.data_dir,settings.net1data_dname)
    datanet2_dir = os.path.join(settings.data_dir,settings.net2data_dname)
    
    if settings.is_overwrite and (os.path.exists(datainput_dir)):
        shutil.rmtree(datainput_dir)
    if settings.is_overwrite and (os.path.exists(datanet1_dir)):
        shutil.rmtree(datanet1_dir)
    if settings.is_overwrite and (os.path.exists(datanet2_dir)):
        shutil.rmtree(datanet2_dir)
    #endregion initialization
    
    #region augmentation
    if settings.augment:
        augment.augment_data(data_dir=settings.data_dir, augtypes_in="n",output_dir=datainput_dir)
    else:
        augment.augment_data(data_dir=settings.data_dir,augtypes_in = None, output_dir=datainput_dir)
    
    create_json_file(datainput_dir)
    #endregion augmentation
      
    try:
        if (args.mode == "train" and not settings.labels_available):
            raise IOError()
    except IOError:  
        print("Selected mode is training. Cannot proceed without target labels")
    
    #region generating train/test datasets 
    data_net = dataset_generator(input_dim, data_dir=datainput_dir,
                                 train_test_split=args.train_test_split, 
                                 no_output_classes=settings.no_output_classes)
    create_json_file(datainput_dir)
    #endregion generating train/test datasets
    
    #region preprocessing
    preprocess_data(settings.data_dir)
    #endregion preprocessing
if __name__ == "__main__":
    main()


"""
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
"""