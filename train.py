import os
import time
import numpy as np
import shutil
import augment
import settings
import datetime
import tensorflow as tf 
import matplotlib.pyplot as plt

#region GPU and tensorflow settings and info
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

#endregion GPU and tensorflow settings and info
from argparser  import args
from tensorflow import keras as K
from preprocess import preprocess_data
from createjson import create_json_file
from dataloader import dataset_generator

from model import unet_3d

#region visualize training
def plot_trainhistory(net, history):
    """
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    fig.savefig(os.path.join(settings.visualizations_dir,'history_accuracy'+net+'.png'), bbox_inches='tight')
    """
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    fig.savefig(os.path.join(settings.visualizations_dir,'history_loss'+str(net)+'.png'), bbox_inches='tight')
   
    fig = plt.figure()
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice coefficient')
    plt.ylabel('dice coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig(os.path.join(settings.visualizations_dir,'history_dice'+str(net)+'.png'), bbox_inches='tight')
#endregion visualize training

#region function to create, train and save network
def train_network(net, input_dim, datanet_dir, model_name):
    from model import dice_coef, soft_dice_coef, dice_loss
    
    #region generating train/test datasets 
    data_net = dataset_generator(input_dim, data_dir=datanet_dir,
                                 train_test_split=args.train_test_split, 
                                 no_output_classes=settings.no_output_classes)
    #endregion generating train/test datasets
    
    
    print("Network {} processing started".format(net))
    #region create network
    model = unet_3d(input_dim=settings.img_size + (settings.no_input_classes,), filters=settings.filters, 
                    no_output_classes=args.output_classes, use_upsampling=settings.use_upsampling, 
                    concat_axis=-1, model_name=model_name)
    
    local_opt = K.optimizers.Adam()
    #accuracy =  tf.keras.metrics.Accuracy() 
    model.compile(loss=dice_loss, metrics=[dice_coef, soft_dice_coef],optimizer=local_opt)
    
    checkpoint = K.callbacks.ModelCheckpoint(model_name, verbose=1, save_best_only=True)
    
    if not os.path.exists(os.path.join("logs","logs_"+model_name)):
        os.mkdir(os.path.join("logs","logs_"+model_name))
        
    logs_dir = os.path.join("logs","logs_"+model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_logs  = K.callbacks.TensorBoard(log_dir=logs_dir)
    callbacks= [checkpoint, tb_logs]
    print("- Network {}: model created.".format(net))
    #endregion create network
    
    #region train network
    print("- Network {}: training began.".format(net))
    
    starttime_train = time.time()
    steps_per_epoch = data_net.no_train // settings.batch_size
    
    history = model.fit(data_net.get_train(), epochs=args.epochs, steps_per_epoch=steps_per_epoch, 
                        validation_data=data_net.get_validate(), callbacks=callbacks, verbose=1)
    
    endtime_train = time.time()
    plot_trainhistory(net,history)
    print("- Network {}: training complete [training time: {:.4f}].".format(net, endtime_train-starttime_train))
    #endregion train network
    
    #region evaluate network
    local_opt = K.optimizers.Adam()
    best_model = K.models.load_model(model_name, custom_objects={"dice_loss": dice_loss,
                                    "dice_coef": dice_coef, "soft_dice_coef": soft_dice_coef})
    
    best_model.compile(loss=dice_loss, metrics=[dice_coef, soft_dice_coef], optimizer=local_opt)
    loss, dice_coef, soft_dice_coef = best_model.evaluate(data_net.get_test())
    
    print("- Network {}: model evaluated [Avg. dice coefficient on test data: {:.4f}].".format(net, dice_coef))
    #endregion evaluate network
    
    #region save trained model
    best_model_name = model_name + "_final"
    best_model.compile(loss="binary_crossentropy", metrics=["accuracy"],optimizer="adam")
    K.models.save_model(best_model, best_model_name, include_optimizer=False)
    print("- Network {}: model saved as {}.".format(net,model_name))
    
    print("Network {} processing complete".format(net))
    #endregion save trained model
    
#endregion function to create, train and save network

def main():
    #region initialization
    try:
        if (args.mode == "train" and not settings.labels_available):
            raise IOError()
    except IOError:  
        print("Selected mode is 'train'. Cannot proceed without target labels to train on.")
        
    if not ((args.network == "1") or (args.network == "2")):
        if not (args.network == "all"): print("Network not specified. Both networks will be trained.")
        else: net="all"
    else: net = int(args.network)

  
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
    if settings.is_overwrite and (os.path.exists(settings.visualizations_dir)):
        shutil.rmtree(settings.visualizations_dir)
        
    if settings.visualize_training and not os.path.exists(settings.visualizations_dir):
        os.makedirs(settings.visualizations_dir)
    #endregion initialization
    
    #region augmentation
    if settings.augment:
        augment.augment_data(data_dir=settings.data_dir, augtypes_in="n",output_dir=datainput_dir)
    else:
        augment.augment_data(data_dir=settings.data_dir,augtypes_in = None, output_dir=datainput_dir)
    #endregion augmentation
      
    #region preprocessing
    create_json_file(datainput_dir)
    preprocess_data(settings.data_dir)
    
    create_json_file(datanet1_dir)
    create_json_file(datanet2_dir)
    #endregion preprocessing
    
    if (net == "all") or (net == 1):
        train_network(1,settings.img_size, datanet1_dir, settings.net1_loc_modelname)

    if (net == "all") or (net == 2):
        #net2_img_size = tuple(int(val/6) for val in settings.img_size)
        train_network(2, settings.img_size, datanet2_dir, settings.net2_seg_modelname)

if __name__ == "__main__":
    main()
