import os
import sys
import time
import warnings
import settings
import datetime
import matplotlib

import numpy   as np
import pandas  as pd
import seaborn as sns
import nibabel as nib

import tensorflow as tf 
import matplotlib.pyplot as plt

from nilearn     import plotting
from sklearn     import metrics
from tensorflow  import keras as K
from prepdataset import prepdata
from dataloader  import DatasetGenerator
from time        import perf_counter
from skimage     import morphology

from skimage.transform      import resize
from evaluate_utilities     import get_distributions, calculate_scores, compare_labels,get_labelComparison_figures
from evaluate_generateplots import generate_boxplots_datasets, generate_boxplots_average 
from evaluate_generateplots import create_confusionMatrix,plot_confMat_distributions, plot_volumeCorrelations

cmap = matplotlib.colors.ListedColormap([plt.cm.get_cmap('viridis',3)(1), 'red'])

network_no = 2

def main():
    
    if network_no == 2:
        network_dir = "data_net2" 
        model_name  = settings.SAVED_MODEL2_NAME
    if network_no == 1:
        network_dir = "data_net1" 
        model_name  = settings.SAVED_MODEL1_NAME

    input_dim        = (settings.TILE_HEIGHT, settings.TILE_WIDTH,settings.TILE_DEPTH)
    testing_datapath = os.path.join(settings.TESTDATA_PATH, network_dir)

    root_dir  = os.path.abspath(settings.ROOT_DIR)
    pred_path = os.path.join(root_dir,"testData/data_net2/predictions")

    warnings.filterwarnings('ignore')

    #region Initialize directories
    if not os.path.isdir(pred_path):
        os.mkdir(pred_path)
    if not os.path.isdir(os.path.join(pred_path,"brains")):
        os.mkdir(os.path.join(pred_path,"brains"))
    if not os.path.isdir(os.path.join(pred_path,"labels")):
        os.mkdir(os.path.join(pred_path,"labels"))
    if not os.path.isdir(os.path.join(pred_path,"outputs")):
        os.mkdir(os.path.join(pred_path,"outputs"))
    #endregion Initialize directories

    #region Data generator
    print("- Starting data generator for network {} ...".format(network_no))

    data_net = DatasetGenerator(input_dim, data_path=testing_datapath, batch_size=settings.BATCH_SIZE,
                                train_test_split=settings.TRAIN_TEST_SPLIT, validate_test_split=settings.VALIDATE_TEST_SPLIT, 
                                number_output_classes=settings.NUMBER_OUTPUT_CLASSES,random_seed=settings.RANDOM_SEED,
                                augment=settings.AUGMENT)

    data_net.print_info()
    print("- Data generator for network {} complete.".format(network_no))

    ds = data_net.get_testn()
    #endregion Data generator

    #region load model
    tf_model = tf.keras.models.load_model(os.path.join(settings.ROOT_DIR,model_name+"_final"), compile=False)
    tf_model.compile(optimizer="adam", loss="binary_crossentropy")
    #endregion load model


    i = 0
    for fname, img, msk in ds:
        i = i+1
    no_files = i

    #region Initialize dataframes
    scores_df  = pd.DataFrame(np.zeros(((no_files),9)), columns=['subject_id','dataset','L|R' ,
                                                                'dice', 'Jacc', 'Kapp', 'Accu','Recl', 'Prec'])

    confmat_df = pd.DataFrame(np.zeros(((no_files),7)), columns=['subject_id', 'dataset' ,'L|R',
                                                                  'true_neg', 'false_pos', 'false_neg', 'true_pos'])

    volumes_df  = pd.DataFrame(np.zeros(((no_files),5)), columns=['subject_id','dataset','L|R' ,
                                                                  'Volumes: Truth', 'Volumes: Predictions'])

    distrib_df = pd.DataFrame(np.zeros(((no_files),207)))

    scores_df.reset_index(drop=True , inplace=True)
    confmat_df.reset_index(drop=True, inplace=True)
    distrib_df.reset_index(drop=True, inplace=True)
    volumes_df.reset_index(drop=True, inplace=True)
    #endregion Initialize dataframes

    i = 0
    for fname, img, msk in ds:

        img_name = os.path.basename(K.backend.get_value(fname)[0].decode('utf-8'))
        msk_name = img_name.replace("t1","labels")

        subject_id = img_name.split("_")[0]
        side = img_name.split("_")[2]

        if subject_id[0:3] == "HAD":
            group = "HAD"
        elif subject_id[0:3] == "ADB":
            group = "ADB"
        elif subject_id[0:3] == "ADN":
            group = "ADNI"
        elif subject_id[0:2] == "CC":
            group = "CC"
        else:
            group = "unclassified"

        prediction_tf = tf_model.predict(img)

        img1  = np.array(np.squeeze(np.squeeze(img,0),3))
        msk1  = np.array(np.squeeze(np.squeeze(msk,0),3))
        pred1 = np.array(np.squeeze(np.squeeze(prediction_tf,0),3))

        #print(subject_id, side, np.round(img1.min(),4), np.round(img1.max(),4))

        msk2  = msk1 
        pred2 = pred1

        tn_dist, fn_dist, fp_dist, tp_dist = get_distributions(msk1,pred1)

        msk2[msk2 >= 0.5] = 1
        msk2[msk2 <  0.5] = 0

        pred2[pred2 >= 0.5] = 1
        pred2[pred2 <  0.5] = 0

        pred2 = morphology.remove_small_holes(pred2.astype(int),500,connectivity=4)
        pred2 = pred2.astype(float)

        confmat = metrics.confusion_matrix(msk2.flatten(),pred2.flatten()).ravel()

        scores = calculate_scores(msk2,pred2)


        msk_hdr = nib.load(os.path.join(root_dir,"testData",network_dir,"target_labels",msk_name)).get_header()

        vox_m = msk_hdr.get_zooms()
        voxvols_m = vox_m[0] * vox_m[1] * vox_m[2]

        volumes_df.loc[i, 'Volumes: Truth']       = np.shape(np.nonzero(msk2 == 1))[1] * voxvols_m
        volumes_df.loc[i, 'Volumes: Predictions'] = np.shape(np.nonzero(pred2 == 1))[1] * voxvols_m

        #print(np.shape(np.nonzero(msk2 == 1))[1]*voxvols_m,"|",np.shape(np.nonzero(pred2 == 1))[1]*voxvols_m)


        predVmsk_nii = compare_labels(msk2, pred2)

        scores_df.iloc[ i:i+1,0:3] = [subject_id, group, side]
        confmat_df.iloc[i:i+1,0:3] = [subject_id, group, side]
        distrib_df.iloc[i:i+1,0:3] = [subject_id, group, side]
        volumes_df.iloc[i:i+1,0:3] = [subject_id, group, side]

        scores_df.iloc[ i:i+1,3:9] = scores
        confmat_df.iloc[i:i+1,3:7] = confmat

        distrib_df.iloc[i:i+1,  3: 54] = tn_dist
        distrib_df.iloc[i:i+1, 54:105] = fn_dist
        distrib_df.iloc[i:i+1,105:156] = fp_dist
        distrib_df.iloc[i:i+1,156:207] = tp_dist

        img_nii      = nib.Nifti1Image(img1    ,np.eye(4))
        pred_nii     = nib.Nifti1Image(pred2   ,np.eye(4))

        nib.save(img_nii     , os.path.join(pred_path,"brains" ,img_name))
        nib.save(pred_nii    , os.path.join(pred_path,"labels" ,msk_name))
        nib.save(predVmsk_nii, os.path.join(pred_path,"outputs",msk_name.replace("labels","predVmsk")))

        i = i+1
        
    evalScores_path = os.path.join(settings.ROOT_DIR,"outputs_evalScores")
    if not os.path.exists(evalScores_path):
        os.mkdir(evalScores_path)
        
    scores_df.to_csv( os.path.join(evalScores_path,'net'+str(network_no)+'-scores.csv') ,index=False)
    confmat_df.to_csv(os.path.join(evalScores_path,'net'+str(network_no)+'-confusion_matrices.csv') ,index=False)
    distrib_df.to_csv(os.path.join(evalScores_path,'net'+str(network_no)+'-prob_distributions.csv') ,index=False)
    volumes_df.to_csv(os.path.join(evalScores_path,'net'+str(network_no)+'-volumes.csv') ,index=False)
    """
    evalScores_path = os.path.join(settings.ROOT_DIR,"outputs_evalScores")
    scores_df  = pd.read_csv(os.path.join(evalScores_path,'net'+str(network_no)+'-scores.csv')) 
    confmat_df = pd.read_csv(os.path.join(evalScores_path,'net'+str(network_no)+'-confusion_matrices.csv'))
    distrib_df = pd.read_csv(os.path.join(evalScores_path,'net'+str(network_no)+'-prob_distributions.csv'))
    volumes_df = pd.read_csv(os.path.join(evalScores_path,'net'+str(network_no)+'-volumes.csv'))
    """
    evalFigures_path = os.path.join(settings.ROOT_DIR,"outputs_evalFigures")
    if not os.path.exists(evalFigures_path):
        os.mkdir(evalFigures_path)
        
    generate_boxplots_datasets(scores_df,evalFigures_path)
    generate_boxplots_average(scores_df,evalFigures_path)
    
    create_confusionMatrix(confmat_df,evalFigures_path)
    plot_confMat_distributions(distrib_df,evalFigures_path)
    
    plot_volumeCorrelations(volumes_df,evalFigures_path)
    
    compFigures_path = os.path.join(settings.ROOT_DIR,"outputs_predVlabelsFigures")
    if not os.path.exists(compFigures_path):
        os.mkdir(compFigures_path)
        
    get_labelComparison_figures(network_no, scores_df,compFigures_path)

if __name__ == "__main__":
    main()
