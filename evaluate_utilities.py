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

from skimage.transform import resize
cmap = matplotlib.colors.ListedColormap([plt.cm.get_cmap('viridis',3)(1), 'red'])

#region metric functions

def dice_coefficient(truth,prediction):
    return 2*np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def jaccard_index(truth,prediction):
    jaccIdx = metrics.jaccard_score(truth.ravel(),prediction.ravel(),average='binary')
    return jaccIdx

def kappa_score(truth,prediction):
    kappaScore = metrics.cohen_kappa_score(truth.ravel(), prediction.ravel())
    return kappaScore

def accuracy_score(truth,prediction):
    accuracyScore = metrics.accuracy_score(truth.ravel(), prediction.ravel())
    return accuracyScore

def recall_score(truth,prediction):
    RecallScore = metrics.recall_score(truth.ravel(), prediction.ravel())
    return RecallScore

def precision_score(truth,prediction):
    PrecisionScore = metrics.precision_score(truth.ravel(), prediction.ravel())
    return PrecisionScore

#endregion metric functions

def calculate_scores(msk, pred):
    
    dice = dice_coefficient(msk,pred)
    Jacc = jaccard_index(msk,pred)
    Kapp = kappa_score(msk,pred)
    Accu = accuracy_score(msk,pred)
    Recl = recall_score(msk,pred)
    Prec = precision_score(msk,pred)
    
    scores = [dice, Jacc, Kapp, Accu, Recl, Prec]
 
    return scores

def get_distributions(msk,pred):
    
    msk = msk.astype(int)
    thresh = 0.5
    
    neg_msk =  np.ones((msk.shape)) - msk
    
    tp_probs = np.round(msk*pred    ,2)     
    tn_probs = np.round(neg_msk*pred,2)  
    
    tp_probs[tp_probs < thresh] = 0 
    tn_probs[tn_probs >=thresh] = 1 
    
    false_mat   = np.int8(pred) - msk  
    
    fp_mat = np.where(false_mat <  1, 0, false_mat)
    fn_mat = np.where(false_mat > -1, 0, false_mat)*(-1)
    
    fp_probs = np.round(fp_mat*pred*neg_msk,2)  
    fn_probs = np.round(fn_mat*pred*msk,2)  
    
    fp_probs[fp_probs < thresh] = 0 
    fn_probs[fn_probs >=thresh] = 1 
    
    tp_dist = np.zeros(51) 
    tn_dist = np.zeros(51)  
    fp_dist = np.zeros(51)
    fn_dist = np.zeros(51) 
    
    prange = np.arange(0.51,1,0.01)
    for pidx , pval in enumerate(prange):
        tp_dist[pidx] = np.count_nonzero(tp_probs[tp_probs == np.round(pval,2)])
        fp_dist[pidx] = np.count_nonzero(fp_probs[fp_probs == np.round(pval,2)])

    nrange = np.arange(0,0.51,0.01)
    for nidx , nval in enumerate(nrange):
        tn_dist[nidx] = np.count_nonzero(tn_probs[tn_probs == np.round(nval,2)])
        fn_dist[nidx] = np.count_nonzero(fn_probs[fn_probs == np.round(nval,2)])
    
    return tn_dist, fn_dist, fp_dist, tp_dist


def compare_labels(msk, pred):
    
    msk  = np.int8(msk)
    pred = np.int8(pred)
    
    ovrlp     = np.multiply(msk,pred)
    ovrlp_nii = nib.Nifti1Image(ovrlp,np.eye(4))

    diff = np.bitwise_xor(msk,pred)
    diff = np.where((diff == 1), 2, diff)
    diff_nii = nib.Nifti1Image(diff,np.eye(4))

    comb = diff + ovrlp
    comb_nii = nib.Nifti1Image(comb,np.eye(4))
    
    return comb_nii


def get_labelComparison_figures(network,scores_df1,compFigures_path):
    
    pred_path = os.path.join(settings.ROOT_DIR,"testData","data_net"+str(network),"predictions")
    scores_gmb  = pd.DataFrame(np.zeros(((len(scores_df1)),4)), columns=['subject_id','dataset','L|R' , 'mean'])

    scores_gmb.iloc[0:len(scores_df1),0:3] = scores_df1.iloc[0:len(scores_df1),0:3]
    scores_gmb.iloc[0:len(scores_df1),  3] = scores_df1.mean(axis=1)
    scores_gmb['mean'] = np.round(scores_gmb['mean'],5)
    
    ID_worst= scores_gmb.iloc[[scores_gmb['mean'].idxmin()]]['subject_id'].values[0]
    LR_worst= scores_gmb.iloc[[scores_gmb['mean'].idxmin()]]['L|R'].values[0]

    ID_best = scores_gmb.iloc[[scores_gmb['mean'].idxmax()]]['subject_id'].values[0]
    LR_best = scores_gmb.iloc[[scores_gmb['mean'].idxmax()]]['L|R'].values[0]

    median = scores_gmb['mean'].median()
    median_idx  = scores_gmb['mean'].sub(median).abs().idxmin()

    ID_med  = scores_gmb.iloc[median_idx,:][0]
    LR_med  = scores_gmb.iloc[median_idx,:][2]
    
    output_name_worst = ID_worst+"_predVmsk_"+LR_worst+"_norm-rC0-n0-d0-sp0-gh0.nii"
    output_name_best  = ID_best +"_predVmsk_"+LR_best +"_norm-rC0-n0-d0-sp0-gh0.nii"
    output_name_med   = ID_med  +"_predVmsk_"+LR_med  +"_norm-rC0-n0-d0-sp0-gh0.nii"
    
    output = nib.load(os.path.join(pred_path,"outputs",output_name_best))
    imgo   = nib.load(os.path.join(pred_path,"brains" ,output_name_best.replace("predVmsk","t1")))

    plotting.plot_roi(roi_img=output,bg_img=None, 
                      output_file=os.path.join( compFigures_path,"best-"+ID_best+"labels.png"),
                      cmap=cmap, black_bg=True, draw_cross=False,cut_coords=(50,70,120))
    plotting.plot_roi(roi_img=output,bg_img=imgo ,
                      output_file=os.path.join(compFigures_path,"best-"+ID_best+"labelsOverlayed.png"),
                      cmap=cmap, black_bg=True, draw_cross=False,cut_coords=(50,70,120))


    output = nib.load(os.path.join(pred_path,"outputs",output_name_med))
    imgo   = nib.load(os.path.join(pred_path,"brains" ,output_name_med.replace("predVmsk","t1")))

    plotting.plot_roi(roi_img=output,bg_img=None , 
                      output_file=os.path.join( compFigures_path,"avg-"+ID_med+"labels.png"),
                      cmap=cmap, black_bg=True, draw_cross=False,cut_coords=(50,70,126))
    plotting.plot_roi(roi_img=output,bg_img=imgo , 
                      output_file=os.path.join( compFigures_path,"avg-"+ID_med+"labelsOverlayed.png"),
                      cmap=cmap, black_bg=True, draw_cross=False,cut_coords=(50,70,126))
    
    output = nib.load(os.path.join(pred_path,"outputs",output_name_worst))
    imgo   = nib.load(os.path.join(pred_path,"brains" ,output_name_worst.replace("predVmsk","t1")))

    plotting.plot_roi(roi_img=output,bg_img=None , 
                      output_file=os.path.join( compFigures_path,"worst-"+ID_worst+"labels.png"),
                      cmap=cmap, black_bg=True, draw_cross=False,cut_coords=(50,70,97))
    plotting.plot_roi(roi_img=output,bg_img=imgo , 
                      output_file=os.path.join( compFigures_path,"worst-"+ID_worst+"labelsOverlayed.png"),
                      cmap=cmap, black_bg=True, draw_cross=False,cut_coords=(50,70,97))