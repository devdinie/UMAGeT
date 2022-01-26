import os
import json
import math
import settings

import numpy     as np
import SimpleITK as sitk

import scipy
import torchio
import skimage
import nilearn.masking
import skimage.morphology

from matplotlib.pyplot import axes
from SimpleITK.SimpleITK import GetArrayFromImage
from numpy.core.fromnumeric import reshape, shape
from scipy.ndimage.measurements import standard_deviation

# These functions are performed regardless of augmentation

def normalize(img,msk):
	
	img_arr  = sitk.GetArrayFromImage(img)
	img_norm_arr = (img_arr - np.mean(img_arr))/(np.std(img_arr))
	img_norm = sitk.GetImageFromArray(img_norm_arr)

	msk_arr = sitk.GetArrayFromImage(msk)
	msk_arr[msk_arr > 0.5 ] =   1
	msk_arr[msk_arr <= 0.5] =   0
	msk_norm = sitk.GetImageFromArray(msk_arr)

	img_norm.CopyInformation(img)
	msk_norm.CopyInformation(msk)

	norm = 1
	return img_norm, msk_norm, norm

def get_roi(msk):
    
    msk_arr = sitk.GetArrayFromImage(msk)

    reserve = int(2)
    X, Y, Z = msk_arr.shape[:3]

    idx = np.nonzero(msk_arr > 0)

    x1_idx = max(idx[0].min() - reserve, 0)
    x2_idx = min(idx[0].max() + reserve, X)
    
    y1_idx = max(idx[1].min() - reserve, 0)
    y2_idx = min(idx[1].max() + reserve, Y)
    
    z1_idx = max(idx[2].min() - reserve, 0)
    z2_idx = min(idx[2].max() + reserve, Z)

    lower_idx = np.array([x1_idx,y1_idx,z1_idx], dtype='int')
    upper_idx = np.array([x2_idx,y2_idx,z2_idx], dtype='int')
    
    #bbox = [x1, y1, z1, x2, y2, z2]
    bbox = [lower_idx[0], lower_idx[1], lower_idx[2], upper_idx[0], upper_idx[1], upper_idx[2]]
    
    return bbox