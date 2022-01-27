import os
import json
import math
import settings

import numpy     as np
import SimpleITK as sitk

import scipy
import skimage
import nilearn.masking
import skimage.morphology

from matplotlib.pyplot import axes
from skimage.transform import resize

from numpy.core.fromnumeric import reshape, shape
from scipy.ndimage.measurements import standard_deviation

# These functions are performed regardless of augmentation

def resample_img(img_nii, msk_nii, input_dim):

    img_nii = img_nii
    msk_nii = msk_nii
    
    reference_size = input_dim

    reference = sitk.GetImageFromArray(np.zeros((input_dim)))
    reference.SetOrigin(msk_nii.GetOrigin())
    reference.SetDirection(msk_nii.GetDirection())

    reference_physicalsize = np.zeros(msk_nii.GetDimension())
    reference_physicalsize[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(msk_nii.GetSize(), msk_nii.GetSpacing(), reference_physicalsize)]
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physicalsize) ]

    msk_resampled = sitk.Resample(msk_nii, reference)
    img_resampled = sitk.Resample(img_nii, msk_resampled)

    return img_resampled, msk_resampled

def normalize_img(img_nii, msk_nii):
	
    img_arr  = sitk.GetArrayFromImage(img_nii)

    img_norm_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
    img_norm_nii     = sitk.GetImageFromArray(img_norm_arr)
    
    img_norm_nii.CopyInformation(img_nii)
    
    return img_norm_nii, msk_nii

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