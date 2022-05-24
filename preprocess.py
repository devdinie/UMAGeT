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

from skimage import exposure
from matplotlib.pyplot import axes
from skimage.transform import resize

from numpy.core.fromnumeric import reshape, shape
from scipy.ndimage.measurements import standard_deviation

# These functions are performed regardless of augmentation

def resample_img(img_nii, msk_nii, input_dim):
    
    reference_size = input_dim

    reference = sitk.GetImageFromArray(np.zeros((input_dim)))
    reference.SetOrigin(msk_nii.GetOrigin())
    reference.SetDirection(msk_nii.GetDirection())

    reference_physicalsize = np.zeros(msk_nii.GetDimension())
    reference_physicalsize[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(msk_nii.GetSize(), msk_nii.GetSpacing(), reference_physicalsize)]
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physicalsize) ]

    reference.SetSpacing(reference_spacing)

    msk_resampled = sitk.Resample(msk_nii, reference)
    img_resampled = sitk.Resample(img_nii, msk_resampled)

    return img_resampled, msk_resampled

def normalize_img(img_nii, msk_nii):
	
    img_arr  = sitk.GetArrayFromImage(img_nii)
    msk_arr  = sitk.GetArrayFromImage(msk_nii)

    img_norm_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))

    img_norm_arr = exposure.equalize_adapthist(img_norm_arr)
    img_norm_nii = sitk.GetImageFromArray(img_norm_arr)
    
    img_norm_nii.CopyInformation(img_nii)
    
    return img_norm_nii, msk_nii

def split_image(img, msk, mid_idx):

    imgL = img[0 : mid_idx      , 0:img.GetSize()[1],0:img.GetSize()[2]]
    imgR = img[mid_idx:mid_idx*2, 0:img.GetSize()[1],0:img.GetSize()[2]]

    mskL = msk[0      : mid_idx ,0:msk.GetSize()[1],0:msk.GetSize()[2]]
    mskR = msk[mid_idx:mid_idx*2,0:msk.GetSize()[1],0:msk.GetSize()[2]]
    
    return imgL, imgR, mskL, mskR

def get_roi(msk, reserve=2):
    
    msk_arr = sitk.GetArrayFromImage(msk)

    msk_arr[msk_arr <= 0] = 0
    msk_arr[msk_arr  > 0] = 1

    reserve = reserve
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

    bbox = [lower_idx[0], lower_idx[1], lower_idx[2], upper_idx[0], upper_idx[1], upper_idx[2]]
  
    msk_nii = sitk.GetImageFromArray(msk_arr)
    msk_nii.CopyInformation(msk)
    
    return msk_nii, bbox