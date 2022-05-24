from locale import normalize
import os
import json
import settings
import numpy   as np
import nibabel as nib

import argparse  as args
import SimpleITK as sitk

from augment       import augment_data
from createjson    import create_jsonFile
from preprocess    import resample_img, normalize_img, split_image, get_roi
from scipy.ndimage import rotate


#region function: get file list from directory
def get_filelist(data_path):
    
    json_filename = os.path.join(data_path, "dataset.json")
    
    try:
        with open(json_filename, "r") as fp:
            experiment_data = json.load(fp)       
    except IOError as e:
        print("File {} doesn't exist. It should be located in 'data/' ".format(json_filename))

    numFiles = experiment_data["numTraining"]

    filenames = {}
    if settings.MODE == "train":
        for idx in range(numFiles):
            filenames[idx] = [os.path.join(experiment_data["training"][idx]["image"]),
                              os.path.join(experiment_data["training"][idx]["label"])]
            
    if settings.MODE == "test":
        for idx in range(numFiles):
            filenames[idx] = [os.path.join(experiment_data["testing"][idx]["image"]),
                              os.path.join(experiment_data["testing"][idx]["label"])]
    return filenames 
#endregion function: get file list from directory
    

def prepdata(data_path=settings.DATA_PATH, augmentation=settings.AUGMENT):
    
    input_dim=[settings.TILE_WIDTH,settings.TILE_HEIGHT,settings.TILE_DEPTH]
    data_filetype = settings.IMAGE_FILETYPE
    create_jsonFile(data_path=data_path)
    
    datapath_net1 = os.path.join(data_path,"data_net1")
    datapath_net2 = os.path.join(data_path,"data_net2")
    
    #region create input directories
    if not os.path.exists(settings.DATAPATH_INPUT):
        os.mkdir(settings.DATAPATH_INPUT)
        os.mkdir(os.path.join(settings.DATAPATH_INPUT,"brains"))
        os.mkdir(os.path.join(settings.DATAPATH_INPUT,"target_labels"))
        
    if not os.path.exists(datapath_net1):
        os.mkdir(datapath_net1)
        os.mkdir(os.path.join(datapath_net1,"brains"))
        os.mkdir(os.path.join(datapath_net1,"target_labels"))
        
    if not os.path.exists(datapath_net2):
        os.mkdir(datapath_net2)
        os.mkdir(os.path.join(datapath_net2,"brains"))
        os.mkdir(os.path.join(datapath_net2,"target_labels"))
    #endregion create input directories
    
    filenames    = get_filelist(data_path=data_path)
    
    for idx in range(0,len(filenames)):

        #region load brains | labels
        imgFile = filenames[idx][0]
        mskFile = filenames[idx][1]
        
        if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
            continue
    
        img_nii = sitk.ReadImage(imgFile, imageIO=data_filetype)
        msk_nii = sitk.ReadImage(mskFile, imageIO=data_filetype)
        #endregion load brains | labels

        #region rename and write non-augmented originals in data_input
        imgFile_aug = os.path.basename(imgFile).replace("_t1"    ,     "_t1_"+"norm"+"-rC00-n00-d00-sp0000-gh0")
        mskFile_aug = os.path.basename(mskFile).replace("_labels", "_labels_"+"norm"+"-rC00-n00-d00-sp0000-gh0")
        
        sitk.WriteImage(img_nii, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgFile_aug))
        sitk.WriteImage(msk_nii, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskFile_aug))
        #endregion rename and write non-augmented originals in data_input
    
    create_jsonFile(data_path=settings.DATAPATH_INPUT)

    if settings.AUGMENT and (settings.MODE != "test"):
        augment_data(data_path=settings.DATAPATH_INPUT)

    filenames_aug = get_filelist(data_path=settings.DATAPATH_INPUT)

    for idx in range(0,len(filenames_aug)):

		#region load data_input (aug) brains | labels
        imgFile_aug = filenames_aug[idx][0]
        mskFile_aug = filenames_aug[idx][1]

        if not (os.path.exists(imgFile_aug) or os.path.exists(mskFile)):
            continue

        imgaug_nii = sitk.ReadImage(imgFile_aug, imageIO=data_filetype)
        mskaug_nii = sitk.ReadImage(mskFile_aug, imageIO=data_filetype)
        #endregion load data_input (aug) brains | labels

        #region Normalize and Resample brains | labels
        img_norm_nii, msk_norm_nii = normalize_img(imgaug_nii, mskaug_nii)
        img_rsmp_nii, msk_rsmp_nii = resample_img(img_norm_nii, msk_norm_nii, input_dim)
		#endregion Normalize and Resample brains | labels

        #region Write aug/non-aug & norm + resampled brains | labels
        sitk.WriteImage(img_rsmp_nii, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgFile_aug))
        sitk.WriteImage(msk_rsmp_nii, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskFile_aug))
		#endregion Write aug/non-aug & norm + resampled brains | labels

        #region Split brains | labels - L|R
        mid_idx = np.around(input_dim[0]/2).astype(int)

        imgL_nii, imgRorg_nii, mskL_nii, mskRorg_nii = split_image(img_rsmp_nii, msk_rsmp_nii, mid_idx)
        imgR_nii = sitk.Flip(imgRorg_nii,[True,False,False])
        mskR_nii = sitk.Flip(mskRorg_nii,[True,False,False])
        #endregion Split brains | labels - L|R

        #region Get ROI from L|R masks
        mskL_nii, bboxL = get_roi(mskL_nii, reserve=15)
        mskR_nii, bboxR = get_roi(mskR_nii, reserve=15)

        [x1_L, y1_L, z1_L, x2_L, y2_L, z2_L] = bboxL
        [x1_R, y1_R, z1_R, x2_R, y2_R, z2_R] = bboxR
        #endregion Get ROI from L|R masks

        #region Crop  L|R brains and masks for network 2
        imgL2_nii = imgL_nii[x1_L:x2_L, y1_L:y2_L, z1_L:z2_L]
        imgR2_nii = imgR_nii[x1_R:x2_R, y1_R:y2_R, z1_R:z2_R]
        
        mskL2_nii = mskL_nii[x1_L:x2_L, y1_L:y2_L, z1_L:z2_L]
        mskR2_nii = mskR_nii[x1_R:x2_R, y1_R:y2_R, z1_R:z2_R]
        #endregion Crop  L|R brains and masks for network 2

        #region Generate L | R masks for network 1
        mskL_nii[x1_L:x2_L, y1_L:y2_L, z1_L:z2_L] = 1
        mskR_nii[x1_R:x2_R, y1_R:y2_R, z1_R:z2_R] = 1
        #endregion Generate L|R masks for network 1

        #region Resample and save brains | labels for network 1
        imgL_rsmp_nii, mskL_rsmp_nii = resample_img(imgL_nii, mskL_nii, input_dim)
        imgR_rsmp_nii, mskR_rsmp_nii = resample_img(imgR_nii, mskR_nii, input_dim)

        sitk.WriteImage(imgL_rsmp_nii, os.path.join(datapath_net1,"brains", os.path.basename(imgFile_aug.replace("_t1","_t1_L"))))
        sitk.WriteImage(imgR_rsmp_nii, os.path.join(datapath_net1,"brains", os.path.basename(imgFile_aug.replace("_t1","_t1_R"))))
		
        sitk.WriteImage(mskL_rsmp_nii, os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile_aug.replace("_labels","_labels_L"))))
        sitk.WriteImage(mskR_rsmp_nii, os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile_aug.replace("_labels","_labels_R"))))
        #endregion Resample and save brains | labels for network 1

        #region Resample and save brains | labels for network 2
        imgL2_rsmp_nii, mskL2_rsmp_nii = resample_img(imgL2_nii, mskL2_nii, input_dim)
        imgR2_rsmp_nii, mskR2_rsmp_nii = resample_img(imgR2_nii, mskR2_nii, input_dim)
        
        sitk.WriteImage(imgL2_rsmp_nii, os.path.join(datapath_net2,"brains", os.path.basename(imgFile_aug.replace("_t1","_t1_L"))))
        sitk.WriteImage(imgR2_rsmp_nii, os.path.join(datapath_net2,"brains", os.path.basename(imgFile_aug.replace("_t1","_t1_R"))))
        
        sitk.WriteImage(mskL2_rsmp_nii, os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile_aug.replace("_labels","_labels_L"))))
        sitk.WriteImage(mskR2_rsmp_nii, os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile_aug.replace("_labels","_labels_R"))))
        #endregion Resample and save brains | labels for network 2

    create_jsonFile(data_path=settings.DATAPATH_INPUT)
    create_jsonFile(data_path=datapath_net1)
    create_jsonFile(data_path=datapath_net2)    