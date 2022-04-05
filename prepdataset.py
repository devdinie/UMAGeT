from locale import normalize
import os
import json
import settings
import numpy   as np
import nibabel as nib

import argparse  as args
import SimpleITK as sitk

from augment	   import augment_data
from createjson    import create_jsonFile
from preprocess    import resample_img, normalize_img, split_image, get_roi
from scipy.ndimage import rotate

def get_filelist(data_path):
	
    json_filename = os.path.join(data_path, "dataset.json")

    try:
	    with open(json_filename, "r") as fp:
		    experiment_data = json.load(fp)
    except IOError as e:
        print("File {} doesn't exist. It should be located in the directory named 'data/data_net1' ".format(json_filename))

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

def prepdata(data_path=settings.DATA_PATH, augmentation=settings.AUGMENT):
        
	input_dim=[settings.TILE_WIDTH,settings.TILE_HEIGHT,settings.TILE_DEPTH]
        
	create_jsonFile(data_path=data_path)
	
	data_filetype = settings.IMAGE_FILETYPE
	
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

	
	#region PREPROCESS ALL PREPARED INPUT DATA brains | labels 	
	filenames    = get_filelist(data_path=data_path)

	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]

		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue
		
		img_nii = sitk.ReadImage(imgFile, imageIO=data_filetype)
		msk_nii = sitk.ReadImage(mskFile, imageIO=data_filetype)

		imgFile_aug = os.path.basename(imgFile).replace("_t1"    , "_t1_"+"norm"+"-rC0-n0-d0-sp0-gh0")
		mskFile_aug = os.path.basename(mskFile).replace("_labels", "_labels_"+"norm"+"-rC0-n0-d0-sp0-gh0")
		
		sitk.WriteImage(img_nii, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgFile_aug))
		sitk.WriteImage(msk_nii, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskFile_aug))

		create_jsonFile(data_path=settings.DATAPATH_INPUT)

	if settings.AUGMENT:
		augment_data(data_path=settings.DATAPATH_INPUT)

	filenames_aug = get_filelist(data_path=settings.DATAPATH_INPUT)
		
	for idx in range(0,len(filenames_aug)):

		#region Read filenames and image - brains & labels
		imgFile_aug = filenames_aug[idx][0]
		mskFile_aug = filenames_aug[idx][1]
			
		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		imgaug_nii = sitk.ReadImage(imgFile, imageIO=data_filetype)
		mskaug_nii = sitk.ReadImage(mskFile, imageIO=data_filetype)
		#endregion Read filenames and image - brains & labels
		
		#region Normalize and Resample
		img_normalized_nii, msk_normalized_nii = normalize_img(imgaug_nii, mskaug_nii)
		img_resampled_nii , msk_resampled_nii  = resample_img(img_normalized_nii, msk_normalized_nii, input_dim)
		#endregion Normalize and Resample

		#region Write augmented and preprocessed full image - brains and labels
		sitk.WriteImage(img_resampled_nii, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgFile_aug))
		sitk.WriteImage(msk_resampled_nii, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskFile_aug))
		create_jsonFile(data_path=settings.DATAPATH_INPUT)
		#endregion Write augmented and preprocessed full image - brains and labels
		
		
		#region generate brains and labels for network 1 and 2
		mid_idx = np.around(input_dim[0]/2).astype(int)

		imgL_nii, imgRorg_nii, mskL_nii, mskRorg_nii = split_image(img_resampled_nii, msk_resampled_nii, mid_idx)

		imgR_nii = sitk.Flip(imgRorg_nii,[True,False,False])
		mskR_nii = sitk.Flip(mskRorg_nii,[True,False,False])

		mskL_nii, bboxL = get_roi(mskL_nii)
		mskR_nii, bboxR = get_roi(mskR_nii)

		[x1_L, y1_L, z1_L, x2_L, y2_L, z2_L] = bboxL
		[x1_R, y1_R, z1_R, x2_R, y2_R, z2_R] = bboxR
		
		imgL2_nii = imgL_nii[x1_L:x2_L, y1_L:y2_L, z1_L:z2_L]
		imgR2_nii = imgR_nii[x1_R:x2_R, y1_R:y2_R, z1_R:z2_R]

		mskL2_nii = mskL_nii[x1_L:x2_L, y1_L:y2_L, z1_L:z2_L]
		mskR2_nii = mskR_nii[x1_R:x2_R, y1_R:y2_R, z1_R:z2_R]

		mskL_nii[x1_L:x2_L, y1_L:y2_L, z1_L:z2_L] = 1
		mskR_nii[x1_R:x2_R, y1_R:y2_R, z1_R:z2_R] = 1

		imgL_resampled_nii, mskL_resampled_nii = resample_img(imgL_nii, mskL_nii, input_dim)
		imgR_resampled_nii, mskR_resampled_nii = resample_img(imgR_nii, mskR_nii, input_dim)

		imgL2_resampled_nii, mskL2_resampled_nii = resample_img(imgL2_nii, mskL2_nii, input_dim)
		imgR2_resampled_nii, mskR2_resampled_nii = resample_img(imgR2_nii, mskR2_nii, input_dim)
		
		#endregion generate brains and labels for network 1 and 2

		#region Write input data for network 1 - brains and labels
		sitk.WriteImage(imgL_resampled_nii, os.path.join(datapath_net1,"brains"       , os.path.basename(imgFile_aug.replace("_t1","_t1_L"))))
		sitk.WriteImage(imgR_resampled_nii, os.path.join(datapath_net1,"brains"       , os.path.basename(imgFile_aug.replace("_t1","_t1_R"))))
		
		sitk.WriteImage(mskL_resampled_nii, os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile_aug.replace("_labels","_labels_L"))))
		sitk.WriteImage(mskR_resampled_nii, os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile_aug.replace("_labels","_labels_R"))))
		
		#endregion WWrite input data for network 1 - brains and labels

		#region WWrite input data for network 2 - brains and labels
		sitk.WriteImage(imgL2_resampled_nii, os.path.join(datapath_net2,"brains"       , os.path.basename(imgFile_aug.replace("_t1","_t1_L"))))
		sitk.WriteImage(imgR2_resampled_nii, os.path.join(datapath_net2,"brains"       , os.path.basename(imgFile_aug.replace("_t1","_t1_R"))))
		
		sitk.WriteImage(mskL2_resampled_nii, os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile_aug.replace("_labels","_labels_L"))))
		sitk.WriteImage(mskR2_resampled_nii, os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile_aug.replace("_labels","_labels_R"))))

		#endregion WWrite input data for network 2 - brains and labels
		
	create_jsonFile(data_path=datapath_net1)
	create_jsonFile(data_path=datapath_net2)
	#endregion PREPROCESS ALL PREPARED INPUT DATA brains | labels
