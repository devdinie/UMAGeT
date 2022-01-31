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
from preprocess    import resample_img, normalize_img
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
    for idx in range(numFiles):
        filenames[idx] = [os.path.join(experiment_data["training"][idx]["image"]),
                          os.path.join(experiment_data["training"][idx]["label"])]
        
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
	filenames    = get_filelist(data_path=settings.DATA_PATH)

	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]

		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue
		
		img_nii = sitk.ReadImage(imgFile, imageIO=data_filetype)
		msk_nii = sitk.ReadImage(mskFile, imageIO=data_filetype)
		

		imgFile_aug = os.path.basename(imgFile).replace("_t1"    , "_t1_"+"norm"+"-rC0-n0-d0-gh0-sp0")
		mskFile_aug = os.path.basename(mskFile).replace("_labels", "_labels_"+"norm"+"-rC0-n0-d0-gh0-sp0")
		
		sitk.WriteImage(img_nii, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgFile_aug))
		sitk.WriteImage(msk_nii, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskFile_aug))

		create_jsonFile(data_path=settings.DATAPATH_INPUT)

	if settings.AUGMENT:
		print("YES! AUGMENT!!!")
		augment_data(data_path=settings.DATAPATH_INPUT)

	filenames_aug = get_filelist(data_path=settings.DATAPATH_INPUT)
		
	for idx in range(0,len(filenames_aug)):
			
		imgFile_aug = filenames_aug[idx][0]
		mskFile_aug = filenames_aug[idx][1]
			
		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		imgaug_nii = sitk.ReadImage(imgFile, imageIO=data_filetype)
		mskaug_nii = sitk.ReadImage(mskFile, imageIO=data_filetype)
		
		img_normalized_nii, msk_normalized_nii = normalize_img(imgaug_nii, mskaug_nii)
		img_resampled_nii, msk_resampled_nii   = resample_img(img_normalized_nii, msk_normalized_nii, input_dim)
		
		sitk.WriteImage(img_resampled_nii, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgFile_aug))
		sitk.WriteImage(img_resampled_nii, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskFile_aug))
		
		print(imgaug_nii.GetSize(),"|", img_resampled_nii.GetSize())
		print(mskaug_nii.GetSize(),"|", msk_resampled_nii.GetSize())
		
		create_jsonFile(data_path=settings.DATAPATH_INPUT)
	#endregion PREPROCESS ALL PREPARED INPUT DATA brains | labels