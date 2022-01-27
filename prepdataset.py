from locale import normalize
import os
import json
import settings
import numpy   as np
import nibabel as nib

import argparse  as args
import SimpleITK as sitk

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

	filenames    = get_filelist(data_path=settings.DATA_PATH)

	#ref_img_size = [settings.TILE_HEIGHT, settings.TILE_WIDTH, settings.TILE_DEPTH]
	#mid_idx      = np.around(ref_img_size[0]/2).astype(int)

	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]

		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		#region PREPROCESS INPUT DATA brains | labels
		
		img_nii = sitk.ReadImage(imgFile, imageIO=data_filetype)
		msk_nii = sitk.ReadImage(mskFile, imageIO=data_filetype)

		img_resampled_nii, msk_resampled_nii   = resample_img(img_nii, msk_nii, input_dim)
		img_normalized_nii, msk_normalized_nii = normalize_img(img_resampled_nii, msk_resampled_nii)

		#endregion PREPROCESS INPUT DATA brains | labels
		
		print(img_nii.GetSize(),"|", img_normalized_nii.GetSize())
		print(msk_nii.GetSize(),"|", msk_normalized_nii.GetSize())

		sitk.WriteImage(img_normalized_nii, os.path.join(settings.DATAPATH_INPUT,"brains"       , os.path.basename(imgFile)))
		sitk.WriteImage(msk_normalized_nii, os.path.join(settings.DATAPATH_INPUT,"target_labels", os.path.basename(mskFile)))

		create_jsonFile(data_path=settings.DATAPATH_INPUT)