import os
import json
import settings
import numpy as np

import argparse  as args
import SimpleITK as sitk

from createjson import create_jsonFile


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
	
	input_dim=np.array((settings.TILE_WIDTH,settings.TILE_HEIGHT,settings.TILE_DEPTH))
	
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

	ref_img_size = [settings.TILE_HEIGHT, settings.TILE_WIDTH, settings.TILE_DEPTH]
	mid_idx      = np.around(ref_img_size[0]/2).astype(int)

	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]

		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		#region READ brains | labels
		reader = sitk.ImageFileReader()
		reader.SetImageIO(data_filetype)
		reader.SetFileName(imgFile)
		img_nii = reader.Execute()

		reader = sitk.ImageFileReader()
		reader.SetImageIO(data_filetype)
		reader.SetFileName(mskFile)
		msk_nii = reader.Execute()
        	#endregion READ brains | labels	

		#region WRITE brains | labels
		writer = sitk.ImageFileWriter()
		writer.SetImageIO("NiftiImageIO")
		writer.SetFileName(os.path.join(settings.DATAPATH_INPUT,"brains" , os.path.basename(imgFile)))
		writer.Execute(img_nii)

		writer = sitk.ImageFileWriter()
		writer.SetImageIO("NiftiImageIO")
		writer.SetFileName(os.path.join(settings.DATAPATH_INPUT,"target_labels", os.path.basename(mskFile)))
		writer.Execute(img_nii)
		#endregion WRITE brains | labels
		