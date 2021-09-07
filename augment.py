import os
import json
import settings

from settings import DATA_PATH


import numpy as np
import SimpleITK as sitk

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

def n4correction(img):
	
	img_corr = sitk.Cast(img, sitk.sitkFloat32)

	n4corrector     = sitk.N4BiasFieldCorrectionImageFilter()
	n4corrected_img = n4corrector.Execute(img_corr)
	
	return n4corrected_img

def rotation(ang_n,dir,img,msk):
	if dir == "CW":
		img = np.rot
	

def gaussian_noise(img):
	noise = np.random.normal(0, .1, img.shape)
	img_wnoise = img + noise

	return sitk.GetImageFromArray(img_wnoise)

def augment_data(data_path):

	print(data_path)

	filenames = get_filelist(data_path)

	no_filenames = len(filenames)
	
	angles    = np.random.choice(11, no_filenames, p=[0.6, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01])
	noise_mat = np.random.choice([0,1], no_filenames, p=[0.6, 0.4])

	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]
		
		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		img = sitk.ReadImage(imgFile, imageIO="NiftiImageIO")
		msk = sitk.ReadImage(mskFile, imageIO="NiftiImageIO")

		msk_thresholded = sitk.GetArrayFromImage(msk)
		msk_thresholded[msk_thresholded > 0.5 ] = 1
		msk_thresholded[msk_thresholded <= 0.5] = 0
		msk = sitk.GetImageFromArray(msk_thresholded)
		
		if not os.path.exists(os.path.join(data_path,"brains_aug")):
			os.mkdir(os.path.join(data_path,"brains_aug"))

		if not os.path.exists(os.path.join(data_path,"target_labels_aug")):
			os.mkdir(os.path.join(data_path,"target_labels_aug"))

		img_n4corr = n4correction(img)

		
		ang =0
		if noise_mat[idx] == 1:
			img_wnoise = gaussian_noise(sitk.GetArrayFromImage(img_n4corr))
			gnoise=1
		else:
			img_wnoise = img_n4corr
			gnoise=0

		sitk.WriteImage(img_n4corr, os.path.join(data_path,"brains_aug" , os.path.basename(imgFile).split(".")[0]+"_t1_n4corrG0r00.nii"))
		sitk.WriteImage(msk, os.path.join(data_path,"target_labels_aug", os.path.basename(mskFile).split(".")[0]+"_labels_n4corrG0r00.nii"))
		

		if angles[idx] > 0 :
			rotation(angles[idx],sitk.GetArrayFromImage(img_wnoise),sitk.GetArrayFromImage(msk))
		

		#img_rot, msk_rot = rotation(sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(msk))
		
		#sitk.WriteImage(img_n4corr, os.path.join(data_path,"brains_aug" , os.path.basename(imgFile).split(".")[0]+"n4"+".nii"))
		#sitk.WriteImage(msk, os.path.join(data_path,"target_labels_aug", os.path.basename(mskFile).split(".")[0]+"n4"+".nii"))

	#return sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(msk)
