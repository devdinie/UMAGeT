import os
import json
import scipy
import settings

from settings   import DATA_PATH
from createjson import create_jsonFile


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

def normalize(img,msk):

	img_arr  = sitk.GetArrayFromImage(img)
	img_norm_arr = (img_arr - np.min(img_arr))/(np.max(img_arr) - np.min(img_arr))
	img_norm = sitk.GetImageFromArray(img_norm_arr)

	msk_arr = sitk.GetArrayFromImage(msk)
	msk_arr[msk_arr > 0.5 ] = 1
	msk_arr[msk_arr <= 0.5] = 0
	msk_norm = sitk.GetImageFromArray(msk_arr)

	norm = 1
	return img_norm, msk_norm, norm

def n4correction(img):
	
	img_corr = sitk.Cast(img, sitk.sitkFloat32)

	n4corrector     = sitk.N4BiasFieldCorrectionImageFilter()
	n4corrected_img = n4corrector.Execute(img_corr)
	
	return n4corrected_img

def rotate(subject_id,img, msk,data_path,angle,axes,norm):
	
	img_rot = scipy.ndimage.rotate(img, angle, axes=axes)
	msk_rot = scipy.ndimage.rotate(msk, angle, axes=axes)
	
	sitk.WriteImage(sitk.GetImageFromArray(img_rot), os.path.join(data_path,"brains"       , subject_id+"_t1_"    +str(norm)+"r"+str(angle)+".nii"))
	sitk.WriteImage(sitk.GetImageFromArray(msk_rot), os.path.join(data_path,"target_labels", subject_id+"_labels_"+str(norm)+"r"+str(angle)+".nii"))
	

def add_noise(data_path,img,msk,sigma_g, sigma_r,img_suff,msk_suff):

	size = msk.shape

	noise_rician   = scipy.stats.rice.rvs(1, sigma_r, size=size)
	noise_gaussian =    np.random.normal( 0, sigma_g, size=size)

	img_wnoise = img + noise_gaussian + noise_rician
	
	sitk.WriteImage(sitk.GetImageFromArray(img_wnoise), os.path.join(img_suff,"n.nii"))
	sitk.WriteImage(sitk.GetImageFromArray(msk) 	  , os.path.join(msk_suff,"n.nii"))
	


def augment_data(data_path):

	print(data_path)

	filenames = get_filelist(data_path)

	no_filenames = len(filenames)	
	
	all_axes    = all_axes = [(1, 0), (1, 2), (0, 2)]
	axes        = np.random.choice(3,no_filenames) 
	
	angles_range=  np.arange(-5, 6, 1)
	rot_prob    =  scipy.stats.norm.pdf(angles_range,0,1)
	angles      =  np.random.choice(angles_range, no_filenames, p=rot_prob)

	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]
		
		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		subject_id = os.path.basename(imgFile).split("_")[0]

		img = sitk.ReadImage(imgFile, imageIO="NiftiImageIO")
		msk = sitk.ReadImage(mskFile, imageIO="NiftiImageIO")

		data_path_aug = os.path.join(data_path,"data_aug")
		if not os.path.exists(data_path_aug):
			os.mkdir(data_path_aug)

		if not os.path.exists(os.path.join(data_path_aug,"brains")):
			os.mkdir(os.path.join(data_path_aug,"brains"))

		if not os.path.exists(os.path.join(data_path_aug,"target_labels")):
			os.mkdir(os.path.join(data_path_aug,"target_labels"))
		
		norm = 0 
		img_norm, msk_norm, norm = normalize(img, msk)

		rotate(subject_id,sitk.GetArrayFromImage(img_norm), sitk.GetArrayFromImage(msk_norm),data_path_aug,angles[idx],all_axes[axes[idx]],norm)
	
	data_path_aug = os.path.join(data_path,"data_aug")
	create_jsonFile(data_path=data_path_aug)

	filenames_aug    = get_filelist(data_path_aug)
	no_filenames_aug = len(filenames_aug)

	sigma_g = np.random.choice(np.arange(0,2.5,0.5),no_filenames_aug)
	sigma_r = np.random.choice(np.arange(0,2.5,0.5),no_filenames_aug)

	for idx in range(0,len(filenames_aug)):
		
		imgFile_aug = filenames_aug[idx][0]
		mskFile_aug = filenames_aug[idx][1]
		
		if not (os.path.exists(imgFile_aug) or os.path.exists(mskFile_aug)):
			continue

		img_aug_suff = os.path.splitext(imgFile_aug)[0]
		msk_aug_suff = os.path.splitext(mskFile_aug)[0]
		add_noise(sitk.GetArrayFromImage(imgFile_aug),sitk.GetArrayFromImage(mskFile_aug),sigma_g, sigma_r,img_aug_suff,msk_aug_suff)
	
	