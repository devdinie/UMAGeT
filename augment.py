import os
import json
import math
from SimpleITK.SimpleITK import GetArrayFromImage
from matplotlib.pyplot import axes
from numpy.core.fromnumeric import reshape, shape
import scipy
import nilearn.masking
from scipy.ndimage.measurements import standard_deviation
import torchio
import skimage
import skimage.morphology
import settings

from settings   import DATA_PATH
from createjson import create_jsonFile

import numpy as np
import SimpleITK as sitk

aug_rotate    = True
aug_normalize = True
aug_noise     = True
aug_deform    = False
aug_ghost     = False
aug_spike     = False

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

def n4correction(img):
	
	img_corr = sitk.Cast(img, sitk.sitkFloat32)

	n4corrector     = sitk.N4BiasFieldCorrectionImageFilter()
	n4corrected_img = n4corrector.Execute(img_corr)
	
	return n4corrected_img

def rotate(img, msk,angle,axes):
	
	img_arr = sitk.GetArrayFromImage(img)
	msk_arr = sitk.GetArrayFromImage(msk)

	img_rot_arr = scipy.ndimage.rotate(img_arr, angle, axes=axes, reshape=False)
	msk_rot_arr = scipy.ndimage.rotate(msk_arr, angle, axes=axes, reshape=False)

	msk_rot_arr[msk_rot_arr > 0.5 ] =   1
	msk_rot_arr[msk_rot_arr <= 0.5] =   0

	img_rot = sitk.GetImageFromArray(img_rot_arr)
	msk_rot = sitk.GetImageFromArray(msk_rot_arr)

	img_rot.CopyInformation(img)
	msk_rot.CopyInformation(msk)
	
	
	dir = 'C' if angle >= 0 else 'A'

	return img_rot, msk_rot, dir
	
def add_noise(img,msk):
	
	img_arr   = sitk.GetArrayFromImage(img)
	
	noise_percentage = np.around(np.random.choice((0.025,0.050,0.075,0.100,0.125),1),2)[0]
	standard_dev     = np.max(img_arr)*noise_percentage

	shape_x = np.int(img_arr.shape[0])
	shape_y = np.int(img_arr.shape[1])
	shape_z = np.int(img_arr.shape[2])

	gaussian_re = np.random.normal(loc=0, scale=standard_dev, size=(shape_x, shape_y, shape_z))
	gaussian_im = np.random.normal(loc=0, scale=standard_dev, size=(shape_x, shape_y, shape_z))

	img_re = img_arr + gaussian_re
	img_im = gaussian_im

	img_wnoise_arr = np.sqrt(np.square(img_re) + np.square(img_im))

	img_wnoise = sitk.GetImageFromArray(img_wnoise_arr)
	img_wnoise.CopyInformation(img)
		
	return img_wnoise, msk
	
def elastic_deformation(img, msk):

	img_arr = sitk.GetArrayFromImage(img)
	msk_arr = sitk.GetArrayFromImage(msk)

	# Display grid to visualize deformation
	#N=25
	#white = np.max(img_arr)
	#img_arr[..., ::N, :, : ] = white
	#img_arr[..., :, ::N, : ] = white
	
	max_displacement = np.random.randint(10,20,3)
	random_elastic = torchio.RandomElasticDeformation(max_displacement=max_displacement)
	
	img_def_arr = random_elastic(np.expand_dims(img_arr,axis=0))
	msk_def_arr = random_elastic(np.expand_dims(msk_arr,axis=0))

	msk_def_arr[msk_def_arr >  np.min(msk_arr)] = 1
	msk_def_arr[msk_def_arr <= np.min(msk_arr)] = 0

	print(img_def_arr.shape,"|",msk_def_arr.shape)
	img_def = sitk.GetImageFromArray(np.squeeze(img_def_arr,axis=0))
	msk_def = sitk.GetImageFromArray(np.squeeze(msk_def_arr,axis=0))

	img_def.CopyInformation(img)
	msk_def.CopyInformation(msk)

	#print(img_def.GetSize(),"|", msk_def.GetSize())
	return img_def, msk_def

def add_ghosting(img, no_ghosts):

	add_ghosts = torchio.RandomGhosting(num_ghosts=np.int(no_ghosts), intensity=1) 
	img_ghosted = add_ghosts(img)

	return img_ghosted

def add_spiking(img):

	img_msk  = sitk.GetArrayFromImage(img)
	img_msk[img_msk>0] = 1

	spikes_list  = [(1, 1), (1, 2), (0, 1), (0, 2), (2, 2)]
	spikes_choice= np.random.choice(5,1) 

	add_spikes = torchio.RandomSpike(num_spikes=spikes_list[spikes_choice[0]])
	img_spiked = add_spikes(img)

	return img_spiked

def augment_data(data_path):

	print(data_path)

	data_path_aug = os.path.join(data_path,"data_aug")
	
	filenames     = get_filelist(data_path)
	no_filenames  = len(filenames)	
	
	#region Initialize AUGMENTATION | Rotation
	if(aug_rotate):
		all_axes    = [(1, 0), (1, 2), (0, 2)]
		axes        = np.random.choice(3,no_filenames) 
		
		angle_limit_neg = -8 
		angle_limit_pos =  8

		prob_div    = np.around(1/(angle_limit_pos - angle_limit_neg),3)
		angles_range= np.arange(angle_limit_neg, angle_limit_pos+1, 1)
		rot_prob    = [prob_div]*(angle_limit_pos+1 - angle_limit_neg) #scipy.stats.norm.pdf(angles_range,0,1)
		
		if sum(rot_prob) != 1:
			sub = 1 - sum(rot_prob)
			rot_prob[0] += sub/2
			rot_prob[len(rot_prob)-1] += sub/2
		
		angles =  np.random.choice(angles_range, no_filenames, p=rot_prob)
	#endregion Initialize AUGMENTATION | Rotation
        
	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]
		
		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		subject_id = os.path.basename(imgFile).split("_")[0]

		print(imgFile,"|",mskFile)
		
		#region READ brains | Labels : Raw data
		ext = os.path.basename(imgFile).split(".")[1]
		data_filetype = "MINCImageIO" if ext == "mnc" else "NiftiImageIO" if ext == "nii" else print("File format incompatible. Consider editing source code.")

		reader = sitk.ImageFileReader()
		reader.SetImageIO(data_filetype)
		reader.SetFileName(imgFile)
		img = reader.Execute()
		
		ext = os.path.basename(mskFile).split(".")[1]
		data_filetype = "MINCImageIO" if ext == "mnc" else "NiftiImageIO" if ext == "nii" else print("File format incompatible. Consider editing source code.")

		reader = sitk.ImageFileReader()
		reader.SetImageIO(data_filetype)
		reader.SetFileName(mskFile)
		msk = reader.Execute()
		#endregion READ brains | Labels : Raw data

		if not os.path.exists(data_path_aug):
			os.mkdir(data_path_aug)

		if not os.path.exists(os.path.join(data_path_aug,"brains")):
			os.mkdir(os.path.join(data_path_aug,"brains"))

		if not os.path.exists(os.path.join(data_path_aug,"target_labels")):
			os.mkdir(os.path.join(data_path_aug,"target_labels"))
		
		#region AUGMENTATION: NORMALIZE
		norm = 0 
		if aug_normalize:
			img_norm, msk_norm, norm = normalize(img, msk)
		else:
			img_norm = img
			msk_norm = msk
		#endregion AUGMENTATION: NORMALIZE

		#region WRITE brains | Labels : No Rotation, Normalized
		
		dir_ang = "C0"
		
		writer = sitk.ImageFileWriter()
		writer.SetImageIO("NiftiImageIO")
		writer.SetFileName(os.path.join(data_path_aug,"brains", subject_id+"_t1_"+"norm"+str(norm)+"-r"+dir_ang+"-n0-d0-gh0-sp0.nii"))
		writer.Execute(img_norm)

		writer = sitk.ImageFileWriter()
		writer.SetImageIO("NiftiImageIO")
		writer.SetFileName(os.path.join(data_path_aug,"target_labels", subject_id+"_labels_"+"norm"+str(norm)+"-r"+dir_ang+"-n0-d0-gh0-sp0.nii"))
		writer.Execute(msk_norm)
		#endregion WRITE brains | Labels : No Rotation, Normalized
		
		#region WRITE brains | Labels : Rotated, Normalized
		if aug_rotate and (np.abs(angles[idx]) != 0):
			img_rot, msk_rot, dir = rotate(img_norm, msk_norm,angles[idx],all_axes[axes[idx]])
			dir_ang = dir+str(np.abs(angles[idx]))

			writer = sitk.ImageFileWriter()
			writer.SetImageIO("NiftiImageIO")
			writer.SetFileName(os.path.join(data_path_aug,"brains", subject_id+"_t1_"+"norm"+str(norm)+"-r"+dir_ang+"-n0-d0-gh0-sp0.nii"))
			writer.Execute(img_rot)

			writer = sitk.ImageFileWriter()
			writer.SetImageIO("NiftiImageIO")
			writer.SetFileName(os.path.join(data_path_aug,"target_labels", subject_id+"_labels_"+"norm"+str(norm)+"-r"+dir_ang+"-n0-d0-gh0-sp0.nii"))
			writer.Execute(msk_rot)
		#endregion WRITE brains | Labels : Rotated, Normalized

	create_jsonFile(data_path=data_path_aug)

	filenames_aug    = get_filelist(data_path_aug)
	no_filenames_aug = len(filenames_aug)

	#region Initialize AUGMENTATION | Noise, Deformation, Ghosting, Spiking
	if aug_noise:
		noise   = np.random.randint(0,2,no_filenames_aug)
	else:
		noise   = np.zeros(no_filenames_aug, dtype=int)

	if aug_deform:
		edeform = np.random.choice([0,1]  , no_filenames_aug, p=[0.50,0.50])
	else:
		edeform = np.zeros(no_filenames_aug, dtype=int)

	if aug_ghost:
		ghosts  = np.random.choice([0,1,2], no_filenames_aug, p=[0.40,0.30,0.30])
	else:
		ghosts  = np.zeros(no_filenames_aug, dtype=int)

	if aug_spike:
		spikes  = np.random.choice([0,1]  , no_filenames_aug, p=[0.50,0.50])
	else:
		spikes  = np.zeros(no_filenames_aug, dtype=int)
	#endregion Initialize AUGMENTATION | Noise, Deformation, Ghosting, Spiking
	
	for idx in range(0,len(filenames_aug)):
		
		#region READ brains | Labels : rotated & normalized 
		imgFile_aug = filenames_aug[idx][0]
		mskFile_aug = filenames_aug[idx][1]
		
		if not (os.path.exists(imgFile_aug) or os.path.exists(mskFile_aug)):
			continue

		ext = os.path.basename(imgFile).split(".")[1]
		data_filetype = "MINCImageIO" if ext == "mnc" else "NiftiImageIO" if ext == "nii" else print("File format incompatible. Consider editing source code.")

		reader = sitk.ImageFileReader()
		reader.SetImageIO(data_filetype)
		reader.SetFileName(imgFile_aug)
		img_aug = reader.Execute()
		
		ext = os.path.basename(mskFile).split(".")[1]
		data_filetype = "MINCImageIO" if ext == "mnc" else "NiftiImageIO" if ext == "nii" else print("File format incompatible. Consider editing source code.")

		reader = sitk.ImageFileReader()
		reader.SetImageIO(data_filetype)
		reader.SetFileName(mskFile_aug)
		msk_aug = reader.Execute()

		img_aug_suff = os.path.splitext(imgFile_aug)[0]
		msk_aug_suff = os.path.splitext(mskFile_aug)[0]
		#endregion READ brains | Labels : rotated & normalized 

		#region AUGMENTATION: ADD NOISE
		if aug_noise and (noise[idx] == 1):
			img_wnoise, msk = add_noise(img_aug,msk_aug)
		else:
			img_wnoise = img_aug
			msk = msk_aug
		#endregion AUGMENTATION: ADD NOISE

		#region AUGMENTATION: DEFORMATION
		if aug_deform and (edeform[idx] == 1):
			img_edef, msk_edef = elastic_deformation(img_wnoise, msk)	
		else:
			img_edef = img_wnoise 
			msk_edef = msk
		#endregion AUGMENTATION: DEFORMATION
		
		#region AUGMENTATION: GHOSTING
		msk_ghosted = msk_edef
		if aug_ghost and (ghosts[idx] == 1):
			img_ghosted = add_ghosting(img_edef, ghosts[idx])
		else:
			img_ghosted = img_edef
		#endregion AUGMENTATION: GHOSTING

		#region AUGMENTATION: SPIKING
		msk_spiked = msk_ghosted
		if aug_spike and (spikes[idx] == 1):
			img_spiked = add_spiking(img_ghosted)
		else:
			img_spiked = img_ghosted
		#endregion AUGMENTATION: SPIKING

		#region WRITE brains | Labels : Noised, Deformed, Ghosted, Spiked
		writer = sitk.ImageFileWriter()
		writer.SetImageIO("NiftiImageIO")
		writer.SetFileName(imgFile_aug.replace("-n0-d0-gh0-sp0","-n"+str(noise[idx])+"-d"+str(edeform[idx])+"-gh"+str(ghosts[idx])+"-sp"+str(spikes[idx])))
		writer.Execute(img_spiked)

		writer = sitk.ImageFileWriter()
		writer.SetImageIO("NiftiImageIO")
		writer.SetFileName(mskFile_aug.replace("-n0-d0-gh0-sp0","-n"+str(noise[idx])+"-d"+str(edeform[idx])+"-gh"+str(ghosts[idx])+"-sp"+str(spikes[idx])))
		writer.Execute(msk_spiked)
		#endregion WRITE brains | Labels : Noised, Deformed, Ghosted, Spiked
	
	
	