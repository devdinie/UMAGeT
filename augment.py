import os
import json
import math
import scipy
import skimage
import settings
import torchio
import numpy     as np
import SimpleITK as sitk
import nilearn.masking
import skimage.morphology

from createjson import create_jsonFile
from preprocess import resample_img

aug_rotate    = True
aug_normalize = True
aug_noise     = True
aug_deform    = True
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
    
def rotate(img, msk,angle,axes):

	img_resampled, msk_resampled= resample_img(img,msk,msk.GetSize())

	img_arr = sitk.GetArrayFromImage(img_resampled)
	msk_arr = sitk.GetArrayFromImage(msk_resampled)

	if angle < 0 :
		angle = 360 + angle

	img_rot_arr = scipy.ndimage.rotate(img_arr, angle, axes=axes, reshape=True)
	msk_rot_arr = scipy.ndimage.rotate(msk_arr, angle, axes=axes, reshape=True)

	msk_rot_arr[msk_rot_arr > 0.5 ] =   1
	msk_rot_arr[msk_rot_arr <= 0.5] =   0

	img_rot = sitk.GetImageFromArray(img_rot_arr)
	msk_rot = sitk.GetImageFromArray(msk_rot_arr)
	
	dir = 'C' if angle <= 300 else 'A'

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

	return img_def, msk_def

def add_spiking(img):

	img_msk  = sitk.GetArrayFromImage(img)
	img_msk[img_msk>0] = 1

	spikes_list  = [(1, 1), (1, 2), (0, 1), (0, 2), (2, 2)]
	spikes_choice= np.random.choice(5,1) 

	add_spikes = torchio.RandomSpike(num_spikes=spikes_list[spikes_choice[0]])
	img_spiked = add_spikes(img)

	return img_spiked

def augment_data(data_path=settings.DATAPATH_INPUT):

	print(data_path)
	
	filenames     = get_filelist(data_path)
	no_filenames  = len(filenames)	

	#region AUGMENTATION | Rotation
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

	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]

		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		subject_id    = os.path.basename(imgFile).split("_")[0]
		data_filetype = "NiftiImageIO"

		img_nii = sitk.ReadImage(imgFile, imageIO=data_filetype)
		msk_nii = sitk.ReadImage(mskFile, imageIO=data_filetype)

		if aug_rotate and (np.abs(angles[idx]) != 0):
			
			img_aug, msk_aug, dir = rotate(img_nii, msk_nii, angles[idx], all_axes[axes[idx]])
			dir_ang = dir+str(np.abs(angles[idx]))

			imgaugFile = os.path.join(data_path,"brains"       , subject_id+"_t1_"+"norm"+"-r"+dir_ang+"-n0-d0-gh0-sp0.nii")
			mskaugFile = os.path.join(data_path,"target_labels", subject_id+"_labels_"+"norm"+"-r"+dir_ang+"-n0-d0-gh0-sp0.nii")

			sitk.WriteImage(img_aug, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgaugFile))
			sitk.WriteImage(msk_aug, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskaugFile))
	
	create_jsonFile(data_path=data_path)
	#endregion AUGMENTATION | Rotation

	
	#region Initialize AUGMENTATION | Noise, Deform
	filenames_rot    = get_filelist(data_path)
	no_filenames_rot = len(filenames_rot)

	if aug_noise:
		noise   = np.random.randint(0,2,no_filenames_rot)
	else:
		noise   = np.zeros(no_filenames_rot, dtype=int)

	if aug_deform:
		edeform = np.random.choice([0,1]  , no_filenames_rot, p=[0.50,0.50])
	else:
		edeform = np.zeros(no_filenames_rot, dtype=int)
	
	if aug_spike:
		spikes  = np.random.choice([0,1]  , no_filenames_rot, p=[0.50,0.50])
	else:
		spikes  = np.zeros(no_filenames_rot, dtype=int)
	#endregion Initialize AUGMENTATION | Noise, Deform

	for idx in range(0,len(filenames_rot)):

		imgFile_rot = filenames_rot[idx][0]
		mskFile_rot = filenames_rot[idx][1]

		if not (os.path.exists(imgFile_rot) or os.path.exists(mskFile_rot)):
			continue

		img_nii = sitk.ReadImage(imgFile_rot, imageIO=data_filetype)
		msk_nii = sitk.ReadImage(mskFile_rot, imageIO=data_filetype)

		#region augmentation - noise
		if aug_noise and (noise[idx] == 1):
			img_nii,  msk_nii = add_noise(img_nii,msk_nii)
		#endregion augmentation - noise
		

		#region augmentation - deformation
		if aug_deform and (edeform[idx] == 1):
			img_nii,  msk_nii = elastic_deformation(img_nii,  msk_nii)	
		#endregion augmentation - deformation

		"""
		#region augmentation - spiking
		if aug_spike and (spikes[idx] == 1):
			img_nii = add_spiking(img_nii)
		#endregion augmentation - spiking
		"""
		
		#region augmentation - write image
		#imgFile_rot.replace("-n0-d0-gh0-sp0","-n"+str(noise[idx])+"-d"+str(edeform[idx])+"-gh"+str(ghosts[idx])+"-sp"+str(spikes[idx]))
		
		img_filename = imgFile_rot.replace("-n0-d0", "-n"+str(noise[idx]) + "-d"+str(edeform[idx]))
		msk_filename = mskFile_rot.replace("-n0-d0", "-n"+str(noise[idx]) + "-d"+str(edeform[idx]))

		sitk.WriteImage(img_nii, os.path.join(settings.DATAPATH_INPUT,"brains"       , img_filename))
		sitk.WriteImage(msk_nii, os.path.join(settings.DATAPATH_INPUT,"target_labels", msk_filename))
		
		create_jsonFile(data_path=data_path)
		#endregion augmentation - write image

	
	


	
