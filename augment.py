import os
import json
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
aug_normalize = False
aug_noise     = True
aug_deform    = True
aug_spike     = True
aug_ghost     = False

data_filetype = settings.IMAGE_FILETYPE #"NiftiImageIO"

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

def add_noise(img,msk, noise_perc):
	
	img_arr   = sitk.GetArrayFromImage(img)
	
	standard_dev     = np.max(img_arr)*noise_perc

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

def elastic_deformation(img, msk, num_ctrl_points, locked_borders, max_displacement):
	
	img_arr = sitk.GetArrayFromImage(img)
	msk_arr = sitk.GetArrayFromImage(msk)

	random_elastic = torchio.RandomElasticDeformation(num_control_points=num_ctrl_points,
							  locked_borders=locked_borders,
							   max_displacement=max_displacement)
				
	img_def_arr = random_elastic(np.expand_dims(img_arr,axis=0))
	msk_def_arr = random_elastic(np.expand_dims(msk_arr,axis=0))

	msk_def_arr[msk_def_arr >  np.min(msk_arr)] = 1
	msk_def_arr[msk_def_arr <= np.min(msk_arr)] = 0

	#print(img_def_arr.shape,"|",msk_def_arr.shape)

	img_def = sitk.GetImageFromArray(np.squeeze(img_def_arr,axis=0))
	msk_def = sitk.GetImageFromArray(np.squeeze(msk_def_arr,axis=0))

	img_def.CopyInformation(img)
	msk_def.CopyInformation(msk)

	return img_def, msk_def

def add_spiking(img, num_spikes, intensity):

	spikes = torchio.RandomSpike(num_spikes=num_spikes,intensity=intensity)
	img_spiked = spikes(img)

	return img_spiked

def augment_data(data_path=settings.DATAPATH_INPUT):

	print(data_path)
	
	filenames     = get_filelist(data_path)
	no_filenames  = len(filenames)	

	if(aug_rotate):

		angle_limit_neg = -15 
		angle_limit_pos =  15

		rot_inc  = 3
		all_axes = [(1, 0), (1, 2), (0, 2)]

	if(aug_noise):
		noise_perc_min  =  2
		noise_perc_max  = 25

		noise_perc_increment = 2

	if(aug_spike):
		no_spikes_arr  = [(1, 1), (1, 2), (0, 1), (0, 2)]
		intensity_arr  = [(1, 2), (1, 3), (1, 4)]

	if(aug_deform):
		
		maxdisplacement_min = 10 ; maxdisplacement_max = 13 ; displacement_inc = 1

		locked_borders   =  2
		ctrl_points_list = np.arange(maxdisplacement_min,maxdisplacement_max)
	
	print("Augmentation - adding augmentation types individually... ")
	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]

		if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
			continue

		subject_id    = os.path.basename(imgFile).split("_")[0]

		img_nii = sitk.ReadImage(imgFile, imageIO=data_filetype)
		msk_nii = sitk.ReadImage(mskFile, imageIO=data_filetype)

		#region AUGMENTATION | rotation only

		# If rotate is true, then for each angle within range, rotate and save
		
		if aug_rotate:
			for rot_angle in range(angle_limit_neg,angle_limit_pos+1,rot_inc):
				if rot_angle == 0:
					img_aug = img_nii
					msk_aug = msk_nii
					dir_ang = "C"+str(f'{np.abs(rot_angle):02}')

					imgaugFile = os.path.join(data_path,"brains"       , subject_id+"_t1_"+"norm"+"-r"+dir_ang+"-n00-d0-sp0000-gh0.nii")
					mskaugFile = os.path.join(data_path,"target_labels", subject_id+"_labels_"+"norm"+"-r"+dir_ang+"-n00-d0-sp0000-gh0.nii")
				else:
					for axes in all_axes:
						img_aug, msk_aug, dir = rotate(img_nii, msk_nii, rot_angle, axes)
						dir_ang = dir+str(f'{np.abs(rot_angle):02}')
						# print(dir_ang,"|",axes)

						imgaugFile = os.path.join(data_path,"brains"       , subject_id+"_t1_"+"norm"+"-r"+dir_ang+"-n00-d00-sp0000-gh0.nii")
						mskaugFile = os.path.join(data_path,"target_labels", subject_id+"_labels_"+"norm"+"-r"+dir_ang+"-n00-d00-sp0000-gh0.nii")

						sitk.WriteImage(img_aug, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgaugFile))
						sitk.WriteImage(msk_aug, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskaugFile))
		
		#endregion AUGMENTATION | rotation only

		#region AUGMENTATION | noise only
		
		if aug_noise:
			img_aug = img_nii
			msk_aug = msk_nii
			for noise_perc in range(noise_perc_min, noise_perc_max+1, noise_perc_increment):

				img_aug, msk_aug = add_noise(img_nii, msk_nii, noise_perc)
				
				imgaugFile = os.path.join(data_path,"brains"       , subject_id+"_t1_"+"norm"+"-rC00"+"-n"+str(f'{noise_perc:02}')+"-d00-sp0000-gh0.nii")
				mskaugFile = os.path.join(data_path,"target_labels", subject_id+"_labels_"+"norm"+"-rC00"+"-n"+str(f'{noise_perc:02}')+"-d00-sp0000-gh0.nii")

				sitk.WriteImage(img_aug, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgaugFile))
				sitk.WriteImage(msk_aug, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskaugFile))
		
		#endregion AUGMENTATION | noise only

		#region AUGMENTATION | spiking only
		
		if aug_spike:
			img_aug = img_nii
			msk_aug = msk_nii
			for no_spikes in no_spikes_arr:
				for intensity in intensity_arr:
					img_aug = add_spiking(img_nii, no_spikes,intensity)

					imgaugFile = os.path.join(data_path,"brains"       , subject_id+"_t1_"    +"norm-rC00-n00"+"-d00"+"-sp"+str(no_spikes[0])+str(no_spikes[1])+str(intensity[0])+str(intensity[1])+"-gh0.nii")
					mskaugFile = os.path.join(data_path,"target_labels", subject_id+"_labels_"+"norm-rC00-n00"+"-d00"+"-sp"+str(no_spikes[0])+str(no_spikes[1])+str(intensity[0])+str(intensity[1])+"-gh0.nii")

					sitk.WriteImage(img_aug, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgaugFile))
					sitk.WriteImage(msk_aug, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskaugFile))
		
		#endregion AUGMENTATION | spiking only
		
		#region AUGMENTATION | deformation only
		if aug_deform:

			img_aug = img_nii
			msk_aug = msk_nii

			img_arr = sitk.GetArrayFromImage(img_nii)
			msk_arr = sitk.GetArrayFromImage(msk_nii)
			
			for max_displacement in range(maxdisplacement_min, maxdisplacement_max+1, displacement_inc):
				
				num_ctrl_points = np.int(np.random.choice(ctrl_points_list))
				
				bounds = np.round(np.array(img_nii.GetSize()) * np.array(img_nii.GetSpacing()) + np.array((0.125,)*3),2)
				grid_spacing = bounds / np.subtract((num_ctrl_points,)*3,(locked_borders,)*3)

				dmax_total = round(np.sqrt(3*np.square(max_displacement)),2)
				
				# Sanity check
				#print(max_displacement,num_ctrl_points,np.round(np.array(grid_spacing)/2,2),(np.round(np.array(grid_spacing)/2,2)<dmax_total),np.all(np.round(np.array(grid_spacing)/2,2)<dmax_total),dmax_total)
				
				if (np.all(np.round(np.array(grid_spacing)/2,2) < dmax_total)):

					img_aug, msk_aug = elastic_deformation(img_aug, msk_aug, num_ctrl_points, locked_borders, max_displacement)

					img_aug.CopyInformation(img_nii)
					msk_aug.CopyInformation(msk_nii)

					imgaugFile = os.path.join(data_path,"brains"       , subject_id+"_t1_"    +"norm-rC00-n00"+"-d"+str(f'{max_displacement:02}')+"-sp0000"+"-gh0.nii")
					mskaugFile = os.path.join(data_path,"target_labels", subject_id+"_labels_"+"norm-rC00-n00"+"-d"+str(f'{max_displacement:02}')+"-sp0000"+"-gh0.nii")

					sitk.WriteImage(img_aug, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgaugFile))
					sitk.WriteImage(msk_aug, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskaugFile))
		#endregion AUGMENTATION | deformation only
	print("Augmentation - adding augmentation types individually complete.")
	create_jsonFile(data_path=data_path)
	

	#region Initialize AUGMENTATION | Combined Augmentatio 
	filenames_comb_aug    = get_filelist(data_path)
	no_filenames_comb_aug = len(filenames_comb_aug)

	if aug_noise:
		noise = np.random.choice([0,1], no_filenames_comb_aug, p=[0.30,0.70])
	else:
		noise = np.zeros(no_filenames_comb_aug, dtype=int)

	if aug_spike:
		spikes  = np.random.choice([0,1]  , no_filenames_comb_aug, p=[0.30,0.70])
	else:
		spikes  = np.zeros(no_filenames_comb_aug, dtype=int)

	if aug_deform:
		edeform = np.random.choice([0,1]  , no_filenames_comb_aug, p=[0.30,0.70])
	else:
		edeform = np.zeros(no_filenames_comb_aug, dtype=int)
	
	print("Augmentation - adding combined augmentation ... ")
	for idx in range(0,len(filenames_comb_aug)):

		imgFile_aug = filenames_comb_aug[idx][0]
		mskFile_aug = filenames_comb_aug[idx][1]

		subject_id    = os.path.basename(imgFile_aug).split("_")[0]

		[augname_norm, augname_r, augname_n, augname_d, augname_sp, augname_gh] = os.path.basename(imgFile_aug).split('_')[2].split('.')[0].split('-')
		aug_inFilename = [augname_norm, augname_r, augname_n, augname_d, augname_sp, augname_gh]

		#print(idx,":",os.path.basename(imgFile_aug),":",aug_inFilename, int(augname_n[1:3]), (not int(augname_n[1:3]) !=0))
		
		if not (os.path.exists(imgFile_aug) or os.path.exists(mskFile_aug)):
			continue

		img_aug = sitk.ReadImage(imgFile_aug, imageIO=data_filetype)
		msk_aug = sitk.ReadImage(mskFile_aug, imageIO=data_filetype)
		
		#region augmentation - noise
		if aug_noise and (not augname_n[1:3] !='00') and (noise[idx] == 1):

			noise_perc = np.random.choice(range(noise_perc_min,noise_perc_max+1))
			img_aug, msk_aug  = add_noise(img_aug, msk_aug, noise_perc)

			n_forFilename  = "n"+str(f'{noise_perc:02}')
			noise_combined = True
		else:
			n_forFilename = "n00"
			noise_combined = False
		#endregion augmentation - noise

		#region augmentation - spiking
		if aug_spike and (not augname_sp[2:6] != '0000') and (spikes[idx] == 1):

			no_spikes = no_spikes_arr[int(np.random.choice(len(no_spikes_arr),1))]
			intensity = intensity_arr[int(np.random.choice(len(intensity_arr),1))]
			
			img_aug = add_spiking(img_aug, no_spikes,intensity)

			sp_forFilename = "sp"+str(no_spikes[0])+str(no_spikes[1])+str(intensity[0])+str(intensity[1])
			spike_combined = True
		else:
			sp_forFilename = "sp0000"
			spike_combined = False
		#endregion augmentation - spiking

		
		#region augmentation - deformation
		if aug_deform and (not augname_d[1:3] != '00') and (edeform[idx] == 1):
			
			num_ctrl_points = np.int(np.random.choice(ctrl_points_list))
			max_displacement = np.random.choice(range(maxdisplacement_min,maxdisplacement_max+1))

			bounds = np.round(np.array(img_aug.GetSize()) * np.array(img_aug.GetSpacing()) + np.array((0.125,)*3),2)
			grid_spacing = bounds / np.subtract((num_ctrl_points,)*3,(locked_borders,)*3)
			
			dmax_total = round(np.sqrt(3*np.square(max_displacement)),2)
			
			if (np.all(np.round(np.array(grid_spacing)/2,2) < dmax_total)):
				img_aug, msk_aug = elastic_deformation(img_aug, msk_aug, num_ctrl_points, locked_borders, max_displacement)
			
			def_forFilename = "d"+str(f'{max_displacement:02}')
			def_combined = True
		else:
			def_forFilename = "d00"
			def_combined = False
		#endregion augmentation - deformation

		
		#region augmentation - write image
		imgaugFile = os.path.join(data_path,"brains"       , subject_id+"_t1_"    +augname_norm+"-"+augname_r+"-"+n_forFilename+"-"+def_forFilename+"-"+sp_forFilename+"-gh0.nii")
		mskaugFile = os.path.join(data_path,"target_labels", subject_id+"_labels_"+augname_norm+"-"+augname_r+"-"+n_forFilename+"-"+def_forFilename+"-"+sp_forFilename+"-gh0.nii")

		#print(os.path.basename(imgaugFile))

		if(noise_combined or spike_combined or def_combined):
			sitk.WriteImage(img_aug, os.path.join(settings.DATAPATH_INPUT,"brains"       , imgaugFile))
			sitk.WriteImage(msk_aug, os.path.join(settings.DATAPATH_INPUT,"target_labels", mskaugFile))
		
		#endregion augmentation - write image
	print("Augmentation - adding combined augmentation complete.")
	create_jsonFile(data_path=data_path)