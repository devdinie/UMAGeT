import os
import json
import settings

import scipy
import numpy     as np
import nibabel   as nib
import SimpleITK as sitk

from augment   import augment_data
from argparser import args
from skimage.transform import resize
from createjson import create_jsonFile

def get_filelist(datapath_net1,datapath_net2):

    if not os.path.exists(datapath_net1):
        os.mkdir(datapath_net1)
        os.mkdir(os.path.join(datapath_net1,"brains"))
        os.mkdir(os.path.join(datapath_net1,"target_labels"))

    if not os.path.exists(datapath_net2):
        os.mkdir(datapath_net2)
        os.mkdir(os.path.join(datapath_net2,"brains"))
        os.mkdir(os.path.join(datapath_net2,"target_labels"))

    json_filename = os.path.join(settings.DATA_PATH, "dataset.json")

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

def get_roi(img, msk):
    
    reserve = int(2)
    X, Y, Z = msk.shape[:3]

    idx = np.nonzero(msk > 0)
    
    x1 = max(idx[0].min() - reserve, 0)
    x2 = min(idx[0].max() + reserve, X)
    
    y1 = max(idx[1].min() - reserve, 0)
    y2 = min(idx[1].max() + reserve, Y)
    
    z1 = max(idx[2].min() - reserve, 0)
    z2 = min(idx[2].max() + reserve, Z)

    bbox = [x1, y1, z1, x2, y2, z2]
    
    return bbox


def prepdata(data_path=settings.DATA_PATH, augmentation=settings.AUGMENT):

    input_dim=np.array((args.tile_width,args.tile_height,args.tile_depth))

    create_jsonFile(data_path=data_path)

    if augmentation:
        augment_data(data_path)
        
        datapath_net1 = os.path.join(settings.DATA_PATH_AUG,"data_net1")
        datapath_net2 = os.path.join(settings.DATA_PATH_AUG,"data_net2")
    
    else:
        datapath_net1 = os.path.join(data_path,"data_net1")
        datapath_net2 = os.path.join(data_path,"data_net2")

    filenames    = get_filelist(datapath_net1,datapath_net2)
    
    for idx in range(0,len(filenames)):

        imgFile = filenames[idx][0]
        mskFile = filenames[idx][1]

        #print("filename: ",imgFile," | ",mskFile)

        if not (os.path.exists(imgFile) or os.path.exists(mskFile)):
            continue

        img = sitk.ReadImage(imgFile, imageIO="NiftiImageIO")
        msk = sitk.ReadImage(mskFile, imageIO="NiftiImageIO")

        #msk_thresholded = sitk.GetArrayFromImage(msk)	
        #msk_thresholded[msk_thresholded > 0.5 ] = 1
        #msk_thresholded[msk_thresholded <= 0.5] = 0
        #msk = sitk.GetImageFromArray(msk_thresholded)

        #print("msk count: ", np.count_nonzero(msk))

        mid = int(img.shape[2]/2)
        print("shape :", img.shape , "| mid :", mid)
    
        if img.shape == msk.shape:

            img_L = img[:,:,0:mid]
            img_R = np.flip(img[:,:,mid:mid*2],axis=2)

            msk_L = msk[:,:,0:mid] 
            msk_R = np.flip(msk[:,:,mid:mid*2],axis=2)

        
            # Sanity checks
            print("img shape: ", img_L.shape,"|",img_R.shape)
            print("msk shape: ", msk_L.shape,"|",msk_R.shape)
            print("msk count L: ", np.count_nonzero(msk_L),"|",np.count_nonzero(msk_L==1))
            print("msk count R: ", np.count_nonzero(msk_R),"|",np.count_nonzero(msk_R==1))
            

            # region LEFT: Brain and label prep
            
            [x1, y1, z1, x2, y2, z2] = get_roi(img_L, msk_L)

            roi_patch = np.ones((x2-x1, y2-y1,z2-z1))

            img_net2L = np.zeros((x2-x1, y2-y1,z2-z1))
            msk_net2L = np.zeros((x2-x1, y2-y1,z2-z1))

            img_net2L = img_L[x1:x2,y1:y2,z1:z2]
            msk_net2L = msk_L[x1:x2,y1:y2,z1:z2]

            img_net2Lnii = sitk.GetImageFromArray(resize(img_net2L,input_dim))
            msk_net2Lnii = sitk.GetImageFromArray(resize(msk_net2L,input_dim))

            sitk.WriteImage(img_net2Lnii, os.path.join(datapath_net2,"brains" , os.path.basename(imgFile).split(".")[0]+"_L.nii"))
            sitk.WriteImage(msk_net2Lnii, os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile).split(".")[0]+"_L.nii"))
            
            msk_net1L = np.zeros(msk_L.shape)
            msk_net1L[:,:,:] = msk_L
            msk_net1L[x1:x2,y1:y2,z1:z2] = roi_patch
            #print("msk count L patch: ", np.count_nonzero(msk_net1L),"|",np.count_nonzero(msk_net1L==1))

            img_net1Lnii = sitk.GetImageFromArray(resize(img_L,input_dim))
            msk_net1Lnii = sitk.GetImageFromArray(resize(msk_net1L,input_dim))

            sitk.WriteImage(img_net1Lnii, os.path.join(datapath_net1,"brains" , os.path.basename(imgFile).split(".")[0]+"_L.nii"))
            sitk.WriteImage(msk_net1Lnii, os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile).split(".")[0]+"_L.nii")) 
            
            # endregion LEFT: Brain and label prep

            # region RIGHT: Brain and label prep

            
            [x1, y1, z1, x2, y2, z2] = get_roi(img_R, msk_R)

            roi_patch = np.ones((x2-x1, y2-y1,z2-z1))

            img_net2R = np.zeros((x2-x1, y2-y1,z2-z1))
            msk_net2R = np.zeros((x2-x1, y2-y1,z2-z1))

            img_net2R  = img_R[x1:x2,y1:y2,z1:z2]
            msk_net2R  = msk_R[x1:x2,y1:y2,z1:z2]

            img_net2Rnii = sitk.GetImageFromArray(resize(img_net2R,input_dim))
            msk_net2Rnii = sitk.GetImageFromArray(resize(msk_net2R,input_dim))
         
            sitk.WriteImage(img_net2Rnii, os.path.join(datapath_net2,"brains" , os.path.basename(imgFile).split(".")[0]+"_R.nii"))
            sitk.WriteImage(msk_net2Rnii, os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile).split(".")[0]+"_R.nii"))

            
            msk_net1R = np.zeros(msk_R.shape)
            msk_net1R[:,:,:] = msk_R
            msk_net1R[x1:x2,y1:y2,z1:z2] = roi_patch
            #print("msk count R patch: ", np.count_nonzero(msk_net1R),"|",np.count_nonzero(msk_net1R==1))

            img_net1Rnii = sitk.GetImageFromArray(resize(img_R,input_dim))
            msk_net1Rnii = sitk.GetImageFromArray(resize(msk_net1R,input_dim))

            sitk.WriteImage(img_net1Rnii, os.path.join(datapath_net1,"brains" , os.path.basename(imgFile).split(".")[0]+"_R.nii"))
            sitk.WriteImage(msk_net1Rnii, os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile).split(".")[0]+"_R.nii"))
            
            # endregion RIGHT: Brain and label prep
            

    create_jsonFile(data_path=datapath_net1)
    create_jsonFile(data_path=datapath_net2)
