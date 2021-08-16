import os
import json
import settings

import scipy
import numpy   as np
import nibabel as nib

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
    
    reserve = 2
    X, Y, Z = msk.shape[:3]

    idx = np.nonzero(msk > 0)

    x1, x2 = idx[0].min() - reserve, idx[0].max() + reserve + 1
    y1, y2 = idx[1].min() - reserve, idx[1].max() + reserve + 1
    z1, z2 = idx[2].min() - reserve, idx[2].max() + reserve + 1

    x1, x2 = max(x1, 0), min(x2, X)
    y1, y2 = max(y1, 0), min(y2, Y)
    z1, z2 = max(z1, 0), min(z2, Z)

    bbox = [x1, y1, z1, x2, y2, z2]

    return bbox


def prepdata(data_path=settings.DATA_PATH ):
        
    create_jsonFile(data_path=data_path)

    datapath_net1 = os.path.join(data_path,"data_net1")
    datapath_net2 = os.path.join(data_path,"data_net2")

    filenames    = get_filelist(datapath_net1,datapath_net2)

    for idx in range(0,len(filenames)):

        imgFile = filenames[idx][0]
        mskFile = filenames[idx][1]

        #print(imgFile,"|",mskFile)

        img = nib.load(imgFile).get_fdata()
        msk = nib.load(mskFile).get_fdata()
        
        img_aff = nib.load(imgFile).get_affine()
        msk_aff = nib.load(mskFile).get_affine()

        mid = int(img.shape[0]/2)

        if img.shape == msk.shape:

            img_L = img[0:mid,:,:]
            img_R = nib.orientations.flip_axis(img[mid:mid*2,:,:],axis=0)

            msk_L = msk[0:mid,:,:] 
            msk_R = nib.orientations.flip_axis(msk[mid:mid*2,:,:],axis=0)

            # LEFT: Brain and label prep network 1
        
            [x1, y1, z1, x2, y2, z2] = get_roi(img_L, msk_L)

            roi_patch = np.ones((x2-x1, y2-y1,z2-z1))

            img_net2L = np.zeros((x2-x1, y2-y1,z2-z1))
            msk_net2L = np.zeros((x2-x1, y2-y1,z2-z1))

            img_net2L = img_L[x1:x2,y1:y2,z1:z2]
            msk_net2L = msk_L[x1:x2,y1:y2,z1:z2]

            img_net2Lnii = nib.Nifti1Image(img_net2L,np.eye(4))
            msk_net2Lnii = nib.Nifti1Image(msk_net2L,np.eye(4))

            nib.save(img_net2Lnii, os.path.join(datapath_net2,"brains" , os.path.basename(imgFile).split(".")[0]+"_L"))
            nib.save(msk_net2Lnii, os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile).split(".")[0]+"_L"))

            msk_net1L = np.zeros(msk_L.shape)
            msk_net1L[:,:,:] = msk_L
            msk_net1L[x1:x2,y1:y2,z1:z2] = roi_patch

            img_net1Lnii = nib.Nifti1Image(img_L,img_aff)
            msk_net1Lnii = nib.Nifti1Image(msk_net1L,msk_aff)

            nib.save(img_net1Lnii, os.path.join(datapath_net1,"brains" , os.path.basename(imgFile).split(".")[0]+"_L"))
            nib.save(msk_net1Lnii, os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile).split(".")[0]+"_L"))
            
            
            # RIGHT: Brain and label prep network 1
            
            [x1, y1, z1, x2, y2, z2] = get_roi(img_R, msk_R)

            roi_patch = np.ones((x2-x1, y2-y1,z2-z1))

            img_net2R = np.zeros((x2-x1, y2-y1,z2-z1))
            msk_net2R = np.zeros((x2-x1, y2-y1,z2-z1))

            img_net2R = img_R[x1:x2,y1:y2,z1:z2]
            msk_net2R = msk_R[x1:x2,y1:y2,z1:z2]

            img_net2Rnii = nib.Nifti1Image(img_net2R,np.eye(4))
            msk_net2Rnii = nib.Nifti1Image(msk_net2R,np.eye(4))

            nib.save(img_net2Rnii, os.path.join(datapath_net2,"brains" , os.path.basename(imgFile).split(".")[0]+"_R"))
            nib.save(msk_net2Rnii, os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile).split(".")[0]+"_R"))

            msk_net1R = np.zeros(msk_R.shape)
            msk_net1R[:,:,:] = msk_R
            msk_net1R[x1:x2,y1:y2,z1:z2] = roi_patch

            img_net1Rnii = nib.Nifti1Image(img_R ,img_aff)
            msk_net1Rnii = nib.Nifti1Image(msk_net1R,msk_aff)

            nib.save(img_net1Rnii, os.path.join(datapath_net1,"brains" , os.path.basename(imgFile).split(".")[0]+"_R"))
            nib.save(msk_net1Rnii, os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile).split(".")[0]+"_R"))

    create_jsonFile(data_path=datapath_net1)
    create_jsonFile(data_path=datapath_net2)