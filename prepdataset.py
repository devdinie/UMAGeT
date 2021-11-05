import os
import json

from numpy.core.records import array
import settings

import scipy
import numpy     as np
import nibabel   as nib
import SimpleITK as sitk

from augment   import augment_data
from argparser import args
from skimage.transform import resize
from createjson import create_jsonFile

def get_filelist(datapath_net1,datapath_net2, augmentation_done=False):

    if not os.path.exists(datapath_net1):
        os.mkdir(datapath_net1)
        os.mkdir(os.path.join(datapath_net1,"brains"))
        os.mkdir(os.path.join(datapath_net1,"target_labels"))

    if not os.path.exists(datapath_net2):
        os.mkdir(datapath_net2)
        os.mkdir(os.path.join(datapath_net2,"brains"))
        os.mkdir(os.path.join(datapath_net2,"target_labels"))
        
    if augmentation_done:
        json_filename = os.path.join(settings.DATA_PATH_AUG, "dataset.json")
    else:
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

def get_roi(msk):
    
    msk_arr = sitk.GetArrayFromImage(msk)

    reserve = int(2)
    X, Y, Z = msk_arr.shape[:3]

    idx = np.nonzero(msk_arr > 0)

    x1_idx = max(idx[0].min() - reserve, 0)
    x2_idx = min(idx[0].max() + reserve, X)
    
    y1_idx = max(idx[1].min() - reserve, 0)
    y2_idx = min(idx[1].max() + reserve, Y)
    
    z1_idx = max(idx[2].min() - reserve, 0)
    z2_idx = min(idx[2].max() + reserve, Z)

    lower_idx = np.array([x1_idx,y1_idx,z1_idx], dtype='int')
    upper_idx = np.array([x2_idx,y2_idx,z2_idx], dtype='int')
    
    #bbox = [x1, y1, z1, x2, y2, z2]
    bbox = [lower_idx[0], lower_idx[1], lower_idx[2], upper_idx[0], upper_idx[1], upper_idx[2]]
    
    return bbox


def prepdata(data_path=settings.DATA_PATH, augmentation=settings.AUGMENT, augmentation_done=False):

    input_dim=np.array((args.tile_width,args.tile_height,args.tile_depth))

    create_jsonFile(data_path=data_path)

    if augmentation:
        #augment_data(data_path)
        create_jsonFile(data_path=settings.DATA_PATH_AUG)

        augmentation_done = True
    
    datapath_net1 = os.path.join(data_path,"data_net1")
    datapath_net2 = os.path.join(data_path,"data_net2")

    data_filetype = "NiftiImageIO"
    #TODO: Change to read filetype later

    filenames    = get_filelist(datapath_net1,datapath_net2, augmentation_done=augmentation_done)

    ref_img_size = [settings.TILE_HEIGHT, settings.TILE_WIDTH, settings.TILE_DEPTH]
    mid_idx      = np.around(ref_img_size[0]/2).astype(int)

    for idx in range(0,len(filenames)):

        imgFile = filenames[idx][0]
        mskFile = filenames[idx][1]

        print("filename: ",imgFile," | ",mskFile)

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

        #region Resample image
        ref_img = sitk.Image(ref_img_size, img_nii.GetPixelIDValue())
        ref_img.SetOrigin(img_nii.GetOrigin())
        ref_img.SetDirection(img_nii.GetDirection())
        ref_img.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(ref_img_size, img_nii.GetSize(), img_nii.GetSpacing())])

        img = sitk.Resample(img_nii, ref_img)
        #endregion Resample image

        #region Resample mask
        ref_msk = sitk.Image(ref_img_size, msk_nii.GetPixelIDValue())
        ref_msk.SetOrigin(msk_nii.GetOrigin())
        ref_msk.SetDirection(msk_nii.GetDirection())
        ref_msk.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(ref_img_size, msk_nii.GetSize(), msk_nii.GetSpacing())])

        msk_resampled = sitk.Resample(msk_nii, ref_img)

        msk_arr = sitk.GetArrayFromImage(msk_resampled)	
        msk_arr[msk_arr > 0.5 ] = 1
        msk_arr[msk_arr <= 0.5] = 0
        msk = sitk.GetImageFromArray(msk_arr)
        msk.CopyInformation(msk_resampled)
        #endregion Resample mask

        #print("***",mid_idx,"|", img.GetSize(), "|", msk.GetSize())

        
        img_L = img[0 : mid_idx      , 0:img.GetSize()[1],0:img.GetSize()[2]]
        img_R = img[mid_idx:mid_idx*2, 0:img.GetSize()[1],0:img.GetSize()[2]]

        msk_L = msk[0      : mid_idx ,0:msk.GetSize()[1],0:msk.GetSize()[2]]
        msk_R = msk[mid_idx:mid_idx*2,0:msk.GetSize()[1],0:msk.GetSize()[2]]

        
        # region LEFT: Brain and label prep

        #region GENERATE brains | labels for NET 1

        if not (np.count_nonzero(sitk.GetArrayFromImage(msk_L))):
            continue

        [x1, y1, z1, x2, y2, z2] = get_roi(msk_L)
        
        imgL_lowerBoundary = img_L.TransformPhysicalPointToIndex(msk_L.TransformIndexToPhysicalPoint(np.array([x1, y1, z1]).tolist()))
        imgL_upperBoundary = img_L.TransformPhysicalPointToIndex(msk_L.TransformIndexToPhysicalPoint(np.array([x2, y2, z2]).tolist()))


        img_net1Lnii = img_L
        img_net1Lnii.CopyInformation(img_L)
        
        msk_net1Larr = sitk.GetArrayFromImage(msk_L)
        msk_net1Larr[x1:x2,y1:y2,z1:z2] = 1

        msk_net1Lnii = sitk.GetImageFromArray(msk_net1Larr)
        msk_net1Lnii.CopyInformation(msk_L)
        #endregion GENERATE brains | labels for NET 1

        #region WRITE LEFT brains | labels for NET 1
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(datapath_net1,"brains" , os.path.basename(imgFile).split(".")[0]+"_L.nii"))
        writer.Execute(img_net1Lnii)

        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile).split(".")[0]+"_L.nii"))
        writer.Execute(msk_net1Lnii)
        #endregion WRITE LEFT brains | labels for NET 1
        
        #region GENERATE LEFT brains | labels for NET 2
        if ((imgL_upperBoundary[0]-imgL_lowerBoundary[0]==0) or (imgL_upperBoundary[1]-imgL_lowerBoundary[1]==0) or (imgL_upperBoundary[2]-imgL_lowerBoundary[2]==0)):
            continue

        imgL_lowerBoundary_list = list(imgL_lowerBoundary)
        imgL_upperBoundary_list = list(imgL_upperBoundary)

        imgL_lowerBoundary_list[0] = max(imgL_lowerBoundary[0], 0)
        imgL_lowerBoundary_list[1] = max(imgL_lowerBoundary[1], 0)
        imgL_lowerBoundary_list[2] = max(imgL_lowerBoundary[2], 0)

        imgL_upperBoundary_list[0] = min(imgL_upperBoundary[0], img_L.GetSize()[0])
        imgL_upperBoundary_list[1] = min(imgL_upperBoundary[1], img_L.GetSize()[1])
        imgL_upperBoundary_list[2] = min(imgL_upperBoundary[2], img_L.GetSize()[2])

        imgL_lowerBoundary = tuple(imgL_lowerBoundary_list)
        imgL_upperBoundary = tuple(imgL_upperBoundary_list)

        imgL_arr      = sitk.GetArrayFromImage(img_L)
        img_net2L_arr = imgL_arr[x1:x2, y1:y2, z1:z2]
        img_net2Lnii  = sitk.GetImageFromArray(img_net2L_arr)

        mskL_arr      = sitk.GetArrayFromImage(msk_L)
        msk_net2L_arr = mskL_arr[x1:x2, y1:y2, z1:z2]
        msk_net2Lnii  = sitk.GetImageFromArray(msk_net2L_arr)
        #endregion GENERATE LEFT brains | labels for NET 2

        #region WRITE LEFT brains | labels for NET 2
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(datapath_net2,"brains" , os.path.basename(imgFile).split(".")[0]+"_L.nii"))
        writer.Execute(img_net2Lnii)

        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile).split(".")[0]+"_L.nii"))
        writer.Execute(msk_net2Lnii)
        #endregion WRITE LEFT brains | labels for NET 2

        # endregion LEFT: Brain and label prep
        

        #region RIGHT: Brain and label prep

        #region GENERATE RIGHT brains | labels for NET 1

        if not (np.count_nonzero(sitk.GetArrayFromImage(msk_R))):
            continue

        imgR_flipped = sitk.Flip(img_R,[True,False,False])
        mskR_flipped = sitk.Flip(msk_R,[True,False,False])

        [x1, y1, z1, x2, y2, z2] = get_roi(mskR_flipped)
        
        imgR_lowerBoundary = imgR_flipped.TransformPhysicalPointToIndex(mskR_flipped.TransformIndexToPhysicalPoint(np.array([x1, y1, z1]).tolist()))
        imgR_upperBoundary = imgR_flipped.TransformPhysicalPointToIndex(mskR_flipped.TransformIndexToPhysicalPoint(np.array([x2, y2, z2]).tolist()))

        img_net1Rnii = imgR_flipped
        img_net1Rnii.CopyInformation(imgR_flipped)
        
        msk_net1Rarr = sitk.GetArrayFromImage(mskR_flipped)
        msk_net1Rarr[x1:x2,y1:y2,z1:z2] = 1

        msk_net1Rnii = sitk.GetImageFromArray(msk_net1Rarr)
        msk_net1Rnii.CopyInformation(mskR_flipped)
        #endregion GENERATE RIGHT brains | labels for NET 1

        #region WRITE RIGHT brains | labels for NET 1
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(datapath_net1,"brains" , os.path.basename(imgFile).split(".")[0]+"_R.nii"))
        writer.Execute(img_net1Rnii)

        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(datapath_net1,"target_labels", os.path.basename(mskFile).split(".")[0]+"_R.nii"))
        writer.Execute(msk_net1Rnii)
        #endregion WRITE RIGHT brains | labels for NET 1

        #region GENERATE RIGHT brains | labels for NET 2
        if ((imgR_upperBoundary[0]-imgR_lowerBoundary[0]==0) or (imgR_upperBoundary[1]-imgR_lowerBoundary[1]==0) or (imgR_upperBoundary[2]-imgR_lowerBoundary[2]==0)):
            continue

        imgR_lowerBoundary_list = list(imgR_lowerBoundary)
        imgR_upperBoundary_list = list(imgR_upperBoundary)

        imgR_lowerBoundary_list[0] = max(imgR_lowerBoundary[0], 0)
        imgR_lowerBoundary_list[1] = max(imgR_lowerBoundary[1], 0)
        imgR_lowerBoundary_list[2] = max(imgR_lowerBoundary[2], 0)

        imgR_upperBoundary_list[0] = min(imgR_upperBoundary[0], imgR_flipped.GetSize()[0])
        imgR_upperBoundary_list[1] = min(imgR_upperBoundary[1], imgR_flipped.GetSize()[1])
        imgR_upperBoundary_list[2] = min(imgR_upperBoundary[2], imgR_flipped.GetSize()[2])

        imgR_lowerBoundary = tuple(imgR_lowerBoundary_list)
        imgR_upperBoundary = tuple(imgR_upperBoundary_list)

        imgR_arr      = sitk.GetArrayFromImage(imgR_flipped)
        img_net2R_arr = imgR_arr[x1:x2, y1:y2, z1:z2]
        img_net2Rnii  = sitk.GetImageFromArray(img_net2R_arr)

        mskR_arr      = sitk.GetArrayFromImage(mskR_flipped)
        msk_net2R_arr = mskR_arr[x1:x2, y1:y2, z1:z2]
        msk_net2Rnii  = sitk.GetImageFromArray(msk_net2R_arr)
        #endregion GENERATE RIGHT brains | labels for NET 2

        #region WRITE RIGHT brains | labels for NET 2
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(datapath_net2,"brains" , os.path.basename(imgFile).split(".")[0]+"_R.nii"))
        writer.Execute(img_net2Rnii)

        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(datapath_net2,"target_labels", os.path.basename(mskFile).split(".")[0]+"_R.nii"))
        writer.Execute(msk_net2Rnii)
        #endregion WRITE RIGHT brains | labels for NET 2

        #endregion RIGHT: Brain and label prep
       
    create_jsonFile(data_path=datapath_net1)
    create_jsonFile(data_path=datapath_net2)
