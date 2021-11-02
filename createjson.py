
import os
import json
import settings
import itertools

def save_json(json_dict, filepath):
    with open(filepath, 'w') as fp:
        json.dump(json_dict, fp)

def create_jsonFile(data_path):
    
    print("********",data_path)
    brains_dir = os.path.join(data_path,"brains")
    labels_dir = os.path.join(data_path,"target_labels")

    no_brains = len(os.listdir(brains_dir))
    no_labels = len(os.listdir(labels_dir))

    #if no_brains == no_labels:
    no_cases = no_brains
    cases_img    = os.listdir(brains_dir)
    cases_msk    = os.listdir(labels_dir)
        
    cases  =list()
    side   =list()

    suffix_img =list()
    suffix_msk =list()

    for file in range(0,no_cases):
        
        suffix_arr     = cases_img[file].split(".")[0].split("_")
        
        #Check:
        #print(file,"|",cases_img[file],"|",suffix_arr[len(suffix_arr)-1],"|",  cases_img[file].split("_",1)[1])
        
        cases.append(cases_img[file].split("_")[0])

        suffix_img.append(cases_img[file].split(".")[0].split("_",1)[1])
        suffix_msk.append(cases_img[file].split(".")[0].split("_",1)[1].replace("t1","labels"))

        if not ((data_path==settings.DATA_PATH) or (data_path==settings.TESTDATA_PATH) or (data_path==settings.DATA_PATH_AUG)):
            side.append(cases_img[file].split("_")[len(suffix_arr)].split(".")[0])

    #else:
        #print("Error: Number of brains and target labels don't match.")

    json_dict                    = {}
    json_dict['name']            = "UNet3D"
    json_dict['description']     = "Segmentation of whole hippocampus"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference']       = "Hippocampal images"
    json_dict['licence']         = ""
    json_dict['release']         = "0.0"
    json_dict['modality']        = { "0": "T1",}
    json_dict['labels']          = { "0": "background",
                                     "1": "Hippocampus" }
    json_dict['numTraining']     = no_cases

    if (data_path != settings.DATA_PATH) and (data_path != settings.DATA_PATH_AUG):
        json_dict['testing']    = [{'image': os.path.join(brains_dir,"%s_%s_%s.nii") % (i,j,k), 
                                     "label": os.path.join(labels_dir,"%s_%s_%s.nii") % (i,j,k)} for (i,j,k) in zip(cases,suffix_img,side)]
    else:
        json_dict['training']   = [{'image': os.path.join(brains_dir,"%s_%s.nii") % (i,j), 
                                    "label": os.path.join(labels_dir,"%s_%s.nii") % (i,k)} for (i,j,k) in zip(cases,suffix_img,suffix_msk)]


    save_json(json_dict, os.path.join(data_path, "dataset.json"))