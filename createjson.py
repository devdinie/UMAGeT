
import os
import json
import settings
import itertools

def save_json(json_dict, filepath):
    with open(filepath, 'w') as fp:
        json.dump(json_dict, fp)

def create_jsonFile(data_path):
    
    brains_dir = os.path.join(data_path,"brains")
    labels_dir = os.path.join(data_path,"target_labels")

    no_brains = len(os.listdir(brains_dir))
    no_labels = len(os.listdir(labels_dir))

    #if no_brains == no_labels:
    no_cases = no_brains
    cases_br    = os.listdir(brains_dir)
        
    cases=list()
    side =list()
    for file in range(0,no_cases):
        cases.append(cases_br[file].split("_")[0])

        if not ((data_path==settings.DATA_PATH) or (data_path==settings.TESTDATA_PATH)):
            side.append(cases_br[file].split("_")[2].split(".")[0])

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

    if data_path != settings.DATA_PATH:
        json_dict['testing']    = [{'image': os.path.join(brains_dir,"%s_t1_%s.nii") % (i,j), 
                                     "label": os.path.join(labels_dir,"%s_labels_%s.nii") % (i,j)} for (i,j) in zip(cases,side)]
    else:
        json_dict['training']    = [{'image': os.path.join(brains_dir,"%s_t1.nii") % i, 
                                     "label": os.path.join(labels_dir,"%s_labels.nii") % i} for i in cases]


    print(cases)
    save_json(json_dict, os.path.join(data_path, "dataset.json"))