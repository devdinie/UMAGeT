
import os
import json
import settings
import itertools

from argparser import args

def save_json(json_dict, filepath):
    with open(filepath, 'w') as fp:
        json.dump(json_dict, fp)

def create_json_file(data_path):
    
    brains_dir = os.path.join(data_path,"brains")
    if os.path.exists(os.path.join(data_path,"target_labels")):
        labels_dir = os.path.join(data_path,"target_labels")
    
    no_cases = len(os.listdir(brains_dir))
    
    json_dict                    = {}
    json_dict['name']            = "UNet3D"
    json_dict['description']     = "Segmentation of whole hippocampus"
    json_dict['reference']       = "Hippocampal images"
    json_dict['modality']        = { "0": "T1",}
    json_dict['labels']          = { "0": "background",
                                     "1": "Hippocampus" }
    
    if not args.mode == "test":
        json_dict['numTraining']= no_cases
        json_dict['training']   = [{"image": os.path.join(brains_dir,img_fname),
                                    "label": os.path.join(labels_dir,img_fname.replace("_t1_","_labels_"))} for img_fname in os.listdir(brains_dir)]
    else:
        json_dict['numTesting'] = no_cases
        
        if os.path.exists(labels_dir):
            json_dict['testing']   = [{"image": os.path.join(brains_dir,img_fname),
                                        "label": os.path.join(labels_dir,img_fname.replace("_t1_","_labels_"))} for img_fname in os.listdir(brains_dir)]
        else:
            json_dict['testing']   = [{"image": os.path.join(brains_dir,img_fname),
                                        "label": os.path.join(labels_dir,img_fname.replace("_t1_","_labels_"))} for img_fname in os.listdir(brains_dir)]
    save_json(json_dict, os.path.join(data_path, "dataset_dict.json"))
