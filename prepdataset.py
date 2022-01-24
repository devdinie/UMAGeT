import os
import json
import settings

def get_filelist():
	
    json_filename = os.path.join(settings.DATA_PATH_AUG, "dataset.json")

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

def prepdata(data_path=settings.DATA_PATH, augmentation=settings.AUGMENT):
	
	input_dim=np.array((args.tile_width,args.tile_height,args.tile_depth))
	
	create_jsonFile(data_path=data_path)
	
	data_filetype = settings.IMAGE_FILETYPE
	
	datapath_net1 = os.path.join(data_path,"data_net1")
	datapath_net2 = os.path.join(data_path,"data_net2")

	#region create input directories
	if not os.path.exists(datapath_net1):
		os.mkdir(datapath_net1)
		os.mkdir(os.path.join(datapath_net1,"brains"))
		os.mkdir(os.path.join(datapath_net1,"target_labels"))
		
	if not os.path.exists(datapath_net2):
		os.mkdir(datapath_net2)
        	os.mkdir(os.path.join(datapath_net2,"brains"))
        	os.mkdir(os.path.join(datapath_net2,"target_labels"))
	#endregion create input directories

	filenames    = get_filelist(datapath_net1,datapath_net2)

	ref_img_size = [settings.TILE_HEIGHT, settings.TILE_WIDTH, settings.TILE_DEPTH]
	mid_idx      = np.around(ref_img_size[0]/2).astype(int)

	print(filenames)
	"""
	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]
	"""
	