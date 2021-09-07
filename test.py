from calculate_volumes import calculate_volumes
import os
import settings
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf


from argparser   import args
from sklearn     import metrics
from argparser   import args
from prepdataset import prepdata, get_filelist
from skimage.transform import resize

def test_intel_tensorflow():

    print("We are using Tensorflow version {}".format(tf.__version__))
    major_version = int(tf.__version__.split(".")[0])
    if major_version >= 2:
	    from tensorflow.python import _pywrap_util_port
	    print("Intel-optimizations (DNNL) enabled:",_pywrap_util_port.IsMklEnabled())
    else:
	    print("Intel-optimizations (DNNL) enabled:",tf.pywrap_tensorflow.IsMklEnabled())

test_intel_tensorflow()

def dice_coefficient(truth,prediction):
    return 2*np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def jaccard_index(truth,prediction):
    #jaccIdx = metrics.jaccard_score(truth.ravel(),prediction.ravel(),average='micro')
    #print(jaccIdx)
    jaccIdx = np.logical_and(truth,prediction)/(np.sum(truth) + np.sum(prediction))
    return jaccIdx#np.logical_and(truth,prediction)/(np.sum(truth) + np.sum(prediction))

def kappa_score(truth,prediction):
    kappaScore = metrics.cohen_kappa_score(truth.ravel(), prediction.ravel())
    return kappaScore

def accuracy_score(truth,prediction):
    accuracyScore = metrics.accuracy_score(truth.ravel(), prediction.ravel())
    return accuracyScore

def recall_score(truth,prediction):
    RecallScore = metrics.recall_score(truth.ravel(), prediction.ravel())
    return RecallScore

def precision_score(truth,prediction):
    PrecisionScore = metrics.precision_score(truth.ravel(), prediction.ravel())
    return PrecisionScore

def calculate_scores():

	"""
	scores_df = pd.DataFrame(columns=('Subject ID','Dice_Left', 'Dice_Right'
				#,'Jaccard_Left', 'Jaccard_Right'
				,'Kappa_Left', 'Kappa_Right',
				#'Recall_Left','Recall_Right','Precision_Left','Precision_Right','Accuracy_Left','Accuracy_Right'
	))
	"""

	#diceL = dice_coefficient(mskL,predL)
	#diceR = dice_coefficient(mskR,predR)
		
	#JaccL = jaccard_index(np.rint(mskL),np.rint(predL))
	#JaccR = jaccard_index(np.rint(mskR),np.rint(predR))

		
	#KappL = kappa_score(np.rint(resize(mskL,input_dim)),np.int16(predL))
	#KappR = kappa_score(np.rint(resize(mskR,input_dim)),np.int16(predR))
	
	#AccuL = accuracy_score(np.rint(resize(mskL,input_dim)),predL.astype('int64'))
	#AccuR = accuracy_score(np.rint(resize(mskR,input_dim)),predR.astype('int64'))

	#ReclL = recall_score(np.rint(resize(mskL,input_dim)),predL.astype('int64'))
	#ReclR = recall_score(np.rint(resize(mskR,input_dim)),predR.astype('int64'))
		
	#PrecL = precision_score(np.rint(resize(mskL,input_dim)),predL.astype('int64'))
	#PrecR = precision_score(np.rint(resize(mskR,input_dim)),predR.astype('int64'))		
		
	#scores_df.loc[idx,:] = [subject_id, diceL, diceR, KappL,KappR]
	#print(scores_df)

def predict_outputs(tf_model, datapath_net,subject_id):
	
	imgL = nib.load(os.path.join(datapath_net,"brains",subject_id+"_t1_L.nii")).get_fdata()
	imgR = nib.load(os.path.join(datapath_net,"brains",subject_id+"_t1_R.nii")).get_fdata()

	imgL_aff = nib.load(os.path.join(datapath_net,"brains",subject_id+"_t1_L.nii")).get_affine()
	imgR_aff = nib.load(os.path.join(datapath_net,"brains",subject_id+"_t1_R.nii")).get_affine()

	mskL = nib.load(os.path.join(datapath_net,"target_labels",subject_id+"_labels_L.nii")).get_fdata()
	mskR = nib.load(os.path.join(datapath_net,"target_labels",subject_id+"_labels_R.nii")).get_fdata()

	imgL = tf.expand_dims(imgL,axis=3)
	imgR = tf.expand_dims(imgR,axis=3)
		
	imgL = imgL[None,:,:,:,:]
	imgR = imgR[None,:,:,:,:]

	predL = tf_model.predict(imgL)
	predR = tf_model.predict(imgR)

	predL = tf.squeeze(predL)
	predR = tf.squeeze(predR)

	nib.save(nib.Nifti1Image(np.float32(predL),imgL_aff),os.path.join(datapath_net,"predictions",subject_id+"_labels_L.nii"))
	nib.save(nib.Nifti1Image(np.float32(predR),imgR_aff),os.path.join(datapath_net,"predictions",subject_id+"_labels_R.nii"))

	print(imgL.shape,"|",mskL.shape,"|",predL.shape)
		
		
def testing():

	prepdata(data_path=args.testdata_path)
		
	datapath_net1 = os.path.join(args.testdata_path,"data_net1")
	datapath_net2 = os.path.join(args.testdata_path,"data_net2")

	filenames = get_filelist(datapath_net1,datapath_net2)
		
	if not os.path.isdir(os.path.join(datapath_net2,"predictions")):
		os.mkdir(os.path.join(datapath_net2,"predictions"))
	

	if not os.path.isdir(os.path.join(datapath_net1,"predictions")):
		os.mkdir(os.path.join(datapath_net1,"predictions"))

	if os.path.isdir(os.path.join(settings.ROOT_DIR,"UNET1_LOCALIZE_final")):
		tf_model1 = tf.keras.models.load_model("UNET1_LOCALIZE_final", compile=False)
		tf_model1.compile(optimizer="adam", loss="binary_crossentropy")
	else:
		print("Model for network 1 (localization) is unavailable. Check if training is complete and model is saved")

	if os.path.isdir(os.path.join(settings.ROOT_DIR,"UNET2_SEGMENT_final")):
		tf_model2 = tf.keras.models.load_model("UNET2_SEGMENT_final", compile=False)
		tf_model2.compile(optimizer="adam", loss="binary_crossentropy")
	else:
		print("Model for network 2 (segmentation) is unavailable. Check if training is complete and model is saved")	

	for idx in range(0,len(filenames)):
		
		imgFile = filenames[idx][0]
		mskFile = filenames[idx][1]

		subject_id = os.path.basename(mskFile).split("_")[0]

		predict_outputs(tf_model1, datapath_net1,subject_id)
		#predict_outputs(tf_model2, datapath_net2,subject_id)
		
		#calculate_scores()
testing()
