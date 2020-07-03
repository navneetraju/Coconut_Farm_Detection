import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import keras.backend as K
import sys

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def iou_score(y_true,y_pred):
  return K.ones(1)

def load_model_util(model_id):
	if(model_id==1):
		try:
			model = load_model('./model_weights/unet_v2.h5',
					custom_objects={'jaccard_distance_loss': jaccard_distance_loss,
									'iou_score':iou_score})
			return model
		except:
			print('Error: Model File Not Found!!')
			sys.exit()
	else:
		model = load_model('model_unet.h5',
                   custom_objects={'jaccard_distance_loss': jaccard_distance_loss,
                                   'iou_score':iou_score})

def process_img(data,model):
	mask_final = np.zeros((data.shape[0],768,768,1),'float32')
	for j in range(0,data.shape[0]):
		final_image=np.zeros((768,768,1))
		print("Processing Image number %d"%(j))
		temp = data[j,:,:,]
		parts=dict()
		parts[1]=temp[0:256,0:256,:]
		parts[2]=temp[0:256,256:512,:]
		parts[3]=temp[0:256,512:768,:]
		parts[4]=temp[256:512,0:256,:]
		parts[5]=temp[256:512,256:512,:]
		parts[6]=temp[256:512,512:768,:]
		parts[7]=temp[512:768,0:256,:]
		parts[8]=temp[512:768,256:512,:]
		parts[9]=temp[512:768,512:768,:]
		for i in range(1,10):
			x=parts[i]                           
			x=np.reshape(x,(1,256,256,3))
			y=model.predict(x/255)               
			y=np.reshape(y,(256,256,1))
			parts[i]=y                            
		final_image[0:256,0:256,:]=parts[1]
		final_image[0:256,256:512,:]=parts[2]
		final_image[0:256,512:768,:]=parts[3]
		final_image[256:512,0:256,:]=parts[4]
		final_image[256:512,256:512,:]=parts[5]
		final_image[256:512,512:768,:]=parts[6]
		final_image[512:768,0:256,:]=parts[7]
		final_image[512:768,256:512,:]=parts[8]
		final_image[512:768,512:768,:]=parts[9]
		final_image_thresh = final_image.copy()
		final_image_thresh[final_image >=0.5] = 255
		final_image_thresh[final_image <0.5] = 0.0
		mask_final[j,:,:,] = final_image.reshape(768,768,1)
		mask_final[j,:,:,] = final_image_thresh.reshape(768,768,1)
	return mask_final

def generate_mask(data,mask_final,Remember_this_number):
	mask_dict = dict()
	next_row = 0
	for k in range(1,int(data.shape[0] //Remember_this_number)+1):
		mask_dict["conc"+str(k)] = np.concatenate([mask_final[i,:,:,] for i in range(0+next_row,Remember_this_number+next_row) ],axis = 1)
		next_row = next_row + Remember_this_number
	rows = []
	for i in range(1,len(mask_dict)+1):
		rows.append(mask_dict["conc"+str(i)])
	concmax =  np.concatenate(rows,axis = 0)
	coco_binary_map = concmax/255
	coco_binary_map = coco_binary_map.reshape(coco_binary_map.shape[0],coco_binary_map.shape[1])
	coco_binary_map = np.invert(coco_binary_map.astype('uint8'))
	coco_binary_map = coco_binary_map - 254
	coco_binary_map = coco_binary_map.astype('float32')
	plt.imsave("mask_generated.png",coco_binary_map,cmap= 'binary')

def generate_blend(data,mask_final,Remember_this_number):
	blend_to_get_a_big_pic = np.zeros((data.shape[0],768,768,3))
	for i in range(0,data.shape[0]):
		backtorgb = cv2.cvtColor(mask_final[i,:,:,],cv2.COLOR_GRAY2RGB)
		blend = ((data[i,:,:,] * 0.70)/255)+ ((backtorgb * 0.30)/255)
		if(blend.max() <= 0.3):
			blend_to_get_a_big_pic[i,:,:,] = blend * 0
		else:
			blend_to_get_a_big_pic[i,:,:,] = blend
	blend_dict = dict()
	next_row = 0
	for k in range(1,int(data.shape[0] //Remember_this_number)+1):
		blend_dict["conc"+str(k)] = np.concatenate([blend_to_get_a_big_pic[i,:,:,] for i in range(0+next_row,Remember_this_number+next_row) ],axis = 1)
		next_row = next_row + Remember_this_number
	rows = []
	for i in range(1,len(blend_dict)+1):
		rows.append(blend_dict["conc"+str(i)])
	concmax =  np.concatenate(rows,axis = 0)
	concmax = np.clip(concmax,0,1)
	plt.imsave("blend_generated.png",concmax)
