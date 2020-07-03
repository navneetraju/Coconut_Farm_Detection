import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import keras.backend as K
from utils import *

if __name__ == "__main__":
	if(len(sys.argv)!=2):
		print('Usage: process.py [Full Directory to Image]')
	ORIGINAL_IMG_PATH=sys.argv[1]
	loaded_data = np.load(ORIGINAL_IMG_PATH)
	data = loaded_data['big']  
	all_conc = dict()
	Remember_this_number  = 10
	next_row = 0
	for k in range(1,int(data.shape[0] //Remember_this_number)+1):
		all_conc["conc"+str(k)] = np.concatenate([data[i,:,:,] for i in range(0+next_row,Remember_this_number+next_row) ],axis = 1)
		next_row = next_row + Remember_this_number
	rows = []
	for i in range(1,len(all_conc)+1):
	rows.append(all_conc["conc"+str(i)])
	concmax =  np.concatenate(rows,axis = 0)
	print('###########################COCOUNT FARM SEGMENTATION#############################')
	print('Choose Model to Segment: ')
	print('1. Custom U-Net\n2. Siamese U-Net')
	print('Enter 1 or 2')
	model_id=input()
	if(model_id=='1'):
		model=load_model_util(1)
	elif(model_id=='2'):
		model=load_model_util(2)
	else:
		print('Wrong Input!')
		sys.exit()
	mask_final=process_img(data,model)	
	try:
		generate_mask(data,mask_final,Remember_this_number)
		print('Binary Mask Generated!')
		print('Check same directory for mask_generated.png')
	except:
		print('Something went wrong,Please try again and ensure all paths are correct')
		sys.exit()
	try:
		generate_blend(data,mask_final,Remember_this_number)
		print('RGB Blend Mask Generated!')
		print('Check same directory for blend_generated.png')
	except:
		print('Something went wrong,Please try again and ensure all paths are correct')
		sys.exit()
	
