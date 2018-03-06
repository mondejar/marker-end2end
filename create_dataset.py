#!/usr/bin/python
################################################################################
#
# Author:  Mondejar-Guerra V.
#
# Create at 5 Feb 2018
# Last modification: 5 Feb 2018
################################################################################

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os
import random
import sys
from draw_corners_on_marker import *

""" 
This code prepares the data for train the model.
Given two directories, the first one with the markers and the second one with the background images.
This code creates a file with:
	- N transformations for each marker M over N different images 
	- The 4 coordinates that enclose each marker M on the image	    		

python marker_dataset_path img_dataset_path number_transforms patchSize
				
Example call:

"""


# Generate a random affine transform over the four corners
def affine_transform(img, mask_img, patchSize, marker_corners, randomDispFactor ):
	 
	# To warp the full image (including white borders)
	corners = np.float32([[0, 0], [0, patchSize], [patchSize, 0], [patchSize, patchSize]])
	cornersT = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])

	w = int(round(randomDispFactor * patchSize))
	h = int(round(randomDispFactor * patchSize))

	for p in range(0,4):
		randValue = random.uniform(0.0, 1.0)
		x =  int(round(((w * randValue)))  + corners[p][0])
		randValue = random.uniform(0.0, 1.0)
		y = int(round(((h * randValue))) + corners[p][1])
		cornersT[p] = np.float32([x, y])

	# Norm to make the patch be centered
	cornersT[:,0] = cornersT[:,0] - np.min(cornersT, 0)[0]
	cornersT[:,1] = cornersT[:,1] - np.min(cornersT, 0)[1]

	maxW = np.max(cornersT, 0)[0] - np.min(cornersT, 0)[0]
	maxH = np.max(cornersT, 0)[1] - np.min(cornersT, 0)[1]

	persT = cv2.getPerspectiveTransform(corners, cornersT)
	warp_img = cv2.warpPerspective(img, persT, (maxH, maxW), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_CONSTANT)

	# Set mask for only select the marker affine
	warp_mask_img = cv2.warpPerspective(mask_img, persT, (maxH, maxW), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_CONSTANT)

	marker_corners = np.array([marker_corners])
	marker_corners_T = cv2.perspectiveTransform(marker_corners, persT)

	return warp_img, warp_mask_img, marker_corners_T

def dynamic_range_compression(img):
	
	randValue = random.uniform(0.0, 1.0)
	a = (0.6 * randValue) + 0.4 #[0.4, 1]
	
	randValue = random.uniform(0.0, 1.0)
	b = 25.0 * randValue#[0,100]	

	rows, cols = img.shape
	for r in range(rows):
		for c in range(cols):
			val = img[r,c]
			img[r,c] = int( np.min([255, a * val + b]))

	return img


def blurring(img):
	x_value = random.uniform(1.0, 15.0)
	y_value = random.uniform(1.0, 15.0)

	img = cv2.blur(img, (int(x_value), int(y_value)), 0)

	return img

# Put the marker in the background image
# This function is needed for the affine transform
# Mask contains the pixels of the marker with 255
#
#   Marker corners:
#
#     0----1
#     |    |
#     2----3
#


def create_dataset(marker_dataset, img_dataset, numWarps, patchSize, outImPath, valFilename, trainFilename, verbose):

	if not os.path.exists(outImPath):
		os.mkdir(outImPath)

	if not os.path.exists(outImPath + 'train_data'):
		os.mkdir(outImPath + 'train_data')

	if not os.path.exists(outImPath + 'val_data'):
		os.mkdir(outImPath + 'val_data')


	# create file
	fileList_Train = open(trainFilename,'w') 
	fileList_Val = open(valFilename,'w') 

	imMarkers = [f for f in listdir(marker_dataset) if isfile(join(marker_dataset, f))]
	imBackgrounds = [f for f in listdir(img_dataset) if isfile(join(img_dataset, f))]
	
	numIm = 0

	train_val_factor = 0.1 # 10% of the warps are exported on the validation folder

	for imMarker in imMarkers:
		# Read marker
		print(marker_dataset + imMarker)
		marker_orig = cv2.imread(marker_dataset + imMarker, 0)

		for w in range(0, numWarps):

			# Pick a random background image
			imBackground = random.choice(imBackgrounds)
			back_img = cv2.imread(img_dataset + imBackground, 0)

			jump_loop = False
			if not back_img is None:
							
				# Resample the background image to the double of patch size
				train_img = cv2.resize(back_img, ( 3 * patchSize, 3 * patchSize)) 	

				# TODO: 
				# Scale the marker at some size between 10-50% of the specified size
				scale_factor = random.uniform(0.2, 0.5)#0.1, 0.5
				marker_size = int(patchSize * scale_factor)

				marker_scale = cv2.resize(marker_orig, (marker_size, marker_size))
				aux_patchSize = 3*patchSize
				# 1 Put the marker in the image and add a white border
				x_pos = np.random.randint(patchSize, (patchSize*2) - marker_size)
				y_pos = np.random.randint(patchSize, (patchSize*2) - marker_size)		

				mask_img = np.zeros((aux_patchSize, aux_patchSize))
				whit_pix = int(marker_size * 0.1)

				mask_img[x_pos:x_pos+marker_size, y_pos:y_pos+marker_size] = np.ones((marker_size, marker_size)) * 255
				train_img[x_pos-whit_pix:x_pos+marker_size+whit_pix, y_pos-whit_pix:y_pos+marker_size+whit_pix] = np.ones((marker_size + 2*whit_pix, marker_size + 2*whit_pix)) * 255
				train_img[x_pos:x_pos+marker_size, y_pos:y_pos+marker_size] = marker_scale


				marker_corners = np.array([[y_pos, x_pos], [y_pos+marker_size, x_pos], [y_pos, x_pos+marker_size], [y_pos+marker_size, x_pos+marker_size]], dtype='float32')


				# Apply affine transform and crop!
				train_img, mask_img, marker_corners = affine_transform(train_img, mask_img, aux_patchSize, marker_corners, 0.3)
				marker_corners = marker_corners[0]

				# Crop the generated regions assuring that the marker is inside the region
				#yx
				min_xy = np.min(marker_corners, axis = 0)
				max_xy = np.max(marker_corners, axis = 0)

				if np.max(max_xy - min_xy) < patchSize:
					random_crop_y = np.random.randint( int(max_xy[0] - patchSize),  np.min([int(min_xy[0]), aux_patchSize-patchSize]))
					random_crop_x = np.random.randint( int(max_xy[1] - patchSize),  np.min([int(min_xy[1]), aux_patchSize-patchSize]))

					train_img = train_img[random_crop_x:random_crop_x + patchSize, random_crop_y:random_crop_y + patchSize]
					mask_img = mask_img[random_crop_x:random_crop_x + patchSize, random_crop_y:random_crop_y + patchSize]			
					
					marker_corners -= (random_crop_y, random_crop_x)

				else:
					jump_loop = True
					continue


				if train_img.shape != (patchSize, patchSize):
					jump_loop = True
					continue
				# Add extra transforms:

				# gray level?
				train_img = dynamic_range_compression(train_img)

				# Blurring
				# Not apply always the bluring!
				apply_blur = random.uniform(0.0, 1.0)
				if apply_blur > 0.3:
					train_img = blurring(train_img)
					
				# Light ? 

				# Last  numWarps - (train_val_factor * numWarps) for validation
				if w > (numWarps - (train_val_factor * numWarps)):
					nameTrainImg = outImPath + 'val_data/' + imMarker[:-5] + "_" + str(w) + '.png'
					nameMaskImg = outImPath + 'val_data/' + imMarker[:-5] + "_" + str(w) + '_mask.png'

					# add line to file
					fileList_Val.write( nameTrainImg + ' ' + nameMaskImg)

					for p in range(0, 4):	
						fileList_Val.write(' ' + str(marker_corners[p][0]) + ' ' + str(marker_corners[p][1]))
					fileList_Val.write('\n')

				else:
					nameTrainImg = outImPath + 'train_data/' + imMarker[:-5] + "_" + str(w) + '.png'
					nameMaskImg = outImPath + 'train_data/' + imMarker[:-5] + "_" + str(w) + '_mask.png'

					# add line to file
					fileList_Train.write( nameTrainImg + ' ' + nameMaskImg)

					for p in range(0, 4):	
						fileList_Train.write(' ' + str(marker_corners[p][0]) + ' ' + str(marker_corners[p][1]))
					fileList_Train.write('\n')	

				# Export patch warp
				cv2.imwrite( nameTrainImg, train_img)

				cv2.imwrite( nameMaskImg, mask_img)	

				# Export segmented mask!

				# Write .bin files for use with caffe 
				if verbose:
					cv2.namedWindow('train_img', cv2.WINDOW_NORMAL)
					cv2.imshow('train_img', train_img)

					cv2.namedWindow('mask_img', cv2.WINDOW_NORMAL)
					cv2.imshow('mask_img', mask_img)

					cv2.namedWindow('train_img_corners', cv2.WINDOW_NORMAL)
					cv2.imshow('train_img_corners', draw_corners_on_marker(train_img, marker_corners.flatten()))

					cv2.waitKey(0)
					cv2.destroyAllWindows()


				if w % 1000 == 0:
					print(str(w) + '/' + str(numWarps))

			else:
				print("Warning: It could not be load background image: " + imBackground)
				w = w-1

			if jump_loop == True:
				w = w-1
	
	fileList_Train.close()
	fileList_Val.close()




if __name__ == "__main__":

	# NOTE
	# check this path dirs!
	patchSize = 128

	# Dir in which the new patches are saved 
	outImPath = 'data/' + str(patchSize) +'/' 

	# Full file path referencing the patches and ground truth
	valFilename = 'data/' + str(patchSize) +'/val_data_list.txt' 
	trainFilename = 'data/' + str(patchSize) +'/train_data_list.txt' 

	verbose = False	# set True to display the process

	create_dataset('/home/mondejar/dataset/markers/', '/home/mondejar/dataset/mirflickr/', 50000, patchSize, outImPath, valFilename, trainFilename, verbose)
    