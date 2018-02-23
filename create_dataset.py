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

python marker_dataset_path img_dataset_path number_transforms patch_size
				
Example call:

"""


# Generate a random affine transform over the four corners
def affine_transform( patchSize, white_border_factor, img, randomDispFactor ):
	
	# 0.115
	white_border = white_border_factor * patchSize

	# To save the correct GT (black borders)
	black_corners = np.array([[white_border, white_border], [white_border, patchSize-white_border], [patchSize-white_border, white_border], [patchSize-white_border, patchSize-white_border]], dtype='float32')
	 
	# To warp the full imagen (including white borders)
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
	warpImg = cv2.warpPerspective(img, persT, (maxW, maxH), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_CONSTANT)

	# Set mask for only select the marker affine
	mask_perspect = np.zeros((patchSize, patchSize, 1), dtype = "uint8") + 255
	mask_perspect = cv2.warpPerspective(mask_perspect, persT, (maxW, maxH), flags=cv2.INTER_LINEAR)


	black_corners = np.array([black_corners])
	black_cornersT = cv2.perspectiveTransform(black_corners, persT)


	return warpImg, persT, mask_perspect, cornersT, black_cornersT[0]

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


# Put the marker in the background image
# This function is needed for the affine transform
# Mask contains the pixels of the marker with 255
#
#   Marker corners:
#
#     0----2
#     |    |
#     1----3
#
def merge_images(train_img, x_pos, y_pos, marker_affin, mask_perspect):
	rows, cols = marker_affin.shape
	
	for x in range(0, cols):
		for y in range(0, rows):
			if mask_perspect[y,x] != 0:
				train_img[y_pos + y, x_pos + x ] = marker_affin[y,x]

	return train_img

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

	train_val_factor = 0.1 # 10% of the warps are employed for validation

	for imMarker in imMarkers:
		# Read marker
		print(marker_dataset + imMarker)
		marker_orig = cv2.imread(marker_dataset + imMarker, 0)

		for w in range(0, numWarps):

			# Pick a random background image
			imBackground = random.choice(imBackgrounds)
			back_img = cv2.imread(img_dataset + imBackground, 0)

			if not back_img is None:
				# Resample background image to specified size
				train_img = cv2.resize(back_img, (patchSize, patchSize)) 				
				
				# Scale the marker at some size between 10-50% of the specified size
				scale_factor = random.uniform(0.3, 0.5)#0.1, 0.5
				marker_size = int(patchSize * scale_factor)
				marker_scale = cv2.resize(marker_orig, (marker_size, marker_size))

				# TODO: add more transformations to the marker

				# gray level?
				marker_scale = dynamic_range_compression(marker_scale)

				# blurring: to the marker or to the global image?

				# Ilumination? non uniform?

				# affine transform
				white_border_factor = 0.115
				marker_affin, persT, mask_perspect, image_corners, gt_corners = affine_transform(marker_size, white_border_factor, marker_scale, 0.25)#0.4)

				rows_marker, cols_marker = marker_affin.shape
			    # and place randomly over the background image
				x_pos = np.random.randint(0, patchSize - cols_marker - 1)
				y_pos = np.random.randint(0, patchSize - rows_marker - 1)				

				train_img = merge_images(train_img, x_pos, y_pos, marker_affin, mask_perspect)


				# Export the image and write the four corners on the file
				gt_corners[:,0] = gt_corners[:,0] + x_pos
				gt_corners[:,1] = gt_corners[:,1] + y_pos

				# Last  numWarps - (train_val_factor * numWarps) for validation
				if w > (numWarps - (train_val_factor * numWarps)):
					nameTrainImg = outImPath + 'val_data/' + imMarker[:-5] + "_" + str(w) + '.png'

					# add line to file
					fileList_Val.write( nameTrainImg)
					for p in range(0, 4):	
						fileList_Val.write(' ' + str(gt_corners[p][0]) + ' ' + str(gt_corners[p][1]))
					fileList_Val.write('\n')

				else:
					nameTrainImg = outImPath + 'train_data/' + imMarker[:-5] + "_" + str(w) + '.png'

					# add line to file
					fileList_Train.write( nameTrainImg)
					for p in range(0, 4):	
						fileList_Train.write(' ' + str(gt_corners[p][0]) + ' ' + str(gt_corners[p][1]))
					fileList_Train.write('\n')	

				# Export patch warp
				cv2.imwrite( nameTrainImg, train_img)	

				# Write .bin files for use with caffe 
				if verbose:
					cv2.namedWindow('train_img', cv2.WINDOW_NORMAL)
					cv2.imshow('train_img', train_img)


					cv2.namedWindow('train_img_corners', cv2.WINDOW_NORMAL)
					cv2.imshow('train_img_corners', draw_corners_on_marker(train_img, gt_corners.flatten()))

					cv2.waitKey(0)
					cv2.destroyAllWindows()


				if w % 1000 == 0:
					print(str(w) + '/' + str(numWarps))

			else:
				print("Warning: It could not be load background image: " + imBackground)
				w = w-1


	
	fileList_Train.close()
	fileList_Val.close()

if __name__ == "__main__":

	# NOTE
	# check this path dirs!

	# Dir in which the new patches are saved 
	outImPath = 'data/128/' 

	# Full file path referencing the patches and ground truth
	valFilename = 'data/128/val_data_list.txt' 
	trainFilename = 'data/128/train_data_list.txt' 

	verbose = False	# set True to display the process

	create_dataset('/home/mondejar/dataset/markers/', '/home/mondejar/dataset/mirflickr/', 22000, 128, outImPath, valFilename, trainFilename, verbose)
    