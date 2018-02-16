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

""" 
This code prepares the data for train the model.
Given two directories, the first one with the markers and the second one with the background images.
This code creates a file with:
	- N transformations for each marker M over N different images 
	- The 4 coordinates that enclose each marker M on the image	    		

python marker_dataset_path img_dataset_path number_transforms patch_size
				
Example call:

python create_dataset.py /home/mondejar/dataset/markers/ /home/mondejar/dataset/mirflickr/ 500 128
"""

# Params
outImPath = 'data_64/train_data/' #'data_simplified/train_data/'						# Dir in which the new patches are saved 
trainFilename = 'data_64/train_data_list.txt' # 'data_simplified/train_data_list.txt'  # Full file path referencing the patches and ground truth
verbose = False 																	# set True to display the process

# Generate a random affine transform over the four corners
def affine_transform( patchSize, img, randomDispFactor ):
	
	corners = np.float32([[0, 0], [0, patchSize], [patchSize, 0], [patchSize, patchSize]])
	cornersT = np.float32([[1, 1], [1, 1], [1, 1], [1, 1]])

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
	warpImg = cv2.warpPerspective(img, persT, (maxW, maxH))

	# Set mask for only select the marker affine
	mask_perspect = np.zeros((patchSize, patchSize, 1), dtype = "uint8") + 255
	mask_perspect = cv2.warpPerspective(mask_perspect, persT, (maxW, maxH))

	return warpImg, persT, mask_perspect, cornersT

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

def main(argv):

	"""
	if len(argv) < 5:
		print 'Error, incorrect number of args:\n python create_dataset.py marker_dataset_path img_dataset_path number_transforms patch_size\n'
		sys.exit(1)

	if not os.path.exists(outImPath):
		os.mkdir(outImPath)
	"""
	marker_dataset 	= argv[1]  #'/home/mondejar/dataset/markers/'  	# Dir that contains the original markers
	img_dataset 	= argv[2] #'/home/mondejar/dataset/mirflickr/' 	# Dir that contains the background images
	numWarps 		= int(argv[3]) 	#100 number of warps per marker
	patchSize 		= int(argv[4]) 	#128 patch size of the resultant image

	# create file
	fileList = open(trainFilename,'w') 
	imMarkers = [f for f in listdir(marker_dataset) if isfile(join(marker_dataset, f))]
	imBackgrounds = [f for f in listdir(img_dataset) if isfile(join(img_dataset, f))]
	
	numIm = 0
	for imMarker in imMarkers:
		# Read marker
		print(marker_dataset + imMarker)
		marker_orig = cv2.imread(marker_dataset + imMarker, 0)

		for w in range(0,numWarps):

			# Pick a random background image
			imBackground = random.choice(imBackgrounds)
			back_img = cv2.imread(img_dataset + imBackground, 0)

			if not back_img is None:
				# Resample background image to specified size
				train_img = cv2.resize(back_img, (patchSize, patchSize)) 				
				
				# Scale the marker at some size between 10-50% of the specified size
				scale_factor = random.uniform(0.1, 0.5)
				marker_size = int(patchSize * scale_factor)
				marker_scale = cv2.resize(marker_orig, (marker_size, marker_size))

				# TODO: add more transformations to the marker

				# gray level?
				marker_scale = dynamic_range_compression(marker_scale)

				# blurring: to the marker or to the global image?

				# Ilumination? non uniform?

				# affine transform
				marker_affin, persT, mask_perspect, gt_corners = affine_transform(marker_size, marker_scale, 0.4) #0.001

				rows_marker, cols_marker = marker_affin.shape
			    # and place randomly over the background image
				x_pos = np.random.randint(0, patchSize - cols_marker - 1)
				y_pos = np.random.randint(0, patchSize - rows_marker - 1)				

				train_img = merge_images(train_img, x_pos, y_pos, marker_affin, mask_perspect)


				# Export the image and write the four corners on the file
				gt_corners[:,0] = gt_corners[:,0] + x_pos
				gt_corners[:,1] = gt_corners[:,1] + y_pos


				# Export patch warp
				nameTrainImg = outImPath + imMarker[:-5] + "_" + str(w) + '.png'
				cv2.imwrite( nameTrainImg, train_img)	

				# add line to file
				fileList.write( nameTrainImg)
				for p in range(0, 4):	
					fileList.write(' ' + str(gt_corners[p][0]) + ' ' + str(gt_corners[p][1]))
				fileList.write('\n')

				# Write .bin files for use with caffe 

				if verbose:
					cv2.namedWindow('train_img', cv2.WINDOW_NORMAL)
					cv2.imshow('train_img', train_img)
					cv2.waitKey(200)
					cv2.destroyAllWindows()

			else:
				print("Warning: It could not be load background image: " + imBackground)
				w = w-1
	fileList.close()

if __name__ == "__main__":
    main(sys.argv)
