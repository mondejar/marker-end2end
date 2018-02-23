# This code read the image files and labels and create a LMDB dataset
# for Caffe compatible
#
# Based on: http://deepdish.io/2015/04/28/creating-lmdb-in-python/
#
#
# Author: Mondejar-Guerra, Victor
# 16/02/2018
################################################################################

caffe_root = '/home/mondejar/caffe-master/'  
# this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np
import lmdb
import cv2
import shutil
import os 

def data_2_LMDB(dirBase, datasetFile, marker_size, output_dir, batch_size):

    img_LMDB_name = output_dir + '/markers_img_LMDB'
    label_LMDB_name = output_dir + '/markers_labels_LMDB'

    fileDB = open(datasetFile,'r') 
    content = fileDB.readlines()
    fileDB.close()

    N = len(content)

    # Let's pretend this is interesting data
    X = np.zeros((N, 1, marker_size, marker_size), dtype=np.uint8)
    Y = np.zeros((N, 1, 1, 8), dtype=np.float32)

    # Read the dataset and set in X,y variables
    for i in range(0, len(content)):
        line_splitted = content[i].split()
        img_path = line_splitted[0]     
        # Convert matrix-images to a single array
        img = cv2.imread(dirBase + img_path, 0)
        X[i,0] = img

        for j in range(1,9):
            Y[i, 0, 0, j-1] = float(line_splitted[j]) / float(marker_size)

    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size_images = X.nbytes * 10
    map_size_labels  = Y.nbytes * 10
    # For multilabels we need to create two separate LMDB directories
    # for image and labels

    # if exist
    if os.path.exists(img_LMDB_name):
        shutil.rmtree(img_LMDB_name)

    if os.path.exists(label_LMDB_name):
        shutil.rmtree(label_LMDB_name)

    images_db = lmdb.open(img_LMDB_name, map_size=map_size_images)
    labels_db = lmdb.open(label_LMDB_name, map_size=map_size_labels)

    images_txn = images_db.begin(write=True)
    labels_txn = labels_db.begin(write=True)

    # txn is a Transaction object
    print("N images: " + str(N))

    item_id = -1
    for i in range(0,N):
        item_id += 1
        # Images
        str_id = '{:08}'.format(i)
        datum_img = caffe.io.array_to_datum(X[i], 1) #, label_value
        # The encode is only essential in Python 3
        images_txn.put(str_id.encode('ascii'), datum_img.SerializeToString())

        # Labels
        datum_label = caffe.io.array_to_datum(Y[i], 1) #, label_value
        labels_txn.put(str_id.encode('ascii'), datum_label.SerializeToString())

        # write batch
        if(item_id + 1) % batch_size == 0:
            images_txn.commit()
            images_txn = images_db.begin(write=True)

            labels_txn.commit()
            labels_txn = labels_db.begin(write=True)

            print (item_id + 1)

    # write last batch
    if (item_id+1) % batch_size != 0:
        images_txn.commit()
        labels_txn.commit()
        print 'last batch'
        print (item_id + 1)


if __name__ == "__main__":

    dirBase = '/home/mondejar/markers_end2end/'
    marker_size = 64

    print("Creating LMDB train data ")
    datasetFile = '/home/mondejar/markers_end2end/data/64/train_data_list.txt'
    data_2_LMDB(dirBase, datasetFile, marker_size, 'LMDB/training', 100)


    print("Creating LMDB validation data ")
    datasetFile = '/home/mondejar/markers_end2end/data/64/val_data_list.txt'
    data_2_LMDB(dirBase, datasetFile, marker_size, 'LMDB/validation', 100)