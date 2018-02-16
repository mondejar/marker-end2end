# Introduction

The idea of this work is based on: https://arxiv.org/abs/1606.03798

- [Objective](#objetive)
  - [Training Dataset](#training-dataset)
  - [Model CNN](#model-cnn)
  - [Test](#Test)
- [Requirements](#requirements)


# Objetive

The goal of this network is the following:

	- given a fiducial square marker in a scene
	- return the four corners of the fiducial square marker (and its ID)

## Training Dataset

To train this model we need thousands of samples.

For that purpose the [MIRFLICKR-25000](http://press.liacs.nl/mirflickr/mirdownload.html) is used. 

For each fiducial marker some perspective transformations are made.
Additionally illumination, blurring and even occlusion changes could do the model more robust.

The input training data consist in:
- One image of size (128 x 128)

The training label consist in:
- Four corners: (x1, y1, x2, y2, x3, y3, x4, y4)


Caffe:

"For LMDB data source you need to separate your data input and your labels by creating two LMDB (one for the data and the second one for the labels). You also have to define two data layers in your network definition, set the same batch size for both of them and disable shuffling for the alignment."
http://blog.kostyaev.me/blog/Multilabel-Dataset


## Model CNN

The error of the network is computed following the eq():

	Dist_Euclidean( predicted_corners - groundTruth_corners)


## Test

For test the funtionallity of this model we use several datasets in which the homographies ground-truth are providad
 (Oxford, Nothredame?? ). 

Additionally, we could extract several patches for each imagen to grow up the number of test-samples.




# Requirements

## [OpenCV](https://github.com/opencv/opencv)
http://opencv.org/

## Python
### python-opencv

```
sudo apt-get install python-opencv
```
### Numpy
```
pip install numpy
```
### python-matplotlib
To display plots on python

```
sudo apt-get install python-matplotlib
```

## [Tensorflow](TF_installation_guide.md)
https://www.tensorflow.org/install/