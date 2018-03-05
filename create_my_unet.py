# Caffe Example:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb

caffe_root = '/home/mondejar/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P


def my_net(data_train_lmdb, label_train_lmdb, data_val_lmdb, label_val_lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, 
        source=data_train_lmdb, ntop=1, transform_param=dict(scale=1./255), include={'phase':caffe.TRAIN})  #, transform_param=dict(scale=1./255) 
        
    n.label = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, 
        source=label_train_lmdb, ntop=1, transform_param=dict(scale=1./255), include={'phase':caffe.TRAIN})  #transform_param=dict(scale=1./255) 

    n.test_data = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, 
        source=data_val_lmdb, ntop=1, transform_param=dict(scale=1./255), include={'phase':caffe.TEST})   #transform_param=dict(scale=1./255) 
    n.test_label = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB,
        source=label_val_lmdb, ntop=1, transform_param=dict(scale=1./255), include={'phase':caffe.TEST})   #transform_param=dict(scale=1./255) 

    # TODO add the copy channels from original size to the next layers! u-net!
    # CROP CONCAT... see unet.prototxt example...

    # Downsampling  (pool)
    #############################################################################################
    # in: 64 x 64
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=32, pad = 1, weight_filler=dict(type='xavier')) 
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    # with padding (64 x 64)     in: 62 x 62 
    
    #n.conv2 = L.Convolution(n.conv1,kernel_size=3, num_output=32, pad = 1, weight_filler=dict(type='xavier'))
    #n.relu2 = L.ReLU(n.conv2, in_place=True)
    # with padding (64 x 64)   in: 60 x 60
    
    n.pool2 = L.Pooling(n.conv1,    kernel_size=3, stride=2, pool=P.Pooling.MAX)
    
    
    ####################################
    # with padding (32 x 32)  in: 30 x 30
    n.conv3 = L.Convolution(n.pool2, kernel_size=3, num_output=64, pad = 1, weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    # with padding (32 x 32)   in: 28 x 28
    #n.conv4 = L.Convolution(n.conv3, kernel_size=3, num_output=64, pad = 1, weight_filler=dict(type='xavier'))
    #n.relu4 = L.ReLU(n.conv4, in_place=True)
    """
    # with padding (32 x 32)   in: 26 x 26
    n.pool4 = L.Pooling(n.relu4,    kernel_size=3, stride=2, pool=P.Pooling.MAX)
    
    
    #######################################

    # with padding (16 x 16)    in: 13 x 13
    n.conv5 = L.Convolution(n.pool4,kernel_size=3, num_output=64, pad = 1, weight_filler=dict(type='xavier'))
    n.relu5 = L.ReLU(n.conv5, in_place=True)

    # with padding (16 x 16)    in: 11 x 11
    n.conv6 = L.Convolution(n.conv5,kernel_size=3, num_output=128, pad = 1, weight_filler=dict(type='xavier'))
    n.relu6 = L.ReLU(n.conv6, in_place=True)
    
    # with padding (16 x 16)     in: 9 x 9
    n.conv7 = L.Convolution(n.conv6,kernel_size=3, num_output=64, pad = 1, weight_filler=dict(type='xavier'))
    n.relu7 = L.ReLU(n.conv7, in_place=True)
    
    # NOTE check num_output at deconv
    n.deconv7 = L.Deconvolution(n.conv7, convolution_param=dict(num_output=32,  kernel_size=2, stride=2))
    n.concat7 = L.Concat(n.conv4, n.deconv7)

    # Upsampling  (deconv)
    """
    #############################################################################################
    # with padding (32 x 32)   in: 18 x 18 ?


    #n.conv8 = L.Convolution(n.concat7, kernel_size=3, num_output=32, pad = 1, weight_filler=dict(type='xavier'))
    n.conv8 = L.Convolution(n.conv3, kernel_size=3, num_output=64, pad = 1, weight_filler=dict(type='xavier'))
    n.relu8 = L.ReLU(n.conv8, in_place=True)
    
    # with padding (32 x 32)   in: 16 x 16 ?
    #n.conv9 = L.Convolution(n.conv8, kernel_size=3, num_output=64, pad = 1, weight_filler=dict(type='xavier'))
    #n.relu9 = L.ReLU(n.conv9, in_place=True)

    # with padding (32 x 32)   in: 14 x 14 ?
    n.deconv9 = L.Deconvolution(n.conv8, convolution_param=dict(num_output=32, kernel_size=2, stride=2))
    n.concat9 = L.Concat(n.conv1, n.deconv9)

    ###################################

    # with padding (64 x 64)   in: 28 x 28 ?
    n.conv10 = L.Convolution(n.concat9, kernel_size=3, num_output=32, pad = 1, weight_filler=dict(type='xavier'))
    n.relu10 = L.ReLU(n.conv10, in_place=True)

    # with padding (64 x 64)   in 26 x 26 ?
    n.conv11 = L.Convolution(n.conv10, kernel_size=3, num_output=32, pad = 1, weight_filler=dict(type='xavier'))
    n.relu11 = L.ReLU(n.conv11, in_place=True)

    # with padding (64 x 64)   in: 24 x 24 ?
    #####################################
    
    # 64 x 64
    n.conv12 = L.Convolution(n.conv11, kernel_size=3, num_output=1, pad = 1, weight_filler=dict(type='xavier'))
    # LOSS
    n.loss =  L.SigmoidCrossEntropyLoss(n.conv12, n.label)
    #n.loss =  L.SoftmaxWithLoss(n.conv12, n.label) #default L1,  L2?
    #AbsVal, Accuracy, ArgMax, BNLL, BatchNorm, BatchReindex, Bias, Concat, ContrastiveLoss, Convolution, Crop, Data, Deconvolution, Dropout, DummyData, ELU, Eltwise, Embed, EuclideanLoss, Exp, Filter, Flatten, HDF5Data, HDF5Output, HingeLoss, Im2col, ImageData, InfogainLoss, InnerProduct, Input, LRN, LSTM, LSTMUnit, Log, MVN, MemoryData, MultinomialLogisticLoss, PReLU, Parameter, Pooling, 
    #Power, Python, RNN, ReLU, Reduction, Reshape, SPP, Scale, Sigmoid, SigmoidCrossEntropyLoss, Silence, Slice, Softmax, SoftmaxWithLoss, Split, TanH, Threshold, Tile, WindowData
    # TODO  test_data and test_label to data and label.... phase TEST

    return n.to_proto()
    
with open('/home/mondejar/markers_end2end/my_unet.prototxt', 'w') as f:
    marker_size = 64#128
    batch_size = 1

    lmdb_dir_tr  = '/home/mondejar/markers_end2end/LMDB/' + str(marker_size) + '/training'
    lmdb_dir_val = '/home/mondejar/markers_end2end/LMDB/' + str(marker_size) + '/validation'

    f.write(str(my_net(lmdb_dir_tr + '/markers_img_LMDB', lmdb_dir_tr + '/markers_labels_mask_LMDB',
        lmdb_dir_val + '/markers_img_LMDB', lmdb_dir_val + '/markers_labels_mask_LMDB', batch_size)))
    
