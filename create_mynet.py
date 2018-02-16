# Caffe Example:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb

caffe_root = '/home/mondejar/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P


def my_net(data_lmdb, label_lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    #n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, transform_param=dict(scale=1./255), ntop=2)
    n.data, n.data2  = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=data_lmdb, transform_param=dict(scale=1./255), ntop=2)
    n.label, n.label2  = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=label_lmdb, ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    
    n.score = L.InnerProduct(n.relu1, num_output=8, weight_filler=dict(type='xavier'))
    n.loss =  L.EuclideanLoss(n.score, n.label)
    
    return n.to_proto()
    
with open('/home/mondejar/markers_end2end/my_net_auto_train.prototxt', 'w') as f:
    f.write(str(my_net('/home/mondejar/markers_end2end/markers_img_LMDB', '/home/mondejar/markers_end2end/markers_labels_LMDB', 100)))
    
