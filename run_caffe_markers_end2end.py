# Caffe Example:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb

caffe_root = '/home/mondejar/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

import cv2
from draw_corners_on_marker import *

batch_size = 100

caffe.set_device(0)

print("\n\n\n Mode GPU")
caffe.set_mode_gpu()
#caffe.set_mode_cpu()


### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('/home/mondejar/markers_end2end/my_net_auto_solver.prototxt')


# each output is (batch size, feature dim, spatial dim)
[(k, v.data.shape) for k, v in solver.net.blobs.items()]


# just print the weight sizes (we'll omit the biases)
[(k, v[0].data.shape) for k, v in solver.net.params.items()]


# TODO make a loop of this!
iter_max = 30000
for iteration in range(0, iter_max):
    #print("Iter " + str(iteration) + " / " + str(iter_max))
    solver.net.forward()  # train net
    #solver.test_nets[0].forward()  # test net (there can be more than one)
    solver.net.backward()

    if iteration % 5000 == 0:
        solver.net.save('net_snapshot/net.caffemodel') 
        print("Iter " + str(iteration) + " loss: ")
        print(solver.net.blobs['loss'])


# we use a little trick to tile the first eight images
#imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')

# verbose
# Display data 

for b in range(0, batch_size):
    gt_corr = np.array(solver.net.blobs['label'].data[b])
    predicted_coor = solver.net.blobs['score'].data[b]
    print('train labels:', gt_corr)
    print('Score output:', predicted_coor)

    # Draw Output
    im = np.array(solver.net.blobs['data'].data[b].reshape(64,64))
    im_pred = draw_corners_on_marker(im, predicted_coor * 255.0)
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', im_pred)
    key = cv2.waitKey(0)
    
    if key == 27:    # Esc key to stop
       sys.exit(0)
    cv2.destroyAllWindows()