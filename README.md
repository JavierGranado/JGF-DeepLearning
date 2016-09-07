# JGF-DeepLearning - Deep Learning algorithms 


VISUALIZE DEEPER OUTUPUT LAYERS:

Visualize deeper layers in CIFAR10 under Tensorflow by displaying images which gain the highest response from neurons. Written for cifar10 model (https://gist.github.com/kukuruza).

TODO: Adapt the code in pretrained Alexnet model under Tensorflow to get the information about these output layers, locate objects in input images and post-processing them (MatLab) to avoid extra-image processing of unwanted objects.


ALEXNET TENSORFLOW IMPLEMENTATION:

Homepage:
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

bvlc_alexnet.py and bvlc_alexnet.npy are generated using https://github.com/ethereon/caffe-tensorflow by converting the AlexNet weights/model from here:
https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet


myalexnet is the implementation of AlexNet in TensorFlow

Weights are available here:
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy

Try myalexnet_forward.py for a version with a placeholder as the input (useful for training). Otherwise see myalexnet.py

