#!/bin/bash
# this script downloads all the model weights for the networks I have been using in our papers

# LeNet
if [ ! -f data/mnist/t10k-images-idx3-ubyte ]; then
  ./data/mnist/get_mnist.sh
fi
if [ ! -d examples/mnist/mnist_test_lmdb ]; then
  ./examples/mnist/create_mnist.sh
fi
if [ ! -f models/lenet/lenet_iter_10000.caffemodel ]; then
  wget http://www.eecg.toronto.edu/~juddpatr/files/caffe_models/lenet_iter_10000.caffemodel -P models/lenet/.
fi
if [ ! -f models/lenet/weights.caffemodel ]; then
  cd models/lenet/
  ln -s lenet_iter_10000.caffemodel weights.caffemodel
  cd -
fi

# Convnet
if [ ! -f ./data/cifar10/data_batch_1.bin ]; then
  ./data/cifar10/get_cifar10.sh
fi
if [ ! -d examples/cifar10/cifar10_test_lmdb ]; then
  ./examples/cifar10/create_cifar10.sh
fi
if [ ! -f models/convnet/cifar10_quick_iter_4000.caffemodel ]; then
  wget http://www.eecg.toronto.edu/~juddpatr/files/caffe_models/cifar10_quick_iter_4000.caffemodel -P models/convnet/.
fi
if [ ! -f models/convnet/weights.caffemodel ]; then
  cd models/convnet/
  ln -s cifar10_quick_iter_4000.caffemodel weights.caffemodel
  cd -
fi

# Imagenet networks

for net in alexnet googlenet nin_imagenet vgg_cnn_s vgg_cnn_m_2048 vgg_19layers; do
  echo "downloading mode for $net"
  if [ -f models/$net/weights.caffemodel ]; then
    echo "  caffemodel already downloaded"
  else
    url=`grep caffemodel_url models/$net/readme.md | cut -d' ' -f2`
    cd models/$net
    wget $url -O weights.caffemodel
    cd -
  fi 
done

echo "follow this example to prepare the imagenet dataset: http://caffe.berkeleyvision.org/gathered/examples/imagenet.html"
echo "you will need to create an account to get access to the dataset"
