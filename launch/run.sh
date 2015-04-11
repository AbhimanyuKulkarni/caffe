#!/bin/bash

CAFFE_DIR=/localhome/juddpatr/caffe

ln -s $CAFFE_DIR/examples
ln -s .skel/libcaffe.so

time .skel/caffe.bin test \
  -model model.prototxt \
  -weights .skel/weights.caffemodel \
  -iterations $1
