#!/bin/bash

<<<<<<< HEAD
if [ "$#" -ne 3 ]; then
  echo "usage: run.sh <model> <weights> <iterations>"
fi

ln -s /localhome/juddpatr/caffe_error/examples
ln -s /localhome/juddpatr/caffe_error/data
ln -s .skel/libcaffe.so

source /localhome/apps/src/caffe/caffe_rc

time .skel/caffe.bin test \
  -model $1 \
  -weights $2 \
  -iterations $3 

=======
CAFFE_DIR=/localhome/juddpatr/caffe

ln -s $CAFFE_DIR/examples
ln -s .skel/libcaffe.so

time .skel/caffe.bin test \
  -model model.prototxt \
  -weights .skel/weights.caffemodel \
  -iterations $1
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
