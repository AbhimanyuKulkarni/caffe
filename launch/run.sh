#!/bin/bash

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

