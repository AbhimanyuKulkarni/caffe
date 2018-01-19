#!/bin/bash

caffeDir=/localhome/juddpatr/caffe

if [ "$#" -ne 1 ]; then
  echo "usage: train.sh <solver>"
fi

ln -s $caffeDir/examples
ln -s $caffeDir/data
ln -s .skel/libcaffe.so

source /localhome/apps/src/caffe/caffe_rc

time .skel/caffe.bin train \
  -solver $1 \

.skel/postprocess.sh
