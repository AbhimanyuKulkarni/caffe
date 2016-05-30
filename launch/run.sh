#!/bin/bash

caffeDir=/localhome/juddpatr/caffe

if [ "$#" -ne 3 ]; then
  echo "usage: run.sh <model> <weights> <iterations>"
fi

ln -s $caffeDir/examples
ln -s $caffeDir/data
ln -s .skel/libcaffe.so

source /localhome/apps/src/caffe/caffe_rc

time .skel/caffe.bin test \
  -model $1 \
  -weights $2 \
  -iterations $3 

.skel/postprocess.sh
