#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=examples/imagenet
DATA=data/ilsvrc12
TOOLS=build/tools

<<<<<<< HEAD
TRAIN_DATA_ROOT=/localhome/apps/src/caffe/datasets/imagenet/set1/ILSVRC2012_sample_10_5/sample_10_train/
VAL_DATA_ROOT=/localhome/apps/src/caffe/datasets/imagenet/set1/ILSVRC2012_sample_10_5/sample_5_val/
=======
DATA_ROOT=/localhome/apps/src/caffe/datasets/imagenet/ILSVRC2012_sample_10_5

TRAIN_DATA_ROOT=$DATA_ROOT/sample_10_train/
TRAIN_DATA_LIST=$DATA_ROOT/sample_10_train.txt
VAL_DATA_ROOT=$DATA_ROOT/sample_5_val/
VAL_DATA_LIST=$DATA_ROOT/sample_5_val.txt
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $TRAIN_DATA_LIST \
    $EXAMPLE/ilsvrc12_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $VAL_DATA_LIST \
    $EXAMPLE/ilsvrc12_val_lmdb

echo "Done."
