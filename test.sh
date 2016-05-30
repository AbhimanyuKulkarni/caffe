#!/bin/bash

MODEL_DIR=$1

./build/tools/caffe test -model $MODEL_DIR/train_val.prototxt -weights $MODEL_DIR/weights.caffemodel -iterations 1
