#!/bin/bash

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
iters=1
gpu="-gpu 0"

showhelp(){
echo "
Usage: ./test.sh models/<net>

Options:
  -i        iterations, default=1
  -c        cpu mode, default=false
"
}

while getopts "h?i:c" opt; do
  case "$opt" in
  h|\?) showhelp; exit 0 ;;
  i)    iters=$OPTARG ;;
  c)    gpu="";;
  esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift


MODEL_DIR=$1

if [ "$MODEL_DIR" = "" ]; then
  showhelp
  exit
fi

cmd="./build/tools/caffe test $gpu -model $MODEL_DIR/train_val.prototxt -weights $MODEL_DIR/weights.caffemodel -iterations $iters"
echo $cmd
exec $cmd
