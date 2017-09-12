import numpy as np
import os
import sys
import time
import re
import code
import argparse

import caffe
from google.protobuf import text_format


def check_file(filename):
    assert os.path.isfile(filename), "%s is not a file" % filename


def read_prototxt(model):
    from caffe.proto import caffe_pb2
    net_param = caffe_pb2.NetParameter()

    print 'reading prototxt',model
    with open(model) as f:
        text_format.Merge(str(f.read()), net_param)

    return net_param


def roundup_to_multiple(x, m):
    return int(np.ceil( x / float(m))) * m

def divide_roundup(x, m):
    return roundup_to_multiple(x,m) / m

def to2d(a):
    a = np.array(a)
    return np.reshape(a, (a.shape[0],-1))

def range(d):
    return (d.min(), d.max())

##################### MAIN ########################################################################


parser = argparse.ArgumentParser(prog='test_net.py', description='Run a network in pycaffe')
parser.add_argument('-n', dest='network', type=str, default = '', help='network name in model directory. \'all\' to run all networks')
parser.add_argument('-b', dest='batches', type=int, help='batches to run')
parser.add_argument('-m', dest='model', type=str, default = '', help='model prototxt')
parser.add_argument('-w', dest='weights', type=str, default = '', help='weight caffemodel')

args = parser.parse_args()
batches = args.batches
network = args.network

if args.network:
    model = os.path.join(args.network,'train_val.prototxt')
    weights = os.path.join(args.network,'weights.caffemodel')
else:
    model = args.model
    weights = args.weights

assert (os.path.exists(model)), "file does not exist"
assert (os.path.exists(weights)), "file does not exist"

sys.path.insert(0, '/home/patrick/python')

caffe.set_mode_gpu()

net_param = read_prototxt(model)

net = caffe.Net(model, weights, caffe.TEST)

print 'variables: net, net_param'
print 'net.layers: vector of layers'
print '\tlayer: blobs (weights), reshape, setup, type'
print 'net.params: vector of weight blobs'
print 'net.blobs: map by layer name'
print '\tblob: count, shape, data, diff'

