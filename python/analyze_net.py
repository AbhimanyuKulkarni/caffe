import argparse
import sys
import os
os.environ['GLOG_minloglevel'] = '2' # supress caffe output
import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from scipy.stats import itemfreq, norm
from my_train import read_net_prototxt

def set_gpu():
    caffe.set_gpu_mode()

def blob_stats(net):
    print "%10s," % '', ','.join(['%12s'%s for s in ['mean','std','min','max']])
    for blob in net.blobs:
        if 'label' in blob:
            continue 
        if 'split' in blob:
            continue 
        data = net.blobs[blob].data
        stats = [data.mean(), data.std(), data.min(), data.max()]
        stats = ["%12.4f" % s for s in stats]
        print "%10s," % blob, ','.join(stats)

def update_model(model, layer, prec, scale):
    netproto = read_net_prototxt(model)
    n = [l.name for l in netproto.layer]
    netproto.layer[n.index(layer)].fwd_act_precision_param.precision = prec
    netproto.layer[n.index(layer)].fwd_act_precision_param.scale = scale
    for l in netproto.layer:
        if l.fwd_act_precision_param.precision > 0:
            print l.name, l.fwd_act_precision_param.precision, l.fwd_act_precision_param.scale
    open(model,'w').write(str(netproto))
