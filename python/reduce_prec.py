import numpy as np
from scipy.stats import itemfreq, norm
import matplotlib.pyplot as plt
import os
import sys
import time
import re
import code
import argparse

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def check_file(filename):
    assert os.path.isfile(filename), "%s is not a file" % filename

def read_prototxt(model):
    net_param = caffe_pb2.NetParameter()

    print 'reading prototxt',model
    with open(model) as f:
        text_format.Merge(str(f.read()), net_param)

    return net_param

def write_prototxt(filename, proto):
    open(filename, 'w').write(str(proto))

def get_layer_by_name(net_proto, name):
    for layer in net_proto.layer:
        if layer.name == name:
            return layer
    return None

def create_prec_dict(layers, mags, precs):
    d = dict()
    for l,m,p in zip(layers, mags, precs):
        d[l] = (m,p)
    return d

def test_net(net, test_it=100):
    test_it_acc = []
    test_it_loss = []
    for test_it in range(test_it):
        net.forward() # run inference on batch
        test_it_acc  += [net.blobs['accuracy'].data[()]]
        test_it_loss += [net.blobs['loss'].data[()]]
    test_acc  = np.mean(test_it_acc)
    test_loss = np.mean(test_it_loss)
    return test_acc, test_loss

def roundup_to_multiple(x, m):
    return int(np.ceil( x / float(m))) * m

def divide_roundup(x, m):
    return roundup_to_multiple(x,m) / m

def to2d(a):
    a = np.array(a)
    return np.reshape(a, (a.shape[0],-1))

def pad_1d(d, size):
    ''' pad weights to fit in the SBs of DaDianNao, also converts fc weights to 4D '''
    dim = 0
    s = list(d.shape)
    assert len(s) == 1, 'd is not 4d'
    d = np.reshape(d,s)
    pad_shape = list(s)
    pad_shape[dim] = roundup_to_multiple(pad_shape[dim],size)
    d_pad = np.zeros(pad_shape, dtype=d.dtype)
    d_pad[:s[0]]  = d
    return d_pad

def pad_2d(d, dim, size):
    ''' pad weights to fit in the SBs of DaDianNao, also converts fc weights to 4D '''
    s = list(d.shape)
    assert len(s) == 2, 'd is not 4d'
    d = np.reshape(d,s)
    pad_shape = list(s)
    pad_shape[dim] = roundup_to_multiple(pad_shape[dim],size)
    d_pad = np.zeros(pad_shape, dtype=d.dtype)
    d_pad[:s[0], :s[1]]  = d
    return d_pad

def pad_4d(d, dim, size):
    ''' pad weights to fit in the SBs of DaDianNao, also converts fc weights to 4D '''
    s = list(d.shape)
    assert len(s) == 4, 'd is not 4d'
    d = np.reshape(d,s)
    pad_shape = list(s)
    pad_shape[dim] = roundup_to_multiple(pad_shape[dim],size)
    d_pad = np.zeros(pad_shape, dtype=d.dtype)
    d_pad[:s[0], :s[1], :s[2] ,:s[3]]  = d
    return d_pad

def drange(d):
    return (d.min(), d.max())

def plot_layer_dynamic_prec(net, net_proto, layer, images=50, verbose=0, perimage=False):
    input_blob = None
    stride = 1
    for l in net_proto.layer:
        if layer in l.name and l.type == 'Convolution':
            print 'found layer', l.name
            input_blob = l.bottom[0]
            prec = l.fwd_act_precision_param.precision 
            scale = l.fwd_act_precision_param.scale 
            if l.convolution_param.stride:
                stride = l.convolution_param.stride[0]
            break
    if not input_blob:
        print "No matching layer found"
        return None

    print prec, scale

    plt.clf()
    # plt.title(net_proto.name + ' ' +  layer)
    plt.title('Fraction of Activations')
    plt.xlabel('Precision [bits]')
    plt.ylim(-0.01,1.01)
    plt.grid(True)
    plt.xticks(range(0,17))

    dynmax = []

    ims = 1
    if perimage: ims = images

    for im in range(ims):

        if perimage:
            data = net.blobs[input_blob].data[im:im+1]
        else:
            data = net.blobs[input_blob].data[0:images]

        n,c,h,w = data.shape
        data = data.swapaxes(1,3) # n w h c
        data = pad_4d(data, 2, stride)
        n,w,h,c = data.shape
        data = data.reshape(n, w, h/stride, c*stride)
        data = pad_4d(data, 3, 16)
        n,w,h,c = data.shape
        data = data.reshape(n,w,h,c/16,16)
        data = data.swapaxes(1,3) # n c h w
        data = data.reshape(-1)

        for gg in range(8,9):
            g = 1<<gg
            d = pad_1d(data, g)
            d = d.reshape(-1,g)
            print d.shape
            dmax = d.max(1)
            dmin = d.min(1)

            ps = [0]
            cs = [0]
            for p in range(1,17):
                pmin = (-2**(p-1)) / scale
                pmax = (2**(p-1)-1) / scale

                c = ((dmax <= pmax) * (dmin >= pmin)).sum()
                
                if verbose: print pmin, pmax, c

                ps += [p]
                cs += [c]

            cs = np.array(cs, dtype=float)
            cs = cs/cs[-1]

            dynmax += [ps[np.argmax(cs)]]

            if perimage:
                label = 'image %d'%im
            else:
                label = g
            plt.plot(ps,cs, label=label)

    plt.axvline(max(dynmax), linestyle='--', color='b', label='dynamic')
    plt.axvline(prec, linestyle=':', color='r', label='static')
    plt.legend()




##################### MAIN ########################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='reduce_prec.py', description='Run a network in pycaffe with reduced precision')
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

    caffe.set_mode_gpu()

    net_proto = read_prototxt(model)

    
    # mag_prec = np.loadtxt( os.path.join(args.network,'mag_prec_100.csv'), delimiter=',' )
    # mags = mag_prec[:,0]
    # precs = mag_prec[:,1]
    mags = np.loadtxt( os.path.join(args.network,'best-mag-error0.csv'), delimiter=',' )
    precs = np.loadtxt( os.path.join(args.network,'best-prec-error0.csv'), delimiter=',' )
    mags = mags[:len(precs)]
    # bits = np.loadtxt( os.path.join(args.network,'custom_bits.csv'), delimiter=',' ,dtype=str)[1:]
    bits = mags + precs + 1
    layers = open(os.path.join(args.network,'layers.csv')).readlines()
    bits = bits.astype(float)
    # precs = bits - mags

    for layer in net_proto.layer:
        if layer.type not in ['Convolution']: continue
        i = np.argmax([layer.name in s for s in layers])
        print layer, i
        b = bits[i]
        p = precs[i]
        scale = 2**(p)
        layer.fwd_act_precision_param.precision = int(b)
        layer.fwd_act_precision_param.scale = scale
        layer.fwd_act_precision_param.quantizer = 3
        layer.fwd_act_precision_param.store_reduced = True

    model = 'temp_net.prototxt'
    write_prototxt(model, net_proto)
    net = caffe.Net(model, weights, caffe.TEST)
    # print test_net(net)
    net.forward()
    print 'mags:', mags
    print 'bits:', bits
