
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import re
import code

def check_file(filename):
    assert os.path.isfile(filename), "%s is not a file" % filename


def read_params(net_name):
    ''' reads parameters of a network from a file
        Inputs:
            net_name    -- name of network, corresponding to directory in models dir

        Returns:
            conv_params -- dictionary mapping [layer name][parameter name] to the parameter value
            layers      -- list of layer names
    '''

    #param_file = caffe_root +  net_name + '_in.csv'
    #param_file = caffe_root +  net_name + '_trace_params.csv'
    param_file = caffe_root + 'models/' + net_name + '/conv_params.csv'
    check_file(param_file)

    net_conv_params = open(param_file,'r')
    lines = net_conv_params.readlines()
    if len(lines) == 0:
        print "WARNING: %s is emtpy" % param_file

    conv_params = {} # dict of dicts of parameters, indexed by layer
    layers = [] # store layers in order

    for l in lines:
        l = l.strip()
        print "line =", l
        layer, input_blob, bottom_size, kernel_size, stride, num_output, pad = l.split(',')
        layers.append(layer)
        conv_params[layer] = {}
        conv_params[layer]['input_blob'] = input_blob
        conv_params[layer]['bottom_size'] = bottom_size
        conv_params[layer]['kernel_size'] = kernel_size
        conv_params[layer]['stride'] = stride
        conv_params[layer]['num_output'] = num_output
        conv_params[layer]['pad'] = pad
    return conv_params, layers


def load_net(net_name):
    ''' load the network definition from the models directory
        Input:
            net_name -- name of network as used in the models directory
        Returns:
            net      -- caffe network object
    '''
    #model = caffe_root + 'models/' + net_name + '/deploy.prototxt'
    model = caffe_root + 'models/' + net_name + '/train_val.prototxt'
    weights = caffe_root + 'models/' + net_name + '/weights.caffemodel'
    check_file(model)
    check_file(weights)
    net = caffe.Net(model, weights, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    check_file(mean_file)
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 50
    batch_size = 50
    net.blobs['data'].reshape(batch_size,3,227,227)

    return net


def roundup_to_multiple(x, m):
    return int(np.ceil( x / float(m))) * m

def divide_roundup(x, m):
    return roundup_to_multiple(x,m) / m

def to2d(a):
    a = np.array(a)
    return np.reshape(a, (a.shape[0],-1))


def write_trace(net_name, batches):
    ''' runs the network for a specified number of batches and saves the inputs to each layer
        Inputs:
            net_name -- name of the network
            batches  -- number of batches to run
        Returns:
            nothing           
    '''

    # we need to get the input_blob to each layer, however each layer names matches its output blob
    # unfortunately the python API doesn't give you this, so we need to get this information from C++
    # and save it in a parameter file
    conv_params, layers = read_params(net_name)

    net = load_net(net_name)

    for b in range(batches):
        print "%s iteration %d" %(net_name, b)

        start = time.time()
        net.forward()
        end = time.time()
        print 'runtime: %.2f' % (end-start)

        if b < skip:
            continue 

        print 'layer, Nb, Ni, Nx, Ny'
        for l, layer in enumerate(layers):
            sane_layer = re.sub('/','-',layer) # sanitize layer name so we can save it as a file (remove /)
            net_dir = '%s/%s' % (trace_dir, net_name)
            savefile = '%s/%s-%d' % (net_dir, sane_layer, b)

            if os.path.isfile(savefile + ".npy"):
                print savefile, "exists, skipping"
                continue

            input_blob = conv_params[layer]['input_blob']
            data = net.blobs[input_blob].data
            if (len(data.shape) == 2):
                (Nb, Ni) = data.shape
                data = data.reshape( (Nb,Ni,1,1) )

            (Nb, Ni, Nx, Ny) = data.shape
            print layer, Nb, Ni, Nx, Ny
            
            if not os.path.exists(net_dir):
                os.makedirs(net_dir)
            print "saving: ", savefile
            np.save(savefile,data)

def write_config(file_handle, net_name):
    ''' write a set of layer parameters for each layer
        Input:
            file_handle -- file handle to write to
            net_name    -- name of network, corresponding to directory in models dir
        Returns:
            nothing
    '''

    conv_params, layers = read_params(net_name)

    net = load_net(net_name)
    
    print "layer, input, Nn, Kx, Ky, stride, pad"
    for l, layer in enumerate(layers):
        sane_layer = re.sub('/','-',layer) # sanitize layer name so we can save it as a file (remove /)

        input_blob = conv_params[layer]['input_blob']
        stride = conv_params[layer]['stride']
        pad = conv_params[layer]['pad']
        data = net.blobs[input_blob].data
        weights = net.params[layer][0].data
        print layer, "D", data.shape, "W", weights.shape

        if (len(weights.shape) == 2):
            (Nn, Ni) = weights.shape
            (Kx, Ky) = (1,1)
        else:
            (Nn, Ni, Kx, Ky) = weights.shape

        if (len(data.shape) == 2):
            (Nb, Ni) = data.shape
            (Nx, Ny) = (1,1)
        else:
            (Nb, Ni, Nx, Ny) = data.shape
            #(Kx, Ky) = (Nx, Ny)

        outstr = ','.join( [str(i) for i in [layer, input_blob, Nn, Kx, Ky, stride, pad]] ) + "\n"
        print outstr
        file_handle.write(outstr)

        
##################### MAIN ########################################################################

import caffe
import argparse

caffe_root      = './'  # this file is expected to be in {caffe_root}/examples
trace_dir       = caffe_root + '/net_traces' # write traces to this directory

# Make sure that caffe is on the python path:
sys.path.insert(0, caffe_root + 'python')

parser = argparse.ArgumentParser(prog='save_net.py', description='Run a network in pycaffe and save a trace of the data input to each layer')
parser.add_argument('network', metavar='network', type=str, help='network name in model directory. \'all\' to run all networks')
parser.add_argument('batches', metavar='batches', type=int, help='batches to run')
parser.add_argument('--skip', type=int,   default=0,          help='batches to skip')
parser.add_argument('-o'    , type=str,   default=trace_dir,  help='output directory for trace files')
parser.add_argument('-p'    , dest='write_params', action='store_true', help='write layer parameters for each net instead of writing trace')
parser.set_defaults(write_params=False)

args = parser.parse_args()
batches = args.batches
network = args.network
skip    = args.skip
write_params = args.write_params
trace_dir = args.o

if network == 'all':
    net_names = ['alexnet', 'nin_imagenet', 'googlenet', 'vgg_cnn_s', 'vgg_cnn_m_2048', 'vgg_19layers']
else:
    network = re.sub('models','',network)
    network = re.sub('/','',network)
    net_names = [network]
    netpath = caffe_root + 'models/' + network
    if not os.path.exists(netpath):
        print "Error: %s does not exist" % netpath
        sys.exit()

caffe.set_mode_cpu()

if write_params:
    for net_name in net_names:
        f = open(net_name + "_trace_params.csv", 'w')
        write_config(f, net_name)
        f.close()

else:
    # run each network and save an ndarray of input data
    for net_name in net_names:
        write_trace(net_name, batches)
    

