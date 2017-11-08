#!/usr/bin/env python

import numpy as np
import os
import sys
import re
import argparse
import yaml
import csv

def roundup_to_multiple(x, m):
    return int(np.ceil( x / float(m))) * m

def divide_roundup(x, m):
    return roundup_to_multiple(x,m) / m

def sparsify(d, p=0.5):
    ''' keep element with probability p, otherwise 0 '''
    return d * np.random.binomial(1, p, d.shape)

def get_default_config():
    doc = """
    Ht: 8
    Wt: 8
    Kt: 64
    """
    config = yaml.load(doc)
    return config

def read_trace_params(param_filename, layer):
    params = dict()
    with open(param_filename, 'r') as csvfile:
        entries = csv.reader(csvfile)
        for row in entries:
            name = row.pop(0)
            if name == layer:
                # outstr = ','.join( [str(i) for i in [name, input_blob, Nn, Kx, Ky, stride, pad]] ) + "\n"
                for k in ['input_blob', 'Nn', 'Kx', 'Ky', 'stride', 'pad']:
                    v = row.pop(0)
                    try:
                        params[k] = int(v)
                    except ValueError:
                        params[k] = v
    return params

def create_idxmap(d):
    s = list(d.shape)
    s.append(len(s))
    idxmap = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            idxmap[i,j] = [i,j]
    return idxmap

def process_tile(act_data, wgt_data):

    print 'act', act_data.shape
    print 'wgt', wgt_data.shape
    time = 0

    # create index map
    act_idx = create_idxmap(act_data)
    wgt_idx = create_idxmap(wgt_data)
    print act_idx
    print wgt_idx

    # linearize tile
    actsize = act_data.size
    wgtsize = wgt_data.size
    act_data = act_data.reshape(actsize)
    wgt_data = wgt_data.reshape(wgtsize)
    act_idx = act_idx.reshape((actsize,-1))
    wgt_idx = wgt_idx.reshape((wgtsize,-1))
    print act_data
    print act_idx.T

    # deparsify

    # chunk into 4

    # compute coordinates

    # map to accumulators

    # count the number of conflicts

    return time

def schedule_act_local(act_data, wgt_data, dims, config, trace_params):
    (N,C,K,H,W,R,S,X,Y) = dims
    stride = trace_params['stride']
    pad = trace_params['pad']
    print 'W', W
    print 'H', H
    total_time = 0
    for n in range(N):
        for k in range(K):
            for c in range(C):
                times = []

                # FIXME: should be dividing by Wt, Ht
                tw = W/config['Wt']
                th = H/config['Ht']
                for w in range(0,W,tw):
                    for h in range(0,H,th):
                        print 'tile',w,h
                        x = w + R - 1
                        y = h + S - 1
                        act = act_data[n,c,x:x+1,y:y+1]
                        wgt = wgt_data[k,c]
                        # times.append( process_tile(act,wgt) )
                print 'debug exit'
                sys.exit()

                total_time += max(times)

def pad_weights(w, dim0):
    ''' pad weights to fit in the SBs of DaDianNao, also converts fc weights to 4D '''
    ws = list(w.shape)
    while len(ws) < 4:
        ws += [1]
    w = np.reshape(w,ws)
    pad_shape = list(w.shape)
    pad_shape[0] = roundup_to_multiple(pad_shape[0],dim0)
    w_pad = np.zeros(pad_shape)
    w_pad[:w.shape[0], :, : ,:]  = w
    return w_pad

def schedule_wgt_local(act_data, wgt_data, dims, config, trace_params):
    (N,C,K,H,W,R,S,X,Y) = dims
    stride = trace_params['stride']
    pad = trace_params['pad']
    Kt = config['Kt']

    total_time = 0
    wgt_pad = pad_weights(wgt_data, Kt)
    for n in range(N):
        for c in range(C):
            for kt in range(0,K,Kt):
                times = []
                for k in range(kt, kt+Kt):
                    print n,c,k
                    act = act_data[n,c]
                    wgt = wgt_data[k,c]
                    times.append( process_tile(act,wgt) )
                    print 'debug exit'
                    sys.exit()

                total_time += max(times)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='scnn.py', description='Performance Model of SCNN')
    parser.add_argument('act_trace', type=str, help='network name in model directory. \'all\' to run all networks')
    args = parser.parse_args()

    try:
        act_filename = args.act_trace
        trace_dir = os.path.dirname(act_filename)
        f = os.path.basename(act_filename)
        _, layer, batch = os.path.splitext(f)[0].split('-')
    except ValueError:
        print "Error: activation trace format error"
        print "expecting ./path/to/act-layer-batch.npy"
        print "got", args.act_trace
        sys.exit(0)

    print 'simulating layer', layer, 'batch', batch
    B

    act_data = np.load(args.act_trace)
    wgt_filename = os.path.join(trace_dir,'wgt-%s.npy'%(layer))
    wgt_data = np.load(wgt_filename)
    param_filename = os.path.join(trace_dir,'trace_params.csv')
    trace_params = read_trace_params(param_filename, layer)
    
    N,C,X,Y     = act_data.shape
    K,Ck,R,S    = wgt_data.shape
    assert C == Ck, "channel depth does not match: act=%s wgt=%s"%(act_data.shape,wgt_data.shape)
    
    stride = trace_params['stride']
    pad = trace_params['pad']

    H = (X + 2 * pad - R) / stride + 1
    W = (Y + 2 * pad - S) / stride + 1

    config = get_default_config()

    dims = (N,C,K,H,W,R,S,X,Y)

    schedule_wgt_local(act_data, wgt_data, dims, config, trace_params)


