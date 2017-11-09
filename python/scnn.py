#!/usr/bin/env python

import numpy as np
import scipy.signal
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
    I: 4
    F: 4
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
    idxmap = np.zeros(s).astype(int)
    for i in range(s[0]):
        for j in range(s[1]):
            idxmap[i,j] = [i,j]
    return idxmap

def gen_random_data():
    a = np.random.randint(0,10,(10,10))
    w = np.random.randint(0,10,(3,3))
    return a,w

def desparsify(data, idx):
    nz = np.where(data != 0)
    do = data[nz]
    io = idx[nz]
    return do, io

def map_accumulator(x,y,X,Y,N):
    ''' map output (x,y) from (X,Y) to one of N accumulator banks '''
    return (x * Y + y) % N

def process_tile(act_data, wgt_data, config, dims, mode='time'):

    (N,C,Ck,K,H,W,R,S,X,Y,stride,pad) = dims

    # create index map
    act_idx = create_idxmap(act_data)
    wgt_idx = create_idxmap(wgt_data)

    # linearize tile
    actsize = act_data.size
    wgtsize = wgt_data.size
    act_data = act_data.reshape(actsize)
    wgt_data = wgt_data.reshape(wgtsize)
    act_idx = act_idx.reshape((actsize,-1))
    wgt_idx = wgt_idx.reshape((wgtsize,-1))
    # print act_data
    # print act_idx.T
    # print wgt_data
    # print wgt_idx.T

    # deparsify
    wgt_data, wgt_idx = desparsify(wgt_data,wgt_idx)
    act_data, act_idx = desparsify(act_data,act_idx)
    # print act_data
    # print act_idx.T
    # print wgt_data
    # print wgt_idx.T

    # chunk into 4
    F,I = [config[k] for k in ['F','I']]
    count = 0
    time = 0
    mult_count = 0
    warp_count = 0 # number of IxF cross products
    
    if mode == 'compute':
        out_data = np.zeros((X,Y))

    for i in range(0, len(act_data), I):
        for f in range(0, len(wgt_data), F):
            warp_count += 1

            # each cycle
            acc = np.zeros(2*F*I)
            for ii in range(i,min(i+I,len(act_data))):
                for ff in range(f,min(f+F,len(wgt_data))):
                    h,w = act_idx[ii]
                    r,s = wgt_idx[ff]

                    x = w - r + pad
                    y = h - s + pad 
                    # print 'h,w',h,w,'r,s',r,s,'->',x,y,
                    if x in range(X) and y in range(Y):
                        # print 'valid',
                        mult_count += 1
                        if mode == 'compute':
                            out_data[x,y] += act_data[ii] * wgt_data[ff] 
                        # map to accumulators
                        acc_idx = map_accumulator(x,y,X,Y,2*F*I)
                        acc[acc_idx] += 1
                        # print acc_idx
            warp_time = acc.max()
            time += warp_time
            count += warp_count
    
            # print 'time',warp_time

    # print 'time', time
    # print 'warp', warp_count, 
    # print 'valid mults', mult_count, 
    # print 'conflict stalls', float(warp_count) / time
    # print 'utilization', float(mult_count) / (warp_count * I * F)

    idle_brick = warp_count * I * F - mult_count
    idle_conflict = (time - warp_count) * I * F

    if mode == 'compute':
        print out_data

    if mode == 'compute':
        return out_data
    else: 
        return time, mult_count, idle_brick, idle_conflict

def verify_comp():

    (N,C,Ck,K,H,W,R,S,X,Y,stride,pad) = \
    (1,1,1,1,4,4,3,3,4,4,1,1)

    dims = (N,C,Ck,K,H,W,R,S,X,Y,stride,pad)
    config = get_default_config()
    ad = np.random.randint(1,10,(H,W))
    wd = np.random.randint(1,10,(R,S))
    ad = sparsify(ad)
    wd = sparsify(wd)
    ref = scipy.signal.convolve2d(ad,wd[::-1,::-1].T,mode='same').T
    out = process_tile(ad, wd, config, dims, mode='compute')
    print out
    print ref
    if (out - ref).any():
        print 'FAIL output does not match reference convolution'
    else:
        print 'PASS'

def schedule_act_local(act_data, wgt_data, dims, config):
    (N,C,Ck,K,H,W,R,S,X,Y,stride,pad) = dims
    print 'W', W
    print 'H', H
    total_time = 0
    for n in range(N):
        for k in range(K):
            print "FIXME"
            sys.exit(1)
            # FIXME: channel tiling for alexnet two towers
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

def pad_1d(d, t):
    ''' pad d to a multiple of t '''
    pad_shape = list(d.shape)
    pad_shape[0] = roundup_to_multiple(pad_shape[0],t)
    d_pad = np.zeros(pad_shape).astype(d.dtype)
    d_pad[:d.shape[0]]  = d
    return d_pad

def pad_2d(d, t):
    ''' pad d to a multiple of t '''
    pad_shape = list(d.shape)
    pad_shape[0] = roundup_to_multiple(pad_shape[0],t)
    d_pad = np.zeros(pad_shape).astype(d.dtype)
    d_pad[:d.shape[0],:]  = d
    return d_pad

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
    (N,C,Ck,K,H,W,R,S,X,Y,stride,pad) = dims
    stride = trace_params['stride']
    pad = trace_params['pad']
    Kt = config['Kt']
    I = config['I']
    F = config['F']

    time = 0
    mults = 0
    idle_bricks = 0
    idle_conflicts = 0
    idle_pe = 0
    wgt_pad = pad_weights(wgt_data, Kt)
    for n in range(N):
        # tiling channels for two towers alexnet
        for ct in range(0,C,Ck):
            for ck in range(Ck):
                print n,ct+ck,ck

                # process Kt filters in parallel on Kt PEs
                for kt in range(0,K,Kt):
                    times = []

                    # distribute work to PEs
                    for k in range(kt, kt+Kt):
                        act = act_data[n,ct+ck]
                        wgt = wgt_data[k,ck]
                        t, mc, ib, ic = process_tile(act,wgt,config, dims)
                        times.append(t)
                        mults += mc
                        idle_bricks += ib
                        idle_conflicts += ic
                        # print 'debug exit'
                        # sys.exit()

                    # this is chip sync
                    maxtime = max(times)
                    ip = (maxtime - np.array(times)).sum() * I * F
                    idle_pe += ip
                    time += maxtime
    
    print 'time', time 
    print 'mults         ', mults 
    print 'idle_bricks   ', idle_bricks 
    print 'idle_conflicts', idle_conflicts 
    print 'idle_pe       ', idle_pe
    tot = mults + idle_bricks + idle_conflicts + idle_pe
    print 'total mult cycles', tot
    print 'sanity', tot/1024


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='scnn.py', description='Performance Model of SCNN')
    parser.add_argument('act_trace', type=str, help='network name in model directory. \'all\' to run all networks')
    parser.add_argument('--short', action='store_true', help='run one image and one channel')
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

    act_data = np.load(args.act_trace)
    wgt_filename = os.path.join(trace_dir,'wgt-%s.npy'%(layer))
    wgt_data = np.load(wgt_filename)
    param_filename = os.path.join(trace_dir,'trace_params.csv')
    trace_params = read_trace_params(param_filename, layer)
    
    N,C,X,Y     = act_data.shape
    K,Ck,R,S    = wgt_data.shape

    if args.short:
        N = 1

    
    wgt_data = sparsify(wgt_data,p=0.25)

    # assert C == Ck, "channel depth does not match: act=%s wgt=%s"%(act_data.shape,wgt_data.shape)
    
    stride = trace_params['stride']
    pad = trace_params['pad']

    H = (X + 2 * pad - R) / stride + 1
    W = (Y + 2 * pad - S) / stride + 1

    config = get_default_config()

    dims = (N,C,Ck,K,H,W,R,S,X,Y,stride,pad)

    schedule_wgt_local(act_data, wgt_data, dims, config, trace_params)


