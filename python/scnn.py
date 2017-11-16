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
    Kc: 16
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

def create_idxmap_4d(d):
    s = list(d.shape)
    s.append(len(s))
    idxmap = np.zeros(s).astype(int)
    for i in range(s[0]):
        for j in range(s[1]):
            for k in range(s[2]):
                for l in range(s[3]):
                    idxmap[i,j,k,l] = [i,j,k,l]
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

def map_accumulator(k,x,y,K,X,Y,N):
    ''' map output (x,y) from (X,Y) to one of N accumulator banks '''
    # return (k + K * (x + X * y)) % N # k,x,y
    # return (k + K * (y + Y * x)) % N # k,y,x
    return ((k & 3) + ((y & 3)<<2) + ((x & 3)<<4))%N

def linearize(data,idx,transpose=False):
    s = data.size
    if transpose:
        data = np.swapaxes(data, -3, -2)
        data = np.swapaxes(data, -2, -1)
        idx  = np.swapaxes(idx , -4 ,-3)
        idx  = np.swapaxes(idx , -3 ,-2)
    data = data.reshape(s)
    idx = idx.reshape((s,-1))
    return data, idx

def stride_groups_slow(data, idx, stride, pad):
    ''' returns 2d list of of data and indexes grouped by stride
        time on 1000x1000 image: 6.82s
        '''
    groups = []
    for sx in range(stride):
        groups.append([])
        for sy in range(stride):
            d = np.array([a for a,i in zip(data,idx) if (i[2]+pad)%stride == sx and (i[3]+pad)%stride == sy])
            i = np.array([i for a,i in zip(data,idx) if (i[2]+pad)%stride == sx and (i[3]+pad)%stride == sy])
            groups[-1].append((d,i))

    return groups

def stride_groups(data, idx, stride, pad):
    ''' returns 2d list of of data and indexes grouped by stride
        time on 1000x1000 image: 1.04s
        '''
    groups = [ [ [[],[]] for j in range(stride) ] for i in range(stride) ]
    if stride == 1:
        groups[0][0][0] = data
        groups[0][0][1] = idx
    else:
        for d,i in zip(data,idx):
            sx = (i[2] + pad) % stride
            sy = (i[3] + pad) % stride
            groups[sx][sy][0].append(d)
            groups[sx][sy][1].append(i)
    return groups

def process_tile(pe_idx, act_data, wgt_data, act_idx, wgt_idx, config, dims, mode='time', debug=False):
    '''
        act_data: 2d array of activations (tx,ty)
        act_idx:  [xx,yy] = [n,c,x,y]
        wgt_data: 3d array of weights (Kc,R,S)
        wgt_idx:  [kk,rr,ss] = [k,c,r,s]
    '''

    (N,C,Ck,K,H,W,R,S,X,Y,stride,pad) = dims

    Wt, Ht = [config[k] for k in ['Wt','Ht']]
    # W = roundup_to_multiple(W,Wt)
    # H = roundup_to_multiple(H,Ht)
    # K = wgt_data.shape[0]
    tw = W/Wt
    th = H/Ht

    w1 = pe_idx[0] * tw
    h1 = pe_idx[1] * th
    w2 = w1 + tw
    h2 = h1 + th
    if debug: print 'output region %d-%d %d-%d'%(w1,w2-1,h1,h2-1)

    if debug:
        print 'input:'
        print 'act'
        print '',act_data
        # print act_idx.T
        print 'wgt'
        print '',wgt_data
        # print wgt_idx.T
        print ''

    # linearize tile
    act_data, act_idx = linearize(act_data, act_idx)
    wgt_data, wgt_idx = linearize(wgt_data, wgt_idx, transpose=True)
    if debug:
        print 'linearize:'
        print '',act_data
        print act_idx.T
        print ''
        print '',wgt_data
        print wgt_idx.T
        print ''

    # deparsify
    wgt_data, wgt_idx = desparsify(wgt_data,wgt_idx)
    act_data, act_idx = desparsify(act_data,act_idx)
    if debug:
        print 'desparsify:'
        print '',act_data
        print act_idx.T
        print ''
        print '',wgt_data
        print wgt_idx.T
        print ''

    time = 0
    mult_count = 0
    warp_count = 0 # number of IxF cross products
    idle_conflict = 0

    if 1 or mode == 'compute':
        out_data = np.zeros((K,W,H))

    act_groups = stride_groups(act_data, act_idx, stride, 0)
    wgt_groups = stride_groups(wgt_data, wgt_idx, stride, pad) # account for padding in the weights only

    for sx in range(stride):
        for sy in range(stride):
            if debug: print 'stride group(%d,%d)'%(sx,sy)
            act_d, act_i = act_groups[sx][sy]
            wgt_d, wgt_i = wgt_groups[sx][sy]

            # chunk into 4
            F,I,Kc = [config[k] for k in ['F','I','Kc']]

            # cycles = roundup_to_multiple(len(act_d),I) * roundup_to_multiple(len(wgt_d),F)
            # mults = len(act_d) * len(wgt_d)

            # iterate through chunks
            for i in range(0, len(act_d), I):
                for f in range(0, len(wgt_d), F):
                    warp_count += 1

                    # each cycle
                    debug_mapping = False
                    if debug_mapping: print '######## cycle %d ######'%time
                    acc = np.zeros(2*F*I)

                    
                    for ii in range(i,min(i+I,len(act_d))):
                        for ff in range(f,min(f+F,len(wgt_d))):
                            n,ca,x,y = act_i[ii]
                            k,cw,r,s = wgt_i[ff]

                            w = (x - r + pad)/stride
                            h = (y - s + pad)/stride

                            # print 'x,y',x,y,'r,s',r,s,'-> w,h',w,h
                            if w in range(W) and h in range(H):
                                # print 'valid',
                                mult_count += 1
                                if mode == 'compute':
                                    out_data[k,w,h] += act_d[ii] * wgt_d[ff] 
                                    # print 'out%s(%4.0f) = a%s(%4.0f) * w%s(%4.0f)' \
                                        # % ([k,w,h],out_data[k,w,h],act_i[ii],act_d[ii],wgt_i[ff],wgt_d[ff])

                                # map to accumulators
                                acc_idx = map_accumulator(k,w,h,K,W,H,2*F*I)
                                acc[acc_idx] += 1
                                if debug_mapping: print 'k,w,h,idx =',k,w,h,acc_idx

                    warp_time = max(acc.max(),1)
                    time += warp_time
                    idle_conflict += (warp_time-1) * I * F

    idle_brick = warp_count * I * F - mult_count
    # idle_conflict = (time - warp_count) * I * F

    if mode == 'compute':
        return out_data
    else: 
        return time, mult_count, idle_brick, idle_conflict

def convolve2d(a,b,stride,pad,debug=False):
    X,Y = a.shape
    R,S = b.shape
    W = (X + 2 * pad - R) / stride + 1
    H = (Y + 2 * pad - S) / stride + 1
    o = np.zeros((W,H))
    for w in range(0,W):
        for h in range(0,H):
            for r in range(R):
                for s in range(S):
                    x = (w * stride + r - pad) 
                    y = (h * stride + s - pad) 
                    if debug: print w,h,'=',x,y,'*',r,s,
                    if x in range(X) and y in range(Y):
                        if debug: print 'valid',
                        o[w][h] += a[x][y] * b[r][s]
                    if debug: print ''
    return o

def verify_comp(p=0,N=1,C=1,Ck=1,K=1,X=5,R=3,stride=1,pad=0,batch=False):

    Y = X
    S = R

    W = (X + 2 * pad - R) / stride + 1
    H = (Y + 2 * pad - S) / stride + 1

    dims = (N,C,Ck,K,H,W,R,S,X,Y,stride,pad)
    config = get_default_config()
    ad = np.random.randint(1,10,(N,C,X,Y))
    wd = np.random.randint(1,10,(K,Ck,R,S))
    ad = sparsify(ad, p=p)
    wd = sparsify(wd, p=p)
    print 'a:'
    print ad
    print 'w:'
    print wd
    if pad == 0:
        mode = 'valid'
    elif pad == (R-1)/2:
        mode = 'same'
    else:
        print 'cannot verify with pad', pad
        return
    # ref = scipy.signal.convolve2d(ad,wd[::-1,::-1].T,mode='same').T
    # ref = scipy.signal.convolve2d(ad[0,0],wd[0,0,::-1,::-1],mode=mode)
    ai = create_idxmap_4d(ad)
    wi = create_idxmap_4d(wd)
    out = process_tile((0,0), ad, wd, ai, wi, config, dims, mode='compute')

    print 'out:'
    print out

    match = True
    for k in range(K):
        ref = convolve2d(ad[0,0], wd[k,0], stride, pad, debug=False)
        print 'ref:'
        print ref

        if out[k].shape == ref.shape:
            if ((out[k] - ref).any()):
                match = False
        else:
            match = False
            # for i in range(out.shape[0]):
                # for j in range(out.shape[1]):
                    # # if out[0,i,j] != ref[i*stride,j*stride]:
                    # if out[0,i,j] != ref[i,j]:
                        # match = False

    if not match:
        print 'FAIL output does not match reference convolution'
    else:
        print 'PASS'
    if batch:
        return match
    else:
        return ad, wd, out

def regression():
    pas = 0
    tot = 0
    for X,R,s,p in [[19,11,4,0],[7,5,1,2],[5,3,1,1],[7,7,2,3],[5,1,1,0],[5,3,1,1],[5,1,2,0]]:
        pas += verify_comp(p=1,X=X,R=R,stride=s,pad=p,batch=True)
        tot += 1
    print 'PASSED %d out of %d tests'%(pas,tot)

def schedule_act_local(act_data, wgt_data, dims, config, debug=False):
    (N,C,Ck,K,H,W,R,S,X,Y,stride,pad) = dims
    Kt = config['Kt']
    Kc = config['Kc']
    I = config['I']
    F = config['F']
    Ht = config['Ht']
    Wt = config['Wt']

    print 'scheduling',config

    time = 0
    mults = 0
    idle_bricks = 0
    idle_conflicts = 0
    idle_pe = 0
    idle_halo = 0
    
    # act_data = pad_4d(act_data, 0, roundup_to_multiple(K,Kc))

    # pad activations to multiple of Ht,Wt
    X = roundup_to_multiple(X,Wt)
    Y = roundup_to_multiple(Y,Ht)
    act_data = pad_4d(act_data, 2, X)
    act_data = pad_4d(act_data, 3, Y)
    print 'padded act_data to', act_data.shape, 'to fit PE array of %d x %d'%(Wt,Ht)

    act_idx = create_idxmap_4d(act_data)
    wgt_idx = create_idxmap_4d(wgt_data)

    for n in range(N):

        # process Kc filters together 
        for kc in range(0,K,Kc):

            # tiling channels for two towers alexnet
            for ct in range(0,C,Ck):
                for ck in range(Ck):
                    print 'scheduling image %d/%d filter group %d channel %d/%d'%(n,N,kc,ct+ck,C)

                    times = []
                    
                    # tile activations across PEs
                    tw = X/Wt
                    th = Y/Ht
                    for pex in range(Wt):
                        for pey in range(Ht):
                            x1 = pex * tw
                            y1 = pey * th
                            x2 = min(x1+tw,X)
                            y2 = min(y1+th,Y)
                            k1 = kc
                            k2 = min(kc+Kc,K)

                            if debug: print 'PE(%2d,%2d) act %d-%d %d-%d' % (pex,pey,x1,x2-1,y1,y2-1)

                            # tile activations and weights
                            act     = act_data[n, ct+ck, x1:x2, y1:y2]
                            act_i   = act_idx [n, ct+ck, x1:x2, y1:y2]
                            wgt     = wgt_data[k1:k2, ck]
                            wgt_i   = wgt_idx [k1:k2, ck]

                            t, mc, ib, ic = process_tile( (pex,pey), act, wgt, act_i, wgt_i, config, dims, debug=debug)

                            times.append(t)
                            mults += mc
                            idle_bricks += ib
                            idle_conflicts += ic

                    # this is chip sync
                    maxtime = max(times)
                    ip = (maxtime - np.array(times)).sum() * I * F
                    idle_pe += ip
                    time += maxtime
                    
                # for ck
            # for ct

            # reslove halos
            # compute the areas of the halo regions around a non edge PE
            # that is, how many psums need to get transfered
            psums = np.outer([R-1-pad,tw,pad],[S-1-pad,th,pad])
            psums[1,1] = 0 # this is the number of local outputs
            max_psums = psums.max() * min(Kc, K-kc)
            
            idle_halo += max_psums * Ht*Wt*I*F
            time += max_psums
        # for kc
    
    print 'time', time 
    print 'mults         ', mults 
    print 'idle_bricks   ', idle_bricks 
    print 'idle_conflicts', idle_conflicts 
    print 'idle_pe       ', idle_pe
    print 'idle_halo     ', idle_halo
    tot = mults + idle_bricks + idle_conflicts + idle_pe + idle_halo
    print 'total_mult_cycles', tot
    print 'sanity (should equal time)', tot/(Ht*Wt*I*F) 
    return time

def test_schedule(pnz=0, N=1, C=1, Ck=1, K=1, X=5, R=3, stride=1, pad=0, debug=True):

    S = R
    Y = X
    W = (X + 2 * pad - R) / stride + 1
    H = (Y + 2 * pad - S) / stride + 1

    dims = (N,C,Ck,K,H,W,R,S,X,Y,stride,pad)
    config = get_default_config()
    config['Ht'] = 8
    config['Wt'] = 8
    ad = np.random.randint(2,10,(N,C,X,Y))
    wd = np.random.randint(1,10,(K,Ck,R,S))
    ad = sparsify(ad, pnz)
    wd = sparsify(wd, pnz)
    return schedule_act_local(ad, wd, dims, config, debug=debug)

def test_scaling():
    ps = np.arange(0.1,1.1,0.1)
    times = []
    for p in ps:
        times.append( test_schedule(pnz=p, X=16, C=16, Ck=16, K=16, R=3, stride=1, pad=0, debug=False) )
    return ps, times

def plot_scaling(ps, times):
    times = np.array(times)
    times = times/times[-1]
    plt.plot(ps,times)



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

    W = (X + 2 * pad - R) / stride + 1
    H = (Y + 2 * pad - S) / stride + 1

    config = get_default_config()

    dims = (N,C,Ck,K,H,W,R,S,X,Y,stride,pad)

    schedule_wgt_local(act_data, wgt_data, dims, config, trace_params)


