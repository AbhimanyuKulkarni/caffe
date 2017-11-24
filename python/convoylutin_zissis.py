import numpy as np
import os
import sys
import time
import re
import code
import argparse
from google.protobuf import text_format
import caffe

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

def drange(d):
    ''' return a tuple with the range of the data '''
    return (d.min(), d.max())

def sparsify(d, p=0.5):
    ''' keep element with probability p, otherwise 0 '''
    return d * np.random.binomial(1, p, d.shape)

def reshape_sb(w, width=4096, brick_size=16):
    ''' 
        Reshape caffe weights to their DaDianNao layout 
    
    Args: 
        w:          caffe weight array (filters, depth, width, height)
        width:      DaDianNao SB width
        brick_size: DaDianNao brick size

    Returns 
        2D array (n rows, row width)
    '''

    nf, ni, nx, ny = w.shape
    assert nf * brick_size == width, 'Width missmatch: %d filters, %d brick size, %d width'%(nf, brick_size, width)
    assert ni % brick_size == 0, 'Channel depth %d not a multiple of brick size %d'%(ni,brick_size)
    sb = np.array(np.reshape(w, (width,-1))) 
    return sb.T

def fill_holes(sb, lookahead, lookaside, interactive=False):
    ''' 
        scans through the weight stream of sb and fills holes by promoting weigths
    Args:
        sb:         2D array with the contents of synapse buffer (n rows, row width)
        lookahead:  how far to look ahead to fill zero holes
    Returns:
        time to process all rows
    '''
    rows, width = sb.shape
    skips = 0
    time = 0
    lookahead_bound = 1
    promoted = 0
    weights_read = 0

    for r in range(rows):

        row_chart_temp = np.zeros((1, 256))
        row_chart_temp.fill(r + 1)
        if interactive: 
            for row in sb:
                print ''.join(['%d'%f for f in row])

        if not sb[r].any():
            if interactive: print 'skipping zero row'
            skips += 1
            continue
        
        for i in range(width):
            if sb[r,i] == 0:
                promoted = 0
                for l in range(r+1, min(r+lookahead+1, rows)):
                    if sb[l,i] != 0:
                        sb[r,i] = sb[l,i] # promote
                        sb[l,i] = 0
                        row_chart_temp[0, i] = l + 1
                        promoted = 1
                        break 
                if (promoted == 0) | (lookahead == 0) : 
                    wrap = 0
                    for k in range(i+1, min(i+lookaside+1, width)):
                        if k % 16 == 0: # the first lane of the next adder tree
                            wrap = 1    # start wrapping 
                        for l in range(r+1, min(r+lookahead_bound+1, rows)):
                            if sb[l,k-(wrap*16)] != 0:    # k is offset by negative wrap*16 if wrap = 1, otherwise it retains its value
                                sb[r,i] = sb[l,k-(wrap*16)] # promote
                                sb[l,k-(wrap*16)] = 0
                                promoted = 1
                                row_chart_temp[0, i] = l + 1
                                break
                        if promoted == 1: 
                            promoted = 0
                            break 
        if interactive: raw_input('')
        time += 1
        weights_read += width
        #chart[r + (warp_num*rows) - skips, tile_num*256:tile_num*256 + 256] = row_chart_temp
    # print 'skipped %d rows' % skips
    return time, weights_read
        
def test(w, show=False):
    f0 = w[0:1,:,:,:]
    wr = np.random.randint(0,10,f0.shape)
    wr = sparsify(wr)
    sb = reshape_sb(wr, brick_size=16)
    if show:
        sb = sb[:16]
    t = fill_holes(sb, lookahead, interactive=show)
    rows = sb.shape[0]
    print 'processed %d/%d rows'%(t,rows)

def test_conv1(la, ls, rows = 3, brick_size=16, bricks_per_row=16):
    # w = np.random.randint(1,10,(16,16,16))
    row_size = brick_size * bricks_per_row
    w = np.repeat([range(1,rows+1)],row_size,axis=0).T
    w = w.reshape((rows,bricks_per_row,brick_size))
    w[:,:,12:] = 0 
    sb = w.reshape(rows,row_size)
    t = fill_holes(sb, la, ls, interactive=True)


def pad_4d(d, dim, size):
    ''' pad weights to '''
    s = list(d.shape)
    assert len(s) == 4, 'd is not 4d'
    d = np.reshape(d,s)
    pad_shape = list(s)
    pad_shape[dim] = roundup_to_multiple(pad_shape[dim],size)
    d_pad = np.zeros(pad_shape, dtype=d.dtype)
    d_pad[:s[0], :s[1], :s[2] ,:s[3]]  = d
    return d_pad

def pad_weights(w, filter=64, brick=16):
    ''' pad weights to fit in the SBs of DaDianNao, also converts fc weights to 4D '''
    ws = list(w.shape)
    while len(ws) < 4:
        ws += [1]
    w = np.reshape(w,ws)
    pad_shape = list(w.shape)
    pad_shape[0] = roundup_to_multiple(pad_shape[0],filter)
    pad_shape[1] = roundup_to_multiple(pad_shape[1],brick)
    w_pad = np.zeros(pad_shape)
    w_pad[:w.shape[0], :w.shape[1]]  = w
    return w_pad

def chip_sync(w, lookahead, lookaside, filter_lanes=64, brick_size=16):
    ''' all tiles operate in lockstep '''
    print 'processing weights', w.shape

    # w = (filters, channels, height, width)
    w_pad = pad_weights(w, filter=filter_lanes, brick=brick_size)
    # w_pad = sparsify(w_pad) # add sparsity for testing

    print 'padding to', w_pad.shape
    filters, channels, width, height = w_pad.shape

    time = 0
    for warp in range(filters / filter_lanes):
        w_warp = w_pad[ warp*filter_lanes : (warp+1)*filter_lanes ]
        sb = reshape_sb(w_warp, width=4096, brick_size=16)
        t = fill_holes(sb, lookahead, lookaside, interactive=False)
        time += t
        rows = sb.shape[0]

        print 'warp %2d processed %d/%d rows'%(warp,t,rows)
    print 'overall time = ', time
    return time

def tile_sync(w, lookahead, lookaside):
    ''' tiles operate in independantly, assuming they can buffer activations to keep busy '''
    print 'processing weights', w.shape

    # w = (filters, channels, height, width)
    w_pad = pad_weights(w, filter=64, brick=16)
    #w_pad = sparsify(w_pad)

    print 'padding to', w_pad.shape
    filters, channels, width, height = w_pad.shape

    tile_times = np.zeros(4)
    for tile in range(4):
        for warp in range(filters / 256):
            w_tile = w_pad[tile*4 + warp*256 : tile*4+16 + warp*256]
            sb = reshape_sb(w_tile, width=256, brick_size=16)
            t = fill_holes(sb, lookahead, lookaside, interactive=False)

            rows = sb.shape[0]
            print 'tile %2d warp %d processed %d/%d rows'%(tile,warp,t,rows)
            tile_times[tile] += t

    time = max(tile_times)
    print 'tile times = ', tile_times
    print 'overall time = ', time
    return time

def tile_sync_buff(w, lookahead, lookaside):
    tiles = 4
    filters_per_tile = 16

    print 'processing weights', w.shape

    # w = (filters, channels, height, width)
    w_pad = pad_weights(w, filter=filters_per_tile*tiles, brick=16)
    print 'padding to', w_pad.shape
    #w_pad = sparsify(w_pad)

    filters, channels, width, height = w_pad.shape

    global chart
    global tile_num
    tot_weights_read = 0
    tile_num = 0
    global warp_num
    global num_rows
    num_rows = (filters*channels*width*height)/(4*256)
    chart = np.zeros((num_rows/4, 4*256))

    warp_width = filters_per_tile * tiles

    tile_times = np.zeros(tiles)
    for tile in range(tiles):
        warp_num = 0
        for warp in range(filters / (warp_width)):
            w0 = tile*filters_per_tile + warp*warp_width
            w1 = (tile+1)*filters_per_tile + warp*warp_width
            w_tile = w_pad[w0:w1] # c = 1111110000000000 ???
            sb = reshape_sb(w_tile, width=filters_per_tile*16, brick_size=16)
            t, w = fill_holes(sb, lookahead, lookaside, interactive=False)
            tot_weights_read += w
            rows = sb.shape[0]
            print 'tile %2d warp %d w[%4d:%4d] processed %d/%d rows'%(tile,warp,w0,w1,t,rows)
            tile_times[tile] += t
            warp_num += 1
        tile_num+=1
    
    min_val = chart.min(axis = 1) #The 'earliest' activation any lane is working on at each t
    max_val = chart.max(axis = 1) #The 'latest' activation any lane is working on at each t

    #Set the empty space at the end to be the last act worked on
    min_val = np.where(min_val == 0, min_val.max(), min_val) 
    max_val = np.where(max_val == 0, max_val.max(), max_val)

    #Find the max difference between any two lanes at each t 
    diff = np.subtract(max_val, min_val)
    print 'Maximum lane difference: ', diff.max()

    #Find where the difference increases (i.e. when a tile skips a full row)
    diff_change = np.ediff1d(diff, to_begin=0)
    #Only count positive changes (negative means the slowest lane caught up)
    diff_change = np.where(diff_change > 0, diff_change, 0) 

    #Is this correct?
    global buffer_size
    buffer_size=8
    stalls = np.where(diff > (buffer_size + lookahead), diff_change, 0)

    total_stalls = np.sum(stalls)
    print 'Total stalls: ', total_stalls

    #Pausing for debug!
    '''
    if(np.any(stalls)):
        raw_input()
    '''

    time = max(tile_times)
    print 'tile times = ', tile_times
    print 'overall time = ', time
    return (time, total_stalls, tot_weights_read) 

def read_repeated_param(param, default):
    if param:
        return param[0]
    else:
        return default

def test_all_layers(net, sync, lookahead, lookaside, params, windows):

    layer_stats = ','.join( ['LAYER_STATS','','times','stalls','weight_reads'] ) + '\n'

    layer_times = []
    layer_weights_read = []
    stall_times = []
    layer_num = 0

    for layer_param in params.layer:
        layer = layer_param.name        
        if layer in banned: continue

        if layer != 'conv1': continue
        group = 1
        stride = 1
        if ('Convolution' not in layer_param.type) and ('InnerProduct' not in layer_param.type): continue
        if 'Convolution' in layer_param.type:
            group = layer_param.convolution_param.group
            stride = read_repeated_param(layer_param.convolution_param.stride, 1)

        w = net.params[layer][0].data
        
        (K,C,H,W) = w.shape
        if(C == 3 and stride > 1):
            print 'folding channels'
            w = w.swapaxes(1,3)
            w = pad_4d(w, 2, stride)
            w = w.reshape(K,W,divide_roundup(H, stride), C*stride)
            w = w.swapaxes(3,1)
            print w.shape
            (K,C,H,W) = w.shape

        t, s, w = sync(w, lookahead, lookaside)

        num_win = windows[layer_num]
        t = t*group*num_win*num_win
        s = s*group*num_win*num_win
        w = w*group*num_win*num_win
        layer_times.append(t)
        layer_weights_read.append(w)
        stall_times.append(s)
        layer_stats += ','.join( ['LAYER_STATS',layer,] + ['%f'%f for f in [t,s,w]] ) + '\n'
        layer_num+=1

    return (np.array(layer_times), np.array(stall_times), np.array(layer_weights_read), layer_stats)

##################### MAIN ########################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='test_net.py', description='Run a network in pycaffe')
    parser.add_argument('-n', dest='network', type=str, default = '', help='network name in model directory. \'all\' to run all networks')
    parser.add_argument('-b', dest='batches', type=int, help='batches to run')
    parser.add_argument('-m', dest='model', type=str, default = '', help='model prototxt')
    parser.add_argument('-w', dest='weights', type=str, default = '', help='weight caffemodel')
   #parser.add_argument('-l',dest='lookahead',type=int, default = 1, help='lookahead offset')

    args = parser.parse_args()
    batches = args.batches
    network = args.network
    #lookahead = args.lookahead

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

    print 'run this in ipython and you can play with these:'
    print 'variables: net, net_param'
    print 'net.layers: vector of layers'
    print '\tlayer: blobs (weights), reshape, setup, type'
    print 'net.params: vector of weight blobs'
    print 'net.blobs: map by layer name'
    print '\tblob: count, shape, data, diff'

    print '\nNOTE: conv times are for ONE WINDOW, need to scale by # windows'
    
    # mux_inputs = [2,4,8]
    mux_inputs = [8]
    file = 'res_tile_buff'
    #windows=np.array([111,56,56,28,28,28,28,28,28,28,28,28,28,28,28,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,7,7,7,7,7,7,7,7,7,7,7,7,1])
    #windows=np.array([55,27,13,13,13,1,1,1])
    windows=np.array([111,56,56,56,56,56,56,56,56,56,56,27,27,28,28,28,28,28,28,28,28,28,28,28,13,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,6,6,7,7,7,7,7,7,7,7,1])
    print windows.shape
    banned = ['loss1/conv', 'loss1/fc', 'loss1/classifier', 'loss2/conv', 'loss2/fc', 'loss2/classifier']
    #if 1:
    for mux in mux_inputs:
        # for la in range(mux):
        for la in [1]:
            ls = mux - 1 - la
            path = file + '_ahead_' + str(la) + '_aside_' + str(ls)
            dump_times = open(path, "w")
            times = test_all_layers(net, tile_sync_buff, la, ls, net_param, windows)
            stall_ratio = times[1]/(times[0] + times[1])
            print >> dump_times, times[3]
            print >> dump_times, 'TILE TIMES:\n', times[0]
            print >> dump_times, 'STALLS:\n', times[1]
            print >> dump_times, 'STALLS/(STALLS+TIMES):\n', stall_ratio
            print >> dump_times, 'WEIGHTS READ:\n', times[2]
            '''
            scaled_times = times[0]
            scaled_stalls = times[1]
            print >> dump_times, 'STALL/TOT RATIO:\n', sum(scaled_stalls)/sum(scaled_times)
            '''
            print >> dump_times, 'TOTAL TIME:\n', sum(times[0])
          
            dump_times.close()
 
