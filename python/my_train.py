#!/usr/bin/env python

import argparse
import sys
import os
import glob
import copy
os.environ['GLOG_minloglevel'] = '2' # supress caffe output
import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import plot_acc_loss
import plot_blob_dists
import dist_stats
import plot_dist_stats
import quantizer
import pickle
from scipy.stats import itemfreq, norm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--solver", help="solver prototxt")
    parser.add_argument( "--train_net", help="train net prototxt")
    parser.add_argument( "--test_net", help="solver prototxt")
    parser.add_argument( "--snapshot_prefix", help="prefix of snapshot file, will go in out_dir", default='snapshot')
    parser.add_argument( "--save_plot", help="save plot to file", action='store_true')
    parser.add_argument( "--show_plot", help="show plot", action='store_true')
    parser.add_argument( "--out_dir", help="directory to store temp files and plots", default="")
    parser.add_argument( "--baseline", help="run baseline solver", action='store_true')
    parser.add_argument( "--itemfreq", help="measure itemfreq of quantized blob", action='store_true')
    parser.add_argument( "--histo", help="dump histogram of each blob", action='store_true')
    parser.add_argument( "--sched", help="schedule: [test,short,full]", type=str, default="full")
    parser.add_argument( "--progressive", help="train with progressively increasing precision", action='store_true')
    parser.add_argument( "--iterperprec", help="iteration per precision", type=int, default=1000)
    parser.add_argument( "--stats", help="collect stats on the data", action='store_true')
    parser.add_argument( "--plot_stats", help="plot stats on the data", action='store_true')
    parser.add_argument( "--test_dists", help="save histograms for each test iteration", action='store_true')
    parser.add_argument( "--stretch", help="stretch range when excessive clipping is detected", action='store_true')
    args = parser.parse_args()
    return args

def get_newest_file(regex):
    # print 'get_newest_file(%s)'%regex
    list_of_files = glob.glob(regex) 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_precision_param(layer, blob):
    if blob == 'fwd_act':
        param = layer.fwd_act_precision_param
    elif blob == 'fwd_wgt':
        param = layer.fwd_wgt_precision_param
    elif blob == 'bwd_grd':
        param = layer.bwd_grd_precision_param
    else:
        raise KeyError, 'Unknown blob %s' % blob
    return param

def read_solver_prototxt(filename):
    s = caffe_pb2.SolverParameter()
    with open(filename) as f:
        text_format.Merge(str(f.read()), s)
    return s

def write_prototxt(filename, proto):
    open(filename, 'w').write(str(proto))

def read_net_prototxt(filename):
    netproto = caffe_pb2.NetParameter()
    with open(filename) as f:
        text_format.Merge(str(f.read()), netproto)
    return netproto

def list2str(l, p=6):
    # return np.array_str(np.array(l), precision=p, suppress_small=True)
    #return "%s" % l
    if len(l) > 16:
        str = list2str(l[:3]) + ' ... ' + list2str(l[-3:])
    else:
        str = ''.join(['{0:{1}.{2}f}'.format(i,p+3,p) for i in l])
    return str

def imbalance(vec, dist='uniform'):
    l = len(vec)
    if dist == 'uniform':
        target = 1./l
        abs_diff = [ abs (v - target) for v in vec ]
        sad = sum(abs_diff)
    elif dist == 'normal':
        # fit to normal distribution with std=1
        # TODO: should std scale with precision?
        x = [-np.inf] +  list(np.arange( -(l/2)+0.5, l/2+0.5 )) + [np.inf] # [-inf, -0.5, 0.5, inf]
        y = norm.cdf(x, scale=1) # slice cdf into bins
        target = np.diff(y) # 
        sad = sum(abs(vec - target))
    else:
        raise TypeError, 'Unknown distribution %s' % dist
    
    return sad

# change these depening on quantization

def intrange(prec):
    intmax =       2 ** (prec - 1) - 1
    intmin = -1 * (2 ** (prec - 1) - 1)
    # intmin = -1 * (2 ** (prec - 1))
    return (intmin, intmax)

def get_nlevels(prec):
    return 2 ** prec - 1

def quantize_cdf(prec, scale, cdf, bins):
    intmin, intmax = intrange(prec)
    abs_max = intmax / scale
    nlevels = get_nlevels(prec)
    i = np.arange(0,nlevels+1)
    level_edges = -abs_max + 2. * i / nlevels * abs_max
    # print 'level_edges=',level_edges 

    bin_edges = [max(0, np.argmax(bins >= l)-1) for l in level_edges]
    bin_edges[0] = 0
    bin_edges[-1] = len(bins)-1
    # print 'bin_edges=',bin_edges 

    cumpop = cdf[bin_edges]
    # print 'cumpop=' ,cumpop 
    pop = np.diff(cumpop)

    return pop, bin_edges

def scale_by_histo(prec, histo, layer_name, blob_name, byrange=False, dist='normal'):
    print 'layer_name =',layer_name
    print 'blob_name =',blob_name
    counts, bins = histo[layer_name][blob_name]
    # print 'count=',counts
    # print 'bin=',bins
    cdf = np.cumsum(counts)
    cdf = cdf/cdf[-1] # normalize to 1
    cdf = np.insert(cdf, 0, 0)
    # print 'cdf=',cdf 

    nbins = len(bins)

    abs_max = max( abs(bins[0]), abs(bins[-1]) )
    # print 'abs_max=',abs_max 

    intmin, intmax = intrange(prec)

    scales = []
    imbas = []

    if byrange:
        print 'scaling by range'
        trails = 1
    else:
        print 'balancing quantization bins'
        trails = 1000

    for j in np.arange(trails):
        scale = float( intmax ) / abs_max 
        # print 'scale=',scale 

        pop, bin_edges = quantize_cdf(prec, scale, cdf, bins)
        imba = imbalance(pop, dist=dist)
        # print 'imba=',imba 

        if imbas and imba > imbas[-1]:
            print 'iteration %d: imba started increasing %f -> %f ' % (j,imbas[-1],imba)
            break

        scales.append(scale)
        imbas.append(imba)

        n = len(bin_edges)
        if bin_edges[n/2-1] == bin_edges[n/2]:
            print 'iteration %d: bins converged' % j
            break

        abs_max  = abs_max * 0.99

    scale = scales[np.argmin(imbas)] # choose the scale with the least imbalance
    #scale = sys.float_info.max
    #scale = scale / (float(prec)/2.)

    print 'scale =',scale, '@ imba =',min(imbas)

    # grow scale with prec
    scale = scale / ( float(1 - 0.5**prec) / 0.5 )
    pop, bin_edges = quantize_cdf(prec, scale, cdf, bins)
    # FIXME negative population?
    if len(pop) > 16:
        print 'balanced quantization:', pop[:2], '...', pop[-2:]
        print 'bin edges:            ', bin_edges[:2], '...', bin_edges[-2:]
    else:
        print 'balanced quantization:', pop
        print 'bin edges:            ', bin_edges
    return scale

def scale_by_range(prec, stats, layer_name, blob_name, grow=True, max_edge=True, stat='absmax'):
    '''
        determine scaling factor from statistics on the data distribution
            grow        increase scaling by 2^(p-2)
            max_edge    use the max as the edge of the bin, not the quantization value
    '''
    stat_name = '-'.join([layer_name, blob_name])
    dmin = stats[stat_name]['min'][0] 
    dmax = stats[stat_name]['max'][0]
    std  = stats[stat_name]['std'][0]
    abs_max = max( abs(dmin), abs(dmax) )
    if grow:
        # scale to -1, 1
        intmin, intmax = np.array(intrange(2)) * ( 2. ** (prec-2) ) # grow with prec
        # intmin, intmax = np.array(intrange(2)) * ( 2. ** ( int( (prec+1) /2 ) -1 ) ) # grow with prec
        # intmin, intmax = np.array(intrange(2)) # don't grow at all
    else:
        intmin, intmax = intrange(prec)
    if dmax != 0:
        smax = float( intmax ) / (dmax)
    else:
        smax = 1e300
    if dmin != 0:
        smin = float( intmin ) / (dmin+1e-300)
    else:   
        smin = -1e300 
    if stat == 'absmax':
        scale = min( abs(smin), abs(smax) )
    # scale = float( intmax ) / abs_max
    elif stat == 'std':
        scale = float( intmax ) / std
    else:
        raise KeyError, 'Unsupported stat %s'%stat
    if max_edge:
        # scale = scale * 3./2
        scale = scale * mescale
    return scale

def set_precision_param(param, prec, scale, quantizer, store_reduced=True):
    param.precision = prec
    param.scale = scale
    param.quantizer = quantizer
    param.store_reduced = store_reduced

def set_prec(param, prec, netproto, stretch_map, stats, layer=''):
    ''' set the precision_param in netproto for a given param 
        param       ['fwd_act','fwd_wgt','bwd_grd']
        prec        precision
        netproto    network prototxt
        stretch_map map[layer][blob] stretch the initial range
        '''

    # statsfile = os.path.dirname(args.solver) + '/stats.pickle'
    # print 'reading stats from', statsfile

    histo = pickle.load(open('histograms.pickle'))

    target_layers = []

    if layer:
        target_layers = [layer]
    else:
        for layer in netproto.layer:
            if layer.type in ['Convolution','InnerProduct']:
                target_layers += [layer.name]
            
        target_layers = target_layers[:4] # just do the last layer

    for layer in netproto.layer:
        if layer.name in target_layers:
            scale = 1
            quantizer = 2

            if layer.name not in stretch_map:
                stretch_map[layer.name] = dict()
            if param not in stretch_map[layer.name]:
                stretch_map[layer.name][param] = 1.;

            if param in 'fwd_act':
                # we don't want to compress the range of first layer activations
                if(layer.bottom == 'data'):
                    scale = scale_by_range(prec, stats, layer.name, 'act_in_data', grow=False, max_edge=False)
                    stretch_map[layer.name][param] = 1; # dont stretch the first layer
                else:
                    scale = scale_by_range(prec, stats, layer.name, 'act_in_data', max_edge=True, stat=scalestat)
                    scale = scale/stretch_map[layer.name][param]
                    scale *= mescale
                set_precision_param(layer.fwd_act_precision_param, prec, scale, quantizer)
            if param in 'fwd_wgt':
                scale=scale_by_range(prec, stats, layer.name, 'wgt_data', max_edge=True, stat=scalestat)
                scale = scale/stretch_map[layer.name][param]
                scale *= mescale
                for p in [layer.fwd_wgt_precision_param, layer.bwd_wgt_precision_param]:
                    set_precision_param(p, prec, scale, quantizer, store_reduced=False)
            if param in 'bwd_grd':
                scale = scale_by_range(prec, stats, layer.name, 'act_out_diff', stat=scalestat)
                scale = scale/stretch_map[layer.name][param]
                scale *= mescale
                set_precision_param(layer.bwd_grd_precision_param, prec, scale, quantizer)

                
            # bins = len(bins)
            print 'set_prec: %10s %7s prec=%d scale=%8.4f' \
                % (layer.name, param, prec, scale),
            if prec == 2:
                bins = get_bins(prec, scale, quantizer)
                bins = list2str(bins, p=4)
                print 'bins', bins
            else:
                print ''

def get_bins(prec, scale, quantizer, edges=False):
    if prec == 0:
        return []
    if prec > 10:
        scale /= 2**(prec-10)
        prec = 10
    intmin, intmax = intrange(prec)
    intbins = range(intmin, intmax+2)
    # scale = scale * ( 2 ** (prec-2) )
    if edges:
        bins = [ float( i -1./2 ) / scale for i in intbins]
        bins[0] = -1e300
        bins[-1] = 1e300
    else:
        bins = [float(i)/scale for i in intbins[:-1]]
    return bins

def set_solver(param, prec, solver_file, sched='full', seed=0xCAFFE):

    print 'reading solver config from',solver_config_path
    s = read_solver_prototxt(solver_config_path)

    s.random_seed = seed

    # Overwrite locations of the train and test networks.
    if train_net_path:
        s.train_net = train_net_path

    if test_net_path:
        s.test_net.append(test_net_path)

    if (s.train_net and s.net):
        print "Error: can't set net, train_net already set to", s.train_net

    # Update net prototxt
    netproto = read_net_prototxt(s.net)
    if prec:
        set_prec(param, prec, netproto)
    print 'writing to', out_net_proto_path
    open(out_net_proto_path, 'w').write(str(netproto))

    # OVERWRITE SOLVER PARAMETERS

    s.net = out_net_proto_path

    if sched == 'full':
        s.test_interval = 100   # Test after every 500 training iterations.
        s.test_iter[0]  = 100   # Test on 100 batches each time we test.
        s.max_iter      = 10000 # no. of times to update the net (training iterations)
        s.snapshot      = 5000  # snapshot every N iterations
    elif sched == 'short':
        s.test_interval = 100   # Test after every 500 training iterations.
        s.test_iter[0]  = 100   # Test on 100 batches each time we test.
        s.max_iter      = 1000  # no. of times to update the net (training iterations)
        s.snapshot      = 100    # snapshot weights every N iterations
    elif sched == 'test':
        s.test_interval = 1     # Test after every 500 training iterations.
        s.test_iter[0]  = 1     # Test on 100 batches each time we test.
        s.max_iter      = 1     # no. of times to update the net (training iterations)
        s.snapshot      = 1     # snapshot weights every N iterations
    else:
        raise NameError('Unknown Schedule \'%s\''%sched)

    if snapshot_prefix:
        s.snapshot_prefix = os.path.join(out_dir, snapshot_prefix)

    s.gamma = 0.0001
    s.power = 0.75

    # write configuration to solver file
    open(solver_file, 'w').write(str(s))

def solve(solver, solver_param):

    global stats_per_blob

    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    niter           = solver_param.max_iter
    test_interval   = solver_param.test_interval
    test_iter       = solver_param.test_iter[0]

    netproto = read_net_prototxt(solver_param.net)

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        
        # store the train loss, [()] extracts scalar from 0d array
        train_loss.append( solver.net.blobs['loss'].data[()] )
        train_acc.append( solver.net.blobs['accuracy'].data[()] )
        
        test_net = solver.test_nets[0]

        # run a full test every so often
        if it % test_interval == 0:
            acc = []
            loss = []
            global test_bin_freq
            test_bin_freq = dict()
            for test_it in range(test_iter):
                test_net.forward() # run inference on batch
                test_net.backward() # run backprop to compute gradients

                if args.stats:
                    get_net_blob_stats(stats_per_blob, solver.test_nets[0], solver_param.net)
                
                if args.itemfreq:
                    update_itemfreq(test_bin_freq, test_net, netproto)

                acc.append( test_net.blobs['accuracy'].data[()] )
                loss.append(test_net.blobs['loss'].data[()] )

            test_acc.append( np.mean(acc) )
            test_loss.append( np.mean(loss) )
            print 'Iteration %5d Test Accuracy = %.4f Test Loss = %.4f' \
                % (           it,                test_acc[-1],  test_loss[-1])

            if args.itemfreq:
                fname = out_dir + '/itemfreq-%s.npy'%(it)
                dump_itemfreq(fname, test_bin_freq, netproto, show=True)

            if args.stats:
                stats_per_blob.end_iter()

            if args.test_dists:
                stats = pickle.load(open('stats.pickle')) 
                plot_blob_dists.plot_blob_dists(test_net, netproto, stats)
                outfile = "%s/plot_stats-%s.pdf"%(out_dir, it)
                print "saving plot to", outfile
                plt.savefig(outfile, format='pdf')

    return train_acc, train_loss, test_acc, test_loss 

def solve_from_snapshot(solver, solver_param, snapshot):

    global stats_per_blob

    train_loss = []
    test_acc = []
    test_loss = []

    niter           = solver_param.max_iter
    test_interval   = solver_param.test_interval
    test_iter       = solver_param.test_iter[0]

    netproto = read_net_prototxt(solver_param.net)

    # initialize parameters to snapshot
    if snapshot:
        print 'initializing weight from', snapshot
        solver.net.copy_from(str(snapshot))

    # the main solver loop
    early_exit = False
    last_it = 0
    # for it in range(niter):
    global total_iter
    it = 0
    while (total_iter < 10000):
        total_iter += 1
        last_it = it + 1
        

        solver.step(1)  # SGD by Caffe
     
        # store the train loss, [()] extracts scalar from 0d array
        train_loss.append( solver.net.blobs['loss'].data[()] )
        
        test_net = solver.test_nets[0]

        early_exit = False

        # test at the initial iteration
        # if it % test_interval == 0:
        # test after test_interval iterations
        if (it+1) % test_interval == 0:
            acc = []
            loss = []
            global test_bin_freq
            test_bin_freq = dict()

            # run the test network for test_iter batches
            for test_it in range(test_iter):
                test_net.forward() # run inference on batch
                test_net.backward() # run backprop to compute gradients

                if args.stats:
                    get_net_blob_stats(stats_per_blob, solver.test_nets[0], solver_proto.net)
                
                if args.itemfreq:
                    update_itemfreq(test_bin_freq, test_net, netproto)

                acc.append( test_net.blobs['accuracy'].data )
                loss.append(test_net.blobs['loss'].data[()] )

            test_acc.append( np.mean(acc) )
            test_loss.append( np.mean(loss) )
            print 'Iteration %5d Test Accuracy =' % it, test_acc[-1], 'Test Loss =', test_loss[-1]

            if args.itemfreq:
                global update
                fname = out_dir + '/itemfreq-%s-%s.npy'%(update,it)
                dump_itemfreq(fname, test_bin_freq, netproto)

            if args.stats:
                stats_per_blob.end_iter()

            # exit early if any blob overflows 
            for layer in netproto.layer:
                if layer.name in test_bin_freq:
                    for param in test_bin_freq[layer.name]:
                        freq = test_bin_freq[layer.name][param]
                        if layer.name != 'conv1' and itemfreq_has_miniscus(freq):
                            print 'miniscus in', layer.name, param
                            if stretch_map[layer.name][param] < 1024:
                                stretch_map[layer.name][param] *= 2;
                                early_exit = True

            # are we making progress? if not, increase precision
            
        # end if test iteration

        if early_exit:
            print 'exiting early at iteration', it
            solver.snapshot()
            break

        it += 1

    return test_acc, train_loss, last_it, early_exit

def load_acc_loss_precs(d, suffix=''):
    acc     = np.load(os.path.join(d, 'test_acc%s.npy' % suffix))
    loss    = np.load(os.path.join(d, 'train_loss%s.npy' % suffix))
    if (os.path.exists(os.path.join(d, 'precs%s.npy' % suffix))):
        precs   = np.load(os.path.join(d, 'precs%s.npy' % suffix))
    else:
        precs   = np.zeros(len(loss)) 
    return acc, loss, precs

def replot_acc_loss(d, save=True):
    acc, loss, precs = load_acc_loss_precs(d, suffix='')
    plot_acc_loss.plot_acc_loss_biter(acc, loss, precs, title='%s'%(d), amin=0.9, lmax=1, newfig=save, iters=True)
    if save:
        outfile = os.path.join(d,'plot_prog.pdf')
        print 'saving plot to', outfile
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()

def replot_glob(dir_glob):
    for d in glob.glob(dir_glob):
        replot_acc_loss(d, save=True)

def replot_acc_loss_dir(d, save=False):
    for dir, _, _ in os.walk(d):
        if dir == d:
            continue
        # _, leaf = os.path.split(dir)
        # _, blob, _, prec = leaf.split('-')
        replot_acc_loss(d, save=save)

def add_itemfreq(f1, f2, use_itemfreq=False):
    if use_itemfreq:
        f = np.array(f1)
        for row in f2:
            item, count = row
            if item in f[:,0]:
                for r, row1 in enumerate(f):
                    if item == row1[0]:
                        f[r,1] += count
            else:
                f = np.concatenate( (f, [row]) )
        f = f[f[:,0].argsort()] # sort by first column
        return f
    else:
        if not np.array_equal(f1[:,0],f2[:,0]):
            print f1
            print f2
            raise KeyError, 'bins dont match'
        ret = f1
        f1[:,1] += f2[:,1]
        return ret

def update_itemfreq(freq_dict, test_net, netproto, use_itemfreq=False):
    ''' get itemfreq for the blob we are quantizing '''
    for layer in netproto.layer:
        if layer.type in ['Convolution','InnerProduct']:

            if layer.fwd_act_precision_param.precision:
                prec = layer.fwd_act_precision_param.precision
                scale = layer.fwd_act_precision_param.scale
                quantizer = layer.fwd_act_precision_param.quantizer
                data = test_net.blobs[layer.bottom[0]].data
                blob = 'fwd_act'
            elif layer.fwd_wgt_precision_param.precision:
                prec = layer.fwd_wgt_precision_param.precision
                scale = layer.fwd_wgt_precision_param.scale
                quantizer = layer.fwd_wgt_precision_param.quantizer
                data = test_net.params[layer.name][0].data
                blob = 'fwd_wgt'
            elif layer.bwd_grd_precision_param.precision:
                prec = layer.bwd_grd_precision_param.precision
                scale = layer.bwd_grd_precision_param.scale
                quantizer = layer.bwd_grd_precision_param.quantizer
                data = test_net.blobs[layer.name].diff
                blob = 'bwd_grd'
            else:
                continue

            if use_itemfreq:
                freq = itemfreq(data)
            else:
                bins = get_bins(prec, scale, quantizer, edges=True)
                levels = get_bins(prec, scale, quantizer)
                counts, _ = np.histogram(data, bins=bins)
                freq = np.concatenate( ([levels],[counts]), axis=0).T

            if layer.name not in freq_dict:
                freq_dict[layer.name] = dict()
                freq_dict[layer.name][blob] = freq
            else:
                freq_dict[layer.name][blob] = add_itemfreq(freq_dict[layer.name][blob], freq)

def dump_itemfreq(filename, freq_dict, netproto, save=True, show=False, short=False):
    for layer in netproto.layer:
        k = layer.name
        if k in freq_dict:
            for b in freq_dict[k]:
                prec        = get_precision_param(layer, b).precision
                scale       = get_precision_param(layer, b).scale
                quantizer   = get_precision_param(layer, b).quantizer
                if show:
                    freq = freq_dict[k][b]
                    tbf = freq[:,1]
                    tbf = tbf / sum(tbf) # normalize
                    if short:
                        print k, list2str(tbf, p=2)
                    else:
                        print '%5s %7s q    '%(k,b),    list2str(freq[:,0])
                        print '%5s %7s P(q) '%('',''),  list2str(tbf)
    
    if save:
        # print 'writing itemfreqs to', fname
        with open(filename, 'w') as f:
            pickle.dump(freq_dict, f)

def print_itemfreq(freq_dict, netproto, short=False):
    for layer in netproto.layer:
        k = layer.name
        if k in freq_dict:
            for b in freq_dict[k]:
                freq = freq_dict[k][b]
                tbf = freq[:,1]
                tbf = tbf / sum(tbf) # normalize
                if short:
                    print '%s%s'%(k, list2str(tbf, p=2)),
                else:
                    print '%5s %7s q    '%(k,b),    list2str(freq[:,0])
                    print '%5s %7s P(q) '%('',''),  list2str(tbf)

def itemfreq_has_miniscus(freq):
    p = 1
    if freq[0,1] > p*freq[1,1]:
    # if freq[0,1] > 0:
        # print 'left miniscus'
        return True
    if freq[-1,1] > p*freq[-2,1]:
    # if freq[-1,1] > 0:
        # print 'right miniscus'
        return True
    return False

def plot_itemfreq(filename, layer='conv1', blob='fwd_act', normed=False):
    freq_dict = np.load(filename)
    f = freq_dict[layer][blob]
    x = f[:,0]
    y = f[:,1]
    if normed:
        y = y / y.sum()
    w = x[1] - x[0]
    plt.clf()
    plt.bar(x,y,width=w,log=False)
    plt.title(filename)
    return f

def myhisto(d):
    return np.histogram(d, bins=1000, normed=True)

def get_net_blob_stats(stats_per_blob, net, net_proto_file, histo=False):

    proto = read_net_prototxt(net_proto_file)

    cdfs = dict()

    for name in net.params:

        cdfs[name] = dict()

        # weights
        stats_per_blob.append_array_stats(name + '-wgt_data', net.params[name][0].data)
        stats_per_blob.append_array_stats(name + '-wgt_diff', net.params[name][0].diff)

        # input activations
        bottom = []
        for l in proto.layer:
            if l.name == name:
                for b in l.bottom:
                    bottom.append(b)

        assert len(bottom) == 1, 'multiple bottoms'
        bottom = bottom[0]

        stats_per_blob.append_array_stats(name + '-act_in_data', net.blobs[bottom].data)
        stats_per_blob.append_array_stats(name + '-act_in_diff', net.blobs[bottom].diff)

        # output activations
        stats_per_blob.append_array_stats(name + '-act_out_data', net.blobs[name].data)
        stats_per_blob.append_array_stats(name + '-act_out_diff', net.blobs[name].diff)

        if histo:
            cdfs[name]['fwd_wgt'] = myhisto(net.params[name][0].data)
            cdfs[name]['fwd_act'] = myhisto(net.blobs[bottom].data)
            cdfs[name]['bwd_grd'] = myhisto(net.blobs[name].diff)

    if histo:
        dumpfile = 'histograms.pickle'
        print 'writing histograms to', dumpfile
        with open(dumpfile,'w') as f:
            pickle.dump(cdfs, f)

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_hist(solver, layer):
    g = solver.net.blobs[layer].diff
    print itemfreq(g)
    plt.clf()
    plt.hist(g.flatten(), bins=100)

def update_netproto_file(net_proto_path, param, prec, out_net_proto_path):
    netproto = read_net_prototxt(net_proto_path)
    if prec:
        set_prec(param, prec, netproto)
    print 'writing to', out_net_proto_path
    open(out_net_proto_path, 'w').write(str(netproto))

def test_fixed(prec, param):
    out_dir = './results/param-%s-prec-%d' % (param,prec)
    make_path(out_dir)
    out_solver_config_path  = out_dir + '/temp_solver.prototxt'
    out_net_proto_path      = out_dir + '/temp_net.prototxt'

    # update solver
    set_solver(param, prec, out_solver_config_path, sched=args.sched)

    ### load the solver and create train and test nets
    solver = caffe.get_solver(out_solver_config_path)
    solver_param = read_solver_prototxt(out_solver_config_path)

    if args.stats:
        global stats_per_blob
        stats_per_blob = dist_stats.distStatsMap()

    train_acc, train_loss, test_acc, test_loss = solve(solver, solver_param)

    np.save(out_dir + '/test_acc-%s-%s.npy'%(param,prec),   test_acc)
    np.save(out_dir + '/train_loss-%s-%s.npy'%(param,prec), train_loss)

    precs = np.ones(len(train_loss))  * prec
    plot_acc_loss.plot_acc_loss_biter(test_acc, train_loss, precs, solver_param.test_interval, '%s %s'%(param,prec), amin=0.9 )

    if args.show_plot:
        plt.show()

    if args.save_plot:
        #outfile = "%s/plot_fixed-%s-%s.pdf"%(out_dir, param, prec)
        outfile = os.path.join(out_dir, "plot_fixed.pdf")
        print "saving plot to", outfile
        plt.savefig(outfile, format='pdf', dpi=1000)

def create_stretch_map(netproto, target_layers):
    stretch_map = dict()
    for layer in netproto.layer:
        if layer.name in target_layers:

            if layer.name not in stretch_map:
                stretch_map[layer.name] = dict()
            if param not in stretch_map[layer.name]:
                stretch_map[layer.name][param] = 1.;
    return stretch_map

def test_net(solver, solver_proto, test_it=100, blobstats=None, blobitemfreq=None):
    netproto = read_net_prototxt(solver_proto.net)
    test_it_acc = []
    test_it_loss = []
    for test_it in range(test_it):
        solver.test_nets[0].forward() # run inference on batch
        solver.test_nets[0].backward() # run backprop to compute gradients
        test_it_acc  += [solver.test_nets[0].blobs['accuracy'].data[()]]
        test_it_loss += [solver.test_nets[0].blobs['loss'].data[()]]
        if blobstats:
            get_net_blob_stats(blobstats, solver.test_nets[0], solver_proto.net, histo=args.histo)
        if blobitemfreq:
            update_itemfreq(blobitemfreq, solver.test_nets[0], netproto)
    if blobstats:
        blobstats.end_iter()
    test_acc  = np.mean(test_it_acc)
    test_loss = np.mean(test_it_loss)
    return test_acc, test_loss

def train_prog(init_solver_file, param, out_dir, prec_list = [0], start_prec=2, end_prec=11):
    plist = list(prec_list)
    prec = plist.pop(0)
    make_path(out_dir)
    temp_solver_file    = os.path.join(out_dir, 'temp_solver.prototxt')
    temp_net_file       = os.path.join(out_dir, 'temp_net.prototxt')

    stats_per_blob = None
    if args.stats:
        stats_per_blob = dist_stats.distStatsMap()

    test_acc = []
    test_loss = []
    train_acc = []
    train_loss = []
    precs = []

    iter = 0
    snap_iter = 0
    update = 0 # count each time the netproto is updated

    solver_proto = read_solver_prototxt(init_solver_file)
    netproto = read_net_prototxt(solver_proto.net)
    stats = pickle.load(open('stats.pickle')) 

    solver_proto.random_seed = 0xCAFFE
    solver_proto.net = temp_net_file  # point solver to new network prototxt
    solver_proto.test_net.append(temp_net_file)  # point solver to new network prototxt
    solver_proto.test_interval = 100      # Test after every 500 training iterations.
    solver_proto.momentum = 0.9
    solver_proto.base_lr = 0.01
    solver_proto.max_iter = ipp
    solver_proto.test_iter[0] = 100        # 100 batches = 10k images (full test set)
    solver_proto.snapshot = solver_proto.max_iter
    solver_proto.snapshot_prefix = os.path.join(out_dir, 'snapshot_update_%d'%update)
    
    write_prototxt(temp_solver_file, solver_proto)

    target_layers = [l.name for l in netproto.layer if l.type in ['Convolution','InnerProduct']]
        
    stretch_map = create_stretch_map(netproto, target_layers)

    # if prec: set_prec(param, prec, netproto, stretch_map)
    # write_prototxt(temp_net_file, netproto)
    # assert os.path.isfile(temp_solver_file)
    # assert os.path.isfile(temp_net_file)
    # solver = caffe.get_solver(temp_solver_file)

    # test net to get distribution of each blob
    write_prototxt(temp_net_file, netproto)
    solver = caffe.get_solver(temp_solver_file)
    stats = dist_stats.distStatsMap()
    for i in range(100):
        solver.test_nets[0].forward() # run inference on batch
        solver.test_nets[0].backward() # run backprop to compute gradients
        get_net_blob_stats(stats, solver.test_nets[0], solver_proto.net, histo=False)
    agg_stats = stats.aggregate()

    # write_prototxt(temp_net_file, netproto)
    # solver = caffe.get_solver(temp_solver_file)
    # print 'testing'
    # solver.test_nets[0].forward() # run inference on batch
    # sys.exit()

    if param == 'grd':
        print 'quantizing one layer at a time, measuring distributions'
        for layer in reversed(target_layers):
            set_prec(param, prec, netproto, stretch_map, stats.aggregate(), layer=layer)
            write_prototxt(temp_net_file, netproto)
            solver = caffe.get_solver(temp_solver_file)
            stats = dist_stats.distStatsMap()
            for i in range(100):
                solver.test_nets[0].forward() # run inference on batch
                solver.test_nets[0].backward() # run backprop to compute gradients
                get_net_blob_stats(stats, solver.test_nets[0], solver_proto.net, histo=False)
            agg_stats = stats.aggregate()
    else:
        if prec: set_prec(param, prec, netproto, stretch_map, agg_stats)
        write_prototxt(temp_net_file, netproto)
        solver = caffe.get_solver(temp_solver_file)

    fname = os.path.join(out_dir, 'net_update_%d.prototxt'%update)
    print 'writing net prototxt to', fname
    write_prototxt(fname, netproto)

    # while iter < solver_proto.max_iter:
    while 1:
        print 'test'
        test_bin_freq = None
        if args.itemfreq:
            test_bin_freq = dict()
        acc, loss = test_net(solver, solver_proto, 
                test_it=solver_proto.test_iter[0], 
                blobstats=stats_per_blob, 
                blobitemfreq=test_bin_freq)
        test_acc  += [acc]
        test_loss += [loss]
        print 'iter %5d test_acc %.4f test_loss %.4f' % (iter, acc, loss),

        if args.itemfreq:
            fname = os.path.join(out_dir,'itemfreq-%s-%s.npy'%(update, snap_iter))
            # dump_itemfreq(fname, test_bin_freq, netproto, show=True, short=True)
            print_itemfreq(test_bin_freq, netproto, short=True)
        print ''

        if args.test_dists:
            plot_blob_dists.plot_blob_dists(solver.test_nets[0], netproto, agg_stats)
            # plot_blob_dists.plot_blob_dists(solver.test_nets[0], netproto)
            outfile = "%s/plot_stats-%s.pdf"%(out_dir, iter)
            print "saving plot_blob_dists to", outfile
            plt.savefig(outfile, format='pdf')
            plt.close('all')

        print 'train'
        for train_it in range(solver_proto.test_interval):
            solver.step(1)
            train_acc  += [solver.net.blobs['accuracy'].data[()]]
            train_loss += [solver.net.blobs['loss'].data[()]]
            precs += [prec]
            iter += 1
            snap_iter += 1

        # exit early if any blob overflows 
        do_update = False
        if args.stretch:
            for layer in netproto.layer:
                if layer.name in test_bin_freq:
                    for param in test_bin_freq[layer.name]:
                        freq = test_bin_freq[layer.name][param]
                        if layer.name != 'conv1' and itemfreq_has_miniscus(freq):
                            print 'miniscus in', layer.name, param
                            if stretch_map[layer.name][param] < 1024:
                                if prec == 2:
                                    stretch_map[layer.name][param] *= 1.1;
                                else:
                                    stretch_map[layer.name][param] *= 2;
                                do_update = True

        if snap_iter == solver_proto.max_iter:
            if not plist:
                break
            prec = plist.pop(0)
            do_update=True

        if do_update:

            print 'test before update'
            test_bin_freq = None
            if args.itemfreq:
                test_bin_freq = dict()
            acc, loss = test_net(solver, solver_proto, 
                    test_it=solver_proto.test_iter[0], 
                    blobstats=stats_per_blob, 
                    blobitemfreq=test_bin_freq)
            print 'up   %5d test_acc %.4f test_loss %.4f' % (iter, acc, loss),

            if args.itemfreq:
                fname = os.path.join(out_dir,'itemfreq-%s-%s.npy'%(update, snap_iter))
                # dump_itemfreq(fname, test_bin_freq, netproto, show=True, short=True)
                print_itemfreq(test_bin_freq, netproto, short=True)
            print ''

            if args.test_dists:
                plot_blob_dists.plot_blob_dists(solver.test_nets[0], netproto, agg_stats)
                outfile = "%s/plot_stats-%s.pdf"%(out_dir, iter-1)
                # print "saving plot to", outfile
                plt.savefig(outfile, format='pdf')
                plt.close('all')

            # take snapshot, update snapshot file
            solver.snapshot()
            snapshot = get_newest_file(os.path.dirname(solver_proto.snapshot_prefix) + "/*caffemodel")
            print 'saving snapshot', snapshot
            # snapshot = solver_proto.snapshot_prefix + "_iter_%d.caffemodel" % (snap_iter)       
            snap_iter = 0
            update += 1
            solver_proto.snapshot_prefix = os.path.join(out_dir, 'snapshot_update_%d'%update)
            write_prototxt(temp_solver_file, solver_proto)

            # quantize weights in memory
            if 0 and quantize_weights:
                for layer in netproto.layer:
                    if layer.type in ['Convolution','InnerProduct']:
                        layer.fwd_wgt_precision_param.store_reduced=True
                update += 1
                solver_proto.snapshot_prefix = os.path.join(out_dir, 'snapshot_update_%d'%update)
                write_prototxt(temp_net_file, netproto)
                print 'writing', temp_net_file
                solver = caffe.get_solver(temp_solver_file)
                snapshot = get_newest_file(os.path.dirname(solver_proto.snapshot_prefix) + "/*caffemodel")
                print 'before quantize initializing weight from', snapshot
                solver.net.copy_from(str(snapshot))
                solver.test_nets[0].share_with(solver.net)
                plot_blob_dists.plot_blob_dists(solver.net, netproto)
                plt.savefig('before.pdf')
                # solver.test_nets[0].forward()
                solver.net.forward()
                solver.snapshot()
                snapshot = get_newest_file(os.path.dirname(solver_proto.snapshot_prefix) + "/*caffemodel")
                print 'after quantize saving snapshot', snapshot
                plot_blob_dists.plot_blob_dists(solver.net, netproto)
                plt.savefig('after.pdf')

            # update prec in net prototxt

            old_proto = copy.deepcopy(netproto)
            set_prec(param, prec, netproto, stretch_map, agg_stats)
            write_prototxt(temp_net_file, netproto)

            # reinitialize solver from snapshot
            solver = caffe.get_solver(temp_solver_file)
            snapshot = get_newest_file(os.path.dirname(solver_proto.snapshot_prefix) + "/*caffemodel")
            assert os.path.isfile(snapshot), 'File does not exist: %s' % snapshot
            print 'initializing weight from', snapshot
            solver.net.copy_from(str(snapshot))
            solver.test_nets[0].share_with(solver.net)

            if 1 and quantize_weights:
                for layer in old_proto.layer:
                    if layer.type in ['Convolution','InnerProduct']:
                        p = layer.fwd_wgt_precision_param.precision
                        s = layer.fwd_wgt_precision_param.scale
                        solver.net.params[layer.name][0].data[...] = quantizer.quantize(solver.net.params[layer.name][0].data, p, s, 2)
                        # solver.test_nets[0].params[layer.name][0].data[...] = quantizer.quantize(solver.test_nets[0].params[layer.name][0].data, p, s, 2)
                        print 'quantized', layer.name
                        print itemfreq(solver.test_nets[0].params[layer.name][0].data)
                solver.test_nets[0].share_with(solver.net)

    np.save(out_dir + '/test_acc.npy',   test_acc)
    np.save(out_dir + '/test_loss.npy', test_loss)
    np.save(out_dir + '/train_acc.npy',   train_acc)
    np.save(out_dir + '/train_loss.npy', train_loss)
    np.save(out_dir + '/precs.npy', precs)

    # plot overall acc and loss 
    title = out_dir
    if (np.mean(test_acc) > 0.95):
        amin=0.9
    else:
        amin=0

    plot_acc_loss.plot_acc_loss_biter(test_acc, train_loss, precs, 
            test_interval=solver_proto.test_interval, 
            title=title,
            amin=amin
            )

    if args.show_plot:
        plt.show()

    if args.save_plot:
        outfile = os.path.join(out_dir, "plot_prog.pdf")
        print "saving plot to", outfile
        plt.savefig(outfile, format='pdf', dpi=1000)

    if args.stats:
        dumpfile = os.path.join(out_dir,'stats.pickle')
        print 'saving stats to', dumpfile
        stats_per_blob.dump_aggregate_pickle(dumpfile)

    if args.plot_stats:
        plot_dist_stats.plot_dist_stats(stats_per_blob.aggregate(), out_dir)

def test_progressive(param, ipp, start_prec=2, end_prec=9, sched='full'):
    
    global out_dir
    out_dir = './results/param-%s-ipp-%d' % (param,ipp)
    make_path(out_dir)
    out_solver_config_path  = out_dir + '/temp_solver.prototxt'
    out_net_proto_path      = out_dir + '/temp_net.prototxt'
    global update
    update = 0

    test_interval = int(np.ceil(ipp/10.))
    if 'test' in sched:
        test_interval = ipp

    if args.stats:
        global stats_per_blob
        stats_per_blob = dist_stats.distStatsMap()

    # reset global variables from previous run
    global stretch_map
    stretch_map = dict()
    global test_bin_freq
    test_bin_freq = dict()

    test_acc = []
    train_loss = []
    precs = []
    outsnap = ''
    print '--------------------------------------------------'
    print 'Progressive training with %s from %d to %d bits' % (param, start_prec, end_prec)
    global prec
    prec = start_prec
    global total_iter
    while(total_iter < 10000):

        print 'reading solver config from', solver_config_path
        s = read_solver_prototxt(solver_config_path)

        update_netproto_file(s.net, param, prec, out_net_proto_path)

        s.random_seed = 0xCAFFE
        s.net = out_net_proto_path  # point solver to new network prototxt
        #s.test_nets.append(out_net_proto_path)  # point solver to new network prototxt
        s.test_interval = test_interval       # Test after every 500 training iterations.
        s.test_iter[0] = 100        # Test on 100 batches each time we test.
        s.max_iter = ipp            # no. of times to update the net (training iterations)
        s.snapshot = ipp
        sp = snapshot_prefix + '_update_%d' % update #  results/snapshot_prec_2_iter_1000.caffemodel
        s.snapshot_prefix = os.path.join(out_dir, sp)
        s.gamma = 0.0001
        s.power = 0.75
        open(out_solver_config_path, 'w').write(str(s))

        solver = caffe.get_solver(out_solver_config_path)

        solver_param = read_solver_prototxt(out_solver_config_path)

        a, l, last_it, early_exit = solve_from_snapshot(solver, solver_param, outsnap)

        test_acc += a
        train_loss += l
        precs += [prec] * len(l)
        update += 1
        #print 'test_acc =' ,test_acc 
        #print 'train_loss =' ,train_loss 
        #print 'precs =',precs 
     
        outsnap = solver_param.snapshot_prefix + "_iter_%d.caffemodel" % (last_it)       
    
        if early_exit:
            print 'early exit'
        # elif prec < end_prec:
            # prec += 1
        else:
            break


    np.save(out_dir + '/test_acc-%s.npy'%(param),   test_acc)
    np.save(out_dir + '/train_loss-%s.npy'%(param), train_loss)
    np.save(out_dir + '/precs-%s.npy'%(param), precs)

    # plot overall acc and loss 
    title = '%s %s %s'%(param, start_prec, end_prec)
    plot_acc_loss.plot_acc_loss_biter(test_acc, train_loss, precs, test_interval, title)

    if args.show_plot:
        plt.show()

    if args.save_plot:
        outfile = "%s/plot_prog-%s.pdf"%(out_dir, param)
        print "saving plot to", outfile
        plt.savefig(outfile, format='pdf', dpi=1000)

    if args.stats:
        dumpfile = out_dir + '/stats.pickle'
        print 'saving stats to', dumpfile
        stats_per_blob.dump_aggregate_pickle(dumpfile)
        plot_dist_stats.plot_dist_stats(stats_per_blob.aggregate(), out_dir)

def debug_exit(s):
    print 'debug exit: %s' % s
    sys.exit()

#################################### MAIN #####################################

if __name__ == '__main__':

    args = parse_args()

    train_net_path          = args.train_net
    test_net_path           = args.test_net
    solver_config_path      = args.solver
    snapshot_prefix         = args.snapshot_prefix

    accs = []
    losses = []
    stats_per_blob = None

    test_bin_freq = dict()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    if args.baseline:
        for seed in range(1,2):
            print 'running solver at full precision with seed %d'%seed
            out_dir = args.out_dir + '/baseline-seed-%d' % (seed)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_solver_config_path  = out_dir + '/temp_solver.prototxt'
            out_net_proto_path      = out_dir + '/temp_net.prototxt'
            set_solver('', 0, out_solver_config_path, sched=args.sched, seed=seed)
            solver = caffe.get_solver(out_solver_config_path)
            solver_param = read_solver_prototxt(out_solver_config_path)

            if args.stats:
                stats_per_blob = dist_stats.distStatsMap()

            train_acc, train_loss, test_acc, test_loss = solve(solver, solver_param)

            precs = np.zeros(len(train_acc))

            np.save(out_dir + '/test_acc.npy',  test_acc)
            np.save(out_dir + '/test_loss.npy', test_loss)
            np.save(out_dir + '/train_acc.npy', train_acc)
            np.save(out_dir + '/train_loss.npy',train_loss)
            np.save(out_dir + '/precs.npy', precs)

            print 'best test_acc', max(test_acc)

            if args.stats:
                dumpfile = out_dir + '/stats.pickle'
                print 'saving stats to', dumpfile
                stats_per_blob.dump_aggregate_pickle(dumpfile)

            plot_acc_loss.plot_acc_loss_biter(test_acc, train_loss, precs, 
                    test_interval=solver_param.test_interval, 
                    title=out_dir, 
                    amin=0.9 
                    )

            if args.save_plot:
                outfile = os.path.join(out_dir, "plot_acc_loss.pdf")
                print "saving plot to", outfile
                plt.savefig(outfile, format='pdf', dpi=1000)

    elif args.progressive:
        prec_list = [2,0]
        prec_str = '_'.join(['%d'%d for d in prec_list]) 
        ipp = 10000
        quantize_weights = 1
        mescale = 1
        # prec_list = range(2,3) + [0]
        # print '!!!!!!!!!!!!!!!!!!!!!!!!!TEST RUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        # for ipp in [10, 20, 50, 100, 200, 500]:
        # for param in ['act','wgt','grd']:
        for param in ['wgt']:
            for quantize_weights in [0,1]:
                for scalestat in ['std']:
                    out_dir = args.out_dir + '/quantize_wgt_storage-%d-store_reduced-scale_by_%s-%.1f-ipp-%d-param-%s-prec-%s' % (quantize_weights, scalestat, mescale, ipp, param, prec_str)
                    # out_dir = args.out_dir + '/test'
                    print 'running experiment', out_dir
                    train_prog(args.solver, param, out_dir, prec_list=prec_list)

    else:   # fixed precision
        start_prec = 2
        end_prec = 2
        print 'Training with fixed precision ' % (start_prec, end_prec)
        for prec in range(start_prec, end_prec+1):
            for param in ['fwd_act','fwd_wgt','bwd_grd']:
            # for param in ['fwd_wgt']:
            # for param in ['bwd_grd']:
                # test_fixed(prec, param)
                out_dir = args.out_dir + '/param-%s-prec-%d_to_full-nostretch' % (param,prec)
                train_prog(args.solver, param, out_dir, start_prec=prec)
                


