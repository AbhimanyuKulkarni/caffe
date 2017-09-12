#!/usr/bin/env python

import argparse
import sys
import os
os.environ['GLOG_minloglevel'] = '2' # supress caffe output
import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import plot_acc_loss
import dist_stats
import pickle
from scipy.stats import itemfreq, norm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--solver", help="solver prototxt")
    parser.add_argument( "--train_net", help="train net prototxt")
    parser.add_argument( "--test_net", help="solver prototxt")
    parser.add_argument( "--snapshot_prefix", help="prefix of snapshot file, will go in out_dir")
    parser.add_argument( "--save_plot", help="save plot to file", action='store_true')
    parser.add_argument( "--show_plot", help="show plot", action='store_true')
    parser.add_argument( "--out_dir", help="directory to store temp files and plots", default="")
    parser.add_argument( "--baseline", help="run baseline solver", action='store_true')
    parser.add_argument( "--itemfreq", help="measure itemfreq of quantized blob", action='store_true')
    parser.add_argument( "--histo", help="dump histogram of each blob", action='store_true')
    parser.add_argument( "--sched", help="schedule: [test,short,full]", type=str, default="full")
    args = parser.parse_args()
    return args

def read_solver_parameter(filename):
    from caffe.proto import caffe_pb2
    s = caffe_pb2.SolverParameter()

    with open(filename) as f:
        text_format.Merge(str(f.read()), s)

    return s

def read_net_prototxt(filename):
    netproto = caffe_pb2.NetParameter()
    with open(filename) as f:
        text_format.Merge(str(f.read()), netproto)
    return netproto

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

def quantize_cdf(prec, scale, cdf, bins):

    abs_max = float( 2 ** (prec-1) - 1 ) / scale
    nlevels = 2 ** (prec) - 1
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

def scale_by_histo(prec, histo, layer_name, blob_name, byrange=False):
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

    scales = []
    imbas = []

    if byrange:
        print 'scaling by range'
        trails = 1
    else:
        print 'balancing quantization bins'
        trails = 1000

    for j in np.arange(trails):
        scale = float( 2 ** (prec-1) - 1 ) / abs_max
        # print 'scale=',scale 

        pop, bin_edges = quantize_cdf(prec, scale, cdf, bins)
        # imba = imbalance(pop, dist='normal')
        imba = imbalance(pop, dist='uniform')
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

    print 'scale =',scale, '@ imba =',min(imbas)
    pop, bin_edges = quantize_cdf(prec, scale, cdf, bins)
    print 'balanced quantization:', pop
    return scale


def scale_by_range(stats, layer_name, blob_name):
    '''
        determine scaling factor from statistics on the data distribution
    '''
    stat_name = '-'.join([layer_name, blob_name])
    dmin = stats[stat_name]['min'][0] # use the last iteration
    dmax = stats[stat_name]['max'][0]
    std  = stats[stat_name]['std'][-1]
    abs_max = max( abs(dmin), abs(dmax) )
    scale = float( 2 ** (prec-1) - 1 ) / abs_max
    # scale = float( 2 ** (prec-1) - 1 ) / std
    return scale

def set_solver(param, prec, scale, solver_file, sched='full'):

    # Read existing solver
    s = caffe_pb2.SolverParameter()

    print 'reading solver config from',solver_config_path
    with open(solver_config_path) as f:
        text_format.Merge(str(f.read()), s)

    s.random_seed = 0xCAFFE

    # Overwrite locations of the train and test networks.
    if train_net_path:
        s.train_net = train_net_path

    if test_net_path:
        s.test_net.append(test_net_path)

    if (s.train_net and s.net):
        print "Error: can't set net, train_net already set to", s.train_net

    stats = pickle.load(open('stats.pickle')) 
    print stats.keys()
    # Update net prototxt

    # Read net prototxt
    netproto = caffe_pb2.NetParameter()
    with open(s.net) as f:
        text_format.Merge(str(f.read()), netproto)

    with open('histograms.pickle') as f:
        histo = pickle.load(f)

    for layer in netproto.layer:
        if layer.type in ['Convolution','InnerProduct']:
            print 'setting precision', layer.name, param, prec, scale
            # support substring matching
            # param='act' will set both fwd_act and bwd_act

            # we don't want to compress the range of first layer activations
            first_act = (param in 'fwd_act' and layer.name in 'conv1')
            scale = scale_by_histo(prec, histo, layer.name, param, byrange=first_act)
            if param in 'fwd_act':
                layer.fwd_act_precision_param.precision=prec
                layer.fwd_act_precision_param.scale=scale
            if param in 'fwd_wgt':
                layer.fwd_wgt_precision_param.precision=prec
                layer.fwd_wgt_precision_param.scale=scale
                # layer.fwd_wgt_precision_param.scale=scale_by_range(stats, layer.name, 'wgt_data')
            if param in 'bwd_act':
                layer.bwd_act_precision_param.precision=prec
                layer.bwd_act_precision_param.scale=scale
                # layer.bwd_act_precision_param.scale=scale_by_range(stats, layer.name, 'act_in_data')
            if param in 'bwd_wgt':
                layer.bwd_wgt_precision_param.precision=prec
                layer.bwd_wgt_precision_param.scale=scale
                # layer.bwd_wgt_precision_param.scale=scale_by_range(stats, layer.name, 'wgt_data')
            if param in 'bwd_grd':
                layer.bwd_grd_precision_param.precision=prec
                layer.bwd_grd_precision_param.scale=scale
                # layer.bwd_grd_precision_param.scale=scale_by_range(stats, layer.name, 'act_out_diff')



    # Write the solver to a temporary file and return its filename.
    print 'writing to', out_net_proto_path
    with open(out_net_proto_path, 'w') as f:
        f.write(str(netproto))

    # OVERWRITE SOLVER PARAMETERS

    s.net = out_net_proto_path

    if sched == 'full':
        s.test_interval = 100  # Test after every 500 training iterations.
        s.test_iter[0] = 100 # Test on 100 batches each time we test.
        s.max_iter = 10000     # no. of times to update the net (training iterations)

        # # Snapshots are files used to store networks we've trained.
        # # We'll snapshot every 5K iterations -- twice during training.
        s.snapshot = 1000
    elif sched == 'short':
        s.test_interval = 10  # Test after every 500 training iterations.
        s.test_iter[0] = 100 # Test on 100 batches each time we test.
        s.max_iter = 100     # no. of times to update the net (training iterations)
        s.snapshot = 10      # snapshot weights every N iterations
    elif sched == 'test':
        s.test_interval = 1  # Test after every 500 training iterations.
        s.test_iter[0] = 1 # Test on 100 batches each time we test.
        s.max_iter = 1     # no. of times to update the net (training iterations)
        s.snapshot = 1      # snapshot weights every N iterations
    else:
        raise NameError('Unknown Schedule \'%s\''%sched)

    if snapshot_prefix:
        s.snapshot_prefix = os.path.join(args.out_dir, snapshot_prefix)
     
    # # EDIT HERE to try different solvers
    # # solver types include "SGD", "Adam", and "Nesterov" among others.
    # s.type = "SGD"

    # # Set the initial learning rate for SGD.
    # s.base_lr = 0.01  # EDIT HERE to try different learning rates

    # # Set momentum to accelerate learning by
    # # taking weighted average of current and previous updates.
    # s.momentum = 0.9

    # # Set weight decay to regularize and prevent overfitting
    # s.weight_decay = 5e-4

    # # Set `lr_policy` to define how the learning rate changes during training.
    # # This is the same policy as our default LeNet.
    # s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75

    # # EDIT HERE to try the fixed rate (and compare with adaptive solvers)
    # # `fixed` is the simplest policy that keeps the learning rate constant.
    # # s.lr_policy = 'fixed'

    # # Display the current training loss and accuracy every 1000 iterations.
    # s.display = 1000


    # # Train on the GPU
    # s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    with open(solver_file, 'w') as f:
        f.write(str(s))

def solve(solver, solver_param):

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
        
        # store the train loss
        # [()] extracts scalar from 0d array
        train_loss.append( solver.net.blobs['loss'].data[()] )
        # print 'Train Loss =',train_loss[-1]
        
        # test_net = solver.test_net[0]
        test_net = solver.net

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            acc = []
            loss = []
            test_bin_freq = dict()
            for test_it in range(test_iter):
                test_net.forward() # run inference on batch
                test_net.backward() # run backprop to compute gradients

                get_net_blob_stats(test_net, solver_param.net)
                
                # examine quantized data:
                if args.itemfreq:
                    for layer in netproto.layer:
                        if layer.type in ['Convolution','InnerProduct']:

                            if layer.fwd_act_precision_param.precision:
                                data = test_net.blobs[layer.bottom[0]].data
                            elif layer.fwd_wgt_precision_param.precision:
                                data = test_net.params[layer.name][0].data
                            elif layer.bwd_grd_precision_param.precision:
                                data = test_net.blobs[layer.name].diff
                            else:
                                data = 0

                            freq = itemfreq(data)
                            if layer.name not in test_bin_freq:
                                test_bin_freq[layer.name] = freq
                            else:
                                test_bin_freq[layer.name]  = add_itemfreq(test_bin_freq[layer.name], freq)

                acc.append( test_net.blobs['accuracy'].data )
                loss.append(test_net.blobs['loss'].data[()] )

            test_acc.append( np.mean(acc) )
            test_loss.append( np.mean(loss) )
            print 'Iteration %5d Test Accuracy =' % it, test_acc[-1], 'Test Loss =', test_loss[-1]

            if args.itemfreq:
                for k in test_bin_freq.keys():
                    print '               ', k
                    print 'quantized value', test_bin_freq[k][:,0]
                    tbf = test_bin_freq[k][:,1]
                    tbf = tbf / sum(tbf) # normalize
                    print 'probability    ', tbf

                fname = out_dir + '/itemfreq-%s-%s-%s.npy'%(param,prec,it)
                print 'writing itemfreqs to', fname
                with open(fname, 'w') as f:
                    pickle.dump(test_bin_freq, f)

            global stats_per_blob
            stats_per_blob.end_iter()

    return test_acc, train_loss

def add_itemfreq(f1, f2):
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

def myhisto(d):
    return np.histogram(d, bins=1000, normed=True)

def get_net_blob_stats(net, net_proto_file):

    proto = read_net_prototxt(net_proto_file)

    global stats_per_blob

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

        if args.histo:
            cdfs[name]['fwd_wgt'] = myhisto(net.params[name][0].data)
            cdfs[name]['fwd_act'] = myhisto(net.blobs[bottom].data)
            cdfs[name]['bwd_grd'] = myhisto(net.blobs[name].diff)

    if args.histo:
        dumpfile = 'histograms.pickle'
        print 'writing histograms to', dumpfile
        with open(dumpfile,'w') as f:
            pickle.dump(cdfs, f)

def debug_exit(s):
    print 'debug exit: %s' % s
    sys.exit()

#################################### MAIN #####################################

if __name__ == '__main__':

    args = parse_args()

    if not args.out_dir:
        out_dir = './'
    else:
        out_dir = args.out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    train_net_path          = args.train_net
    test_net_path           = args.test_net
    solver_config_path      = args.solver
    snapshot_prefix         = args.snapshot_prefix
    out_solver_config_path  = out_dir + '/temp_solver.prototxt'
    out_net_proto_path      = out_dir + '/temp_net.prototxt'

    accs = []
    losses = []

    caffe.set_mode_gpu()
    caffe.set_device(0)


    if args.baseline:
        print 'running solver at full precision'
        solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
        solver = caffe.get_solver(solver_config_path)
        solver_param = read_solver_parameter(solver_config_path)

        stats_per_blob = dist_stats.distStatsMap()

        test_acc, train_loss = solve(solver, solver_param)

        print "RESULT test_acc",test_acc
        print "RESULT train_loss",train_loss
        # print stats_per_blob

        dumpfile = out_dir + '/stats.pickle'
        print 'saving stats to', dumpfile
        stats_per_blob.dump_aggregate_pickle(dumpfile)

    else:
        for prec in range(2,7):
            # for param in ['fwd_act','fwd_wgt','bwd_act','bwd_wgt','bwd_grd']:
            for param in ['fwd_act','fwd_wgt','bwd_grd']:

                scale = 2**prec
                # if 'wgt' in param:
                    # scale  = 2**prec

                print 'experiment: %s %s %s\n\n'%(param, prec, scale)

                # update solver
                set_solver(param, prec, scale, out_solver_config_path, sched=args.sched)

                ### load the solver and create train and test nets
                solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
                solver = caffe.get_solver(out_solver_config_path)
                solver_param = read_solver_parameter(out_solver_config_path)

                stats_per_blob = dist_stats.distStatsMap()

                test_acc, train_loss = solve(solver, solver_param)

                print "RESULT test_acc",test_acc
                # print "RESULT train_loss",train_loss
                # print stats_per_blob

                plot_acc_loss.plot_acc_loss(test_acc, train_loss, '%s %s'%(param,prec) )

                np.save(out_dir + '/test_acc-%s-%s.npy'%(param,prec),   test_acc)
                np.save(out_dir + '/train_loss-%s-%s.npy'%(param,prec), train_loss)

                if args.show_plot:
                    plt.show()

                if args.save_plot:
                    outfile = "%s/plot-%s-%s.pdf"%(out_dir, param, prec)
                    print "saving plot to", outfile
                    plt.savefig(outfile, format='pdf', dpi=1000)

                if args.sched == 'test':
                    debug_exit('test schedule: only one iteration')


