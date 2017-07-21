#!/usr/bin/env python

import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe import layers as L, params as P
import argparse
from google.protobuf import text_format
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--solver", help="solver prototxt")
    parser.add_argument( "--train_net", help="train net prototxt")
    parser.add_argument( "--test_net", help="solver prototxt")
    parser.add_argument( "--snapshot_prefix", help="prefix of snapshot file, including path")
    parser.add_argument( "--save_plot", help="save plot to file", default="")
    parser.add_argument( "--show_plot", help="show plot", action='store_true')
    parser.add_argument( "--out_dir", help="directory to store temp files and plots", default="")
    args = parser.parse_args()
    return args

def set_solver(param, prec):

    # Read existing solver
    from caffe.proto import caffe_pb2
    s = caffe_pb2.SolverParameter()

    print 'reading solver config from',solver_config_path
    with open(solver_config_path) as f:
        text_format.Merge(str(f.read()), s)

    print s

    s.random_seed = 0xCAFFE

    # Overwrite locations of the train and test networks.
    if train_net_path:
        s.train_net = train_net_path

    if test_net_path:
        s.test_net.append(test_net_path)

    if (s.train_net and s.net):
        print "Error: can't set net, train_net already set to", s.train_net

    # Update net prototxt

    # Read net prototxt
    netproto = caffe_pb2.NetParameter()
    with open(s.net) as f:
        text_format.Merge(str(f.read()), netproto)

    for layer in netproto.layer:
        print layer.name, layer.type
        if layer.type in ['Convolution','InnerProduct']:
            print 'setting precision'
            if param == 'fwd_act':
                layer.fwd_act_precision_param.precision=prec
            if param == 'fwd_wgt':
                layer.fwd_wgt_precision_param.precision=prec
            if param == 'bwd_act':
                layer.bwd_act_precision_param.precision=prec
            if param == 'bwd_wgt':
                layer.bwd_wgt_precision_param.precision=prec
            if param == 'bwd_grd':
                layer.bwd_grd_precision_param.precision=prec

    # Write the solver to a temporary file and return its filename.
    with open(out_net_proto_path, 'w') as f:
        f.write(str(netproto))

    # OVERWRITE SOLVER PARAMETERS

    # s.test_interval = 500  # Test after every 500 training iterations.
    # s.test_iter.append(100) # Test on 100 batches each time we test.

    # s.max_iter = 10000     # no. of times to update the net (training iterations)
     
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

    # # Snapshots are files used to store networks we've trained.
    # # We'll snapshot every 5K iterations -- twice during training.
    s.snapshot = 10
    if snapshot_prefix:
        s.snapshot_prefix = snapshot_prefix

    # # Train on the GPU
    # s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    with open(out_solver_config_path, 'w') as f:
        f.write(str(s))

def solve(solver, niter, test_interval ):

    train_loss = list()
    test_acc = list()

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        
        # store the train loss
        # [()] extracts scalar from 0d array
        train_loss.append( solver.net.blobs['loss'].data[()] )
        print 'Train Loss =',train_loss[-1]
        
        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            acc = list()
            for test_it in range(100):
                solver.test_nets[0].forward() # run inference on batch
                acc.append(solver.test_nets[0].blobs['accuracy'].data)
            test_acc.append( np.mean(acc) )
            print 'Test Accuracy =',test_acc[-1]

    return test_acc, train_loss

def plot_acc_loss(test_acc, train_loss, test_interval, title):

    _, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot( np.arange(len(train_loss)), train_loss)
    ax2.plot( test_interval * np.arange(len(test_acc)), test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('{} Test Accuracy: {:.2f}'.format(title, test_acc[-1]))

# print 'debug exit'
# sys.exit()

#################################### MAIN #####################################

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
out_plot_prefix         = out_dir + '/' + args.save_plot

accs = []
losses = []


for param in ['fwd_act','fwd_wgt','bwd_act','bwd_wgt','bwd_grd']:
    for prec in range(1,8):

        print 'experiment: %s %s\n\n'%(param, prec)

        set_solver(param, prec)

        ### load the solver and create train and test nets
        solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
        solver = caffe.get_solver(out_solver_config_path)

        ### solve
        niter = 20  
        test_interval = niter/10

        test_acc, train_loss = solve(solver, niter, test_interval)

        plot_acc_loss(test_acc, train_loss, test_interval, '%s %s'%(param,prec) )

        if args.show_plot:
            plt.show()

        if args.save_plot:
            outfile = "%s-%s-%s.pdf"%(out_plot_prefix, param, prec)
            print "saving plot to", outfile
            plt.savefig(outfile, format='pdf', dpi=1000)


