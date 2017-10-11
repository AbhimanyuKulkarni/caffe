#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import pickle
# import caffe
# from caffe.proto import caffe_pb2
# from google.protobuf import text_format
# import plot_acc_loss
# import dist_stats

def exit():
    print 'debug exit'
    sys.exit()

def plot_lines(data, labels, title='', xlabel='', ylabel='', newfig=True):

    if newfig:
        fig = plt.gcf()
        fig.clear() # overwrite existing figure

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for vec, lab in zip(data, labels):
        plt.plot(vec, label=lab)

    plt.legend()

def plot_dist_stats(stats_map, out_dir):
    """
    Generate a plot of stats over iterations for each of the entries in stats_map 
    
    Args:
        stats_map:  a DistStatMap: stats_map[blob][stat][iteration]
        out_dir:    directory to saves plot pdfs 
    """

    for blob, stats in stats_map.iteritems():

        stat_names = stats.keys()
        stat_vals = [v for v in stats.itervalues()]

        plot_lines(
                stat_vals, 
                stat_names, 
                xlabel='test iterations', 
                title=blob,
            )

        outfile = "%s/plot-%s.pdf"%(out_dir, blob)
        print "saving plot to", outfile
        plt.savefig(outfile, format='pdf', dpi=200)

def plot_dist_stats_grid(stats_map):
    ''' plots stat line graphs for blob and layer in a grid'''

    layers = []
    blobs = []
    for k in stats_map.keys():
        layer, blob = k.split('-')
        if blob not in ['act_in_data','wgt_data','act_out_diff', 'wgt_diff']:
            continue
        if layer not in layers:
            layers.append(layer)
        if blob not in blobs:
            blobs.append(blob)

    layers = sorted(layers)
    blobs = sorted(blobs)
    print "layers =",layers
    print "blobs =",blobs
    h = len(layers)
    w = len(blobs)

    fig = plt.figure(figsize=(8.27,11.69)) # A4

    i = 1
    for l, layer in enumerate(layers):
        for b, blob in enumerate(blobs): 
            k = '-'.join([layer,blob])
            stats = stats_map[k]
            stat_names = stats.keys()
            stat_vals = [v for v in stats.itervalues()]

            fig.add_subplot(h,w,i)
            plot_lines(
                    stat_vals, 
                    stat_names, 
                    xlabel='test iterations', 
                    title=k,
                    newfig=False
                )

            # plt.tick_params(axis='y', which='both', left='off', labelleft='off')
            # plt.locator_params(axis='x', nticks=3)
            # plt.ylim( (0, ymax) )
            # plt.text(0, ymax, '%.2f' % zero_frac, verticalalignment = 'top' ) 
            i += 1

    plt.tight_layout()

#################################### MAIN #####################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "pickle", help="pickle containing map of dist stats")
    parser.add_argument( "--out_dir", help="directory to store temp files and plots", default=".")
    args = parser.parse_args()

    stats_map = None
    with open(args.pickle) as f:
        stats_map = pickle.load(f)

    plot_dist_stats(stats_map, args.out_dir)


