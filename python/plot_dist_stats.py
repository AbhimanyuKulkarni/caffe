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

def plot_lines(data, labels, title='', xlabel='', ylabel=''):

    fig = plt.gcf()
    fig.clear() # overwrite existing figure

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for vec, lab in zip(data, labels):
        plt.plot(vec, label=lab)

    plt.legend()

#################################### MAIN #####################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "pickle", help="pickle containing map of dist stats")
    parser.add_argument( "--out_dir", help="directory to store temp files and plots", default=".")
    args = parser.parse_args()

    stats_map = None
    with open(args.pickle) as f:
        stats_map = pickle.load(f)

    for blob, stats in stats_map.iteritems():

        stat_names = stats.keys()
        stat_vals = [v for v in stats.itervalues()]

        plot_lines(
                stat_vals, 
                stat_names, 
                xlabel='test iterations', 
                title=blob,
            )

        outfile = "%s/plot-%s.pdf"%(args.out_dir, blob)
        print "saving plot to", outfile
        plt.savefig(outfile, format='pdf', dpi=200)



