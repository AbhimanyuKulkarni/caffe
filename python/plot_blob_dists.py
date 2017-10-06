#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import pickle


def plot_blob_dists(net, netproto, distStats=None):
    ''' plots a histogram for each blob and layer '''
    #plt.clf()
    #plt.ylabel('accuracy')
    #plt.xlabel('precision')

    layers = []
    blobs = ['fwd_act','fwd_wgt','bwd_grd']
    for layer in netproto.layer:
        if layer.type in ['Convolution','InnerProduct']:
            layers.append(layer)

    h = len(layers)
    w = len(blobs)

    #fig = plt.figure(figsize=(4*w,3*h))
    #fig = plt.figure(figsize=(8,11))
    fig = plt.figure(figsize=(8.27,11.69)) # A4


    i = 1
    for l, layer in enumerate(layers):
        for b, blob in enumerate(blobs): 
            if blob == 'fwd_act':
                data = net.blobs[layer.bottom[0]].data
                stat = layer.name + '-act_in_data'
            elif blob == 'fwd_wgt':
                data = net.params[layer.name][0].data
                stat = layer.name + '-wgt_data'
            elif blob == 'bwd_grd':
                data = net.blobs[layer.name].diff
                stat = layer.name + '-act_out_diff'
            else:
                print "Error: unknown blob", blob

            fig.add_subplot(h,w,i)
            plt.title(layer.name + '-' + blob)

            # todo: scale y by next biggest bin
            counts, edges, _ = plt.hist(data.flatten(),normed=True,bins=100)
            sort_counts = np.sort(counts)
            plt.tick_params(axis='y', which='both', left='off', labelleft='off')
            plt.locator_params(axis='x', nticks=3)
            ymax = sort_counts[-2] * 2
            plt.ylim( (0, ymax) )
            zero_frac = sort_counts[-1] * 1. / sort_counts.sum()
            plt.text(0, ymax, '%.2f' % zero_frac, verticalalignment = 'top' ) 
            if distStats:
                dmin = min(distStats[stat]['min'])
                dmax = max(distStats[stat]['max'])
                plt.xlim((dmin,dmax))
            i += 1

    plt.tight_layout()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "acc_file", nargs="+", help="accuracy npy")
    args = parser.parse_args()
