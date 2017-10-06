#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import itertools

def plot_acc_vs_prec(data, precs, labels):
    plt.clf()
    plt.title('Precision sensitivity per blob')
    plt.ylabel('accuracy')
    plt.xlabel('precision')

    ax = plt.gca()
    # ax.axhline(1, color='lightgray', linestyle=':')
    ax.grid(True, which='both', linestyle=':')
    ax.minorticks_off()


    markers=itertools.cycle(['^', 'D', 'o', 'p', 's', 'v', '>', '*'])

    for idx, label in enumerate(labels):
        plt.plot(precs[idx], data[idx], label=label, marker=next(markers), linestyle='--')
    plt.legend()

def sorting(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    # l1 and l2 has to be numpy arrays
    idx = np.argsort(l1)
    return list(l1[idx]), list(l2[idx])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "acc_file", nargs="+", help="accuracy npy")
    parser.add_argument( "--max", action='store_true', help="plot the max accuracy per run")
    args = parser.parse_args()
    
    labels = []
    data = []
    precs = []

    for acc in args.acc_file:

        dirname = os.path.dirname(acc)
        filename = os.path.basename(acc)
        file, _ = os.path.splitext(filename)
        try:
            _, blob, prec = file.split('-')
        except ValueError:
            print "Error: file does not follow convention file-blob-prec.npy:", filename

        test_acc = np.load(acc)

        if (args.max):
            datum = max(test_acc) # use the maximum accuracy for plotting
        else:
            datum = test_acc[-1] # use the maximum accuracy for plotting

        print acc, datum

        prec = int(prec)

        if blob not in labels:
            labels.append(blob)
            precs.append([prec])
            data.append([datum])
        else:
            blob_idx = labels.index(blob)
            data[blob_idx].append(datum)
            precs[blob_idx].append(prec)
            precs[blob_idx], data[blob_idx] = sorting(precs[blob_idx], data[blob_idx])

    print ''
    
    # data = [n, x], precs = x, labels = n
    plot_acc_vs_prec(data, precs, labels)

    # print csv
    print '        ,', ','.join(['%6d'%p for p in precs[0]])
    for row, label in zip(data, labels):
        print label , ',', 
        print ','.join(['%.4f'%r for r in row])

    # savefile = os.path.join(dirname, 'acc_vs_prec.pdf')
    savefile = 'acc_vs_prec.pdf'
    print 'saving figure to', savefile
    plt.savefig(savefile)
        # plt.show()
