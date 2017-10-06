#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def plot_acc_loss(test_acc, train_loss, title):

    test_interval = len(train_loss)/len(test_acc)

    #fig = plt.gcf()
    #fig.clear() # overwrite existing figure
    fig = plt.figure(figsize=(8,6))

    ax1 = plt.gca()

    # plot loss
    ax1.plot( np.arange(len(train_loss)), train_loss, 'lightblue' )
    ax1.set_ylim(0,3)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')

    # plot accuracy
    ax2 = ax1.twinx()
    x = test_interval * np.arange(1,len(test_acc)+1) - 1
    ax2.plot( x, test_acc, 'green' )
    ax2.set_ylim(0,1)
    ax2.set_ylabel('test accuracy')
    ax2.set_title(title)

    # annotate max accuracy
    argmax = np.argmax(test_acc)
    ymax = test_acc[argmax]
    xmax = x[argmax]
    ax2.annotate('%.4f' % ymax, xy=(xmax,ymax), xytext=(xmax, ymax+0.01))
    ax2.plot(xmax, ymax, marker='*')

def plot_acc_loss_biter(test_acc, train_loss, precs, test_interval, title):
    ''' plot accuracy and loss over biters
        biters are bits of precision * iterations
        an estimation of time on bit serial hardware
    '''

    marker='|'
    #test_interval = int(float(len(train_loss))/len(test_acc))

    # end time of each iteration
    biters = np.cumsum(precs)

    if len(test_acc) * test_interval != len(train_loss):
        print 'Warning: test_acc %s missaligned with train_loss %s' % (len(test_acc),len(train_loss))

    #fig = plt.gcf()
    #fig.clear() # overwrite existing figure
    fig = plt.figure(figsize=(8,6))

    ax1 = plt.gca()
    ax1.set_xlabel('biters')
    plt.title(title)

    # annotate prec boundaries
    # also determine test_idx since we test before and after changing precision
    test_idx = []
    for i,p in enumerate(precs):
        if (i+1) % test_interval == 0:
            test_idx.append(i)
        #if i < len(precs)-1 and precs[i] != precs[i+1]:
        if i > 0 and precs[i] != precs[i-1]:
            plt.axvline(biters[i], color='lightgrey', zorder=-1)
            #test_idx.append(i)

    # plot loss
    i = np.arange(0, len(train_loss)+0)
    x = biters[i]
    ax1.plot( x, train_loss, 'lightblue', marker=marker )
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('train loss')

    # plot accuracy
    ax2 = ax1.twinx()
    # test iterations happen after test_interval iterations
    # e.g. i = 0, 100, 200
    acc_idx = np.arange(0, len(test_acc))
    i = test_interval * acc_idx
    #i = np.arange(0, len(test_acc), test_interval) 
    x = biters[test_idx]
    ax2.plot( x, test_acc, 'green', marker=marker )
    ax2.set_ylim(0,1.1)
    ax2.axhline(1, color='lightgrey', linestyle=':')
    ax2.set_ylabel('test accuracy')
    plt.xlim(left=0)


    # annotate max accuracy
    argmax = np.argmax(test_acc)
    ymax = test_acc[argmax]
    imax = test_idx[argmax]
    xmax = biters[imax]
    ax2.annotate('%.4f' % ymax, xy=(xmax,ymax), xytext=(xmax, ymax+0.01), ha='right', va='bottom')
    ax2.plot(xmax, ymax, marker='*')

    return ax1, ax2



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "acc", help="accuracy npy")
    parser.add_argument( "loss", help="loss npy")
    args = parser.parse_args()

    test_acc = np.load(args.acc)
    train_loss = np.load(args.loss)

    plot_acc_loss(test_acc, train_loss, 'test')

    plt.show()
