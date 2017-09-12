#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def plot_acc_loss(test_acc, train_loss, title):

    test_interval = len(train_loss)/len(test_acc)

    fig = plt.gcf()
    fig.clear() # overwrite existing figure

    ax1 = plt.gca()

    # plot loss
    ax1.plot( np.arange(len(train_loss)), train_loss, 'lightblue' )
    ax1.set_ylim(0,3)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')

    # plot accuracy
    ax2 = ax1.twinx()
    ax2.plot( test_interval * np.arange(len(test_acc)), test_acc, 'green' )
    ax2.set_ylim(0,1)
    ax2.set_ylabel('test accuracy')
    ax2.set_title(title)

    # annotate max accuracy
    argmax = np.argmax(test_acc)
    ymax = test_acc[argmax]
    xmax = test_interval * argmax
    ax2.annotate('%.4f' % ymax, xy=(xmax,ymax), xytext=(xmax, ymax+0.01))
    ax2.plot(xmax, ymax, marker='*')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "acc", help="accuracy npy")
    parser.add_argument( "loss", help="loss npy")
    args = parser.parse_args()

    test_acc = np.load(args.acc)
    train_loss = np.load(args.loss)

    plot_acc_loss(test_acc, train_loss, 'test')

    plt.show()
