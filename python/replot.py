#!/usr/bin/env python
# reads npy files corresponding to provided plot pdfs
# uses them to regenerate plots with new parameters

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from plot_acc_loss import plot_acc_loss


parser = argparse.ArgumentParser(
        description='Reads npy files corresponding to provided plots and reruns plot_acc_loss, overwriting plots'
        )
parser.add_argument( "plot", nargs='*', help="plot pdfs")
args = parser.parse_args()

for plot_file in args.plot:

    dir, plot_filename = os.path.split(plot_file)
    name, ext = os.path.splitext(plot_filename)
    prefix, param, prec = name.split('-')

    test_acc_file   = '%s/test_acc-%s-%s.npy'   % (dir, param, prec)
    train_loss_file = '%s/train_loss-%s-%s.npy' % (dir, param, prec)
    outfile         = '%s/plot-%s-%s.pdf'       % (dir, param, prec)

    test_acc = np.load(test_acc_file)
    train_loss = np.load(train_loss_file)

    plot_acc_loss(test_acc, train_loss, param + '-' + prec)

    print "saving plot to", outfile
    plt.savefig(outfile, format='pdf', dpi=1000)
