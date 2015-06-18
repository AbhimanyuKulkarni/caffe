#!/bin/bash

echo lenet
get_uniform-mag-prec.pl lenet-uniform-mag-prec-100/ | filter_bandwidth.pl | tolerance.pl
get_optimal_mag_prec.pl lenet-find_optimal-mag_prec/ | filter_bandwidth.pl | tolerance.pl

echo convnet
get_uniform-mag-prec.pl convnet-uniform-mag-prec-100/ | filter_bandwidth.pl  | tolerance.pl
get_optimal_mag_prec.pl convnet-find_optimal-mag_prec/ | filter_bandwidth.pl | tolerance.pl

echo alexnet
get_uniform_bandwidth.pl alexnet-uniform-max_data_mag-prec0-100/ | filter_bandwidth.pl | tolerance.pl
get_optimal_accuracy.pl alexnet-find_optimal-max_data_mag | filter_bandwidth.pl | tolerance.pl

echo nin
get_uniform_bandwidth.pl nin_imagenet-uniform-max_data_mag-prec0-100/ | filter_bandwidth.pl | tolerance.pl
get_optimal_accuracy.pl nin_imagenet-find_optimal-max_data_mag/ | filter_bandwidth.pl | tolerance.pl

echo googlenet
get_uniform_bandwidth.pl googlenet-uniform-max_data_mag-prec2-100/ | filter_bandwidth.pl | tolerance.pl
get_optimal_accuracy.pl googlenet-find_optimal-max_data_mag/ | filter_bandwidth.pl | tolerance.pl
