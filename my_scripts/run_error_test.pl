#!/usr/bin/perl

use bignum;

chdir("~/source/caffe_error");

#my $outputDir = "run/gaussian-0-mean";
my $outputDir = "run/test";
my $errorType = 2;
my $model = "examples/cifar10/cifar10_quick_train_test.prototxt";
my $weights = "examples/cifar10/cifar10_quick_iter_4000.caffemodel";
my $iterations = "50";

#my @dirs = ("gaussian-0-mean","add-exp","sub-exp");

#for (my $e=1; $e<=2; $e++){
#  print "outputDir=$outputDir\n";
  $errorType = 3;
  $outputDir = "run/bitflip_sef";
  if (! -d "$outputDir"){
    mkdir("$outputDir");
  }
  print "CAFFE_ERROR_TYPE=$errorType\n";
  for (my $i = 0.0; $i <= 0.000; $i += 0.00001){
    print "CAFFE_ERROR=$i\n";
    $ENV{"CAFFE_ERROR"} = "$i";
    $ENV{"CAFFE_ERROR_TYPE"} = "$errorType";
    my $runStr = "./build/tools/caffe test -model $model -weights $weights -iterations $iterations > $outputDir/error_$i 2>&1";
    print "$runStr\n";
    system($runStr);
  }
