#!/usr/bin/perl
#
# This script updates a caffe model prototxt file with a parameter and value for a given layer
# the parameter will be updated in place or added to the end of the layer if not found
#
# Be advised that this script make some assumptions about formatting
#   - only 1 open/close bracket per line

use Getopt::Std;

getopts('');

if ($#ARGV < 3) {
  print "usage: set_layer_param.pl <file> <layer> <parameter> <value>\n";
  print "options:\n";
  exit;
}

my $filename = $ARGV[0];
my $layer = $ARGV[1];
my $param = $ARGV[2];
my $value = $ARGV[3];

my @layers = split /,/, $layer;

if ($partial){
  #print "searching for layers named \".*$layer.*\"\n";
}

# read file to array
open(my $fh, "<$filename") or die $!;
my @lines = <$fh>;
close $fh;

# open for writing
open(my $fh, ">$filename") or die $!;

my $depth=0;
my $inLayer=0;
my $done=0;
foreach my $line (@lines){
  if ($line =~ /{/) {
    $depth++;
  }
  if ($line =~ /}/) {
    $depth--;
  }
  
  # find the right layer by name
  #
  if ($line =~ m/name:\s*\"([^\"]+)\"/) {
    $name = $1;
    #print "name = $name\n";
    if (grep /^$name$/, @layers 
        or ($layers[0] eq '*' and $depth==1)
        ) {
      #print "match\n";
      $inLayer = 1;
      $done = 0;
    }
  }
  
  # now look for a good place to set the parameter
  if ($inLayer){

    # found param, update value
    if ($line =~ /^(\s*$param:)/){
      $line = "$1 $value\n";
      $done = 1;
    }

    # reached the end of the layer
    # add parameter here
    if ($depth == 0 and $done == 0) {
      print $fh "  $param: $value\n";
      $done = 1;
    }

  }
  
  # reached the end of the topmost block, no longer in layer definition
  if ($depth == 0){
    $inLayer = 0;
  }

  print $fh "$line";
}

close $fh;

if ($done == 0){
  die "Error: could not find layer $layer\n";
}

