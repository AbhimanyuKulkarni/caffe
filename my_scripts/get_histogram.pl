#!/usr/bin/perl

use Statistics::Histogram;

my $logFile = shift(@ARGV);

open (my $logFH, "<$logFile") or die $!;

my @data = <$logFH>;
chomp @data;

print get_histogram(\@data);

