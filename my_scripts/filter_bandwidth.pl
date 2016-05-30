#!/usr/bin/perl
#
# reads stdin
# input format:
# ... , bandwidth, accuracy
# find pareto frontier with low bandwidth, high accuracy
# print the input lines in the pareto frontier

use Scalar::Util qw(looks_like_number);

@best = ();
foreach(<>){
  s/,\s*$//;
  ($bn,$an) = (split /\s*[,%]\s*/)[-2..-1];
  next unless (looks_like_number($bn));
  next unless (looks_like_number($an));
  $new = $_;

#  print "considering $_\n";

  $foundBetter=0;
  foreach $b (0..$#best) {

    # get a BEST point from the pareto frontier
    ($bb,$ab) = (split /\s*[,%]\s*/,$best[$b])[-2..-1];

    $equal = 0;
    if ($bn == $bb and $an == $ab) {
      $equal = 1;
    }

    # if N has higher bandwidth, lower accuracy then B is strictly better, don't include N
    if ($bn >= $bb and $an <= $ab){
      $foundBetter=1;
#      print "FOUND BETTER\n";
    }

    # if N has lower bandwidth, higher accuracy than B, delete B
    if (not $equal and $bn <= $bb and $an >= $ab){
#      print "deleting    $best[$b]\n";
      delete $best[$b];
      $b--;
    }
  }

  unless ($foundBetter){
    push @best, $new;
#    print "adding      $new\n";
  }
}

@best = sort @best;
foreach (@best) {
  print "$_" unless m/^\s*$/;
}
