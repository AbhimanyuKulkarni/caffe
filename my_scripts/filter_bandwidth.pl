#!/usr/bin/perl
#

use Scalar::Util qw(looks_like_number);

  @best = ();
  foreach(<>){
    s/,\s*$//;
    ($bn,$an) = (split /[,%]/)[-2..-1];
    next unless (looks_like_number($bn));
    next unless (looks_like_number($an));
    $new = $_;

    $foundBetter=0;
    foreach $b (0..$#best) {
      ($bb,$ab) = (split /[,%]/,$best[$b])[-2..-1];

      if ($bn >= $bb and $an <= $ab){
        $foundBetter=1;
      }

      if ($bn <= $bb and $an >= $ab){
        delete $best[$b];
        $b--;
      }

    }

    unless ($foundBetter){
      push @best, $new;
    }
  }

  @best = sort @best;
  foreach (@best) {
    print "$_" unless m/^\s*$/;
  }
