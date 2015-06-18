#!/usr/bin/perl
#

use Scalar::Util qw(looks_like_number);

@tolerances = qw(0.01 0.02 0.05 0.1);

  @arr = ();
  foreach(<>){
    s/,\s*$//;
    ($bn,$an) = (split /[,%]/)[-2..-1];
    next unless (looks_like_number($bn));
    next unless (looks_like_number($an));
    push (@arr,$_);
  }

  foreach $tol (@tolerances) {
    $best = 1;
    $bestLine = "";
    foreach(@arr){
      chomp;
      ($bn,$an) = (split /[,%]/)[-2..-1];
      if ( (1-$tol) < $an) {
        if ( $bn <= $best ) {
          $best = $bn;
          $bestLine = $_;
        }
      }
    }
    ($bn,$an) = (split /[,%]/,$bestLine)[-2..-1];
      print "$tol, $bestLine\n";
#    print "$tol% $bestLine\n";
  }

  
