#!/usr/bin/perl

sub grepFile {
  my $grepStr = $_[0];
  my $file = $_[1];
  my @matches = `grep -a "$grepStr" $file | tail -n 1`;
#  die "Multiple matches for $grepStr in $file\n" if (scalar(@matches) > 1);
#  die "No matches for $grepStr in $file\n" if (scalar(@matches) < 1);
  if ( scalar(@matches) == 0 ){
    return "";
  }
  return $matches[0];
}

while (scalar(@ARGV)){
  my $runDir = shift(@ARGV);
  print "$runDir\n";

  if ( ! -d "$runDir") {
    die "Error $runDir is not a directory\n";
  }

  my @files = glob ("$runDir/*/stderr");
  my @stats = ();
  my @failed = ();

  print "filename, param, accuracy\n";
  foreach $file (@files){
    my $grepStr = "caffe\.cpp.*accuracy";
    $_ = grepFile($grepStr, $file);

    if ( /Binary file/ ) {
      print "$file filetype is binary\n";
    }
    my $accuracy = -1;
    if ( /^$/ ){ # empty string
      push @failed, $file;
    } else {
      /accuracy = (\d+\.\d+)/;
      $accuracy = $1;
    }

    $run = (split(/\//,$file))[-2];
    $_ = $run;
    /([\w\d]+)-([\w\d_]+)-([0-9\.]+)/;
    my $layer = $1;
    my $param = $3;

    push @array, sprintf("%-30s , %5s , %.4f , %.4f", $file ,$layer ,$param ,$accuracy);
    print "$array[-1]\n";
  }


  if (scalar @failed){
    print "Failed runs: \n\t" . (join "\n\t", @failed) . "\n";
  }
}
