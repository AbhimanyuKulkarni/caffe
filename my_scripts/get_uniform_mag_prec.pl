#!/usr/bin/perl

$caffeDir = "/localhome/juddpatr/caffe";

if ( scalar(@ARGV) < 1 ) {
  print "usage: get_accuracy.pl <run dir>\n";
  exit 1;
}

while (scalar(@ARGV)){
  my $runDir = shift(@ARGV);
  print "$runDir\n";

  if ( ! -d "$runDir") {
    die "Error $runDir is not a directory\n";
  }
  $runDir =~ /^([^-]+)-/;
  $net = $1;
  $launchDir = "$caffeDir/models/$net";
  $baseline = `cat $launchDir/$net.baseline`;
  $numStages = `wc -l $launchDir/$net.layers`;

  my @files = glob ("$runDir/*");
  my @stats = ();
  my @failed = ();
  my %hash;
  my $paramName;

  foreach $file (@files){
    my $accuracy = -1;
    my $grepStr = "caffe\.cpp:188.*accuracy";
    $_ = grepFile($grepStr, "$file/stderr");
    if ( /^$/ ){ # empty string
      push @failed, $file;
    } else {
      /accuracy = (\d+\.\d+)/;
      $accuracy = $1;
    }

    $_ = $file;
    s/.*\/(.*)\/?$/$1/; # strip out path

    $layer, $paramName, $param;
    # convention: layer-parameterName-parameter value
    if (m/^(prec)?(\d+)-([\w\d_]+)-([0-9\.]+)$/) {
    $layer = $2;
    $paramName = $3;
    $param = $4 + 0.0;

    } elsif (m/^(layer_all)-([0-9\.]+)-([0-9\.]+)$/) {
    $layer = $2;
    $paramName = $1;
    $param = $3 + 0.0;
    } else {
      die "format not recognised: $_\n";
    }

    $relAcc = $accuracy/$baseline;

    $hash{$layer}{$param} = $relAcc;

    push @array, sprintf("%-30s , %5s , %.4f , %.4f", $file ,$layer ,$param ,$relAcc);
#    print $array[-1] . "\n"
  }

  my %layer_hash = ();
  foreach (@array) {
    my $l = (split /\s*,\s*/)[1];
    #print "layer = '$l'\n";
    $layer_hash{$l}++;
  }

  my %param_hash = ();
  foreach (@array) {
    my $p = (split /\s*,\s*/)[2];
    $param_hash{$p+0.0}++;
  }

# sanity check
  unless ($runDir =~ m/uniform/) {
    foreach (keys %layer_hash) {
      my $num_params = scalar(keys %param_hash);
      if ($layer_hash{$_} != $num_params) {
        print "$_ (" . $layer_hash{$_} . ") != number of params ($num_params)\n";
        #print "layers: " . join(',',(keys %layer_hash)) . "\n";
      }
    }
    foreach (keys %param_hash) {
      my $num_layers = scalar(keys %layer_hash);
      if ($param_hash{$_ + 0.0} != $num_layers) {
        print "$_ (" . $param_hash{$_} . ") != number of layers ($num_layers)\n";
      }
    }
  }

# print csv:
#       , param1 , param2 ,
# layer1, data1-1, data1-2,
# layer2, data2-1, data2-2,
  my @params = sort {$a<=>$b}(keys %param_hash);
  my @layers = sort (keys %layer_hash);

  print "bandwidth, accuracy\n";
  foreach my $param (@params) {
    foreach my $layer (@layers) {
      $layer =~ /(\d+)/;
      $otherParam = $1;

      $integer = $param + 1;
      $fraction = sprintf "%d", $otherParam;
      $bits = $param + $otherParam + 1;
      $relBW = $bits/32;
#printf("%s %f,%f\n", join(',',(("$param+$otherParam+1")x$numStages)), $relBW, $hash{$layer}{$param});
#printf("%s,%f,%f\n", join('-',(("$bits")x$numStages)), $relBW, $hash{$layer}{$param});
      printf("%s,%f,%f\n", join('-',(("$integer.$fraction")x$numStages)), $relBW, $hash{$layer}{$param});
    }
  }



  if (scalar @failed){
    print "Failed runs: \n\t" . (join "\n\t", @failed) . "\n";
  }


  sub grepFile {
    my $grepStr = $_[0];
    my $file = $_[1];
    my @matches = `grep -a "$grepStr" $file`;
#  die "Multiple matches for $grepStr in $file\n" if (scalar(@matches) > 1);
#  die "No matches for $grepStr in $file\n" if (scalar(@matches) < 1);
    if ( scalar(@matches) == 0 ){
      return "";
    }
    return $matches[0];
  }
}

