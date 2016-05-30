#!/usr/bin/perl

$caffeDir = "/localhome/juddpatr/caffe";
$launchDir = "$caffeDir/launch";

$d = ","; # delimeter 
$print1d=0;
$print2d=1;
$printThresh=0;

if ( scalar(@ARGV) < 1 ) {
  print "usage: get_accuracy.pl <run dir> \n";
  exit 1;
}

#my $thresh = 1;
#if (not -d $ARGV[-1]){
#  $thresh = pop(@ARGV);
#}

while (scalar(@ARGV)){
  my $runDir = shift(@ARGV);
  print "$runDir\n" if $print1d;

  if ( ! -d "$runDir") {
    die "Error $runDir is not a directory\n";
  }
  $runDir =~ /^([^-]+)-/;
  $net = $1;
  if ($net eq "") {
    $runDir =~ /^(.*)\//;
    $net = $1;
  }
  if ($net eq "") {
    $net = $runDir;
  }

  $baseline = `cat $caffeDir/models/$net/$net.baseline`;
  if ($print1d){
    print "net = $net\n";
    print "baseline = $baseline\n";
  }

  my @files = glob ("$runDir/*");
  my @stats = ();
  my @failed = ();
  my @array = ();
  my %hash;
  my $paramName;

  # 1. Build hash{layer}{param}
  foreach $file (@files){
    my $accuracy = -1;
    my $grepStr = "caffe\.cpp.*accuracy";
    $_ = grepFile($grepStr, "$file/stderr");
    chomp;
    /(final accuracy = \d+\.\d+)/;
    $accStr=$1;
    print "$file $accStr\n" if $print1d;
    if ( length($_) == 0 ){ # empty string
      push @failed, $file;
    } else {
      /accuracy = (\d+\.\d+)/;
      $accuracy = $1;
    }

    $_ = $file;
    s/.*\/(.*)\/?$/$1/; # strip out path

    # convention: *-layer-parameterName-parameter value
    /([\w\d_]+)-([\w\d_]+)-([-0-9\.]+)$/;
    # convention: layer-parameterName-parameter value
    #/^([\w\d-_]+)-([\w\d_]+)-([-0-9\.]+)$/;
    # convention: mag-#-prec-#
    #/^mag-(\d+).*(prec)-(\d+)$/;
    # convention: mag-#-prec-#
    #/^mag(\d+)-(bits)(\d+)$/;
    my $layer = $1;
    $paramName = $2;
    my $param = $3 + 0.0;
#    print "layer='$layer' paramName='$paramName' param='$param'\n";

    $relAcc = $accuracy/$baseline;

    $hash{$layer}{$param} = $relAcc;

    $file =~ s/,/_/g; # sanitize csv
    push @array, sprintf("%-30s , %5s , %.4f , %.4f", $file ,$layer ,$param ,$relAcc);
#    print $array[-1] . "\n"
  }

#  foreach my $l (keys %hash){
#    foreach my $p ( keys %{$hash{$l}} ){
#      print "$l $p $hash{$l}{$p}\n";
#    }
#  }
  
  my %layer_hash = ();
  foreach (@array) {
    my $l = (split /\s*,\s*/)[1];
#    print "layer = '$l'\n";
    $layer_hash{$l}++;
  }

  my %param_hash = ();
  foreach (@array) {
    my $p = (split /\s*,\s*/)[2];
#    print "param = '$p'\n";
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
  my @params = sort by_number (keys %param_hash);
  my @layers = sort by_number (keys %layer_hash);

  print "\n";
  $pwd = `pwd`;
  chomp($pwd);
  print "$pwd/$runDir\n";
  printf "%-7s$d", $paramName;
  foreach (@layers){ printf "%-5s$d", $_; }
  print "\n";
  foreach my $param (@params) {
    printf("%-7s$d", $param);
    foreach my $l (0..$#layers) {
      my $layer = $layers[$l];
      my $acc = $hash{$layer}{$param};
      printf("%.3f$d ", $acc);
    }
    printf("\n");
  }
  printf("\n");

  # get bit profiles for a certain threshold
#  for my $thresh (0.9,0.99,1){
  if ($printThresh){
    my @best = ();
    for (my $thresh = 0.9; $thresh <= 1; $thresh += 0.01){
      @best = (16) x @layers;
      foreach my $param (@params) {
        foreach my $l (0..$#layers) {
          my $layer = $layers[$l];
          my $acc = $hash{$layer}{$param};
          if ($best[$l] == 16 and $acc >= $thresh){
            $best[$l] = $param;
          }
        }
      }
      print "acc_ge_$thresh," . (join ",",@best) . "\n";
    }

    print "base_plus_1";
    foreach (@best){
      printf ",%d", $_+1;
    }
    print "\n";
  }

  if (0 and scalar @failed){
    print "Failed runs: \n\t" . (join "\n\t", @failed) . "\n";
  }

}

###############################################################################

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

# used for sorting by the first number in the string
sub by_number {
  my ( $anum ) = $a =~ /(\d+)/;
  my ( $bnum ) = $b =~ /(\d+)/;
  ( $anum || 0 ) <=> ( $bnum || 0 );
}
