#!/usr/bin/perl
# parses custom profile caffe results and calculates the computation based on the profile
# e.g. alexnet/layer_custom-max_data_mag-10,8,8,8,8


$caffeDir = "/localhome/juddpatr/caffe";

$write          = 0 ;                       # write best profiles to :
$outputFilename = "profiles_power2.csv" ;
$outputFilename = "profiles.csv" ;

$threshold  =1;   # print best config > threshold
$showThresh =0.75; # only show accuracies above this threshold
$clampAcc   =0;   # convert accuracy > 1 to 1


# initialize best profile for each threshold to 16,16....
@thresholds = (0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1);
@thresholds = (1);

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
  #print "$runDir\n";

  if ( ! -d "$runDir") {
    die "Error $runDir is not a directory\n";
  }

  # rundir is net-other
  $runDir =~ /^([^-]+)-?/;
  $net = $1;
  if ($net eq "") {
    $runDir =~ /^(.*)\//;
    $net = $1;
  }
  $net =~ s/\///g;

#  print "net = $net\n";
  $baseline = `cat $caffeDir/models/$net/$net.baseline`;
  @compPerLayer = `cat $caffeDir/models/$net/comp.txt`;
  chomp(@compPerLayer);
  #print "baseline = $baseline\n";

  my @files = `find $runDir -mtime -1 -name "stderr"`;
  chomp(@files);
  my @files = glob ("$runDir/*/stderr");
  my @stats = ();
  my @failed = ();

  #print "profile, computation, accuracy\n";
  foreach $file (@files){
    $done = grepFile("caffe run complete", $file);
    my $grepStr = "caffe\.cpp.*accuracy";
    $_ = grepFile($grepStr, $file);

    if ( /Binary file/ ) {
      print "$file filetype is binary\n";
    }
    my $accuracy = -1;
    if ( /^$/ or not $done){ # empty string
      push @failed, $file;
    } else {
      /accuracy = (\d+\.\d+)/;
      $accuracy = $1;
    }

    $run = (split(/\//,$file))[-2];
    $file =~ s/,/./g;
#    print "run = $run\n";
    $_ = $run;
    /([\w\d]+)-([\w\d_]+)-([0-9\.,]+)/;
    my $p = $3;
#    print "profile = $p\n";
    @profile = split /,/,$p;
    
    my $comp=0;
    my $baseComp=0;
    die "Can't find profile in $run\n" if ($#profile < 1);
    for $i (0..$#profile){
      #$profile[$i]++;
      $bits = $profile[$i];
      $c = $compPerLayer[$i];
      $comp += $bits * $c;
      $baseComp += 16 * $c;
    }

    if ($baseline){
      $accuracy = $accuracy/$baseline;
    }
    
    die "baseComp=$baseComp\n" if $baseComp == 0;
    my $relComp = $comp/$baseComp;

    if ($clampAcc and $accuracy > 1){$accuracy = 1;}
    push @array, sprintf("%-50s , %.4f , %.4f", (join '-',@profile) , $relComp , $accuracy);
    print "$array[-1]\n" if $accuracy >= $showThresh;
  }
  #print "\n";
  
  %bestMap = undef;
  foreach (@thresholds){
    @profile = (16) x @profile;
    $bestMap{"$_"} = sprintf("%-50s , %.4f , %.4f", (join '-',@profile) , 1 , 1);
  }

  # find the best profiles
  foreach $line (@array){
    ($pf,$comp,$acc) = split /\s*,\s*/, $line;
    foreach (@thresholds){
      ($oldPf,$bestComp,$bestAcc) = split /\s*,\s*/, $bestMap{"$_"};
      if ($acc >= $_ and $comp < $bestComp){
        $bestMap{"$_"} = $line;
      }
    }
  }


  if (scalar @failed){
    #print "Failed runs: \n\t" . (join "\n\t", @failed) . "\n";
  }
  #print "\n";
}

# write best profiles to file
if ($threshold){
  $outfile="$caffeDir/models/$net/$outputFilename";
  print "writing best profiles to $outfile\n" if $write;
  printf "%-40s,%-6s,%-6s,%-6s\n", "profile:$net", "comp", "acc", "thresh";
  open ($fh, ">$outfile") or die "$! $outfile\n";
  foreach $thresh (@thresholds){
    ($pf,$comp,$acc) = split /\s*,\s*/, $bestMap{$thresh};
    if ($write){$pf =~ s/-/,/g;}
    print $fh "acc_ge_$thresh," . $pf . "\n" if $write;
    printf "%-40s,%.4f,%.4f,%.4f\n", $pf, $comp, $acc, $thresh;
  }
  close $fh;
}
