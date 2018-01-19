#!/usr/bin/perl
# parses custom profile caffe results and calculates the computation based on the profile
# e.g. alexnet/layer_custom-max_data_mag-10,8,8,8,8


$caffeDir = "/localhome/juddpatr/caffe";

$write          = 0 ;                       # write best profiles to :
$outputFilename = "profiles_power2.csv" ;
$outputFilename = "profiles.csv" ;
$csv = 0;
$tex = 1;


$threshold  =1;   # print best config > threshold
$showThresh =0.0; # only show accuracies above this threshold
$clampAcc   =1;   # convert accuracy > 1 to 1


# initialize best profile for each threshold to 16,16....
@thresholds = (1);
@thresholds = (0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1);
@thresholds = (0.90,0.95,0.98,0.99,1);
@thresholds = (1,0.99,0.98,0.95,0.90);

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

sub isUniform {
  my @arr = @_;
  my $v = $arr[0];
  my $match = 1;
  foreach (@arr) {
    $match = 0 if ($_ != $v);
  }
  return $match;
}

my $uniform = 0;
my $mixed = 0;
if ("$ARGV[0]" eq "-u"){
  shift(@ARGV);
  $uniform = 1;
}
if ("$ARGV[0]" eq "-m"){
  shift(@ARGV);
  $mixed = 1;
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
      if (/accuracy = (\d+\.?\d*)/) {
        $accuracy = $1;
      } else {
        print "no accuracy in $file\n";
      }
    }

    $run = (split(/\//,$file))[-2];
    #$file =~ s/,/./g;
    #print "run = $run\n";
    $_ = $run;
    #/([\w\d]+)-([\w\d_]*)-([0-9\.,]+)/;
    /-([0-9\.,]+)-?/;
    my $p = $1;
    #print "profile = $p\n";
    @profile = split /,/,$p;
    
    next if ($uniform and not isUniform(@profile));
    next if ($mixed and  isUniform(@profile));

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
    my $pf = join '-', @profile;
    my $pf = $file;
    push @array, sprintf("%-50s\t%.4f\t%.4f" , $pf, $relComp , $accuracy);
    print "$array[-1]\n" if $accuracy >= $showThresh;
  }
  #print "\n";
  
  # initialize bestMap to the 16 bit config
  %bestMap = undef;
  foreach (@thresholds){
    @profile = (16) x @profile;
    my $pf = join ',',@profile;
    $bestMap{"$_"} = sprintf("%-50s\t%.4f\t%.4f", $pf , 1 , 1);
  }

  # find the best profiles
  foreach $line (@array){
    ($pf,$comp,$acc) = split /\s*\t\s*/, $line;
    foreach (@thresholds){
      ($oldPf,$bestComp,$bestAcc) = split /\s*\t\s*/, $bestMap{"$_"};
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
  printf "%-40s,%-6s,%-6s,%-6s\n", "profile:$net", "comp", "acc", "thresh" if $csv;
  printf "%s & %-40s & %s \n", "threshold", "profile", "compute" if $tex;
  open ($fh, ">$outfile") or die "$! $outfile\n";
  foreach $thresh (@thresholds){
    ($pf,$comp,$acc) = split /\s*\t\s*/, $bestMap{$thresh};
    
    $pf =~ /-((\d+,)+\d+)\/.*-((\d+,)+\d+)/;
    my $mlist = $1;
    my $blist = $3;
    my @marr = split /,/, $mlist;
    my @barr = split /,/, $blist;
    my @profileArr = ();
    for (my $i=0; $i < @barr; $i++){
      my $m = $marr[$i];
      my $b = $barr[$i];
      my $p = $m - $b;
      #push(@profileArr, "\$$b^{$p}\$");
      push(@profileArr, "\$$b,$p\$");
    }
    my $profileStr = join '-', @profileArr;

    if ($write){$pf =~ s/-/,/g;}
    print $fh "acc_ge_$thresh," . $pf . "\n" if $write;
    printf "%-40s,%.4f,%.4f,%.4f\n", $profileStr, $comp, $acc, $thresh if $csv;
    printf "%2d\\\% & %-40s & %.2f \\\\\n", 100 - ($thresh) * 100, $profileStr, $comp if $tex;
  }
  close $fh;
}
