#!/usr/bin/perl
# produce csv:
# layer1 config, layer2 config, .... ,layerN config, accuracy

use Data::Dumper;
use Scalar::Util qw(looks_like_number);

$caffeDir = "/localhome/juddpatr/caffe";


sub grepFile {
  my $grepStr = $_[0];
  my $file = $_[1];
  my @matches = `grep "$grepStr" $file | tail -n 1`;
#  die "Multiple matches for $grepStr in $file\n" if (scalar(@matches) > 1);
#  die "No matches for $grepStr in $file\n" if (scalar(@matches) < 1);
  if ( scalar(@matches) == 0 ){
    return "";
  }
  return $matches[0];
}

use Getopt::Std;
getopts('b');
$best = $opt_b;

my @runDirs = @ARGV;

if ( ! scalar(@runDirs) ) {
  print "usage: ./script.pl runDir1 [runDir2 ..]\n";
  exit;
}

$first = 1;

$startDir = `pwd`;


# net-find-optimal-param/param-config/layersN
foreach $dir (@runDirs) { # alexnet-find-optimal
  $dir =~ /([^-]+)-([^-]+)-(.*)/;
  $net = $1;
  $modelDir = "$caffeDir/models/$net";

  # read in datasizes
  open ($ds, "<$modelDir/$net.datasize") or die "$! $net.datasize\n";
  @datasizes = ();
  foreach (<$ds>){
    ($idx,$data,$weight) = (split /,/,$_);
    if (looks_like_number($idx)){
      push @datasizes, $data;
    }
  }
  close $ds;
  $baseline = `cat $modelDir/$net.baseline`;
  chomp($baseline);

  #calculate baseline bandwidth
  @totalBitsPerElement = (0) x @datasizes;
  $totalBits = 0;
  foreach $i (0..$#datasizes){
    $totalBitsPerElement[$i] = 32;
    $totalBits += $datasizes[$i] * $totalBitsPerElement[$i];
  }

  $baselineBandwidth = $totalBits * 8 / (1024 * 1024);
#print "baseline bandwidth = $baselineBandwidth";

  chdir ($startDir);
  chdir ($dir);

  foreach $iDir ( glob ("*") )  { # max_data_mag-1,2,3,4,5,5
    next if ( not -d $iDir or $iDir =~ m/^\./ );
    my @files = glob ("$iDir/*/stderr");
    my @failed = ();

    # header
    if ($first){
      print "config, bandwidth, accuracy\n";
      print "" . (join '-', @totalBitsPerElement) . ", 1, $baseline\n";
      $first = 0;
    }

    $run = $iDir;
    $run =~ m/[^-]+-([\d,]+)/;
    $config = $1;
    @params = split /,/, $config;

    foreach (@files){
      $file = $_;
      /\/layers(\d+)\//;
      $layerNum = $1;
      @tempParams = @params;
      if ($tempParams[$layerNum-1] > 0){
        $tempParams[$layerNum-1]--;
      }

      my $grepStr = "caffe\.cpp.*accuracy";
      $_ = grepFile($grepStr, $file);
      my $accuracy = -1;
      if ( /^$/ ){ # empty string
        push @failed, $file;
      } else {
        /accuracy = (\d+\.\d+)/;
        $accuracy = $1;

        $relAcc = $accuracy/$baseline;

        

#die "datasizes dont match the # of parameters\n" if ($#datasizes != $#tempParams);

        @totalBitsPerElement = (0) x @datasizes;
        @bitStr = ();
        $totalBits = 0;
        $half = scalar(@tempParams)/2;
        foreach $i (0..$#datasizes){
          $integer = $tempParams[$i] + 1;
          $fraction = $tempParams[$i+$half];
          $bitStr[$i] = "$integer.$fraction";
          $totalBitsPerElement[$i] = ($tempParams[$i] + 1 + $tempParams[$i+$half]);
          $totalBits += $datasizes[$i] * $totalBitsPerElement[$i];
        }

        $bandwidth = $totalBits * 8 / (1024 * 1024);
        $relBW = $bandwidth/$baselineBandwidth;
#printf "%s, %.4f, %.4f\n", (join '-', @totalBitsPerElement), $relBW, $relAcc;
        printf "%s, %.4f, %.4f\n", (join '-', @bitStr), $relBW, $relAcc;

      }
    }
  }
  

}
if (scalar @failed){
  print "Failed runs: \n\t" . (join "\n\t", @failed) . "\n";
}
