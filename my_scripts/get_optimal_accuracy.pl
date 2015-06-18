#!/usr/bin/perl
# produce csv:
# layer1 config, layer2 config, .... ,layerN config, accuracy

use Data::Dumper;
use Scalar::Util qw(looks_like_number);

$launchDir = "/localhome/juddpatr/caffe_error/launch";


sub grepFile {
  my $grepStr = $_[0];
  my $file = $_[1];
  my @matches = `grep "$grepStr" $file`;
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
  $dir =~ /^([^-]+)-/;
  $net = $1;

  # read in datasizes
  open ($ds, "<$launchDir/$net.datasize") or die "$! $net.datasize\n";
  @datasizes = ();
  foreach (<$ds>){
    ($idx,$data,$weight) = (split /,/,$_);
    if (looks_like_number($idx)){
      push @datasizes, $data;
    }
  }
  close $ds;
  $prec = `cat $launchDir/$net.prec`;
  chomp($prec);
  $baseline = `cat $launchDir/$net.baseline`;
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
  print "$net: $baseline\n";

  foreach $iDir ( glob ("*") )  { # max_data_mag-1,2,3,4,5,5
    next if ( not -d $iDir or $iDir =~ m/^\./ );
    my @files = glob ("$iDir/*/stderr");
    my @failed = ();

    # header
    if ($first){
      foreach $i (1..$#datasizes+1){
        print " stage$i,";
      }
      print " bandwidth, accuracy\n";
      print "" . (join ', ', @totalBitsPerElement) . ", 1, $baseline\n";
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

      my $grepStr = "caffe\.cpp:18[78].*accuracy";
      $_ = grepFile($grepStr, $file);
      my $accuracy = -1;
      if ( /^$/ ){ # empty string
        push @failed, $file;
      } else {
        /accuracy = (\d+\.\d+)/;
        $accuracy = $1;

        $relAcc = $accuracy/$baseline;

        die "datasizes dont match the # of parameters\n" if ($#datasizes != $#tempParams);

        @totalBitsPerElement = (0) x @datasizes;
        @bitStr = (0) x @datasizes;
        $totalBits = 0;
        foreach $i (0..$#datasizes){
          $integer = $tempParams[$i] + 1;
          $totalBitsPerElement[$i] = ($tempParams[$i] + 1 + $prec);
          $bitStr[$i] = "$integer.$prec";
          $totalBits += $datasizes[$i] * $totalBitsPerElement[$i];
        }

        $bandwidth = $totalBits * 8 / (1024 * 1024);
        $relBW = $bandwidth/$baselineBandwidth;
        unless ($relAcc > 1.1) { # skip lines that are obviously wrong
#          printf "%s, %.4f, %.4f\n", (join '-', @bitStr), $relBW, $relAcc;
            printf "%s, %.4f, %.4f\n", (join '-', @totalBitsPerElement), $relBW, $relAcc;
#          printf "%s %% %.4f %% %.4f\n", (join '-', @totalBitsPerElement), $relBW, $relAcc;
        }

      }
    }
  }
  

}
if (scalar @failed){
  print "Failed runs: \n\t" . (join "\n\t", @failed) . "\n";
}
