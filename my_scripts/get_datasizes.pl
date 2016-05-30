#!/usr/bin/perl

use Getopt::Std;

$expected="datasize (name,datain,dataout,data,weight):loss3/top-5,50050, 1,50051,0";

sub grepFile {
  my $grepStr = $_[0];
  my $file = $_[1];
  my @matches = `grep -a "$grepStr" $file`;
  return @matches;
}

getopts('v');
$verb = $opt_v;

my $caffeDir = "/localhome/juddpatr/caffe";

# for each batch directory
while (scalar(@ARGV)) {
  my $runDir = shift (@ARGV);

  if ( ! -d "$runDir") {
    die "Error $runDir is not a directory\n";
  }

  my @files = glob ("$runDir/*/stderr");
  my @stats = ();
  my @failed = ();

  $runDir =~ /([^-]*)-/;
  $net = $1;
  open ($layerFile, "<$caffeDir/models/$net/$net.layers") or die "$! $net.layers";
  @layers = <$layerFile>;
  chomp(@layers);

  %datasize = ();
  foreach $file (@files){
    print "$file\n";
    my $grepStr = "error.cpp.*datasize";
    @matches = grepFile($grepStr, $file);

# get stats for each sublayer
# for runs with multiple iterations this will only use the last one
    if ($verb) {
      print "layer, data_in, data_out, data, weights\n";
    }else{
      print "layer, data, weights\n";
    }
    foreach (@matches) {
#      print "$_\n";
      /datasize.*:(.*)/;
      ($name,$di,$do,$d,$w) = split /\s*,\s*/, $1;
      die "Error: wrong format:\n\t$_\n\texpected: $expected\n" if $d eq "" or $w eq "";
#print "$1 : $d, $w\n";
      if ($verb) {
        $datasize{$name} = "$di,$do,$d,$w";
      }else{
        $datasize{$name} = "$d,$w";
      }
    }

    foreach $layer (0 .. $#layers){
      @sublayers = split /,/, $layers[$layer];
      if ($verb) {
        @sum = (0,0,0,0);
      }else{
        @sum = (0,0);
      }
      foreach (@sublayers) {
        $counts = $datasize{$_};
#print "$_ = $counts\n";
        foreach $i (0..$#sum){
          $sum[$i] += (split /,/,$counts)[$i];
        }
      }
      print "$layer, " . join (',',@sum) . "\n";
    }
  }
}

