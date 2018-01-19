#!/usr/bin/perl

$caffeDir = "/localhome/juddpatr/caffe";

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

  # rundir is net-other
  $runDir =~ /^([^-]+)-?/;
  $net = $1;
  if ($net eq "") {
    $runDir =~ /^(.*)\//;
    $net = $1;
  }
  $net =~ s/\///g;

  print "net = $net\n";
  $baseline = 1;
  if ( -f "$caffeDir/models/$net/$net.baseline") {
    $baseline = `cat $caffeDir/models/$net/$net.baseline`;
    print "baseline = $baseline\n";
  }

  my @files = `find $runDir -mtime -1 -name "stderr"`;
  chomp(@files);
  my @files = glob ("$runDir/*/stderr");
  my @stats = ();
  my @failed = ();

  print "filename, param, accuracy\n";
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
    $_ = $run;
    /([\w\d]+)-([\w\d_]+)-([0-9\.]+)/;
    my $layer = $1;
    my $param = $3;
    
    if ($baseline){
      $accuracy = $accuracy/$baseline;
    }

    $file =~ s/,/-/g;
    push @array, sprintf("%-30s , %5s , %.4f , %.4f", $file ,$layer ,$param ,$accuracy);
    print "$array[-1]\n";
  }


  if (scalar @failed){
    print "Failed runs: \n\t" . (join "\n\t", @failed) . "\n";
  }
  print "\n";
}

sub file2arr(){
  my $filename=shift; 
  my $fh = myopen("<$filename");
  my @arr = <$fh>;
  chomp(@arr);
  return @arr;
}

sub myopen() {
  my $filename=shift; 
  open (my $fh, "$filename") or die "$! $filename";
  return $fh; 
}
