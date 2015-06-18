#!/usr/bin/perl

$me = `whoami`;
chomp($me);

# attributes will not be printed in the given order
@attrs = (
    "Iwd"
    ,"ClusterId"
    ,"ProcId"
  );
$attrStr = join ',',@attrs;
@resp = `condor_q -l -attributes "$attrStr" $me`;

while (1){
  $line = shift(@resp);
  last if ($line =~ /Submitter/) or not scalar(@resp);
}
    
%job_hash = ();
%batches = ();
@jobs = ();

$job = "";
foreach (@resp) {
  if (/^\s*$/) {
    push (@jobs, $job);
#   print "$job\n";
    @fields = split /;/,$job;
#    foreach $i (0..$#fields){print "$i: $fields[$i]\n";}
    $k = "$fields[2].$fields[3]";
#   print "k=$k\n";
    $job_hash{"$k"} = "$fields[1]";
    $job="";

    $batch = $fields[1];
    $batch =~ s/.*results//;
    $batch =~ s/\/[^\/]+$//;
    $batches{"$fields[2]"} = $batch;
    next;
  }
  /(\w+)\s*=\s*(.*)/;
  $key = $1;
  $val = $2;

  if ($key eq "Iwd"){
#print $key;
    $val =~ s/"//g;
#$val =~ s/.*results\///;
  }
  $job = $job . ";$val";
}

@queue = ();
@resp = `condor_q $me`;
chomp(@resp);

foreach (keys(%batches)){
  printf "%5s %s\n", $_, $batches{$_};
}
print "\n";

printf "%-9s %11s %-10s %1s  %7s %s\n" ,"ID", "START_TIME", "RUNTIME", "S", "PROGRESS", "DIR";
foreach (@resp){
  next unless m/\d+\.\d+\s+$me/;
  @cols = split /\s+/, $_;
  $longInfo = $job_hash{$cols[0]};
  $_ = `grep Batch $longInfo/stderr | tail -n 1`;
  /Batch\s+(\d+)/;
  $batch=$1;
  $longInfo =~ s/.*results\///;
  printf "%-9s %5s %5s %10s %s   %3d/100 %s\n" ,$cols[0], $cols[2], $cols[3], $cols[4], $cols[5], $batch, $longInfo;
}

