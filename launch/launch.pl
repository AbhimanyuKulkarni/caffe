#!/usr/bin/perl
#

use POSIX qw(strftime);

my $caffeDir = "/localhome/juddpatr/caffe/";

my $resultDir = "/aenao-99/juddpatr/caffe/results";
my $batchTitle = "test-imagenet";
my $model = "";
my $weights = "$caffeDir/examples/cifar10/cifar10_quick_iter_4000.caffemodel";

my $timestamp = strftime "%F-%H-%M", localtime;
my $batchName = "$batchTitle-$timestamp";

open($fh, ">job.submit") or die "could not open job.submit for writing\n";
print $fh <<'END_MSG';
Universe = vanilla
Getenv = True
Requirements = ( Activity == "Idle") && ( Arch == "X86_64" ) && regexp( ".*fc16.*", TARGET.CheckpointPlatform )
Executable = run.sh
Output = stdout
Error = stderr
Log = condor.log
Rank = (TARGET.Memory*1000 + Target.Mips) + ((TARGET.Activity =?= "Idle") * 100000000) - ((TARGET.Activity =?= "Retiring" ) * 100000000 )
Notification = error
Copy_To_Spool = False
Should_Transfer_Files = yes
When_To_Transfer_Output = ON_EXIT
END_MSG


my $batchDir = $resultDir . "/$batchName";
#die "$batchDir exists\n" if ( -d $batchDir );
system ("rm -rf $batchDir");
mkdir ($batchDir, 0755);

# make skeleton dir
mkdir ("$batchDir/.skel",0755);
system ("cp $caffeDir/build/tools/caffe.bin $batchDir/.skel/.") and die $!;
system ("cp $caffeDir/.build_release/lib/libcaffe.so $batchDir/.skel/.") and die $!;
system ("cp $weights $batchDir/.skel/weights.caffemodel") and die $!;

my @fileList = (
    "$caffeDir/launch/run.sh"
    , "model.prototxt"
);

my $fileListStr = join(',', @fileList);
print $fh "Transfer_Input_Files = ../.skel, $fileListStr\n";
print $fh "Transfer_Output_Files = run.sh\n";

foreach my $i (1, 10, 100, 1000, 10000){
  my $runDir = $batchDir . "/run$i";
  if (-d $runDir){
    print "$runDir exists, clobber?";
    my $in = <>;
    exit if ( $in !~ /^\s*[yY]\s*$/ );
  }
  mkdir $runDir, 0755;
  system ("cp $caffeDir/examples/cifar10/cifar10_quick_train_test.prototxt $runDir/model.prototxt") and die $!;
  print $fh "InitialDir =  $runDir\n";
#  print $fh "transfer_output_files = /aenao-99/juddpatr/caffe/results/run$i\n";
  print $fh "Args = $i\n";
  print $fh "Queue\n\n";
}

my $retcode = system("condor_submit job.submit");
if ($retcode != 0) {
  print "condor_submit returned $retcode\n";
}
system("cp job.submit $batchDir/.skel/.") and die $!;

