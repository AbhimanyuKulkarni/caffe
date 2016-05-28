#!/usr/bin/perl
#

<<<<<<< HEAD
use Getopt::Std;
use POSIX qw(strftime);
use Scalar::Util qw(looks_like_number);

sub my_system;

my $timestamp = strftime "%F-%H-%M", localtime;

getopts('rtn');
my $rerun = $opt_r;
my $test = $opt_t;
my $norun = $opt_n;

my $caffeDir = "/localhome/juddpatr/caffe";
my $resultDir = "/aenao-99/juddpatr/caffe/results";

my %weight_hash = (
      'convnet' => "$caffeDir/examples/cifar10/cifar10_quick_iter_4000.caffemodel"
    , 'alexnet' => "$caffeDir/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
    , 'googlenet' => "$caffeDir/models/bvlc_googlenet/bvlc_googlenet.caffemodel"
    , 'vgg_devils' => "$caffeDir/models/vgg_devils/VGG_CNN_S.caffemodel"
    , 'nin_imagenet' => "$caffeDir/models/nin_imagenet/nin_imagenet.caffemodel"
    , 'lenet' => "$caffeDir/examples/mnist/lenet_iter_10000.caffemodel"
    );
my %model_hash = (
      'convnet' => "$caffeDir/models/my5/convnet.prototxt"
    , 'alexnet' => "$caffeDir/models/my5/alexnet.prototxt"
    , 'googlenet' => "$caffeDir/models/my5/googlenet.prototxt"
    , 'vgg_devils' => "$caffeDir/models/vgg-devils/VGG_CNN_S_deploy.prototxt"
    , 'nin_imagenet' => "$caffeDir/models/my5/nin.prototxt"
    , 'lenet' => "$caffeDir/models/my5/lenet.prototxt"
    );

my $run = "$caffeDir/launch/run.sh";

#-------------------------------------------------------------------------------------------

$param = "weight_precision";
$param = "data_precision";
$param = "max_data_mag";
$param = "";

$net = "googlenet";
$net = "nin_imagenet";
$net = "convnet";
$net = "alexnet";
$net = "lenet";

$iters = 1;
$skip = 0;

#for $net ("lenet","convnet","alexnet","nin_imagenet","googlenet") {
for $dummy (""){

$prec = `cat $net.prec`;
chomp($prec);

open ($fh, "<$net.pareto") or die "$! $net.pareto\n";
@mags = (<$fh>);
chomp @mags;

$batchTitle = "inter-stage-$param-$iters";
$batchTitle = "per-stage-$param-$iters";
$batchTitle = "$net-incremental-optimal-$param-$iters";
$batchTitle = "$net-histo";
$batchTitle = "$net-stepdown-$param-$iters";
$batchTitle = "$net-incremental-uniform-$param-$iters";
$batchTitle = "$net-per-layer-$param-prec$prec-$iters";
$batchTitle = "$net-test-$iters";
$batchTitle = "$net-iters";
$batchTitle = "$net-baseline-$iters";
$batchTitle = "$net-uniform-$param-$iters";
$batchTitle = "$net-uniform-mag-prec-$iters";
$batchTitle = "$net-per-layer-$param-$iters";
$batchTitle = "$net-validate-uniform-$param-prec$prec-$iters";
$batchTitle = "$net-validate-custom-$param-prec$prec-$iters";
$batchTitle = "$net-validate-custom-mag-prec-set2";
$batchTitle = "$net-baseline-test-$iters";

@values = 0..15;

#---------------------------------------------------------------------------------------------

$batchName = "$batchTitle";
$batchDir = $resultDir . "/$batchName";

$model = $model_hash{$net};
$weights = $weight_hash{$net};

# create common batch-job.submit for all jobs in this batch
open($fh, ">batch-job.submit") or die "could not open batch-job.submit for writing\n";
print $fh <<'END_MSG';
Universe = vanilla
Getenv = True
Requirements = (Activity == "Idle") && ( Arch == "X86_64" ) && regexp( ".*fc16.*", TARGET.CheckpointPlatform ) && ( RemoteHost == "aenao-26.eecg.toronto.edu" )
=======
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
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e
Executable = run.sh
Output = stdout
Error = stderr
Log = condor.log
Rank = (TARGET.Memory*1000 + Target.Mips) + ((TARGET.Activity =?= "Idle") * 100000000) - ((TARGET.Activity =?= "Retiring" ) * 100000000 )
Notification = error
Copy_To_Spool = False
<<<<<<< HEAD
Should_Transfer_Files = no
#When_To_Transfer_Output = ON_EXIT
END_MSG

if ("$net" eq "googlenet") {
  print $fh "+AccountingGroup = \"long_jobs.juddpatr\"\n";
  print $fh "request_memory = 2048\n";
} else {
  print $fh "request_memory = 1024\n";
}
close $fh;


# create batch dir and skeleton dir
if ($rerun) {
  die "$batchDir does not exists\n" unless ( -d $batchDir );
} else {
  if (-d $batchDir and not $test){
    print "$batchDir exists, clobber (y/n)?";
    my $in = <>;
    exit if ( $in !~ /^\s*[yY]\s*$/ );
    my_system("rm -rf $batchDir");
  }
  my_system("mkdir $batchDir");

  # make skeleton dir
  my_system("mkdir $batchDir/.skel");
  my_system ("cp $caffeDir/build/tools/caffe.bin $batchDir/.skel/.");
  my_system ("cp $caffeDir/build/lib/libcaffe.so $batchDir/.skel/.");
}

my_system("cp batch-job.submit $batchDir/.skel/.") ;


open ($layerFile, "<$net.layers") or die "$! $net.layers";
@layers = <$layerFile>;
chomp(@layers);

my $first = 1;

print "Preparing $batchDir\n";

# create individual submit script
my_system("cp batch-job.submit job.submit");
open($fh, ">>job.submit") or die "could not open submit for append\n";

#foreach $layer (@layers){
#foreach $layer (0..$#layers){
#foreach $layer ("_all"){
foreach $layer ("_custom"){

#  sleep(60) if not $first;
#  $first = 0;

  # strip '/'s from layer name when to create runDir
  $layer_dir = "layer";
  if (looks_like_number($layer)){
    $layer_dir .= sprintf("%02d",$layer);
  } else {
    $layer_dir .= $layer;
  }
  $layer_dir =~ s/\//_/g;

  printf("%s-%s\n", $layer_dir, $param) if not $test and not $rerun;

  # precisions/magnitude
#  for $value (@values) {
#    print " $value" if not $test and not $rerun;
  for $mag (@mags) {
    $jobName = sprintf("%s-%s-%s",$layer_dir,$param,$mag);
    $jobName = "test";
    print "$jobName\n";

# testing
#  foreach $value (1) {
#    my $jobName = "run";
    my $runDir = $batchDir . "/" . $jobName;

    if ($rerun and -d $runDir) {
      # did this run succeed?
      if (system("grep \"final accuracy\" $runDir/stderr >/dev/null") == 0){
        next; # if so, skip
      } else {
        print "\nRerunning $runDir\n";
      }
    } else { 
      # setup runDir
      my_system("mkdir $runDir");

      # copy files to runDir
      my_system("cp $model $runDir/model.prototxt");
      my_system("cp $run $runDir/run.sh");
      my_system("ln -s $batchDir/.skel/. $runDir/.skel");


      # set parameters in model 
      if ("$param" ne "") {
        if ($layer =~ m/all/) {
          $l = join ',', @layers;
          my_system("perl set_layer_param.pl $runDir/model.prototxt \"$l\" $param $value");
          my_system("perl set_layer_param.pl $runDir/model.prototxt \"$l\" data_precision $prec");
        }  elsif ($layer =~ m/custom/) {
          @val = split /[,-]/,$mag;
          foreach $i (0..$#val){
            ($mag,$prec) = split /\./, $val[$i];
            $mag--;
            my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$i] max_data_mag $mag");
            my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$i] data_precision $prec");
          }
        } else {
          # individual layer 
          my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$layer] $param $value");
        }
      }
    }
    
    # always do with, run.sh will delete
    if ($net eq "convnet") {
      #copy leveldb files
#my_system("cp -r $caffeDir/examples/cifar10/cifar10_test_leveldb $runDir/.");
    }

    $args = "model.prototxt $weights $iters";
    my_system("echo \"$args\" > $runDir/args");

    # append job details to submit script
    print $fh "InitialDir =  $runDir\n";
    print $fh "Args = $args\n";
    print $fh "Queue\n\n";
    last if ($batchTitle =~ m/baseline/);

  } # foreach value
  print "\n" if not $test and not $rerun;
  last if ($batchTitle =~ m/baseline/);
} # foreach layer

  close $fh;
  my_system("condor_submit job.submit") unless $norun;

}# foreach net

#----------------------------------------------------------------------------------------------




# system call wrapper for testing
sub my_system {
  my $cmd = shift(@_);
  if ($test) {
    print "$cmd\n";
  } else {
    system ("$cmd") and die "failed to run $cmd";
  }
}
=======
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
>>>>>>> 7c8c68eca96cc3284f8dd2e7920b6e254063b65e

