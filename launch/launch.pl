#!/usr/bin/perl
# launch script, run ./launch.pl -h for help

use Getopt::Std;
use Scalar::Util qw(looks_like_number);

getopts('rtnh');
my $rerun = $opt_r;
my $test = $opt_t;
my $norun = $opt_n;
my $help = $opt_h;

if ($help) {
usage();
exit;
}

#my $caffeDir = "/localhome/juddpatr/caffe";
my $caffeDir = "/home/ubuntu/caffe";
my $modelDir = "$caffeDir/models";
#my $resultDir = "/aenao-99/juddpatr/caffe/results";
my $resultDir = "/home/ubuntu/caffe/results";

my $run = "$caffeDir/launch/run.sh";

#-------------------------------------------------------------------------------------------

$param = "max_data_mag";
$param = "data_precision";

$iters      = 100;
$skip       = 0;
$batchSize  = 0;    # 0 to use default

my @nets = ("lenet","convnet","alexnet","nin_imagenet","googlenet","vgg_cnn_s","vgg_cnn_m_2048","vgg_19layers");
my @nets = ("lenet","convnet","nin_imagenet","bvlc_googlenet","vgg_cnn_s","vgg_cnn_m_2048","vgg_19layers"); #MS: No alexnet, change googlenet to bvlc_
my @nets = ("lenet","convnet","nin_imagenet","bvlc_googlenet","vgg_cnn_s","vgg_cnn_m_2048","vgg_19layers"); #MS: No alexnet, change googlenet to bvlc_
for $net (@nets) {
  $netDir="$modelDir/$net";
  $netDir="$caffeDir/metafiles/$net";

  $useFixedPrec = 0;
#  $prec = (file2arr("$netDir/$net.prec"))[0]; #MS

  $useFixedMag = 0;
  $mag = 2;

  @mags = file2arr("$netDir/$net.pareto");
  @layers = file2arr("$netDir/$net.layers");

  $profile = (file2arr("$netDir/best-mag-error0.csv"))[0];
  die "Error: profile=$profile from $netDir/best-mag-error0.csv\n" if $profile eq "";
  print "profile: $profile\n";

  $batchTitle = "per-layer-bits-custom-mag-$iters";
  $batchTitle = "baseline-$iters";
  $batchTitle = "per-layer-$param-$iters";
  $batchTitle = "per-layer-bits-custom-mag-error0-$iters";

  @values = (1);

#---------------------------------------------------------------------------------------------

  $batchDir = $resultDir . "/$batchTitle/$net";

  $model = "$caffeDir/models/$net/train_val.prototxt";
  $weights = "$caffeDir/models/$net/weights.caffemodel";

  if ( not -d "$resultDir/$batchTitle" ){
    my_system("mkdir $resultDir/$batchTitle");
  }

  # create common batch-job.submit for all jobs in this batch
  write_batch_job($net);

  # create batch dir and skeleton dir
  if ($rerun) {
    die "$batchDir does not exists\n" unless ( -d $batchDir );
  } else {
    print "Preparing $batchDir\n";
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

  # create individual submit script
  my_system("cp batch-job.submit job.submit");
  open($fh, ">>job.submit") or die "could not open submit for append\n";

  $jobCount=0;
#foreach $layer (0..$#layers){
  foreach $layer ("_all"){
  #foreach $layer ("_custom"){

    $layer_dir = "layer";
    if (looks_like_number($layer)){
      $layer_dir .= sprintf("%02d",$layer);
    } else {
      $layer_dir .= $layer;
    }
    # strip '/'s from layer name when to create runDir
    $layer_dir =~ s/\//_/g;

    printf("%s-%s", $layer_dir, $param) if not $test and not $rerun;

    for $value (@values) {
      print " $value" if not $test and not $rerun;
  #   $jobName = sprintf("%s-%s-%02d",$layer_dir,$param,$value);

  #    for $dummy (0) {
  #     $jobName = sprintf("%s-%s-%s",$layer_dir,$param,$mag);
    #   $jobName = "test";



# testing
#  $mag_prec = 0;
#  foreach $mag (0..16) {
#    foreach $prec (0..16) {
      #$jobName = sprintf "mag-$profile-layer%02d-bits-%02d", $layer, $value;
      $jobName = sprintf "layer%02d-%s-%02d", $layer, $param, $value;
      if ($batchName =~ /baseline/){ $jobName = "baseline"; }

      my $runDir = $batchDir . "/" . $jobName;

      my $runDirExists = (-d $runDir);

      if ($rerun and -d $runDir) {
        # did this run succeed?
        $ret = system("grep \"final accuracy\" $runDir/stderr >/dev/null");
        if ($ret == 0){
          next; # if so, skip
        } else {
          print "Rerunning $runDir\n";
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
            # set all layers to uniform param
            $l = join ',', @layers;
            #my_system("perl set_layer_param.pl $runDir/model.prototxt \"$l\" $param $value");
            if ($mag_prec) {
              my_system("perl set_layer_param.pl $runDir/model.prototxt \"$l\" max_data_mag $mag");
              my_system("perl set_layer_param.pl $runDir/model.prototxt \"$l\" data_precision $prec");
            } else {
              my_system("perl set_layer_param.pl $runDir/model.prototxt \"$l\" $param $value");
            }
            if ($batchSize){
              my_system("perl set_layer_param.pl $runDir/model.prototxt \"data\" batch_size 1");
            }
            if ($useFixedPrec){
              my_system("perl set_layer_param.pl $runDir/model.prototxt \"$l\" data_precision $prec");
            }
            if ($useFixedMag){
              my_system("perl set_layer_param.pl $runDir/model.prototxt \"$l\" max_data_mag $mag");
            }
          } elsif ($layer =~ m/custom/ or $batchTitle =~ /custom/) {
            @val = split /[,-]/, $profile;
            foreach $i (0..$#layers){
              #($mag,$prec) = split /\./, $val[$i];
              $mag = $val[$i];
              $mag--; # drop sign bit
              #  $wprec = $wprec_h{$net};
              #  $wprec--; # drop sign bit
                if ($batchTitle =~ /per-layer-bits-custom-mag/){
                  my $bits = $value;
                  my $msb = $mag;
                  my $lsb = $msb - $bits + 1; 
                  my $prec = -1*$lsb;
                  my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$i] max_data_mag $msb");
                  if ($i == $layer){
                    my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$i] data_precision $prec");
                  }
                } else {
                  my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$i] max_data_mag $mag");
                  my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$i] data_precision $prec");
                }
#            my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$i] weight_precision $wprec");
            }
          } else {
            # individual layer 
            my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$layer] $param $value");
            if ($useFixedPrec){
              my_system("perl set_layer_param.pl $runDir/model.prototxt $layers[$layer] data_precision $prec");
            }
          }
        }
      }

      $args = "model.prototxt $weights $iters";
      my_system("echo \"$args\" > $runDir/args"); # so we can run locally: ./run.sh `cat args`

        # append job details to submit script
        print $fh "InitialDir =  $runDir\n";
        print $fh "Args = $args\n";
        print $fh "Queue\n\n";
        $jobCount++;
        last if ($batchTitle =~ m/baseline/);

      } # foreach value
#    } 
    print "\n" if not $test and not $rerun;
    last if ($batchTitle =~ m/baseline/);
  } # foreach layer

  close $fh;
  if ($jobCount > 0){
    my_system("condor_submit job.submit") unless $norun;
  }

}# foreach net



#----------------------------------------------------------------------------------------------




# system call wrapper for testing
sub my_system {
  my $cmd = shift(@_);
  if ($test) {
    print "cmd: $cmd\n";
  } else {
    system ("$cmd") and die "failed to run $cmd";
  }
}

sub write_batch_job {
  my $net = shift(@_);
  open($fh, ">batch-job.submit") or die "could not open batch-job.submit for writing\n";
  print $fh <<'END_MSG';
Universe = vanilla
Getenv = True
Requirements = (Activity == "Idle") && ( Arch == "X86_64" ) && regexp( ".*fc16.*", TARGET.CheckpointPlatform )
Executable = run.sh
Output = stdout
Error = stderr
Log = condor.log
Rank = (TARGET.Memory*1000 + Target.Mips) + ((TARGET.Activity =?= "Idle") * 100000000) - ((TARGET.Activity =?= "Retiring" ) * 100000000 )
Notification = error
Copy_To_Spool = False
Should_Transfer_Files = no
#When_To_Transfer_Output = ON_EXIT
END_MSG

  if ("$net" eq "googlenet") {
    print $fh "+AccountingGroup = \"long_jobs.juddpatr\"\n";
    print $fh "request_memory = 2048\n";
  } elsif ($net =~ "vgg_19") {
    print $fh "request_memory = 6144\n";
  } elsif ($net =~ "vgg") {
    print $fh "request_memory = 5120\n";
  } else {
    print $fh "request_memory = 1024\n";
  }
  close $fh;
}

sub usage () {
print <<END_HELP;
launches caffe jobs on condor
options:

  -t
      test: prints commands instead of executing them

  -n
      norun: does everything but launch the job

  -r
      rerun: rerun any failed runs
END_HELP
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
