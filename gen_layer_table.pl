#!/usr/bin/perl
#
#
#
print <<'END';
\begin{table*}[ht]
\centering
\begin{tabular}{|l|l|l|}
\hline
\textbf{Network} & \textbf{Source} & \textbf{Layer} & \textbf{Caffe Layers} \\
\hline
END

foreach (@ARGV){
    next if /my5/;
    next if /bvlc/;
    /\/([^\/]+)\.layers/;
    my $net = $1;
    $net =~ s/nin_imagenet/nin/;
    open (FILE, "$_") or die "$! $_\n";
    my $n=1;
    my $col0 = $net;
    print "\\hline\n";
    while(<FILE>){
        chomp;
        s/_/\\_/g;
        if (/\//) {
            my @layers = split /,/;
            foreach (@layers) {
                s/.*\///;
            }
            my $base = $_;
            $base =~ s/\/.*//;
            $_ = "$base\/{" . (join ' , ',@layers) . "}";
            $_ = "$base\/*";
        }
        print "$col0 & & Layer $n & $_ \\\\\n";
        print "\\hline\n";
        $n++;
        $col0="";
    }
}
print <<'END';
\end{tabular}
\caption{}
\label{table:layers}
\end{table*}
END
