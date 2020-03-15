#!/bin/bash

nj=4
cmd=run.pl
source_name="fbank"
global_cmvn=true
cmvn_file=
compress=false
cmvn_option="--norm-means=true --norm-vars=false"
add_delta=false
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<apply-cmvn-dir>]  [<log-dir>]";
   echo "e.g.: $0 data/train  data/train_cmvn  data/train_cmvn/log"
   echo "Note: <log-dir> defaults to <apply-cmvn-dir>/log, and <apply-cmvn-dir> defaults to <data-dir>_cmvn"
   echo "Options: "
   echo "  --global-cmvn (true|false)                       # whether do global cmvn, defulat ture"
   echo "  --add-delta (true|false)                       # whether add delta, defulat false"
   echo "  --cmvn-file                                      # gloval cmvn file"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

sourcedir=$1
if [ $# -ge 2 ]; then
    targetdir=$2
else
    targetdir=${1}_cmvn
fi
if [ $# -ge 3 ]; then
    logdir=$3
else
    logdir=${1}_cmvn/log
fi

mkdir -p $targetdir
mkdir -p $logdir

cp $sourcedir/{text,spk2utt,utt2spk,wav.scp} $targetdir/
utils/validate_data_dir.sh --no-text --no-feats $targetdir || exit 1;
rootdir=`pwd`
targetdir="${rootdir}/${targetdir}"
if [ -f $sourcedir/feats.scp ]; then
    mkdir -p $targetdir/data
    split_scps=""
    for n in $(seq $nj); do
        split_scps="$split_scps $logdir/feats.$n.scp"
    done
    utils/split_scp.pl $sourcedir/feats.scp $split_scps || exit 1;
    if $add_delta; then
        if $global_cmvn; then
            if [ -f $cmvn_file ]; then
                $cmd JOB=1:$nj $logdir/${source_name}_cmvn.JOB.log \
                apply-cmvn $cmvn_option $cmvn_file scp:$logdir/feats.JOB.scp ark:- \| \
                add-deltas --delta-order=2 ark:- ark:- \| \
                copy-feats --compress=$compress ark:- ark,scp:$targetdir/data/${source_name}_cmvn.JOB.ark,$targetdir/data/${source_name}_cmvn.JOB.scp || exit 1;
            else
                echo "Error: no cmvn file: $cmvn_file exist";
                exit 1;
            fi
        else
            $cmd JOB=1:$nj $logdir/${source_name}_cmvn.JOB.log \
            apply-cmvn --utt2spk=ark:$sourcedir/utt2spk scp:$sourcedir/cmvn.scp scp:$logdir/feats.JOB.scp ark:- \| \
            add-deltas --delta-order=2 ark:- ark:- \| \
            copy-feats --compress=$compress ark:- ark,scp:$targetdir/data/${source_name}_cmvn.JOB.ark,$targetdir/data/${source_name}_cmvn.JOB.scp || exit 1;
        fi
    else
        if $global_cmvn; then
            if [ -f $cmvn_file ]; then
                $cmd JOB=1:$nj $logdir/${source_name}_cmvn.JOB.log \
                apply-cmvn $cmvn_option $cmvn_file scp:$logdir/feats.JOB.scp ark:- \| \
                copy-feats --compress=$compress ark:- ark,scp:$targetdir/data/${source_name}_cmvn.JOB.ark,$targetdir/data/${source_name}_cmvn.JOB.scp || exit 1;
            else
                echo "Error: no cmvn file: $cmvn_file exist";
                exit 1;
            fi
        else
            $cmd JOB=1:$nj $logdir/${source_name}_cmvn.JOB.log \
            apply-cmvn --utt2spk=ark:$sourcedir/utt2spk scp:$sourcedir/cmvn.scp scp:$logdir/feats.JOB.scp ark:- \| \
            copy-feats --compress=$compress ark:- ark,scp:$targetdir/data/${source_name}_cmvn.JOB.ark,$targetdir/data/${source_name}_cmvn.JOB.scp || exit 1;
        fi

        fi
else
    echo "Error: no feats.scp find in source data dir ";
    exit 1;
fi

rm $logdir/feats.*.scp

for n in $(seq $nj); do
  cat $targetdir/data/${source_name}_cmvn.$n.scp || exit 1;
done > $targetdir/feats.scp

