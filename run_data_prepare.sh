#/bin/bash                                                                      
                                                                                
echo "$0 $@"                                                                    
                                                                                
[ -f cmd.sh ] && . ./cmd.sh                                                     
[ -f path.sh ] && . ./path.sh                                                   
                                                                                
export LC_ALL=C
                                                                                
stage=1
nj=80

if [ -z $1 ]; then
    source_data_dir=my_data/mobvoi_hotword_to_openslr/mobvoi_hotword_dataset
else
    source_data_dir=$1
fi
resource_file_dir=${source_data_dir}_resources
if [ $stage -le 0 ]; then
    for x in train dev test;
    do
		for subset in p n;
		do
    	    data_local=data/$subset/$x
    	    mkdir -p $data_local
    	    python prepare_kaldi_file.py $source_data_dir \
					${resource_file_dir}/${subset}_${x}.json \
					$data_local/wav.scp \
					$data_local/utt2spk \
					$data_local/text
    	done
    	
		data=data/$x
    	mkdir -p $data
    	for y in wav.scp utt2spk text;
    	do
    	        cat data/{p,n}/${x}/$y | sort > $data/$y
    	done
    	utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
	done
fi
src_dir=data
fbank_dir=fbank

if [ $stage -le 1 ]; then
    for x in train dev test;
    do
        mkdir -p $fbank_dir/$x
        utils/copy_data_dir.sh $src_dir/$x $fbank_dir/$x
        steps/make_fbank.sh --cmd "$train_cmd" --nj $nj \
                            --fbank-config conf/fbank.conf $fbank_dir/$x \
                            $fbank_dir/$x/log $fbank_dir/$x/data
    done
fi

if [ $stage -le 2 ]; then
    compute-cmvn-stats scp:$fbank_dir/train/feats.scp $fbank_dir/train.ark
    for x in train dev test;
    do
        apply_cmvn.sh --cmd "$train_cmd" --nj $nj --source-name "fbank" \
                --global-cmvn true --cmvn-file $fbank_dir/train.ark \
                --cmvn-option "--norm-means=true --norm-vars=true" \
                $fbank_dir/$x
    done
fi

