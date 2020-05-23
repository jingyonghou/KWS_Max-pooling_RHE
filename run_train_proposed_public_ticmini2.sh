#!/bin/bash
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
set -e 
stage=1

path=./
data=$path/fbank
label_dir=$path/info
if [ $stage -le 0 ]; then
	for x in train dev test; do
		x=${x}_cmvn
		feat-to-len scp:$data/$x/feats.scp ark,t:$data/$x/lens.scp
		python prepare_torch_scp.py $data/$x/feats.scp $label_dir/labels.scp $data/$x/lens.scp 1500 $data/$x/torch.scp
	done
fi
for seed in 11; do # 22 33 44 55; do
for ohem in 50; do #50 10000 25 15 0; do 
for max_ratio in 10 20; do
for learning_rate in 0.010 0.008 0.004 0.020; do 
train_scp=$data/train_cmvn/torch.scp
dev_scp=$data/dev_cmvn/torch.scp
test_scp=$data/test_cmvn/torch.scp
feat_dim=`feat-to-dim scp:$data/train_cmvn/feats.scp -`
layer_type=gru
if [ $layer_type == gru ]; then
	hidden_dim=128
	num_layers=2
	dropout=0.5
elif [ $layer_type == tcn ]; then
	hidden_dim=64
	num_layers=2
	dropout=0.0
elif [ $layer_type == wavenet ]; then
	hidden_dim=64
	num_layers=
	dropout=
else
	exit 1
fi

optimizer="noam"
left_context=0
right_context=0
filler=0
input_dim=$((($left_context+$right_context+1)*$feat_dim))
output_dim=2
proto=Ticmini2_Public_ConstraintMaxPooling
weight_decay=5e-5
batch_size=400
halving_factor=0.7
clamp="1e-6"
#clamp="0"
previous_model=""
num_p=1
num_n=1
gamma_p=0.0 # if gamma = 0, that means we use CE, otherwise, we use Focal loss
gamma_n=0.0 # if gamma = 0, that means we use CE, otherwise, we use Focal loss
gpu_num=1
spec_augment=0
constraint=2
constraint_type="edge"
constraint_l=30
constraint_r=30
random_ng="True"
random_ng="False"
debug="-m pdb"
debug=""
do_test="1"
save_dir="$path/exp/${proto}_random-ng${random_ng}_sa${spec_augment}_rhe${ohem}_ratio${max_ratio}_constraint${constraint}_ct${constraint_type}_cl${constraint_l}_cr${constraint_r}_num-p${num_p}_num-n${num_n}_${layer_type}_nl${num_layers}_hd${hidden_dim}_opt${optimizer}_bs${batch_size}_lr${learning_rate}_gamma-p${gamma_p}_gamma-n${gamma_n}_hf${halving_factor}_wd${weight_decay}_dp${dropout}_lc${left_context}_rc${right_context}_clamp${clamp}_seed${seed}"
mkdir -p $save_dir

echo "Input dim: $input_dim"
echo "Hidden feature dim: $hidden_dim"
echo "Output dim: $output_dim"
echo "Exp Dir: $save_dir"
echo "Train Scp: ${train_scp}"
head $train_scp -n 1
echo "Dev Scp: ${dev_scp}"
head $dev_scp -n 1

if [ $stage -le 1 ]; then
    #CUDA_VISIBLE_DEVICES=$gpu_num python $debug train_max_pooling_binary.py \
    $cuda_cmd $save_dir/train_log.txt python $debug train_max_pooling_binary.py \
            --seed=${seed} --train=1 --test=0 \
            --encoder=$layer_type \
			--random-n=$random_ng \
            --spec-augment=$spec_augment \
			--ohem=$ohem \
			--max-ratio=$max_ratio \
            --constraint=$constraint \
			--constraint-type=$constraint_type \
			--cl=$constraint_l \
			--cr=$constraint_r \
            --num-p=$num_p \
            --num-n=$num_n \
            --input-dim=$input_dim \
            --hidden-dim=$hidden_dim \
            --num-layers=$num_layers \
            --output-dim=$output_dim \
            --dropout=$dropout \
            --left-context=$left_context \
            --right-context=$right_context \
            --max-epochs=20 \
            --min-epochs=15 \
            --batch-size=$batch_size \
            --learning-rate=$learning_rate \
			--optimizer=${optimizer} \
			--init-weight-decay=$weight_decay \
			--gamma-p=$gamma_p \
			--gamma-n=$gamma_n \
			--clamp=$clamp \
            --halving-factor=$halving_factor \
            --load-model=$previous_model \
            --start-halving-impr=0.01 \
            --end-halving-impr=0.001 \
            --use-cuda=1 \
            --multi-gpu=0 \
            --train-scp=$train_scp \
            --dev-scp=$dev_scp \
            --num-workers=5 \
            --save-dir=$save_dir \
            --log-interval=10   #| tee $save_dir/log.txt
fi

best_model=$save_dir/final.mdl 
decode_output=ark:$save_dir/dev_post.ark
if [ $stage -le 2 ]; then
    # test and get roc
    $cuda_cmd $save_dir/dev_log.txt python $debug train_max_pooling_binary.py \
            --seed=10 --train=0 --test=1 \
            --encoder=$layer_type \
            --input-dim=$input_dim \
            --hidden-dim=$hidden_dim \
            --num-layers=$num_layers \
            --output-dim=$output_dim \
            --dropout=$dropout \
            --left-context=$left_context \
            --right-context=$right_context \
            --batch-size=$batch_size \
            --load-model=$best_model \
            --use-cuda=1 \
            --multi-gpu=0 \
            --test-scp=$dev_scp \
            --num-workers=5 \
            --output-file=$decode_output \
            --log-interval=10 | tee -a $save_dir/log.txt
fi
#hixiaowen
for keyword in hixiaowen nihaowenwen; do
	if [ $keyword == hixiaowen ]; then
		ignore_keyword=0; negative_duration=11.81; num_positive=3680; tag=""
	#nihaowenwen
	elif [ $keyword == nihaowenwen ]; then
		ignore_keyword=0; negative_duration=11.77; num_positive=3677; tag=""
	fi
	
	if [ $stage -le 3 ]; then
	    # get score
	    python get_score_by_label.py \
				--ignore-keyword=$ignore_keyword \
				--smooth-window=1 \
				"$decode_output" \
				$keyword \
				$dev_scp \
				"$save_dir/dev_${keyword}_${ignore_keyword}_score.txt" 
	fi
	
	if [ $stage -le 4 ]; then
	    python compute_det.py --sliding-window=100 \
	                          --start-threshold=0.0 \
	                          --end-threshold=1.0 \
	                          --threshold-step=0.01 \
	                            $save_dir/dev_${keyword}_${ignore_keyword}_score.txt \
	                            $save_dir/dev_${keyword}_${ignore_keyword}_roc.txt \
	                            $negative_duration $num_positive 
	fi
done
if [ $do_test == 1 ]; then

decode_output=ark:$save_dir/test_post.ark
if [ $stage -le 5 ]; then
    # test and get roc
    #CUDA_VISIBLE_DEVICES=$gpu_num python $debug train_max_pooling_binary.py \
    $cuda_cmd $save_dir/test_log.txt python $debug train_max_pooling_binary.py \
            --seed=10 --train=0 --test=1 \
            --encoder=$layer_type \
            --input-dim=$input_dim \
            --hidden-dim=$hidden_dim \
            --num-layers=$num_layers \
            --output-dim=$output_dim \
            --dropout=$dropout \
            --left-context=$left_context \
            --right-context=$right_context \
            --batch-size=$batch_size \
            --load-model=$best_model \
            --use-cuda=1 \
            --multi-gpu=0 \
            --test-scp=$test_scp \
            --num-workers=5 \
            --output-file=$decode_output \
            --log-interval=10 | tee -a $save_dir/log.txt
fi
for keyword in hixiaowen nihaowenwen; do
	# hixiaowen
	if [ $keyword == hixiaowen ]; then
		ignore_keyword=0; negative_duration=28.32; num_positive=10641; tag=""
	# nihaowenwen
	elif [ $keyword == nihaowenwen ]; then
		ignore_keyword=0; negative_duration=28.19; num_positive=10640; tag=""
	fi
	
	if [ $stage -le 6 ]; then
	    # get score
	    echo "python get_score.py --ignore-keyword=$ignore_keyword --smooth-window=1 '$decode_output' $keyword '$save_dir/test_${keyword}_${ignore_keyword}_score.txt'"
	    python get_score_by_label.py \
				--ignore-keyword=$ignore_keyword \
				--smooth-window=1 \
				"$decode_output" \
				$keyword \
				$test_scp \
				"$save_dir/test_${keyword}_${ignore_keyword}_score.txt" 
	fi
	
	if [ $stage -le 7 ]; then
	    python compute_det.py --sliding-window=100 \
	                          --start-threshold=0.0 \
	                          --end-threshold=1.0 \
	                          --threshold-step=0.01 \
	                            $save_dir/test_${keyword}_${ignore_keyword}_score.txt \
	                            $save_dir/test_${keyword}_${ignore_keyword}_roc.txt \
	                            $negative_duration $num_positive 
	fi
done
fi
done
done
done
done
