dir=./exp/Ticmini2_Public_ConstraintMaxPooling_random-ngFalse_sa1_ohem200_ratio10_constraint2_ctedge_cl30_cr30_num-p1_num-n1_gru_nl2_hd128_optnoam_bs400_lr0.010_gamma-p0.0_gamma-n0.0_hf0.7_wd5e-5_dp0.5_lc0_rc0_clamp1e-6_seed11
python draw_det.py --x-min=0.0 --x-max=3.0 \
                   --y-min=0.0 --y-max=3.0 \
                   --line-style="- -" \
                   --color="r b" \
                   gru_rhe.png \
                   "RHE Hi Xiaowen " $dir/test_hixiaowen_0_roc.txt \
                   "RHE NiHao Wenwen " $dir/test_nihaowenwen_0_roc.txt \



