dir=./
python draw_det.py --x-min=0.0 --x-max=5.0 \
                   --y-min=1.0 --y-max=8.0 \
                   --line-style="- - -" \
                   --color="r b g" \
                   gru_hixiaowen.png \
                   "RHM " $dir/test_hixiaowen_0_roc.txt \

python draw_det.py --x-min=0.0 --x-max=5.0 \
                   --y-min=1.0 --y-max=8.0 \
                   --line-style="- - -" \
                   --color="r b g" \
                   gru_nihaowenwen.png \
                   "S1" $dir/test_nihaowenwen_0_roc.txt \


