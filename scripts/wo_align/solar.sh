#!/bin/bash

model_name=MFRS
is_training=1
use_align=0

root_path_name=./datasets/
data_path_name=Solar.txt
model_id_name=Solar
data_name=Solar
enc_in=137

use_dc=1
use_embed=1
rs_len=96
rs_type=sin
Q=1
manual_pbp=0
manual_hbp=""

lr=0.001
lradj=(type3 type3 type3 type5)



pred_len=(96 192 336 720)
gpu=(0 1 2 3)

for i in 0 1 2 3
do
    nohup python -u run.py \
        --model $model_name \
        --is_training $is_training \
        --use_align $use_align \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name \
        --data $data_name \
        --enc_in $enc_in \
        --use_dc $use_dc \
        --use_embed $use_embed \
        --rs_len $rs_len \
        --rs_type $rs_type \
        --Q $Q \
        --manual_pbp $manual_pbp \
        --manual_hbp "$manual_hbp" \
        --learning_rate $lr \
        --lradj ${lradj[${i}]} \
        --pred_len ${pred_len[${i}]} \
        --gpu ${gpu[${i}]} \
    > logs/${model_id_name}_${use_align}_${pred_len[${i}]}.txt &
done


