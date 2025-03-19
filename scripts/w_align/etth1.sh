#!/bin/bash

model_name=MFRS
is_training=0
use_align=1
L=8640

root_path_name=./datasets/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
enc_in=7

use_dc=0
use_embed=1
rs_len=96
rs_type=sin
Q=3
manual_pbp=8640
manual_hbp="2 3 4"

lr=(0.0002 0.0002 0.0001 0.0001)
lradj=type1



pred_len=(96 192 336 720)
gpu=(0 1 2 3)

for i in 0 1 2 3
do
    nohup python -u run.py \
        --model $model_name \
        --is_training $is_training \
        --use_align $use_align \
        --L $L \
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
        --manual_hbp "${manual_hbp}" \
        --learning_rate ${lr[${i}]} \
        --lradj $lradj \
        --pred_len ${pred_len[${i}]} \
        --gpu ${gpu[${i}]} \
    > logs/${model_id_name}_${use_align}_${pred_len[${i}]}.txt &
done