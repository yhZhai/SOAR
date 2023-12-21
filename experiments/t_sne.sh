#!/bin/zsh

# Generate t-SNE visualization

cur_dir=$pwd
tmp_file=tmp/tmp.json

gpu_id=$1
config=$2
checkpoint=$3
out=$4  # end with format suffix
title=${5:-"t-SNE Visualization"}

CUDA_VISIBLE_DEVICES=$gpu_id python experiments/t_sne/t_sne_generation.py $config $checkpoint --out $tmp_file
python experiments/t_sne/t_sne_visualization.py -src $tmp_file -out $out -title $title
