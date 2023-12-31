#!/usr/bin/env bash

# set up environment
conda env create -f environment.yml
source activate gym
pip install "mmcv<2.0.0"
pip install --upgrade youtube-dl
pip install cv2

DATA_DIR="../../../data/gym"
ANNO_DIR="../../../data/gym/annotations"
python download.py ${ANNO_DIR}/annotation.json ${DATA_DIR}/videos

conda deactivate
conda remove -n gym --all
