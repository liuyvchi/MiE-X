#!/usr/bin/env bash

python simulate_data_3fold.py \
--gpu_ids 2 \
--checkpoints_dir ./checkpoints/ \
--name ./experiment_yh_2/ \
--load_epoch 25 \
