#!/usr/bin/env bash

python simulate_data.py \
--gpu_ids 1 \
--checkpoints_dir ./checkpoints/ \
--name ./experiment_yh_2/ \
--load_epoch 25 \
