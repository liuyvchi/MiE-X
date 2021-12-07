#!/usr/bin/env bash

python train.py \
--gpu_ids 0 \
--data_dir ./EmotioNet/sample_dataset_yh_2 \
--checkpoints_dir ./checkpoints/ \
--name ./experiment_yh_2/ \
--batch_size  25 \
--n_threads_train 4 \
