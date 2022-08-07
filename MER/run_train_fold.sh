#!/usr/bin/env bash

python ./train_fold_nov.py \
--gpu_ids 1 \
--Loading_path ./Loading_file_new/5w_real+ck+prior_fold/ \
--batch_size  32 \
--num_workers 4 \