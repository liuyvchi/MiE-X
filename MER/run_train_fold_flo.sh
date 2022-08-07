#!/usr/bin/env bash

python ./train_fold_flo.py \
--gpu_ids 1 \
--Loading_path /home/user/Yuchi/iccv2019/MER/Loading_file/macro_micro_flo_3fold \
--batch_size  32 \
--num_workers 4 \
