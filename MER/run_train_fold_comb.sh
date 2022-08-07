#!/usr/bin/env bash

python ./train_comb_flod.py \
--gpu_ids 0 \
--Loading_path /home/user/Yuchi/iccv2019/MER/Loading_file/micro_flo_3fold \
--batch_size  32 \
--num_workers 4 \
