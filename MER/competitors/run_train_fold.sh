#!/usr/bin/env bash

python ./train_fold_nov.py \
--gpu_ids 1 \
--Loading_path ./../Loading_file/macro_micro_3fold \
--batch_size  32 \
--num_workers 4 \

#macro_micro_3fold
#5w_real+ck+prior_fold