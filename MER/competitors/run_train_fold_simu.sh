#!/usr/bin/env bash

python ./train_fold_simu_cat_nov.py \
--gpu_ids 0 \
--Loading_path ./../Loading_file_new/5w_real+ck+prior_fold/ \
--batch_size  32 \
--num_workers 4 \

#macro_micro_3fold
#5w_real+ck+prior_fold