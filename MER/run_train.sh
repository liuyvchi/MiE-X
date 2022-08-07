#!/usr/bin/env bash

python ./train_nov.py \
--gpu_ids 1 \
--train_data_path ./Loading_file_new/5w_ck/simulate_ck_train.txt \
--validation_data_path ./Loading_file_new/5w_ck+prior/simulate_ck+prior_vali.txt \
--test_data_path ./Loading_file/Micro_data_0.txt \
--test_label_path ./Loading_file/Micro_label_0.txt \
--batch_size  32 \
--num_workers 4 \
