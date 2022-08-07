#!/usr/bin/env bash
echo
python ./train_fold_simu_apex_fineT.py \
--gpu_ids 1 \
--Loading_path ./../Loading_file_new/5w_real+ck+prior_fold/ \
--Loading_path_f ./../Loading_file_new/micro_3fold/ \
--batch_size  96 \
--num_workers 16 \
| tee ./log/flow/flow_apex.log