#!/usr/bin/env bash
echo
python ./train_fold_simu_apex_fineT_5fold.py \
--gpu_ids 2 \
--Loading_path ./../Loading_file_new/5w_real+ck+prior_fold/ \
--Loading_path_f ./../Loading_file_new/MMEW_5fold/ \
--batch_size  64 \
--num_workers 16 \
| tee ./log/flow/flow_MMEW_5fold_withPre.log