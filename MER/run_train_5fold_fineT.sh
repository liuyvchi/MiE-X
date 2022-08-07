#!/usr/bin/env bash
echo
python ./train_fold_simu_fineT_5fold.py \
--gpu_ids 2 \
--Loading_path ./Loading_file_new/5w_real+ck+prior_fold/ \
--Loading_path_f ./Loading_file_new/MMEW_5fold/ \
--batch_size  32 \
--num_workers 4 \
| tee ./log/branches/nomiex_MMEW_5fold.log
#| tee ./log/branches/MiEX_pretrain_fineT_microMacro_0.log