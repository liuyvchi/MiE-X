#!/usr/bin/env bash
echo
python ./train_fold_simu_fineT.py \
--gpu_ids 2 \
--Loading_path ./Loading_file_new// \
--Loading_path_f ./Loading_file_new/micro_3fold/ \
--batch_size  64 \
--num_workers 4 \
| tee ./log/flow/flow_DTSCNN.log
#| tee ./log/branches/SAMMpretrain_micro_0.log
#| tee ./log/branches/MiEX_pretrain_fineT_microMacro_0.log
#SMIC CASMEII_cropped, SAMM_cropped
