#!/usr/bin/env bash

python test.py \
--gpu_ids 2 \
--input_path  ./EmotioNet/generated_imgs/WechatIMG2.jpeg_out_base.png \
--output_dir ./EmotioNet/generated_imgs/ \
--checkpoints_dir ./checkpoints/ \
--name ./experiment_yh_2/ \
--load_epoch 20 \
