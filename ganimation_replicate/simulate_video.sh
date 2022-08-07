#!/usr/bin/env bash

python simulate_video_AUexp.py --mode test --data_root datasets/celebA --gpu_ids 2,3 --ckpt_dir ckpts/celebA/ganimation/190327_161852/ --load_epoch 30
#set '--interpolate_len 1' if you don't need linear interpolation.
#use '--save_test_gif' to generate animated images.