#!/bin/bash
# launch tensorboard
tensorboard --logdir=/workspace/mnt/YOLOX_outputs/mft_pg_yolox_exp --host 0.0.0.0 --port 6006 &

python train.py -f mft_pg_yolox_exp.py -b 32 --cache