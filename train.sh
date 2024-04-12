#!/bin/bash
# launch tensorboard
tensorboard --logdir=YOLOX_outputs --port=6006 &

python train.py -f mft_pg_yolox_exp.py -b 16 --cache