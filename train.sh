CUDA_VISIBLE_DEVICES=1 python tools/train.py --exp_file exps/default/yolox_s_custom.py \
                                             --devices 0 \
                                             --batch-size 8 \
                                             --fp16