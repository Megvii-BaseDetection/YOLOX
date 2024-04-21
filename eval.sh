#!/bin/bash

# evaluate retrained model
python eval.py -f mft_pg_yolox_exp.py -b 16 -c YOLOX_outputs/mft_pg_yolox_exp/best_ckpt.pth --result_file data/annotations/inference_result.json --nms 0.45 --conf 0.1 --class_agnostic

# evaluate current production model
tools/download_prod_model.sh
python eval.py -f object-detection-yolox_pytorch1.4/mft_pg_yolox_exp.py -b 16 -c object-detection-yolox_pytorch1.4/YOLOX_outputs/mft_pg_yolox_exp/best_ckpt.pth --result_file data/annotations/prod_inference_result.json  --nms 0.45 --conf 0.1 --class_agnostic
