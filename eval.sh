#!/bin/bash
python eval.py -f mft_pg_yolox_exp.py -b 16 -c YOLOX_outputs/mft_pg_yolox_exp/best_ckpt.pth --result_file data/annotations/inference_result.json
