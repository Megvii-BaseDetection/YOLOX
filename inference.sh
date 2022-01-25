python tools/demo_ocr.py image -n yolox-s \
-c /home/ecnu-lzw/bwz/ocr-gy/YOLOX/YOLOX_outputs/voc_ocr_2027_noaug30/best_ckpt.pth \
-f /home/ecnu-lzw/bwz/ocr-gy/YOLOX/myconfig/voc_ocr_2027_noaug30.py \
--path /home/ecnu-lzw/bwz/ocr-gy/steelDatasets/datasets2022_unlabeled_img \
--conf 0.25 --nms 0.3 --save_result --device gpu

# cd /home/ecnu-lzw/bwz/ocr-gy/YOLOX
# ./inference.sh
