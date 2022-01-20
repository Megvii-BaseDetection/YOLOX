python tools/demo_ocr.py image -n yolox-s \
-c /home/ecnu-lzw/bwz/ocr-gy/YOLOX/YOLOX_outputs/voc_ocr/best_ckpt.pth \
-f /home/ecnu-lzw/bwz/ocr-gy/YOLOX/myconfig/voc_ocr.py \
--path /home/ecnu-lzw/bwz/ocr-gy/YOLOX/datasets/VOCdevkit/VOC2022/JPEGImages \
--conf 0.25 --nms 0.3 --tsize 640 --save_result --device gpu

# cd /home/ecnu-lzw/bwz/ocr-gy/YOLOX
# ./inference.sh
