python tools/train.py \
-f /home/ecnu-lzw/bwz/ocr-gy/YOLOX/myconfig/voc_ocr.py \
-d 2 -b 64 --fp16 -o \
-c /home/ecnu-lzw/bwz/ocr-gy/YOLOX/models/yolox_s.pth \
--cache

# cd /home/ecnu-lzw/bwz/ocr-gy/YOLOX
# ./train.sh
# tensorboard --logdir='/home/ecnu-lzw/bwz/ocr-gy/YOLOX/YOLOX_outputs/voc_ocr'
