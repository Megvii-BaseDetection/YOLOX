python tools/demo_ocr.py image -n yolox-s \
-f /home/ecnu-lzw/bwz/ocr-gy/YOLOX/myconfig/voc_ocr_2023.py \
--path /home/ecnu-lzw/bwz/ocr-gy/YOLOX/datasets/VOCdevkit/VOC2022/JPEGImages \
--conf 0.25 --nms 0.3 --save_result --device gpu --trt

# cd /home/ecnu-lzw/bwz/ocr-gy/YOLOX
# ./inferenceTRT.sh
