python tools/export_onnx.py \
-c /home/ecnu-lzw/bwz/ocr-gy/YOLOX/YOLOX_outputs/voc_ocr_2023/best_ckpt.pth \
-f /home/ecnu-lzw/bwz/ocr-gy/YOLOX/myconfig/voc_ocr_2023.py \
--output-name=yolox_s2023.onnx \
--dynamic --no-onnxsim

# /home/ecnu-lzw/bwz/ocr-gy/YOLOX/pth2onnx.sh