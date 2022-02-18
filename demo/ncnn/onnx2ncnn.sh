cd /home/ecnu-lzw/bwz/ocr-gy/ncnn
./build/tools/onnx/onnx2ncnn /home/ecnu-lzw/bwz/ocr-gy/YOLOX/yolox_s2023.onnx ./outputs/yolox_s2023.param ./outputs/yolox_s2023.bin
# /home/ecnu-lzw/bwz/ocr-gy/ncnn/onnx2ncnn.sh
# ./build/tools/ncnnoptimize ./outputs/yolox_s2023.param ./outputs/yolox_s2023.bin ./outputs/yolox_s2023_fix.param ./outputs/yolox_s2023_fix.bin 65536
