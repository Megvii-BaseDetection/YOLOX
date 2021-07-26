python tools/demo.py image -n yolox-s \
                           -c pretrained_models/yolox_s.pth.tar \
                           --path assets/dog.jpg \
                           --conf 0.25 \
                           --nms 0.45 \
                           --tsize 640 \
                           --save_result \
                           --device gpu
