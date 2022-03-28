# YOLOX-Python-MegEngine

Python version of YOLOX object detection base on [MegEngine](https://github.com/MegEngine/MegEngine).

## Tutorial

### Step1: install requirements

```
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

### Step2: convert checkpoint weights from torch's path file

```
python3 convert_weights.py -w yolox_s.pth -o yolox_s_mge.pkl
```

### Step3: run demo

This part is the same as torch's python demo, but no need to specify device.

```
python3 demo.py image -n yolox-s -c yolox_s_mge.pkl --path ../../../assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result
```

###  [Optional]Step4: dump model for cpp inference

> **Note**: result model is dumped with `optimize_for_inference` and `enable_fuse_conv_bias_nonlinearity`.

```
python3 dump.py -n yolox-s -c yolox_s_mge.pkl --dump_path yolox_s.mge
```
