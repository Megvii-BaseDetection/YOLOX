# Train YOLO Format Dataset

This page explains how to train your own custom data in YOLO format.

We take an example of fine-tuning YOLOX-S model on VOC dataset to give a more clear guide.

## 0. Before you start
* Clone this repo and follow the [README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/README.md) to install YOLOX.
* Prepare your own dataset in YOLO format.


## 1. Create your Exp file
We provide a simple example exp  [here](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/example/yolo_format/yolox_s.py) to simply start your training.

**NOTE**: While using `YoloFormatDataset`,  please make sure that annotation file could be got  by change suffix and replace `images` to `labels`.

For example:  if path of an image is **/path/to/images/a/b/001.jpg**, then the annotation could be found in **/path/to/labels/a/b/001.txt**.

Some argument that you might use in `YoloFormatDataset`:
* data_dir: dir of  your self-defined YOLO format dataset.
* anno_file: filename that contains all used image, default to  "train.txt".   
* normalized_anno: whether your annotation is normalized or not. Default to `True`.
If your annotations looks like absolute value of coordinates,  please use `False`
* anno_format: Format of your boxes'  annotation.  Support `"xyxy"` and `"cxcywh"` now. Default to "cxcywh".
* label_first_anno: whether the first place of your annotation is label. For example, annotations like (0.12, 0.12, 0.23, 0.23, 18) should set this argument to `False`. Default to `True`.

## 2. Train/Test
Just follow  [README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/README.md). Please use `-f` arg with your exp file.

For example, run
```shell 
python3 tools/train.py -f your_file.py
```
could simply start your training.
