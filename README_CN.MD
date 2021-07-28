<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## 介绍
YOLOX是YOLO的anchor-free版本，设计更简单但性能更好！旨在弥合研究和工业界之间的差距，更多细节请参考我们在Arxiv上的论文。[点击这里](https://arxiv.org/abs/2107.08430).

<img src="assets/git_fig.png" width="1000" >

## 更新!!
* 【2021/07/26】 现已支持[MegEngine](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/MegEngine)（天元）部署.
* 【2021/07/20】 我们在 [Arxiv](https://arxiv.org/abs/2107.08430)发布了我们的技术报告。

## 即将推出
- [ ] YOLOX-P6和更大的模型.
- [ ] Objects365 预训练模型.
- [ ] Transformer 模块.
- [ ] 增加更多需要改善的功能.

## 基准

#### 标准模型.
|模型 |尺寸 |mAP<sup>test<br>0.5:0.95 | 速度V100<br>(ms) | 参数量<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.4      | 11.1 |63.7 | 185.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth) |

#### 轻量模型.
|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| 权重下载 |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/nano.py) |416  |25.3  | 0.91 |1.08 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdcREey-krhLtdtSnxolxiUBjWMy6EFdiaO9bdOwZ5ygCQ?e=yQpdds)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |31.7 | 5.06 |6.45 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EYtjNFPqvZBBrQ-VowLcSr4B6Z5TdTflUsr_gO2CwhC3bQ?e=SBTwXj)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny.pth) |

## 快速开始

<details>
<summary>安装</summary>

第一步. 安装 YOLOX
```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```
第二步. 安装 [apex](https://github.com/NVIDIA/apex).

```shell
# 如果不想训练模型，可跳过这步。
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
第三步. 安装 [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

</details>

<details>
<summary>演示</summary>

1.从基准表下载预训练模型。

2.使用 -n 或 -f 来设定检测器的配置。以图片为例：

```shell
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth.tar --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
或者
```shell
python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth.tar --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
视频演示:
```shell
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth.tar --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```


</details>

<details>
<summary>如何在COCO上复现我们的成果</summary>

1. 准备COCO数据集
```shell
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

2. 通过-n 设定检测器配置在COCO上复现我们的结果:

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o
                         yolox-m
                         yolox-l
                         yolox-x
```
* -d: GPU数量
* -b: 总批次大小, 我们推荐 b 的大小为 GPU数量 * 8
* --fp16: mixed precision training

使用 -f的时候，上述命令等价于下面:

```shell
python tools/train.py -f exps/default/yolox-s.py -d 8 -b 64 --fp16 -o
                         exps/default/yolox-m.py
                         exps/default/yolox-l.py
                         exps/default/yolox-x.py
```

</details>


<details>
<summary>评估</summary>

我们支持批量测试以进行快速评估：

```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth.tar -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
                         yolox-m
                         yolox-l
                         yolox-x
```
* --fuse: 融合 conv and bn 
* -d: 用于评估的 GPU 数量。默认：将使用所有可用的GPU。
* -b: 所有 GPU 上的总批大小 GPUs

复现速度测试，使用以下命令
```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth.tar -b 1 -d 1 --conf 0.001 --fp16 --fuse
                         yolox-m
                         yolox-l
                         yolox-x
```

</details>


<details open>
<summary>教程</summary>

*  [自定义数据集训练](docs/train_custom_data.md).

</details>

## 部署


1.  [MegEngine C++/Python 部署](./demo/MegEngine)
2.  [ONNX export and an ONNXRuntime](./demo/ONNXRuntime)
3.  [TensorRT  C++/Python 部署](./demo/TensorRT)
4.  [ncnn in C++/Java部署](./demo/ncnn)
5.  [OpenVINO C++/Python 部署](./demo/OpenVINO)


## 参考资源
* 支持视频的 ncnn android 应用程序： [ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)
* 支持Tengine: [Tengine](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolox.cpp) from [BUG1989](https://github.com/BUG1989)
* YOLOX + ROS2 Foxy: [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) from [Ar-Ray](https://github.com/Ar-Ray-code)
* YOLOX 部署 DeepStream: [YOLOX-deepstream](https://github.com/nanmi/YOLOX-deepstream) from [nanmi](https://github.com/nanmi)
* YOLOX ONNXRuntime C++ 演示: [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/ort/cv/yolox.cpp) from [DefTruth](https://github.com/DefTruth)

## 引用 YOLOX
如果您在研究中使用 YOLOX，请使用以下 BibTeX 条目引用我们的工作:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
