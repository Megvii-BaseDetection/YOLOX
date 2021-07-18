# User Guide for Deploy YOLOX on OpenVINO

This toturial includes a C++ demo for OpenVINO, as well as some converted models.

### Download OpenVINO models.
| Model | Parameters | GFLOPs | Test Size | mAP |
|:------| :----: | :----: | :---: | :---: | 
|  [YOLOX-Nano](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.res101.fpn.coco.800size.1x) |  0.91M  | 1.08 | 416x416 | 25.3 |
|  [YOLOX-Tiny](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.fpn.coco.800size.1x) | 5.06M     | 6.45 | 416x416 |31.7 |
|  [YOLOX-S](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 9.0M | 26.8 | 640x640 |39.6 | 
|  [YOLOX-M](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 25.3M | 73.8 | 640x640 |46.4 |
|  [YOLOX-L](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 54.2M | 155.6 | 640x640 |50.0 | 
|  [YOLOX-X](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 99.1M | 281.9 | 640x640 |51.2 | 
|  [YOLOX-Darknet53](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 63.72M | 185.3 | 640x640 |47.3 | 

## Install OpenVINO Toolkit

Please visit [Openvino Homepage](https://docs.openvinotoolkit.org/latest/get_started_guides.html) for more details.

## Set up the Environment

### For Linux

**Option1. Set up the environment tempororally. You need to run this command everytime you start a new shell window.**

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

**Option2. Set up the environment permenantly.**

*Step1.* For Linux:
```shell
vim ~/.bashrc 
```

*Step2.* Add the following line into your file:

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

*Step3.* Save and exit the file, then run:

```shell
source ~/.bashrc
```


## Convert model

1. Export ONNX model
   
   Please refer to the [ONNX toturial]() for more details. **Note that you should set --opset to 10, otherwise your next step will fail.**

2. Convert ONNX to OpenVINO 

   ``` shell
   cd <INSTSLL_DIR>/openvino_2021/deployment_tools/model_optimizer
   ```

   Install requirements for convert tool

   ```shell
   sudo ./install_prerequisites/install_prerequisites_onnx.sh
   ```

   Then convert model.
   ```shell
   python3 mo.py --input_model <ONNX_MODEL> --input_shape <INPUT_SHAPE> [--data_type FP16]
   ```
   For example:
   ```shell
   python3 mo.py --input_model yolox.onnx --input_shape (1,3,640,640) --data_type FP16
   ```  

## Build 

### Linux
```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
mkdir build
cd build
cmake ..
make
```

## Demo

### c++

```shell
./yolox_openvino <XML_MODEL_PATH> <IMAGE_PATH> <DEVICE>
```
