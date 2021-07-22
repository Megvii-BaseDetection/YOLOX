# YOLOX-OpenVINO in Python

This toturial includes a Python demo for OpenVINO, as well as some converted models.

### Download OpenVINO models.
| Model | Parameters | GFLOPs | Test Size | mAP | Weights |
|:------| :----: | :----: | :---: | :---: | :---: |
|  [YOLOX-Nano](../../../exps/default/nano.py) |  0.91M  | 1.08 | 416x416 | 25.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EeWY57o5wQZFtXYd1KJw6Z8B4vxZru649XxQHYIFgio3Qw?e=ZS81ce)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano_openvino.tar.gz) |
|  [YOLOX-Tiny](../../../exps/default/yolox_tiny.py) | 5.06M     | 6.45 | 416x416 |31.7 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ETfvOoCXdVZNinoSpKA_sEYBIQVqfjjF5_M6VvHRnLVcsA?e=STL1pi)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny_openvino.tar.gz) |
|  [YOLOX-S](../../../exps/default/yolox_s.py) | 9.0M | 26.8 | 640x640 |39.6 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EXUjf3PQnbBLrxNrXPueqaIBzVZOrYQOnJpLK1Fytj5ssA?e=GK0LOM)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s_openvino.tar.gz) |
|  [YOLOX-M](../../../exps/default/yolox_m.py) | 25.3M | 73.8 | 640x640 |46.4 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EcoT1BPpeRpLvE_4c441zn8BVNCQ2naxDH3rho7WqdlgLQ?e=95VaM9)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m_openvino.tar.gz) |
|  [YOLOX-L](../../../exps/default/yolox_l.py) | 54.2M | 155.6 | 640x640 |50.0 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZvmn-YLRuVPh0GAP_w3xHMB2VGvrKqQXyK_Cv5yi_DXUg?e=YRh6Eq)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l_openvino.tar.gz) |
|  [YOLOX-Darknet53](../../../exps/default/yolov3.py) | 63.72M | 185.3 | 640x640 |47.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EQP8LSroikFHuwX0jFRetmcBOCDWSFmylHxolV7ezUPXGw?e=bEw5iq)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53_openvino.tar.gz) | 
|  [YOLOX-X](../../../exps/default/yolox_x.py) | 99.1M | 281.9 | 640x640 |51.2 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZFPnLqiD-xIlt7rcZYDjQgB4YXE9wnq1qaSXQwJrsKbdg?e=83nwEz)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x_openvino.tar.gz) |

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

   Please refer to the [ONNX toturial](../../ONNXRuntime). **Note that you should set --opset to 10, otherwise your next step will fail.**

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
   python3 mo.py --input_model yolox.onnx --input_shape [1,3,640,640] --data_type FP16 --output_dir converted_output
   ```

## Demo

### python

```shell
python openvino_inference.py -m <XML_MODEL_PATH> -i <IMAGE_PATH> 
```
or
```shell
python openvino_inference.py -m <XML_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s <SCORE_THR> -d <DEVICE>
```

