## ONNXRuntime Demo in Python

This doc introduces how to convert you pytorch model into onnx, and how to run an onnxruntime demo to verify your convertion.

### Download ONNX models.
| Model | Parameters | GFLOPs | Test Size | mAP |
|:------| :----: | :----: | :---: | :---: | 
|  [YOLOX-Nano](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.res101.fpn.coco.800size.1x) |  0.91M  | 1.08 | 416x416 | 25.3 |
|  [YOLOX-Tiny](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.fpn.coco.800size.1x) | 5.06M     | 6.45 | 416x416 |31.7 |
|  [YOLOX-S](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 9.0M | 26.8 | 640x640 |39.6 | 
|  [YOLOX-M](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 25.3M | 73.8 | 640x640 |46.4 |
|  [YOLOX-L](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 54.2M | 155.6 | 640x640 |50.0 | 
|  [YOLOX-X](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 99.1M | 281.9 | 640x640 |51.2 | 
|  [YOLOX-Darknet53](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | 63.72M | 185.3 | 640x640 |47.3 | 

### Convert Your Model to ONNX

First, you should move to <YOLOX_HOME> by:
```shell
cd <YOLOX_HOME>
```
Then, you can:

1. Convert a standard YOLOX model by -n:
```shell
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth.tar
```
Notes:
* -n: specify a model name. The model name must be one of the [yolox-s,m,l,x and yolox-nane, yolox-tiny, yolov3]
* -c: the model you have trained
* -o: opset version, default 11. **However, if you will further convert your onnx model to [OpenVINO](), please specify the opset version to 10.**
* --no-onnxsim: disable onnxsim
* To customize an input shape for onnx model,  modify the following code in tools/export.py:

    ```python
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    ```

2. Convert a standard YOLOX model by -f. By using -f, the above command is equivalent to:

```shell
python3 tools/export_onnx.py --output-name yolox_s.onnx -f exps/yolox_s.py -c yolox_s.pth.tar
```

3. To convert your customized model, please use -f:

```shell
python3 tools/export_onnx.py --output-name your_yolox.onnx -f exps/your_yolox.py -c your_yolox.pth.tar
```

### ONNXRuntime Demo

Step1.
```shell
cd <YOLOX_HOME>/yolox/deploy/demo_onnxruntime/
```

Step2. 
```shell
python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```
Notes:
* -m: your converted onnx model
* -i: input_image
* -s: score threshold for visualization.
* --input_shape: should be consistent with the shape you used for onnx convertion.