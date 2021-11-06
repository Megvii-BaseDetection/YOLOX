# YOLOX-CPP-ncnn

Cpp file compile of YOLOX object detection base on [ncnn](https://github.com/Tencent/ncnn).  
YOLOX is included in ncnn now, you could also try building from ncnn, it's better.

## Tutorial

### Step1
Clone [ncnn](https://github.com/Tencent/ncnn) first, then please following [build tutorial of ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build) to build on your own device.

### Step2
Use provided tools to generate onnx file.
For example, if you want to generate onnx file of yolox-s, please run the following command:
```shell
cd <path of yolox>
python3 tools/export_onnx.py -n yolox-s
```
Then, a yolox.onnx file is generated.

### Step3
Generate ncnn param and bin file.
```shell
cd <path of ncnn>
cd build/tools/ncnn
./onnx2ncnn yolox.onnx model.param model.bin
```

Since Focus module is not supported in ncnn. Warnings like:
```shell
Unsupported slice step ! 
```
will be printed. However, don't  worry!  C++ version of Focus layer is already implemented in yolox.cpp.

### Step4
Open **model.param**, and modify it.
Before (just an example):
```
295 328
Input            images                   0 1 images
Split            splitncnn_input0         1 4 images images_splitncnn_0 images_splitncnn_1 images_splitncnn_2 images_splitncnn_3
Crop             Slice_4                  1 1 images_splitncnn_3 647 -23309=1,0 -23310=1,2147483647 -23311=1,1
Crop             Slice_9                  1 1 647 652 -23309=1,0 -23310=1,2147483647 -23311=1,2
Crop             Slice_14                 1 1 images_splitncnn_2 657 -23309=1,0 -23310=1,2147483647 -23311=1,1
Crop             Slice_19                 1 1 657 662 -23309=1,1 -23310=1,2147483647 -23311=1,2
Crop             Slice_24                 1 1 images_splitncnn_1 667 -23309=1,1 -23310=1,2147483647 -23311=1,1
Crop             Slice_29                 1 1 667 672 -23309=1,0 -23310=1,2147483647 -23311=1,2
Crop             Slice_34                 1 1 images_splitncnn_0 677 -23309=1,1 -23310=1,2147483647 -23311=1,1
Crop             Slice_39                 1 1 677 682 -23309=1,1 -23310=1,2147483647 -23311=1,2
Concat           Concat_40                4 1 652 672 662 682 683 0=0
...
```
* Change first number for 295 to 295 - 9 = 286(since we will remove 10 layers and add 1 layers, total layers number should minus 9). 
* Then remove 10 lines of code from Split to Concat, but remember the last but 2nd number: 683.
* Add YoloV5Focus layer After Input (using previous number 683):
```
YoloV5Focus      focus                    1 1 images 683
```
After(just an example):
```
286 328
Input            images                   0 1 images
YoloV5Focus      focus                    1 1 images 683
...
```

### Step5
Use ncnn_optimize to generate new param and bin:
```shell
# suppose you are still under ncnn/build/tools/ncnn dir.
../ncnnoptimize model.param model.bin yolox.param yolox.bin 65536
```

### Step6
Copy or Move yolox.cpp file into ncnn/examples, modify the CMakeList.txt, then build yolox

### Step7
Inference image with executable file yolox, enjoy the detect result:
```shell
./yolox demo.jpg
```

## Acknowledgement

* [ncnn](https://github.com/Tencent/ncnn)
