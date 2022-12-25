# YOLOX-CPP-ncnn

Cpp file compile of YOLOX object detection base on [ncnn](https://github.com/Tencent/ncnn).  

## Tutorial

### Step1
Clone [ncnn](https://github.com/Tencent/ncnn) first, then please following [build tutorial of ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build) to build on your own device.

### Step2
First, we try the original onnx2ncnn solution by using provided tools to generate onnx file.
For example, if you want to generate onnx file of yolox-s, please run the following command:
```shell
cd <path of yolox>
python3 tools/export_onnx.py -n yolox-s
```
Then a yolox.onnx file is generated.

### Step3
Generate ncnn param and bin file.
```shell
cd <path of ncnn>
cd build/tools/ncnn
./onnx2ncnn yolox.onnx model.param model.bin
```

Since Focus module is not supported in ncnn. You will see warnings like:
```shell
Unsupported slice step!
```
However, don't worry on this as a C++ version of Focus layer is already implemented in yolox.cpp.

### Step4
Open **model.param**, and modify it. For more information on the ncnn param and model file structure, please take a look at this [wiki](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure).

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
* Change first number for 295 to 295 - 9 = 286 (since we will remove 10 layers and add 1 layers, total layers number should minus 9). 
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
Copy or Move yolox.cpp file into ncnn/examples, modify the CMakeList.txt to add our implementation, then build.

### Step7
Inference image with executable file yolox, enjoy the detect result:
```shell
./yolox demo.jpg
```

### Bounus Solution:
As ncnn has released another model conversion tool called [pnnx](https://zhuanlan.zhihu.com/p/427620428) which directly finishs the pytorch2ncnn process via torchscript, we can also try on this.

```shell
# take yolox-s as an example
python3 tools/export_torchscript.py -n yolox-s -c /path/to/your_checkpoint_files
```
Then a `yolox.torchscript.pt` will be generated. Copy this file to your pnnx build directory (pnnx also provides pre-built packages [here](https://github.com/pnnx/pnnx/releases/tag/20220720)).

```shell
# suppose you put the yolox.torchscript.pt in a seperate folder
./pnnx yolox/yolox.torchscript.pt inputshape=[1,3,640,640]
# for zsh users, please use inputshape='[1,3,640,640]'
```
Still, as ncnn does not support `slice` op as we mentioned in [Step3](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ncnn/cpp#step3). You will still see the warnings during this process.

Then multiple pnnx related files will be genreated in your yolox folder. Use `yolox.torchscript.ncnn.param` and `yolox.torchscript.ncnn.bin` as your converted model. 

Then we can follow back to our [Step4](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ncnn/cpp#step4) for the rest of our implementation.

## Acknowledgement

* [ncnn](https://github.com/Tencent/ncnn)
