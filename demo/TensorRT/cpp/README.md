# YOLOX-TensorRT in C++

As YOLOX models is easy to converted to tensorrt using [torch2trt gitrepo](https://github.com/NVIDIA-AI-IOT/torch2trt), 
our C++ demo will not include the model converting or constructing like other tenorrt demos.


## Step 1: Prepare serialized engine file

Follow the trt [python demo README](../python/README.md) to convert and save the serialized engine file.


## Step 2: build the demo

Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT.

Install opencv with ```sudo apt-get install libopencv-dev```.

build the demo:

```shell
mkdir build
cd build
cmake ..
make
```

Move the 'model_trt.engine' file generated from Step 1 (saved at the exp output dir) to the build dir:

```shell
mv /path/to/your/exp/output/dir/model_trt.engine .
```

Then run the demo:

```shell
./yolox -d /your/path/to/yolox/assets
```

or

```shell
./yolox -d <img dir>
```
