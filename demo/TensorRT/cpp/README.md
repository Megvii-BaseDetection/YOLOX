# YOLOX-TensorRT in C++

As YOLOX models are easy to convert to tensorrt using [torch2trt gitrepo](https://github.com/NVIDIA-AI-IOT/torch2trt), 
our C++ demo does not include the model converting or constructing like other tenorrt demos.


## Step 1: Prepare serialized engine file

Follow the trt [python demo README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/TensorRT/python/README.md) to convert and save the serialized engine file.

Check the 'model_trt.engine' file generated from Step 1, which will be automatically saved at the current demo dir.


## Step 2: build the demo

Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT.

And you should set the TensorRT path and CUDA path in CMakeLists.txt.

If you train your custom dataset, you may need to modify the value of `num_class`.

```c++
const int num_class = 80;
```

Install opencv with ```sudo apt-get install libopencv-dev``` (we don't need a higher version of opencv like v3.3+). 

build the demo:

```shell
mkdir build
cd build
cmake ..
make
```

Then run the demo:

```shell
./yolox ../model_trt.engine -i ../../../../assets/dog.jpg
```

or

```shell
./yolox <path/to/your/engine_file> -i <path/to/image>
```

