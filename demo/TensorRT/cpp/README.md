# YOLOX-TensorRT in C++

As YOLOX models is easy to converted to tensorrt using [torch2trt gitrepo](https://github.com/NVIDIA-AI-IOT/torch2trt), 
our C++ demo will not include the model converting or constructing like other tenorrt demos.


## Step 1: Prepare serialized engine file

Follow the trt [python demo README](../python/README.md) to convert and save the serialized engine file.

Check the 'model_trt.engine' file generated from Step 1, which will automatically saved at the current demo dir.


## Step 2: build the demo

Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT.

And you should set the TensorRT path and Cuda path in CMakeLists.txt.

If you train you custom datasets just one classes ,and you should change the number of your datasets.

```c++
const int num_class = 80;
```

Install opencv with ```sudo apt-get install libopencv-dev```.

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

