# YOLOX-CPP-MegEngine

Cpp file compile of YOLOX object detection base on [MegEngine](https://github.com/MegEngine/MegEngine).

## Tutorial

### Step1: install toolchain

	* host: sudo apt install gcc/g++ (gcc/g++, which version >= 6) build-essential git git-lfs gfortran libgfortran-6-dev autoconf gnupg flex bison gperf curl zlib1g-dev gcc-multilib g++-multilib cmake
 * cross build android: download [NDK](https://developer.android.com/ndk/downloads)
   	* after unzip download NDK, then export NDK_ROOT="path of NDK"

### Step2: build MegEngine

```shell
git clone https://github.com/MegEngine/MegEngine.git

# then init third_party
 
export megengine_root="path of MegEngine"
cd $megengine_root && ./third_party/prepare.sh && ./third_party/install-mkl.sh

# build example:
# build host without cuda:   
./scripts/cmake-build/host_build.sh
# or build host with cuda:
./scripts/cmake-build/host_build.sh -c
# or cross build for android aarch64: 
./scripts/cmake-build/cross_build_android_arm_inference.sh
# or cross build for android aarch64(with V8.2+fp16): 
./scripts/cmake-build/cross_build_android_arm_inference.sh -f

# after build MegEngine, you need export the `MGE_INSTALL_PATH`
# host without cuda: 
export MGE_INSTALL_PATH=${megengine_root}/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_ON/Release/install
# or host with cuda: 
export MGE_INSTALL_PATH=${megengine_root}/build_dir/host/MGE_WITH_CUDA_ON/MGE_INFERENCE_ONLY_ON/Release/install
# or cross build for android aarch64: 
export MGE_INSTALL_PATH=${megengine_root}/build_dir/android/arm64-v8a/Release/install
```
* you can refs [build tutorial of MegEngine](https://github.com/MegEngine/MegEngine/blob/master/scripts/cmake-build/BUILD_README.md) to build other platform, eg, windows/macos/ etc!

### Step3: build OpenCV

```shell
git clone https://github.com/opencv/opencv.git

git checkout 3.4.15 (we test at 3.4.15, if test other version, may need modify some build)
```

- patch diff for android:

```
# ```
#     diff --git a/CMakeLists.txt b/CMakeLists.txt
#     index f6a2da5310..10354312c9 100644
#     --- a/CMakeLists.txt
#     +++ b/CMakeLists.txt
#     @@ -643,7 +643,7 @@ if(UNIX)
#        if(NOT APPLE)
#          CHECK_INCLUDE_FILE(pthread.h HAVE_PTHREAD)
#          if(ANDROID)
#     -      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log)
#     +      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log z)
#          elseif(CMAKE_SYSTEM_NAME MATCHES "FreeBSD|NetBSD|DragonFly|OpenBSD|Haiku")
#            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} m pthread)
#          elseif(EMSCRIPTEN)
    
# ```
```

- build for host

```shell
cd root_dir_of_opencv
mkdir -p build/install
cd build
cmake -DBUILD_JAVA=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install 
make install -j32
```

* build for android-aarch64

```shell
cd root_dir_of_opencv
mkdir -p build_android/install
cd build_android

cmake -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_NDK="$NDK_ROOT"  -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 -DBUILD_JAVA=OFF -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install ..

make install -j32
```

* after build OpenCV, you need export  `OPENCV_INSTALL_INCLUDE_PATH ` and `OPENCV_INSTALL_LIB_PATH`

```shell
# host build: 
export OPENCV_INSTALL_INCLUDE_PATH=${path of opencv}/build/install/include
export OPENCV_INSTALL_LIB_PATH=${path of opencv}/build/install/lib
# or cross build for android aarch64:
export OPENCV_INSTALL_INCLUDE_PATH=${path of opencv}/build_android/install/sdk/native/jni/include
export OPENCV_INSTALL_LIB_PATH=${path of opencv}/build_android/install/sdk/native/libs/arm64-v8a
```

###  Step4: build test demo

```shell
run build.sh

# if host:
export CXX=g++
./build.sh
# or cross android aarch64
export CXX=aarch64-linux-android21-clang++
./build.sh
```

### Step5: run demo

> **Note**: two ways to get `yolox_s.mge` model file
>
> * reference to python demo's `dump.py` script.
> * For users with code before 0.1.0 version, wget yolox-s weights [here](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.mge).
> * For users with code after 0.1.0 version, use [python code in megengine](../python) to generate mge file.

```shell
# if host:
LD_LIBRARY_PATH=$MGE_INSTALL_PATH/lib/:$OPENCV_INSTALL_LIB_PATH ./yolox yolox_s.mge ../../../assets/dog.jpg cuda/cpu/multithread <warmup_count> <thread_number>

# or cross android
adb push/scp $MGE_INSTALL_PATH/lib/libmegengine.so android_phone
adb push/scp $OPENCV_INSTALL_LIB_PATH/*.so android_phone
adb push/scp ./yolox yolox_s.mge android_phone
adb push/scp ../../../assets/dog.jpg android_phone

# login in android_phone by adb or ssh
# then run: 
LD_LIBRARY_PATH=. ./yolox yolox_s.mge dog.jpg cpu/multithread <warmup_count> <thread_number> <use_fast_run> <use_weight_preprocess>  <run_with_fp16>

# * <warmup_count> means warmup count, valid number >=0
# * <thread_number> means thread number, valid number >=1, only take effect `multithread` device
# * <use_fast_run> if >=1 , will use fastrun to choose best algo
# * <use_weight_preprocess> if >=1, will handle weight preprocess before exe
# * <run_with_fp16> if >=1, will run with fp16 mode
```

## Bechmark

* model info: yolox-s @ input(1,3,640,640)					

* test devices

```
  * x86_64  -- Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz					
  * aarch64 -- xiamo phone mi9					
  * cuda    -- 1080TI @ cuda-10.1-cudnn-v7.6.3-TensorRT-6.0.1.5.sh @ Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
```

  | megengine @ tag1.4(fastrun + weight\_preprocess)/sec | 1 thread |
  | ---------------------------------------------------- | -------- |
  | x86\_64                                              | 0.516245 |
  | aarch64(fp32+chw44)                                  | 0.587857 |

  | CUDA @ 1080TI/sec   | 1 batch    | 2 batch   | 4 batch   | 8 batch   | 16 batch  | 32 batch | 64 batch |
  | ------------------- | ---------- | --------- | --------- | --------- | --------- | -------- | -------- |
  | megengine(fp32+chw) | 0.00813703 | 0.0132893 | 0.0236633 | 0.0444699 | 0.0864917 | 0.16895  | 0.334248 |

## Acknowledgement

* [MegEngine](https://github.com/MegEngine/MegEngine)
* [OpenCV](https://github.com/opencv/opencv)
* [NDK](https://developer.android.com/ndk)
* [CMAKE](https://cmake.org/)
