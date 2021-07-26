#!/usr/bin/env bash
set -e

if [ -z $CXX ];then
    echo "please export you c++ toolchain to CXX"
    echo "for example:"
    echo "build for host:                                        export CXX=g++"
    echo "cross build for aarch64-android(always locate in NDK): export CXX=aarch64-linux-android21-clang++"
    echo "cross build for aarch64-linux:                         export CXX=aarch64-linux-gnu-g++"
    exit -1
fi

if [ -z $MGE_INSTALL_PATH ];then
    echo "please refsi ./README.md to init MGE_INSTALL_PATH env"
    exit -1
fi

if [ -z $OPENCV_INSTALL_INCLUDE_PATH ];then
    echo "please refs ./README.md to init OPENCV_INSTALL_INCLUDE_PATH env"
    exit -1
fi

if [ -z $OPENCV_INSTALL_LIB_PATH ];then
    echo "please refs ./README.md to init OPENCV_INSTALL_LIB_PATH env"
    exit -1
fi

INCLUDE_FLAG="-I$MGE_INSTALL_PATH/include -I$OPENCV_INSTALL_INCLUDE_PATH"
LINK_FLAG="-L$MGE_INSTALL_PATH/lib/ -lmegengine -L$OPENCV_INSTALL_LIB_PATH -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs"
BUILD_FLAG="-static-libstdc++ -O3 -pie -fPIE -g"

if [[ $CXX =~ "android" ]]; then
    LINK_FLAG="${LINK_FLAG} -llog -lz"
fi

echo "CXX: $CXX"
echo "MGE_INSTALL_PATH: $MGE_INSTALL_PATH"
echo "INCLUDE_FLAG: $INCLUDE_FLAG"
echo "LINK_FLAG: $LINK_FLAG"
echo "BUILD_FLAG: $BUILD_FLAG"

echo "[" > compile_commands.json
echo "{" >> compile_commands.json
echo "\"directory\": \"$PWD\"," >> compile_commands.json
echo "\"command\": \"$CXX yolox.cpp -o yolox ${INCLUDE_FLAG} ${LINK_FLAG}\"," >> compile_commands.json
echo "\"file\": \"$PWD/yolox.cpp\"," >> compile_commands.json
echo "}," >> compile_commands.json
echo "]" >> compile_commands.json
$CXX yolox.cpp -o yolox ${INCLUDE_FLAG} ${LINK_FLAG} ${BUILD_FLAG}

echo "build success, output file: yolox"
if [[ $CXX =~ "android" ]]; then
    echo "try command to run:"
    echo "adb push/scp $MGE_INSTALL_PATH/lib/libmegengine.so android_phone"
    echo "adb push/scp $OPENCV_INSTALL_LIB_PATH/*.so android_phone"
    echo "adb push/scp ./yolox yolox_s.mge android_phone"
    echo "adb push/scp ../../../assets/dog.jpg android_phone"
    echo "adb/ssh to android_phone, then run: LD_LIBRARY_PATH=. ./yolox yolox_s.mge dog.jpg cpu/multithread <warmup_count> <thread_number> <use_fast_run> <use_weight_preprocess>"
else
    echo "try command to run: LD_LIBRARY_PATH=$MGE_INSTALL_PATH/lib/:$OPENCV_INSTALL_LIB_PATH ./yolox yolox_s.mge ../../../assets/dog.jpg cuda/cpu/multithread <warmup_count> <thread_number> <use_fast_run> <use_weight_preprocess>"
fi
