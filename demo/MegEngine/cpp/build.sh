#!/usr/bin/env bash
set -e

if [ -z $CXX ];then
    echo "please export you c++ toolchain to CXX"
    echo "for example:"
    echo "build for host:                                        export CXX=g++"
    echo "cross build for aarch64-android(always locate in NDK): export CXX=aarch64-linux-android29-clang++"
    echo "cross build for aarch64-linux:                         export CXX=aarch64-linux-gnu-g++"
    exit -1
fi

if [ -z $MGE_INSTALL_PATH ];then
    echo "please export megengine install dir to MGE_INSTALL_PATH"
    echo "please refs https://github.com/MegEngine/MegEngine/blob/master/scripts/cmake-build/BUILD_README.md to build MegEngine"
    exit -1
fi

if [ -z $OPENCV_INSTALL_INCLUDE_PATH ];then
    echo "please export opencv install include dir to OPENCV_INSTALL_INCLUDE_PATH"
    echo "please refs https://github.com/opencv/opencv to build sdk or download prebuild sdk from https://github.com/opencv/opencv/releases"
    exit -1
fi

if [ -z $OPENCV_INSTALL_LIB_PATH ];then
    echo "please export opencv install libs dir to OPENCV_INSTALL_LIB_PATH"
    echo "please refs https://github.com/opencv/opencv to build sdk or download prebuild sdk from https://github.com/opencv/opencv/releases"
    exit -1
fi

INCLUDE_FLAG="-I$MGE_INSTALL_PATH/include -I$OPENCV_INSTALL_INCLUDE_PATH"
LINK_FLAG="-L$MGE_INSTALL_PATH/lib/ -lmegengine -L$OPENCV_INSTALL_LIB_PATH -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs"
BUILD_FLAG="-static-libstdc++ -O3 -pie -fPIE -g"

echo "CXX: $CXX"
echo "MGE_INSTALL_PATH: $MGE_INSTALL_PATH"
echo "INCLUDE_FLAG: $INCLUDE_FLAG"
echo "LINK_FLAG: $LINK_FLAG"
echo "BUILD_FLAG: $BUILD_FLAG"

echo "$CXX yolox.cpp -o yolox ${INCLUDE_FLAG} ${LINK_FLAG}" > compile_commands.json
$CXX yolox.cpp -o yolox ${INCLUDE_FLAG} ${LINK_FLAG}

echo "build success, output file: yolox"
echo "try command to run: LD_LIBRARY_PATH=$MGE_INSTALL_PATH/lib/:$OPENCV_INSTALL_LIB_PATH ./yolox yoloxs.mge dog.jpg cuda/cpu/multithread"
