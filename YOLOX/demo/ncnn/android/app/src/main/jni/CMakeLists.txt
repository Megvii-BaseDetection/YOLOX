project(yoloXncnn)

cmake_minimum_required(VERSION 3.4.1)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210525-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yoloXncnn SHARED yoloXncnn_jni.cpp)

target_link_libraries(yoloXncnn
    ncnn

    jnigraphics
)
