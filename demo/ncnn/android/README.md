# YOLOX-Android-ncnn

Andoird app of YOLOX object detection base on [ncnn](https://github.com/Tencent/ncnn)


## Tutorial

### Step1

Download ncnn-android-vulkan.zip from [releases of ncnn](https://github.com/Tencent/ncnn/releases). This repo uses
[20210525 release](https://github.com/Tencent/ncnn/releases/download/20210525/ncnn-20210525-android-vulkan.zip) for building.

### Step2

After downloading, please extract your zip file. Then, there are two ways to finish this step:
* put your extracted directory into **app/src/main/jni**
* change the **ncnn_DIR** path in **app/src/main/jni/CMakeLists.txt** to your extracted directory

### Step3
Download example param and bin file from [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ESXBH_GSSmFMszWJ6YG2VkQB5cWDfqVWXgk0D996jH0rpQ?e=qzEqUh) or [github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s_ncnn.tar.gz). Unzip the file to **app/src/main/assets**.

### Step4
Open this project with Android Studio, build it and enjoy!

## Reference

* [ncnn-android-yolov5](https://github.com/nihui/ncnn-android-yolov5)
