import os
import glob
import cv2
import shutil
def CopyFile(dir_s,format):
    i = 1
    c_count = 0
    for file in glob.glob(os.path.join(dir_s,"*"+format)):
        print(i,':: ',file)
        legth = len(format)
        file_name = file.split("/")[-1]
        print("file_name: ",file_name)
        file_jpg = file_name[:-legth]+".jpg"
        print("file_jpg: ",file_jpg)
        img = cv2.imread(os.path.join(dir_s,file_jpg))
        h,w,c = img.shape
        print(str(w)+"x"+str(h))
        folder  = os.path.join(dir_s, "("+str(w)+"x"+str(h)+")")
        print(folder)
        if not os.path.exists(folder):
            print(c_count,"=============================")
            os.mkdir(folder)
        else:
            print("error!! can not create folder JPEGImages!!")
        shutil.copyfile(file, os.path.join(folder,file_name))
        shutil.copyfile(os.path.join(dir_s,file_jpg), os.path.join(folder,file_jpg))
        c_count = c_count+1
        i = i+1
    print('Finish Copying File!!')
    return
path = "/Users/julialu/Desktop/visdrone-3/valid"
CopyFile(path,".xml")
#CopyFile(path,"*.jpg",500)
