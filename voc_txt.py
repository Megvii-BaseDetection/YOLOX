# https://github.com/roboflow-ai/YOLOX/blob/main/voc_txt.py
import os
import random
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("no directory specified, please input target directory")
    exit()

root_path = sys.argv[1]

xmlfilepath = root_path + 'VOC2007/Annotations/'
os.mkdir(xmlfilepath)
imagefilepath = root_path + 'VOC2007/JPEGImages/'
os.mkdir(imagefilepath)

# Move annotations to annotations folder
for filename in os.listdir(root_path):
    if filename.endswith('.xml'):
        with open(os.path.join(root_path, filename)) as f:
            Path(root_path + filename).rename(xmlfilepath + filename)

    if filename.endswith('.jpg'):
        with open(os.path.join(root_path, filename)) as f:
            Path(root_path + filename).rename(imagefilepath + filename)


txtsavepath = root_path + '/VOC2007/ImageSets/Main'

if not os.path.exists(root_path):
    print("cannot find such directory: " + root_path)
    exit()

if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

trainval_percent = 0.9
train_percent = 0.8
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size:", tv)
print("train size:", tr)

ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
