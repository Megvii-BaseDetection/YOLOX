'''Python Code to Convert OpenImage Dataset into VOC XML format. 
Author: https://github.com/AtriSaxena
Please see Read me file to know how to use this file.
'''

from xml.etree.ElementTree import Element, SubElement, Comment
import xml.etree.cElementTree as ET
#from ElementTree_pretty import prettify
import cv2
import os
from pathlib import Path
from shutil import move
import argparse

parser = argparse.ArgumentParser(description = 'Convert OIDV4 dataset to VOC XML format')
parser.add_argument('--sourcepath',type = str, default = 'dataset/', help ='Path of class to convert')
parser.add_argument('--dest_path',type=str, required=True, default='Annotation/',help='Path of Dest XML files')
args = parser.parse_args()

ids = []
for file in os.listdir(args.sourcepath): #Save all images in a list
    filename = os.fsdecode(file)
    if filename.endswith('.jpg'):
        ids.append(filename[:-4])

for fname in ids: 
    myfile = os.path.join(args.dest_path,fname +'.xml')
    myfile = Path(myfile)
    if not myfile.exists(): #if file is not existing 
        txtfile = os.path.join(args.sourcepath, 'Label', fname + '.txt') #Read annotation of each image from txt file
        f = open(txtfile,"r")
        imgfile = os.path.join(args.sourcepath, fname +'.jpg')
        img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED) #Read image to get image width and height
        top = Element('annotation')
        child = SubElement(top,'folder')
        child.text = 'open_images_volume'

        child_filename = SubElement(top,'filename')
        child_filename.text = fname +'.jpg'

        child_path = SubElement(top,'path')
        child_path.text = '/mnt/open_images_volume/' + fname +'.jpg'

        child_source = SubElement(top,'source')
        child_database = SubElement(child_source, 'database')
        child_database.text = 'Unknown'

        child_size = SubElement(top,'size')
        child_width = SubElement(child_size,'width')
        child_width.text = str(img.shape[1])

        child_height = SubElement(child_size,'height')
        child_height.text = str(img.shape[0])

        child_depth = SubElement(child_size,'depth')
        if len(img.shape) == 3: 
            child_depth.text = str(img.shape[2])
        else:
            child_depth.text = '3'
        child_seg = SubElement(top, 'segmented')
        child_seg.text = '0'
        for x in f:     #Iterate for each object in a image. 
            x = list(x.split())
            child_obj = SubElement(top, 'object')

            child_name = SubElement(child_obj, 'name')
            child_name.text = x[0] #name

            child_pose = SubElement(child_obj, 'pose')
            child_pose.text = 'Unspecified'

            child_trun = SubElement(child_obj, 'truncated')
            child_trun.text = '0'

            child_diff = SubElement(child_obj, 'difficult')
            child_diff.text = '0'

            child_bndbox = SubElement(child_obj, 'bndbox')

            child_xmin = SubElement(child_bndbox, 'xmin')
            child_xmin.text = str(int(float(x[1]))) #xmin

            child_ymin = SubElement(child_bndbox, 'ymin')
            child_ymin.text = str(int(float(x[2]))) #ymin

            child_xmax = SubElement(child_bndbox, 'xmax')
            child_xmax.text = str(int(float(x[3]))) #xmax

            child_ymax = SubElement(child_bndbox, 'ymax')
            child_ymax.text = str(int(float(x[4]))) #ymax

        tree = ET.ElementTree(top)
        save = fname+'.xml'
        tree.write(save)
        move(fname+'.xml', myfile)
