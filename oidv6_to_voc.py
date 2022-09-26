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
        txtfile = os.path.join(args.sourcepath, 'labels', fname + '.txt') #Read annotation of each image from txt file
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

            ### bodycloth
            if(x[0] == 'human_eye'):
                x[0] = 'eye'
            elif(x[0] == 'human_beard'):
                x[0] = 'beard'
            elif(x[0] == 'human_mouth'):
                x[0] = 'mouth'
            elif(x[0] == 'human_body'):
                x[0] = 'body'
            elif(x[0] == 'human_foot'):
                x[0] = 'foot'
            elif(x[0] == 'human_leg'):
                x[0] = 'leg'
            elif(x[0] == 'human_ear'):
                x[0] = 'ear'
            elif(x[0] == 'human_hair'):
                x[0] = 'hair'
            elif(x[0] == 'human_head'):
                x[0] = 'head'
            elif(x[0] == 'human_face'):
                x[0] = 'face'
            elif(x[0] == 'human_arm'):
                x[0] = 'arm'
            elif(x[0] == 'human_nose'):
                x[0] = 'nose'
            elif(x[0] == 'human_hand'):
                x[0] = 'hand'
            elif(x[0] == 'brassiere'):
                x[0] = 'bra'
            elif(x[0] == 'miniskirt'):
                x[0] = 'skirt'

            ### food
            elif(x[0] == 'common_fig'):
                x[0] = 'fig'
            elif(x[0] == 'egg_(food)'):
                x[0] = 'egg'
            elif(x[0] == 'garden_asparagus'):
                x[0] = 'asparagus'
            elif(x[0] == 'submarine_sandwich'):
                x[0] = 'sandwich'
                
            # indoor
            elif(x[0] == 'loveseat'):
                x[0] = 'couch'
            elif(x[0] == 'studio_couch'):
                x[0] = 'couch'
            elif(x[0] == 'kitchen_&_dining_room_table'):
                x[0] = 'table'
            elif(x[0] == 'computer_keyboard'):
                x[0] = 'keyboard'
            elif(x[0] == 'computer_mouse'):
                x[0] = 'mouse'
            elif(x[0] == 'power_plugs_and_sockets'):
                x[0] = 'power socket'

            # outdoor
            elif(x[0] == 'waste_container'):
                x[0] = 'garbage can'
            elif(x[0] == 'vehicle_registration_plate'):
                x[0] = 'license plate'

            # animal
            elif(x[0] == 'bat_(animal)'):
                x[0] = 'Bat'
            elif(x[0] == 'jaguar_(animal)'):
                x[0] = 'Jaguar'


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


