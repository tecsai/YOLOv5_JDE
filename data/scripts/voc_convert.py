# import sys
# import imp
# imp.reload(sys)
# sys.setdefaultencoding("utf8")

"""
将VOC格式的标注文件转换为yolo-txt格式
class_id x-center y-center w h
class_id x-center y-center w h
class_id x-center y-center w h
...
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from xml.dom.minidom import parse

sets=['train', 'val']

""" DSAI """
### VOC
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# ### Power48
# classes = ["GD01", "GC01", "GC02", "GC03", "GC04", "GC05", "GX01", "GS01", "II01", "II02", "IS01", "IR01", "IX01", "IX02", "IX03", "IX04", "IS02", "IS03",
# "IX05", "IX06", "OL01", "OL02", "ON01", "IX07", "IX08", "IX09", "IX11", "IX12", "II03", "II04", "OT05", "GC06", "IS04", "IS05", "IR02", "IX10", "IX13",
# "IX14", "IX15", "IX16", "GC07", "IX17", "II05", "IS06", "IX19", "II06", "GL01", "GL02"]

### ISR30Car
classes = ["car"]

### YIWEI-UAV
# classes = ["UAV01"]

### DroneView_Vehicle_20221028
# classes = ["hat", "person"]

### DroneView_Vehicle_20221028
# classes = ["ignored regions", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]

### DroneView_Vehicle_20221028
# classes = ["BL01", "DJ01", "LJ01", "FW01", "KD01", "MM01", "ZZ01", "XSG01", "crack"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(data_set, image_id):  # [DSAI]
    in_file = open('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/Annotations/%s.xml'%(image_id))
    out_file = open('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/%s/%s.txt'%(data_set, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        if(bb[0]>=0 and bb[1]>=0 and bb[2]>=0 and bb[3]>=0):
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


"""
创建images和labels文件夹，并在images文件夹中设置(生成)train.txt和val.txt文件
"""
for image_set in sets:  # [DSAI]
    if not os.path.exists('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/'):
        os.makedirs('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/')
    if not os.path.exists('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/train/'):
        os.makedirs('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/train/')
    if not os.path.exists('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/val/'):
        os.makedirs('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/val/')
    if not os.path.exists('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/images/'):
        os.makedirs('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/images/')
    if not os.path.exists('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/images/train/'):
        os.makedirs('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/images/train/')
    if not os.path.exists('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/images/val/'):
        os.makedirs('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/images/val/')

    image_ids = open('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/images/%s.txt'%(image_set), 'w')
    # print(list_file)
    for image_id in image_ids:
        list_file.write('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/images/%s.jpg\n'%(image_id))
        convert_annotation(image_set, image_id)
    list_file.close()