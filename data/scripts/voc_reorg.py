"""
将文件重新组织，复制到个文件夹中
"""

import shutil
import os

DatasetName = "DroneView_Vehicle_20221028"

f = open('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/' + DatasetName + '/images/train.txt', 'r')
lines = f.readlines()
for line in lines:
    FileName = line.split('/')[-1][:-1]
    SrcFile = "/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/" + DatasetName + "/JPEGImages/" + FileName
    if (os.path.exists(SrcFile)):
        print(SrcFile)
        os.system("cp "+ SrcFile + " /2T/001_AI/3001_YOLOv5_JDE/002_Datasets/" + DatasetName + "/images/train")


f = open('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/' + DatasetName + '/images/val.txt', 'r')
lines = f.readlines()
for line in lines:
    FileName = line.split('/')[-1][:-1]
    SrcFile = "/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/" + DatasetName + "/JPEGImages/" + FileName
    if (os.path.exists(SrcFile)):
        print(SrcFile)
        os.system("cp "+ SrcFile + " /2T/001_AI/3001_YOLOv5_JDE/002_Datasets/" + DatasetName + "/images/val")


# f = open('/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/val.txt', 'r')
# lines = f.readlines()
# for line in lines:
#     FileName = line.split('/')[-1][:-1]
#     SrcFile = "/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/" + FileName
#     SrcFile = SrcFile.replace('jpg', 'txt')
#     if (os.path.exists(SrcFile)):
#         os.system("cp "+ SrcFile + " /2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/labels/val")