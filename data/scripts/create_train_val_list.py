# coding:utf-8

"""
按照一定的比例划分数据集为trainval, train, val, test
"""

import os
import random


trainval_percent = 0.98
train_percent = 0.95
basepath = "/2T/001_AI/3001_YOLOv5_JDE/002_Datasets"
xmlfilepath = "DroneView_Vehicle_20221028/Annotations"

txtsavepath = "DroneView_Vehicle_20221028/ImageSets/Main/"

# 如果ImageSets不存在，则创建
if os.path.exists(os.path.join(basepath, txtsavepath)) == False:
    os.makedirs(os.path.join(basepath, "DroneView_Vehicle_20221028/ImageSets"))
    os.makedirs(os.path.join(basepath, "DroneView_Vehicle_20221028/ImageSets/Main"))


total_xml = os.listdir(os.path.join(basepath, xmlfilepath))


num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(os.path.join(basepath, txtsavepath) + 'trainval.txt', 'w')
file_test = open(os.path.join(basepath, txtsavepath) + 'test.txt', 'w')
file_train = open(os.path.join(basepath, txtsavepath) + 'train.txt', 'w')
file_val = open(os.path.join(basepath, txtsavepath) + 'val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
