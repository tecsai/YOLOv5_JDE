"""
指定ImgPath, 自动匹配标签路径
"""

import os

ImgPath = '/2T/001_AI/3001_YOLOv5_JDE/002_Datasets/DroneView_Vehicle_20221028/JPEGImages'
ImgFiles = os.listdir(ImgPath)

for file in ImgFiles:
    img_file = os.path.join(ImgPath, file)
    anno_file = img_file.replace("JPEGImages", "Annotations")
    anno_file = anno_file.replace("jpg", "xml")
    if not os.path.exists(anno_file):
        print(img_file)
        os.remove(img_file)