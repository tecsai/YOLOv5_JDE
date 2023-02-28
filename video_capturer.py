import os
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import numpy as np
import cv2


def open_video(video_path):
    videoCapture = cv2.VideoCapture(video_path)
    # obtain resolution of source video
    FPS = videoCapture.get(cv2.CAP_PROP_FPS)
    SIZE = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    FRAME_COUNT = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS: ", FPS)
    print("SIZE: ", SIZE)
    print("FRAME_COUNT: ", FRAME_COUNT)
    return videoCapture, FPS, SIZE, FRAME_COUNT



if __name__=="__main__":
    original_video_res = '/2T/001_AI/009_YOLOv4/005_EvalRepo/helmet_demo.mp4'
    vc, fps, size, frames = open_video(original_video_res)

    for i in range(int(frames)):
        ret, frame = vc.read()
        # print("/2T/001_AI/009_YOLOv4/005_EvalRepo/helmet_demo/"+"%06d.jpg"%(i))
        cv2.imwrite("/2T/001_AI/009_YOLOv4/005_EvalRepo/helmet_demo/"+"%06d.jpg"%(i), frame)


