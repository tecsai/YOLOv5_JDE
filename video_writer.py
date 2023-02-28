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
    #获得码率及尺寸
    FPS = videoCapture.get(cv2.CAP_PROP_FPS)
    SIZE = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    FRAME_COUNT = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS: ", FPS)
    print("SIZE: ", SIZE)
    print("FRAME_COUNT: ", FRAME_COUNT)
    return videoCapture, FPS, SIZE, FRAME_COUNT

def save_video(img_path, vid_path):
    images = []
    eval_list = os.listdir(img_path)

    print("Totasl frames: ", len(eval_list))
    for image_id in range(1, len(eval_list)+1):
        image_path = os.path.join(img_path, '%06d.jpg'%image_id)
        print("CHECK: {}".format(image_path))
        if(os.path.exists(image_path) == False):
            print("{} is not exist.".format(image_path))
            break
        else:
            images.append(image_path)
    
    tmp_img = cv2.imread(images[0])
    h = tmp_img.shape[0]
    w = tmp_img.shape[1]
    fps = 30

    print("h = ", h)
    print("w = ", w)

    fourcc = 'mp4v'  # output video codec
    vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    
    for image in images:
        tmp_img = cv2.imread(image)
        new_h = tmp_img.shape[0]
        new_w = tmp_img.shape[1]
        if new_h != h or new_w != w:
            print("image: {} has different h or w")
            break
        vid_writer.write(tmp_img)

        
# 待合并的两个视频分辨率必须相同
def MergeVideo(EoVideo, DensityVideo, dest_path):
    total_frame = 0
    eo_vc, eo_fps, eo_size, eo_total_frame = open_video(EoVideo)
    ds_vc, ds_fps, ds_size, ds_total_frame = open_video(DensityVideo)
    if eo_total_frame > ds_total_frame:
        total_frame = ds_total_frame
    else:
        total_frame = eo_total_frame
    print("EO: {}, {}".format(eo_size[0], eo_size[1]))
    print("DS: {}, {}".format(ds_size[0], ds_size[1]))


    fourcc = 'mp4v'  # output video codec
    fps = 25
    vid_writer = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*fourcc), fps, (int(eo_size[0])*2, int(eo_size[1])))

    for frame in range(int(total_frame)):
        eo_ret, eo_frame = eo_vc.read()
        ds_ret, ds_frame = ds_vc.read()
        if eo_ret == True and ds_ret == True:
            merge_image = cv2.hconcat([eo_frame, ds_frame])
            # cv2.imwrite(os.path.join(dest_path, str(frame)+'.jpg'), merge_image)
            vid_writer.write(merge_image)




def Video_Predict(eval_repo_dir, eval_res_dir):
    '''
    Show one estimated density-map.
    root_dir: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cpu")
    model=CSRNetV2().to(device)

    ckpt = torch.load(model_param_path)
    model.load_state_dict(ckpt['model'], strict=False)
    # model.load_state_dict(torch.load(model_param_path))

    eval_videos = []
    eval_list = os.listdir(eval_repo_dir)
    for video in eval_list:
        eval_videos.append(os.path.join(eval_repo_dir, video))

    # img_trans = Compose([ToTensor(), Normalize(mean=[0.5,0.5,0.5], std=[0.225,0.225,0.225])])
    img_trans = Compose([ToTensor()])

    model.eval()
    
    for video in eval_videos:
        print("VIDEO: ", video)
        # fourcc = 'mp4v'  # output video codec
        vc, fps, size, total_frame = open_video(video)
        # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        #读帧
        ret, frame = vc.read()
        count = 0
        while ret :
            # cv2.imwrite(os.path.join(eval_res_dir, str(count)+'.jpg'), frame)
            img = frame[:, :, ::-1].copy()  # BGR2RGB
            img = img.transpose(2, 0, 1)  # HWC2CHW
            img = img.astype(np.float32)
            img /= 255.0
            img_tensor = torch.from_numpy(img)

            img_tensor = img_tensor.to(device)
            # forward propagation
            img_tensor = img_tensor.unsqueeze(0) #  # NCHW
            et_dmap = model(img_tensor).detach()
            
            et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            total_num = et_dmap.sum()
            plt.imsave(os.path.join(eval_res_dir, str(count)+'.jpg'), et_dmap, cmap=CM.jet)

            print("frame_id: ", count)
            print("===> et_dmap.sum = ", total_num)

            """ 读取下一帧 """
            ret, frame = vc.read() #获取下一帧
            count += 1

        # img = Image.open(os.path.join(eval_repo_dir, image))
        # print("img.shape: ", np.array(img).shape)
        # img = img_trans(img) # 2CHW, 2Float, 2Normalization
        # print("img.shape: ", img.shape)
        # print("img.numpy.shape: ", img.numpy().shape)

        # """ [SAI-KP]保存数据值txt文件，用于部署时比对调试 """
        # # tempf = img.numpy().reshape(1280*720*3)
        # # np.savetxt("image.txt", tempf, fmt='%10.6f')



        # original_img = img
        # print("original img.shape: ", img.shape)

        # img = img.to(device)
        # # forward propagation
        # img = img.unsqueeze(0) #  # NCHW
        # et_dmap = model(img).detach()
        
        # et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
        # total_num = et_dmap.sum()
        # plt.imsave(os.path.join(eval_res_dir, image), et_dmap, cmap=CM.jet)

        # print("image: ", image)
        # print("===> et_dmap.sum = ", total_num)




if __name__=="__main__":
    eval_images_res = '/2T/001_AI/3001_YOLOv5_JDE/006_EvalRes/DroneView_Vehicle_20221028/Group_1'
    eval_video_res = '/2T/001_AI/3001_YOLOv5_JDE/006_EvalRes/DroneView_Vehicle_20221028/DroneView_Vehicle_20221028_Demo.mp4'
    # 推理
    # Video_Predict(eval_repo, eval_images_res) 
    # 保存为视频
    save_video(eval_images_res, eval_video_res)

