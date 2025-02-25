import cv2
import os
import shutil
import math
import re
import numpy as np


def video2frame(video_path, out_frame_path, output_folders):
    print("---------------数据开始解析！----------------")
    # 定义文件夹数量
    num_folders = len(output_folders)
    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出文件夹
    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if not os.path.exists(out_frame_path):
        os.makedirs(out_frame_path)

    # 逐帧读取视频并将帧保存为图片
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            # 将帧保存为图片
            cv2.imwrite(out_frame_path + "frame{:04d}.jpg".format(i), frame)
        else:
            break

    # 将所有图片按照编号从小到大排序
    sorted_frames = sorted(os.listdir(out_frame_path))

    # 计算每个文件夹应该包含的图片数量
    num_frames = [math.floor(total_frames / num_folders)] * num_folders
    remainder = total_frames % num_folders
    for i in range(remainder):
        num_frames[i] += 1

    # 将所有图片分为六部分，分别保存到六个文件夹中
    start = 0
    for i in range(len(output_folders)):
        end = start + num_frames[i]
        folder = output_folders[i]
        for j in range(start, end):
            # 将图片从原路径移动到新路径
            shutil.move(out_frame_path + sorted_frames[j], folder + sorted_frames[j])
        start = end
    print("---------------数据解析完成！----------------")
    return width, height, fps


def frame2video(output_folders,height, width, fps,load_path):
    # 读取所有文件夹中的图片
    all_frames = []
    for folder in output_folders:
        frames = os.listdir(folder)
        frames = sorted(frames)
        frames = [folder + frame for frame in frames]
        all_frames.extend(frames)

    all_frames.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 最好再看看图片顺序对不

    frame_size = (height, width)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv版本是3
    videoWriter = cv2.VideoWriter(load_path, fourcc, fps, frame_size)
    count = 1
    for i in all_frames:
        im_name = i
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        count += 1
        if (count % 200 == 0):
            print(im_name)

    videoWriter.release()
    print("---------------Merge finish！----------------")
