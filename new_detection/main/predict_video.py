import multiprocessing as mp
import sys
import argparse
import torch.utils
from torch.autograd import Variable
from model import *
import time
from multi_read_data import *
import torch.multiprocessing as mp
from video_utils import *
import os
import cv2

parser = argparse.ArgumentParser("detection")
parser.add_argument('--video_path', type=str, default="/Users/mayohoshi/Library/CloudStorage/OneDrive-个人/论文、报告、项目、大作业/Night-vehicle-detection-system/new_detection/video/input/test2.mp4", help='location of the input data')
parser.add_argument('--load_path', type=str, default='/Users/mayohoshi/Library/CloudStorage/OneDrive-个人/论文、报告、项目、大作业/Night-vehicle-detection-system/new_detection/video/output/enhance_test2_2.mp4', help='location of the merge data ')
parser.add_argument('--e_model', type=str, default="/Users/mayohoshi/Library/CloudStorage/OneDrive-个人/论文、报告、项目、大作业/Night-vehicle-detection-system/new_detection/weights/enhance_weights/medium.pt", help='enhacne_model')
parser.add_argument('--d_model', type=str, default="weights/detect_weights/yolov8s.pt", help='detect_model')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()

# 定义输出文件夹的路径
output_folders = ["video/output_folder_1/", "video/output_folder_2/", "video/output_folder_3/",
                  "video/output_folder_4/", "video/output_folder_5/", "video/output_folder_6/"]

enhance_folders = ["result/enhance/output_folder_1/", "result/enhance/output_folder_2/",
                   "result/enhance/output_folder_3/",
                   "result/enhance/output_folder_4/", "result/enhance/output_folder_5/",
                   "result/enhance/output_folder_6/"]

def enhance(data_path, model, save_path):
    os.makedirs(save_path, exist_ok=True)
    TestDataset = MemoryFriendlyLoader(img_dir=data_path, task='test')

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=False, num_workers=0)
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input)
            image_name = image_name[0].split('/')[-1].split('.')[0]
            im = model(input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))

            u_path = save_path + '/' + u_name
            im.save(u_path, "png")

def run(i_path, model, o_path):
    enhance(i_path, model, o_path)

def frame2video(frame_folders, width, height, fps, output_path):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # 使用 mp4v 编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for folder in frame_folders:
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".png"):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Failed to read image {img_path}")
                    continue
                out.write(img)

    out.release()
    print(f"Video successfully saved to {output_path}")

def main():
    mp.set_start_method('spawn')  # 设置进程启动方式
    s_t = time.time()

    # 检查文件是否存在
    if not os.path.exists(args.e_model):
        print(f"Error: {args.e_model} does not exist")
    else:
        print(f"File {args.e_model} exists")

    # 强制使用 CPU
    device = torch.device('cpu')
    print('Using CPU')

    # 加载模型时指定 map_location 参数
    model = Network(args.e_model).to(device)
    model.eval()
    dmodel = YOLO(args.d_model)  # load an official model
    
    # 定义视频文件路径
    video_path = args.video_path
    # 定义输出图片的路径
    out_frame_path = "video/output_frames/"

    # 合成视频的路径
    load_path = args.load_path

    # v-f
    width, height, fps = video2frame(video_path, out_frame_path, output_folders)
    print(f"Video to frame conversion done: width={width}, height={height}, fps={fps}")
    i_path1 = output_folders[0]
    i_path2 = output_folders[1]
    i_path3 = output_folders[2]
    i_path4 = output_folders[3]
    i_path5 = output_folders[4]
    i_path6 = output_folders[5]

    o_path1 = enhance_folders[0]
    o_path2 = enhance_folders[1]
    o_path3 = enhance_folders[2]
    o_path4 = enhance_folders[3]
    o_path5 = enhance_folders[4]
    o_path6 = enhance_folders[5]
    run(i_path1, model, o_path1)
    run(i_path2, model, o_path2)
    run(i_path3, model, o_path3)
    run(i_path4, model, o_path4)
    run(i_path5, model, o_path5)
    run(i_path6, model, o_path6)

    frame2video(enhance_folders, width, height, fps, load_path)
    print(f"Video saved to {load_path}")

    # 检查合成视频文件是否存在
    if not os.path.exists(load_path):
        print(f"Error: {load_path} does not exist")
        return

    results = dmodel.predict(load_path, save=True, show=False, project="result/predict",
                            name="video")  # predict on an image
    print("Total Cost Time:", time.time() - s_t)

if __name__ == '__main__':
    main()