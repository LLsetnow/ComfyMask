import cv2
from cv2.gapi import video
import numpy as np
import mediapipe as mp
import time
import os
import json
import cv2
from tqdm import tqdm

from mask import MediaPipeSegmenter

def process_single_video(video_path, output_dir, video_count = 0):
    """处理单个视频"""
    # 初始化分割器
    segmenter = MediaPipeSegmenter()
    pose_video = os.path.join(output_dir, f"Pose{video_count}.mp4")
    video_name = os.path.basename(video_path)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_pose = cv2.VideoWriter(pose_video, fourcc, fps, (width, height))
    try:
        for frame_count in tqdm(range(frames), desc=f"Processing {video_name}", unit="frame"):
            ret, frame = cap.read()
            
            if not ret:
                break
            out_pose.write(segmenter.detect_pose_and_hands(frame))
    finally:
        cap.release()
        out_pose.release()
        segmenter.release()

def main():
    # 初始化分割器
    
    # 输入图像路径
    video_path = r"D:\AI_Graph\输入\原视频_16fps"  # 替换为你的输入图像路径
    output_dir = r"D:\AI_Graph\输入\输入视频整合"  # 替换为你的输出图像路径
    video_idx = 0
    if os.path.isdir(video_path):
        video_list = [os.path.join(video_path, file) for file in os.listdir(video_path) if file.endswith(".mp4")]
        for video_path in video_list:
            process_single_video(video_path, output_dir, video_idx)
            video_idx += 1
    else:
        process_single_video(video_path, output_dir, video_idx)


if __name__ == "__main__":
    main()