import cv2
import numpy as np
import json
import os

def mouse_callback(event, x, y, flags, param):
    """
    鼠标回调函数。
    
    参数:
        event: 鼠标事件类型
        x: 鼠标光标的X坐标
        y: 鼠标光标的Y坐标
        flags: 鼠标事件标志
        param: 用户数据
    """
    global positive_points, negative_points, frame, current_frame_points, current_frame_index, cap, scale_factor
    
    if event == cv2.EVENT_LBUTTONDOWN:
        positive_points.append((x, y))
        current_frame_points["positive"].append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        negative_points.append((x, y))
        current_frame_points["negative"].append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # 滚轮向上，切换到上一帧
            if current_frame_index > 0:
                current_frame_index -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                current_frame_points = {"positive": [], "negative": []}
        else:  # 滚轮向下，切换到下一帧
            if current_frame_index < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1:
                current_frame_index += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                current_frame_points = {"positive": [], "negative": []}

def get_points_from_video(video_path: str, json_dir: str):
    """
    从视频文件中获取点坐标。
    
    参数:
        video_path: 输入视频文件的路径
        json_dir: 保存JSON文件的目录
    """
    global positive_points, negative_points, frame, scale_factor, current_frame_points, frame_data, current_frame_index, cap
    
    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return      
    
    # 获取总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 读取第一帧  
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from: {video_path}")
        cap.release()
        return
    
    # 检查并根据需要缩放帧大小
    max_width, max_height = 720, 1200
    height, width = frame.shape[:2]
    scale_factor = 1.0
    
    if width > max_width or height > max_height:
        scale_factor = min(max_width / width, max_height / height)
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # 设置鼠标回调
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)
    
    # 创建带滑块的控件窗口
    cv2.namedWindow("Control")
    cv2.createTrackbar("Frame", "Control", 0, total_frames - 1, lambda x: None)
    
    # 初始化帧数据
    frame_data = {}
    current_frame_index = 0
    current_frame_points = {"positive": [], "negative": []}
    
    # 主循环
    while True:
        # 显示当前帧
        cv2.imshow("Video", frame)
        
        # 更新滑块位置
        cv2.setTrackbarPos("Frame", "Control", current_frame_index)
        
        # 处理按键事件
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # 退出
            break
        elif key == ord('s'):  # 提交当前帧点坐标
            if current_frame_points["positive"] or current_frame_points["negative"]:
                frame_data[current_frame_index] = {
                    "positive": current_frame_points["positive"],
                    "negative": current_frame_points["negative"]
                }
                print(f"Frame {current_frame_index} points submitted:")
                print(f"Positive points: {current_frame_points['positive']}")
                print(f"Negative points: {current_frame_points['negative']}")
                current_frame_points = {"positive": [], "negative": []}
        elif key == ord('c'):  # 清除当前帧点坐标
            current_frame_points = {"positive": [], "negative": []}
            positive_points.clear()
            negative_points.clear()
            print(f"Frame {current_frame_index} points cleared")
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
            ret, frame = cap.read()
            if scale_factor != 1.0:
                frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        elif key == 81:  # 左箭头键 - 上一帧
            if current_frame_index > 0:
                current_frame_index -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                current_frame_points = {"positive": [], "negative": []}
        elif key == 83:  # 右箭头键 - 下一帧
            if current_frame_index < total_frames - 1:
                current_frame_index += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                current_frame_points = {"positive": [], "negative": []}
        
        # 检查滑块位置
        trackbar_pos = cv2.getTrackbarPos("Frame", "Control")
        if trackbar_pos != current_frame_index:
            current_frame_index = trackbar_pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
            ret, frame = cap.read()
            if scale_factor != 1.0:
                frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            current_frame_points = {"positive": [], "negative": []}
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 如果帧被缩放，将点坐标缩放回原始尺寸
    for frame_idx in frame_data:
        if scale_factor != 1.0:
            frame_data[frame_idx]["positive"] = [(int(x / scale_factor), int(y / scale_factor)) for x, y in frame_data[frame_idx]["positive"]]
            frame_data[frame_idx]["negative"] = [(int(x / scale_factor), int(y / scale_factor)) for x, y in frame_data[frame_idx]["negative"]]
    
    # 将点坐标保存到JSON文件
    name, _ = os.path.splitext(os.path.basename(video_path))
    json_path = os.path.join(json_dir, f"{name}.json")

    with open(json_path, "w") as f:
        json.dump(frame_data, f)
    
    print(f"Points saved to {json_path}")
    
    # 清除点坐标以便处理下一个视频
    positive_points.clear()
    negative_points.clear()
    frame_data = {}
    current_frame_points = {"positive": [], "negative": []}

def process_videos(video_path: str, json_dir: str):
    """
    处理目录中的所有视频文件或单个视频文件。
    
    参数:
        video_path: 输入视频文件或目录的路径
        json_dir: 保存JSON文件的目录
    """
    if os.path.isfile(video_path):
        get_points_from_video(video_path, json_dir)
    elif os.path.isdir(video_path):
        for filename in os.listdir(video_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                file_path = os.path.join(video_path, filename)
                get_points_from_video(file_path, json_dir)
    else:
        print(f"Error: Invalid path: {video_path}")

def main():
    video_path = r"D:\AI_Graph\视频\输入\MultiScene.mp4"
    json_dir = r"D:\AI_Graph\视频\输入\sam坐标"
    process_videos(video_path, json_dir)   

# 全局变量
positive_points = []
negative_points = []
frame = None
scale_factor = 1.0
current_frame_points = {"positive": [], "negative": []}
frame_data = {}
current_frame_index = 0
cap = None

if __name__ == "__main__":
    main()
