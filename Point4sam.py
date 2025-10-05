"""
通过读取视频的第一帧。通过鼠标左键为正样本,右键为负样本,通过键盘的q键退出。并将样本点坐标写入json文件。
"""
import cv2
import numpy as np
import json
import os

def draw_star(image, center, color, size=15):
    """在图像上绘制五角星"""
    points = []
    for i in range(5):
        angle = np.pi * 2 * i / 5 - np.pi / 2
        x = center[0] + size * np.cos(angle)
        y = center[1] + size * np.sin(angle)
        points.append((int(x), int(y)))
        
        angle_inner = np.pi * 2 * (i + 0.5) / 5 - np.pi / 2
        x_inner = center[0] + size * 0.4 * np.cos(angle_inner)
        y_inner = center[1] + size * 0.4 * np.sin(angle_inner)
        points.append((int(x_inner), int(y_inner)))
    
    points = np.array(points, np.int32)
    cv2.fillPoly(image, [points], color)

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function.
    
    Args:
        event: Mouse event type
        x: X coordinate of the mouse cursor
        y: Y coordinate of the mouse cursor
        flags: Mouse event flags
        param: User data
    """
    global positive_points, negative_points, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        positive_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        negative_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)   

def get_points_from_video(video_path: str, json_dir: str):
    """
    Get points from a video file.
    
    Args:
        video_path: Path to the input video file
        json_dir: Directory to save the JSON file
    """
    global positive_points, negative_points, frame, scale_factor
    
    # Read video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return      
    
    # Read first frame  
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from: {video_path}")
        cap.release()
        return
    
    # Check and scale frame size if needed
    max_width, max_height = 720, 1200
    height, width = frame.shape[:2]
    scale_factor = 1.0
    
    if width > max_width or height > max_height:
        scale_factor = min(max_width / width, max_height / height)
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # Set up mouse callback
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)
    
    # Show video and wait for user input
    while True:
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Scale points back to original coordinates if frame was scaled
    if scale_factor != 1.0:
        positive_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in positive_points]
        negative_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in negative_points]
    
    # Save points to JSON file
    points_dict = {
        "positive": positive_points,
        "negative": negative_points
    }
    name, _ = os.path.splitext(os.path.basename(video_path))
    json_path = os.path.join(json_dir, f"{name}.json")

    with open(json_path, "w") as f:
        json.dump(points_dict, f)
    
    print(f"Points saved to {json_path}")
    
    # Clear points for next video
    positive_points.clear()
    negative_points.clear()    
    
def process_videos(video_path: str, json_dir: str):
    """
    Process all video files in a directory or a single video file.
    
    Args:
        video_path: Path to the input video file or directory
        json_dir: Directory to save the JSON files
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

def mark_points_on_video(video_path: str, json_dir: str, output_dir: str):
    """
    在视频第一帧上标记JSON文件中的点，并保存到输出目录
    
    Args:
        video_path: 视频文件路径
        json_dir: JSON文件目录
        output_dir: 输出图像目录
    """
    # 读取视频第一帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from: {video_path}")
        cap.release()
        return
    
    cap.release()
    
    # 读取JSON文件
    name, _ = os.path.splitext(os.path.basename(video_path))
    json_path = os.path.join(json_dir, f"{name}.json")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return
    
    with open(json_path, "r") as f:
        points_dict = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制五角星标记
    for point in points_dict.get("positive", []):
        draw_star(frame, point, (0, 255, 0))  # 绿色五角星
    
    for point in points_dict.get("negative", []):
        draw_star(frame, point, (0, 0, 255))  # 红色五角星
    
    # 保存标记后的图像
    output_path = os.path.join(output_dir, f"{name}.png")
    cv2.imencode('.png', frame)[1].tofile(output_path)
    print(f"Marked image saved to {output_path}")

def main():
    video_path = r"D:\AI_Graph\视频\输入\MultiScene.mp4"
    json_dir = r"D:\AI_Graph\视频\输入\sam坐标"
    output_dir = r"D:\AI_Graph\视频\输入\sam坐标"
    process_videos(video_path, json_dir) 
    mark_points_on_video(video_path, json_dir, output_dir)
    
# Global variables to store clicked points and scale factor
positive_points = []
negative_points = []
frame = None
scale_factor = 1.0

if __name__ == "__main__":
    main()