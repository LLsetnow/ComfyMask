# 加载yolo10m模型，为视频添加检测点
from tabnanny import verbose
from unittest import TestResult
import cv2
import json
import os
from tqdm import tqdm
import numpy as np
from omegaconf import base
from ultralytics import YOLO
import onnxruntime


class YOLOProcessor:
    """
    使用YOLO模型处理视频，检测人体和人脸，并将结果写入JSON文件。
    """

    def __init__(self, video_path, output_json_dir, body_model_path, face_model_path, 
                frame_interval=1000, frame_sequence=None):
        """
        初始化YOLO处理器。

        Args:
            video_path (str): 视频文件路径或文件夹路径。
            output_json_path (str): 输出JSON文件路径。
            frame_interval (int, optional): 每隔多少帧处理一帧。默认为1（每帧都处理）。
            frame_sequence (list, optional): 指定帧序列。如果提供，则忽略frame_interval。
        """
        self.video_path = video_path
        self.output_json_dir = output_json_dir
        self.frame_interval = frame_interval
        self.frame_sequence = frame_sequence
        self.model_body = YOLO(body_model_path)  # 加载人体检测模型
        self.model_face = YOLO(face_model_path)  # 加载人脸检测模型
        self.results = {}

    def process_video(self, video_path):
        """
        处理视频并生成检测结果。

        Args:
            video_path (str, optional): 视频文件路径。如果未提供，则使用初始化时的路径。
        """
        base_name = os.path.basename(video_path)
        video_name, _ = os.path.splitext(base_name)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        frame_count = 0
        frame_data = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检查是否需要处理当前帧
            if self.frame_sequence is not None:
                if frame_count not in self.frame_sequence:
                    frame_count += 1
                    continue
            elif frame_count % self.frame_interval != 0:
                frame_count += 1
                continue


            current_frame_points = {"positive": [], "negative": []}
            current_box = None

            flag_face = True
            flag_body = True

            # 检测人脸
            face_results = self.model_face(frame, verbose=False)
            for result in face_results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取人脸边界框坐标
                if len(boxes) > 0:
                    x1, y1, x2, y2 = [int(coord) for coord in boxes[0]]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    face_down_y = y2
                    current_frame_points["negative"].append((center_x, center_y))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                else:
                    flag_face = False
                    current_frame_points["positive"].append((0, 0))
                    print(f"没有检测到人脸: {video_name}_{frame_count}")

            # 检测人体
            body_results = self.model_body(frame, verbose=False)
            max_conf = -1
            best_box = None
            for result in body_results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
                confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        if confidences[i] > max_conf:
                            max_conf = confidences[i]
                            best_box = [int(coord) for coord in box.tolist()]
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                if flag_face:
                    y1 = face_down_y
                current_box = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                flag_body = False
                current_box = (0, 0, 0, 0)
                print(f"没有检测到人体: {video_name}_{frame_count}")
            
            frame_data[frame_count] = {
                "positive": current_frame_points["positive"],
                "negative": current_frame_points["negative"],
                "box": current_box
                }
            
            # cv2.imshow("Detection", frame)
            # cv2.waitKey(1)
            # 保存当前帧识别图像
            file_path = os.path.join(self.output_json_dir, f"{video_name}_{frame_count}.jpg")
            cv2.imencode('.jpg', frame)[1].tofile(file_path)
            frame_count += 1

        self.results = frame_data
        cap.release()
        cv2.destroyAllWindows()

    def save_results(self):
        """
        将检测结果保存到JSON文件。
        """
        with open(self.output_json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
            self.results = {}

    def run(self):
        """
        运行处理流程。
        """
        if os.path.isfile(self.video_path):
            self.process_video(self.video_path)
            base_name = os.path.basename(self.video_path)
            video_name, _ = os.path.splitext(base_name)
            self.output_json_path = os.path.join(self.output_json_dir, f"{video_name}.json")
            self.save_results()
            print(f"检测结果已保存到: {self.output_json_path}")
        elif os.path.isdir(self.video_path):
            video_files = [f for f in os.listdir(self.video_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for video_file in tqdm(video_files, desc="为所有视频添加检测点和识别框", unit="video"):
                video_path = os.path.join(self.video_path, video_file)
                base_name = os.path.basename(video_path)
                video_name, _ = os.path.splitext(base_name)
                self.process_video(video_path)
                self.output_json_path = os.path.join(self.output_json_dir, f"{video_name}.json")
                self.save_results()
            print(f"检测结果已保存到: {self.output_json_dir}")
        else:
            raise ValueError(f"无效的路径: {self.video_path}")


if __name__ == "__main__":
    # 示例用法
    name = "我不是gay 胸肌 - 抖音"
    video_path = f"D:\AI_Graph\输入\原视频_16fps\{name}.mp4"  # 替换为实际视频路径
    video_path = r"D:\AI_Graph\输入\原视频_16fps"  # 替换为实际视频路径
    output_json_dir = "D:\AI_Graph\输入\sam坐标"  # 替换为输出JSON文件路径
    body_model_path = r"D:\AI_Graph\tools\checkpoints\yolo11l.pt"  # 替换为人体检测模型路径
    face_model_path = r"D:\AI_Graph\tools\checkpoints\face_yolov8m.pt"  # 替换为人脸检测模型路径
    frame_interval = 48  # 每隔n帧处理一帧
    frame_sequence = [0, 5, 10, 40, 70, 100, 150, 200, 250, 300]  # 或指定帧序列

    processor = YOLOProcessor(video_path, output_json_dir, body_model_path, face_model_path, 
                            frame_interval, frame_sequence)
    processor.run()