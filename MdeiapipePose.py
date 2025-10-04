import cv2
import mediapipe as mp
import numpy as np
import os

class MediaPipePoseDetector:
    def __init__(self, enable_pose=True, enable_hands=True, enable_face=True):
        """
        初始化姿势检测器
        :param enable_pose: 是否开启骨骼检测
        :param enable_hands: 是否开启手指检测
        :param enable_face: 是否开启面部检测
        """
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        self.enable_face = enable_face

        # 初始化mediapipe模块
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh

        # 初始化检测器
        self.pose = self.mp_pose.Pose(
            model_complexity=1,  # 更高的模型复杂度，提高检测精度
            smooth_landmarks=True,  # 平滑关键点
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) if enable_pose else None
        self.hands = self.mp_hands.Hands() if enable_hands else None
        self.face = self.mp_face.FaceMesh() if enable_face else None

    def process_video(self, input_path, output_path):
        """
        处理视频文件，检测姿势并保存为黑底视频
        :param input_path: 输入视频路径
        :param output_path: 输出视频路径
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        # 获取视频参数
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 创建黑底视频输出
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 创建黑底画布
            black_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # 检测姿势
            if self.enable_pose:
                results_pose = self.pose.process(frame_rgb)
                if results_pose.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        black_canvas, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # 检测手指
            if self.enable_hands:
                results_hands = self.hands.process(frame_rgb)
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            black_canvas, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # 检测面部
            if self.enable_face:
                results_face = self.face.process(frame_rgb)
                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            black_canvas, face_landmarks, self.mp_face.FACEMESH_CONTOURS)

            # 写入输出视频
            out.write(black_canvas)

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# 示例用法
if __name__ == "__main__":

    input_path = r"D:\AI_Graph\视频\输入\原视频_16fps\她命真好一句新赛季不会玩你连夜上号打排位我问你可以带我吗你说鼠标吃老鼠药挂了 瓦学弟 瓦学姐 cos 甜妹 - 抖音.mp4"

    index = 0
    out_name = str(index) + "Pose.mp4"
    output_path = f"D:\AI_Graph\视频\输入\pose\{out_name}"
    detector = MediaPipePoseDetector(enable_pose=True, enable_hands=True, enable_face=False)
    detector.process_video(input_path, output_path)