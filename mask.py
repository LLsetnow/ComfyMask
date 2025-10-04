"""
图像分割工具模块

该模块提供基于 MediaPipe 和 YOLOv8 的图像分割功能，支持人物分割和脸部分割。

主要功能:
- MediaPipeSegmenter: 基于 MediaPipe 实现的人物和脸部分割。
- YOLOv8Segmenter: 基于 YOLOv8 实现的人物分割和近似脸部分割。

类说明:
1. MediaPipeSegmenter:
   - get_person_mask: 获取人物分割遮罩。
   - get_face_mask: 获取精确的脸部分割遮罩。
   - release: 释放资源（可选）。

2. YOLOv8Segmenter:
   - get_person_mask: 获取人物分割遮罩。
   - get_face_mask: 获取近似脸部分割遮罩（基于人物框上半部分）。
   - get_detailed_face_mask: 结合外部检测器获取更精确的脸部遮罩。

使用示例:
    from mask import MediaPipeSegmenter, YOLOv8Segmenter

    # 使用 MediaPipe 分割
    segmenter = MediaPipeSegmenter()
    person_mask = segmenter.get_person_mask(image_rgb)
    face_mask = segmenter.get_face_mask(image_rgb)

    # 使用 YOLOv8 分割
    segmenter = YOLOv8Segmenter(model_path='yolov8n-seg.pt')
    person_mask = segmenter.get_person_mask(image_rgb)
    face_mask = segmenter.get_face_mask(image_rgb)

注意事项:
- MediaPipe 适用于实时应用  ，且精度较低，YOLOv8 适用于高精度需求，且精度较高。
- 确保输入图像为 RGB 格式。
- YOLOv8 需要额外安装 ultralytics 和 torch。

日期: 2025-10-02
"""
import cv2
import os
from cv2.gapi import mask
import torch
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import shutil
from sam2.build_sam import build_sam2_video_predictor
# from ultralytics import YOLO

class MediaPipeSegmenter:
    """MediaPipe分割器类"""

    def __init__(self, model_selection=0, face_detection_confidence=0.5):
        # 初始化MediaPipe模型
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        mp_face_mesh = mp.solutions.face_mesh

        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=face_detection_confidence
        )

    def get_person_mask(self, image_rgb, threshold=0.3):
        """获取人物分割遮罩"""
        # 进行人物分割
        results_seg = self.selfie_segmentation.process(image_rgb)

        if results_seg.segmentation_mask is None:
            return None

        height, width = image_rgb.shape[:2]
        condition = results_seg.segmentation_mask > threshold
        person_mask = np.zeros((height, width), dtype=np.uint8)
        person_mask[condition] = 255

        return person_mask

    def get_face_mask(self, image_rgb):
        """获取脸部遮罩"""
        # 进行脸部网格检测
        results_face = self.face_mesh.process(image_rgb)

        height, width = image_rgb.shape[:2]
        face_mask = np.zeros((height, width), dtype=np.uint8)

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                # 获取所有关键点的像素坐标
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    landmarks.append((x, y))
                # 使用凸包来生成脸部遮罩
                hull = cv2.convexHull(np.array(landmarks))
                cv2.fillConvexPoly(face_mask, hull, 255)

        return face_mask

    def release(self):
        """释放资源（如果需要）"""
        # MediaPipe的模型通常不需要手动释放
        pass
class YOLOv8Segmenter:
    """YOLOv8分割器类"""

    def __init__(self, model_path='yolov8n-seg.pt', confidence_threshold=0.5, device='auto'):
        """
        初始化YOLOv8分割器

        参数:
        model_path: YOLOv8模型路径，默认为yolov8n-seg.pt（轻量版分割模型）
        confidence_threshold: 检测置信度阈值
        device: 运行设备，'auto'、'cpu'或'cuda'
        """
        # 加载YOLOv8模型
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device

        # 设置模型推理设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 人物类别ID（COCO数据集中person类别为0）
        self.person_class_id = 0

        print(f"YOLOv8模型已加载，使用设备: {self.device}")

    def get_person_mask(self, image_rgb, threshold=0.3):
        """获取人物分割遮罩"""
        # 使用YOLOv8进行推理
        results = self.model(image_rgb, device=self.device, conf=self.confidence_threshold, verbose=False)

        if len(results) == 0 or results[0].masks is None:
            return None

        height, width = image_rgb.shape[:2]
        person_mask = np.zeros((height, width), dtype=np.uint8)

        # 获取第一个结果（假设单张图片）
        result = results[0]

        # 检查是否有检测到的人物
        if result.boxes is not None and len(result.boxes) > 0:
            # 获取所有检测框的类别
            class_ids = result.boxes.cls.cpu().numpy()
            # 获取所有检测框的置信度
            confidences = result.boxes.conf.cpu().numpy()

            # 筛选出人物类别
            person_indices = [i for i, class_id in enumerate(class_ids)
                              if class_id == self.person_class_id and confidences[i] >= self.confidence_threshold]

            # 如果有检测到的人物，合并所有人物掩码
            if person_indices and result.masks is not None:
                for idx in person_indices:
                    # 获取单个实例的掩码
                    instance_mask = result.masks.data[idx].cpu().numpy()

                    # 调整掩码尺寸匹配原图
                    instance_mask = cv2.resize(instance_mask, (width, height))

                    # 应用阈值并合并到总掩码中
                    instance_mask_binary = (instance_mask > threshold).astype(np.uint8) * 255
                    person_mask = cv2.bitwise_or(person_mask, instance_mask_binary)

        return person_mask

    def get_face_mask(self, image_rgb):
        """
        获取脸部遮罩
        注意：YOLOv8分割模型不直接提供脸部分割，这里使用人物检测框的上半部分作为近似脸部区域
        或者可以结合其他面部检测方法
        """
        height, width = image_rgb.shape[:2]
        face_mask = np.zeros((height, width), dtype=np.uint8)

        # 使用YOLOv8进行推理
        results = self.model(image_rgb, device=self.device, conf=self.confidence_threshold, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return face_mask

        result = results[0]

        # 获取所有检测框的类别
        class_ids = result.boxes.cls.cpu().numpy()
        # 获取所有检测框的坐标
        boxes = result.boxes.xyxy.cpu().numpy()

        # 筛选出人物类别
        person_indices = [i for i, class_id in enumerate(class_ids)
                          if class_id == self.person_class_id]

        # 为每个人物创建近似脸部区域
        for idx in person_indices:
            x1, y1, x2, y2 = boxes[idx].astype(int)

            # 计算人物高度
            person_height = y2 - y1

            # 定义脸部区域为人物框的上半部分（可根据需要调整比例）
            face_ratio = 0.4  # 脸部占人物高度的比例
            face_height = int(person_height * face_ratio)

            # 计算脸部区域坐标
            face_x1 = x1
            face_y1 = y1
            face_x2 = x2
            face_y2 = y1 + face_height

            # 确保坐标在图像范围内
            face_x1 = max(0, face_x1)
            face_y1 = max(0, face_y1)
            face_x2 = min(width, face_x2)
            face_y2 = min(height, face_y2)

            # 在脸部掩码上绘制矩形区域
            cv2.rectangle(face_mask, (face_x1, face_y1), (face_x2, face_y2), 255, -1)

        return face_mask

    def get_detailed_face_mask(self, image_rgb, face_detector=None):
        """
        获取更精确的脸部遮罩
        可以结合其他面部检测方法（如MediaPipe）来提高精度

        参数:
        face_detector: 可选的外部面部检测器
        """
        # 如果没有提供外部面部检测器，使用默认的近似方法
        if face_detector is None:
            return self.get_face_mask(image_rgb)

        # 否则使用外部面部检测器
        height, width = image_rgb.shape[:2]
        face_mask = np.zeros((height, width), dtype=np.uint8)

        # 这里可以集成MediaPipe或其他面部检测方法
        # 例如，如果face_detector是MediaPipeFaceMesh实例:
        # results = face_detector.process(image_rgb)
        # 然后根据面部关键点生成掩码

        # 示例代码（需要根据实际使用的面部检测器调整）
        try:
            # 假设face_detector有process方法，返回面部关键点
            results = face_detector.process(image_rgb)
            if hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        landmarks.append((x, y))

                    # 使用凸包生成脸部遮罩
                    hull = cv2.convexHull(np.array(landmarks))
                    cv2.fillConvexPoly(face_mask, hull, 255)
        except Exception as e:
            print(f"使用外部面部检测器失败: {e}, 回退到默认方法")
            face_mask = self.get_face_mask(image_rgb)

        return face_mask

    def release(self):
        """释放资源"""
        # YOLOv8模型通常不需要手动释放，但可以清理GPU内存
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
class HybridSegmenter:
    """混合分割器，可以组合使用YOLOv8和MediaPipe"""

    def __init__(self, yolo_model_path='yolov8n-seg.pt', use_mediapipe_face=True):
        self.yolo_segmenter = YOLOv8Segmenter(yolo_model_path)
        self.use_mediapipe_face = use_mediapipe_face

        if use_mediapipe_face:
            # 初始化MediaPipe面部检测
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.1
            )
        else:
            self.face_mesh = None

    def get_person_mask(self, image_rgb, threshold=0.3):
        return self.yolo_segmenter.get_person_mask(image_rgb, threshold)

    def get_face_mask(self, image_rgb):
        if self.use_mediapipe_face and self.face_mesh is not None:
            return self.yolo_segmenter.get_detailed_face_mask(image_rgb, self.face_mesh)
        else:
            return self.yolo_segmenter.get_face_mask(image_rgb)

    def release(self):
        self.yolo_segmenter.release()
        if self.face_mesh is not None:
            self.face_mesh.close()
def process_frame_with_sam2(
    predictor,
    inference_state,
    frame: np.ndarray,
    points: list,
    labels: list,
    frame_idx: int) -> np.ndarray:
    """
    逐帧处理函数，返回当前帧的蒙版。
    
    Args:
        predictor: SAM2 预测器对象
        inference_state: 推理状态
        frame: 当前帧（RGB 格式）
        points: 用户选择的点坐标列表
        labels: 点标签（1=正样本，0=负样本）
        frame_idx: 当前帧索引
    """
    # 转换输入格式
    input_points = np.array(points, dtype=np.float32)
    input_labels = np.array(labels, dtype=np.int32)

    # 处理当前帧
    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=1,
        points=input_points,
        labels=input_labels,
    )

    # 生成蒙版
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return mask.astype(np.uint8) * 255
def process_video_with_sam2(video_path) -> np.ndarray:
    """
    逐帧处理函数，返回当前帧的蒙版。
    
    Args:
        video_path: 视频路径
    """
    # 初始化SAM
    inference_state = predictor.init_state(video_path=video_path) # 加载(跟踪)视频
    points = np.array(positive_points + negative_points, dtype=np.float32)
    labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
    # predictor.reset_state(inference_state) # 重置跟踪
    print(f"正样本点{len(positive_points)}个，负样本点{len(negative_points)}个")
    print(f"正样本点{positive_points}")
    print(f"负样本点{negative_points}")
    
    # 从frame_idx帧 增加跟踪点
    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=DrawPointFrame,
        obj_id=1,
        points=points,
        labels=labels,
    )
    # 双击扩大识别范围
    # _, _, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=0,
    #     obj_id=1,
    #     points=points,
    #     labels=labels,
    # )

    # 在整个视频中运行传播，并在字典中收集结果
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments
def combine_and_plot_masks(video_segments, frame_index):
    """
    将指定帧的所有mask合成并可视化
    :param video_segments: 嵌套字典结构的mask数据
    :param frame_index: 帧索引
    """
    if frame_index not in video_segments:
        print(f"Error: Frame index {frame_index} not found.")
        return

    frame_masks = video_segments[frame_index]
    if not frame_masks:
        print(f"Error: No masks found for frame {frame_index}.")
        return

    # 初始化合成mask（全False）
    combined_mask = None

    # 遍历所有obj_id的mask并合成
    for obj_id, mask in frame_masks.items():
        if combined_mask is None:
            combined_mask = mask.copy()
        else:
            combined_mask = np.logical_or(combined_mask, mask)

    # 移除多余的维度（例如从 (1, 1024, 576) 变为 (1024, 576)）
    combined_mask = np.squeeze(combined_mask)

    # 检查combined_mask是否为2D数组
    if combined_mask.ndim != 2:
        print(f"Error: Combined mask shape {combined_mask.shape} is invalid.")
        return
    return combined_mask
def load_points_from_json(video_path, json_folder = "D:\AI_Graph\视频\输入\sam坐标"):
    """
    从 json_folder文件夹内读取同名的json文件, 并从文件中读取正负点数据

    参数:
    json_folder: JSON 文件所在的文件夹路径
    video_path: 输入视频的路径

    返回:
    positive_points: 正点列表
    negative_points: 负点列表
    """
    import json
    import os

    # 从视频路径中提取文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(json_folder, f"{video_name}.json")

    # 初始化返回列表
    positive_points = []
    negative_points = []

    # 检查 JSON 文件是否存在
    if not os.path.exists(json_path):
        print(f"JSON 文件不存在: {json_path}")
        return positive_points, negative_points

    # 读取 JSON 文件内容
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取正负点数据
    positive_points = data.get("positive", [])
    negative_points = data.get("negative", [])

    return positive_points, negative_points
fram4samPoint_global = None
def click_event(event, x, y, flags, param):
    """
    鼠标点击事件处理函数
    """
    global fram4samPoint_global
    if fram4samPoint_global is None or fram4samPoint_global.size == 0:
        print("Error: fram4samPoint_global is invalid or empty.")
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        positive_points.append((x, y))
        cv2.circle(fram4samPoint_global, (x, y), 5, (0, 255, 0), -1)  # Green for positive points
        cv2.imshow('First Frame', fram4samPoint_global)
    elif event == cv2.EVENT_RBUTTONDOWN:
        negative_points.append((x, y))
        cv2.circle(fram4samPoint_global, (x, y), 5, (0, 0, 255), -1)  # Red for negative points
        cv2.imshow('First Frame', fram4samPoint_global)
def fill_above_min_y(face_mask):
    """
    将面部所在位置以上的所有像素点涂白

    参数:
    face_mask: 输入的面部遮罩（二值图像，白色为255，黑色为0）

    返回:
    处理后的遮罩
    """
    if np.max(face_mask) == 0:  # 如果没有检测到面部
        return face_mask

        # 找到所有白色像素点的坐标
    y_coords, x_coords = np.where(face_mask == 255)

    if len(y_coords) == 0:  # 如果没有白色像素点
        return face_mask

    # 找到坐标的边界值
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)

    y_mid = int((min_y + max_y) / 2)

    # 创建结果掩码，初始化为原始掩码
    result_mask = face_mask.copy()

    # 使用向量化操作创建条件区域
    y_indices, x_indices = np.ogrid[:face_mask.shape[0], :face_mask.shape[1]]
    condition = (y_indices < y_mid) & (x_indices > min_x) & (x_indices < max_x)

    # 将条件区域涂白
    result_mask[condition] = 255

    return result_mask
def fill_below_y(face_mask, y):
    """
    将面部遮罩中白色像素点的最大值以下的所有区域涂白

    参数:
    face_mask: 输入的面部遮罩（二值图像，白色为255，黑色为0）

    返回:
    处理后的遮罩
    """

    # 创建一个与原始遮罩相同大小的全白图像
    result_mask = np.ones_like(face_mask) * 255

    # 将最大y坐标以上的区域恢复为原始遮罩
    result_mask[:y + 1, :] = face_mask[:y + 1, :]

    return result_mask
def apply_dilation_and_squarization(mask, dilation_kernel_size, square_size):
    """
    对遮罩进行膨胀和方块化处理

    参数:
    mask: 输入遮罩
    dilation_kernel_size: 膨胀核大小
    square_size: 方块大小

    返回:
    处理后的遮罩
    """
    # 1. 膨胀处理
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    if dilation_kernel_size == 0:
        dilated_mask = mask
    else:
        dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)

    # 2. 方块化处理
    if square_size == 0:
        return dilated_mask

    # 创建一个与原始遮罩相同大小的全零矩阵
    squared_mask = np.zeros_like(dilated_mask)

    # 获取膨胀后遮罩的轮廓
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 获取轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 将矩形区域划分为square_size x square_size的方块
        for i in range(0, w, square_size):
            for j in range(0, h, square_size):
                # 计算当前方块的位置和大小
                block_x = x + i
                block_y = y + j
                block_w = min(square_size, w - i)
                block_h = min(square_size, h - j)

                # 检查当前方块区域内是否有足够的白色像素
                block_region = dilated_mask[block_y:block_y + block_h, block_x:block_x + block_w]
                white_pixels = np.sum(block_region == 255)
                total_pixels = block_w * block_h

                # 如果白色像素比例超过阈值，则填充整个方块
                if white_pixels / total_pixels > 0.1:  # 可调节的阈值
                    squared_mask[block_y:block_y + block_h, block_x:block_x + block_w] = 255

    return squared_mask
def detect_skin_mask(image_rgb):
    """
    输入: image_rgb (RGB格式的numpy数组)
    输出: mask (单通道，皮肤区域为255，其余为0)
    """
    # 将 RGB 转换到 HSV
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # 设定肤色范围 (HSV)，可以根据需要调整
    # 常用的人类皮肤色阈值
    lower_hsv = np.array([0, 40, 120], dtype=np.uint8)   # H, S, V 下限
    upper_hsv = np.array([30, 200, 220], dtype=np.uint8) # H, S, V 上限

    # 生成蒙板
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    return mask
def remove_small_components(mask, n):
    """
    输入:
        mask: 单通道二值图像 (白=255, 黑=0)
        n: 面积阈值，小于此值的连通域会被去掉
    输出:
        new_mask: 处理后的遮罩
    """
    # 确保 mask 是二值图
    mask = (mask > 0).astype(np.uint8) * 255

    # 查找连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 创建输出掩码
    new_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # 0是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= n:
            new_mask[labels == i] = 255

    return new_mask
def process_single_video(video_path, output_root, video_count, display = False):
    """
    核心代码
    处理单个视频，输出对应的遮罩视频
    """
    # 复制输入视频到 output_root 目录下
    origin_video_path = os.path.join(output_root, f"OriginVideo{video_count}.mp4")
    shutil.copy(video_path, origin_video_path)
    video_name, _ = os.path.splitext(os.path.basename(video_path))

    # person_mask_video = os.path.join(output_root, f"PersonMask{video_count}.mp4")
    # face_mask_video = os.path.join(output_root, f"FaceMask{video_count}.mp4")
    body_mask_video = os.path.join(output_root, f"BodyMask{video_count}.mp4")
    background_video = os.path.join(output_root, f"Background{video_count}.mp4")

    # 初始化分割器
    if USE_MEDIAPIPE:
        segmenter = MediaPipeSegmenter(model_selection=0, face_detection_confidence=FACE_THRESHOLD)
    if USE_YOLO:
        segmenter = YOLOv8Segmenter(model_path="D:\\AI_Graph\\视频\\遮罩视频\\model\\yolov8x-seg.pt", confidence_threshold=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out_person = cv2.VideoWriter(person_mask_video, fourcc, fps, (width, height), isColor=False)
    # out_face = cv2.VideoWriter(face_mask_video, fourcc, fps, (width, height), isColor=False)
    out_body = cv2.VideoWriter(body_mask_video, fourcc, fps, (width, height), isColor=False)
    out_final = cv2.VideoWriter(background_video, fourcc, fps, (width, height))

    # 如果使用sam识别主体
    if SAM_FLAG:
        # 读取sam识别点坐标
        if SAM_POINT_CLICK:
            # 窗口点选
            global fram4samPoint_global, positive_points, negative_points
            cap4samPoint = cv2.VideoCapture(video_path)
            cap4samPoint.set(cv2.CAP_PROP_POS_FRAMES, DrawPointFrame)
            ret, fram4samPoint_global = cap4samPoint.read()
            cv2.imshow('First Frame', fram4samPoint_global)
            cv2.setMouseCallback('First Frame', click_event)

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()
        else:
            # 读取json文件
            positive_points, negative_points = load_points_from_json(video_path)
        
        # # 初始化SAM
        # inference_state = predictor.init_state(video_path=video_path)
        # input_points = np.array(positive_points + negative_points, dtype=np.float32)
        # input_labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
        # predictor.reset_state(inference_state) # 重置跟踪

        # 获取所有帧的sam处理结果
        video_segments = process_video_with_sam2(video_path)
    
    try:
        
        # 处理所有帧
        for frame_count in tqdm(range(DrawPointFrame, frames), desc=f"Processing {video_name}", unit="frame"):
            ret, frame = cap.read()

            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if SAM_FLAG:
                # person_mask = process_frame_with_sam2(predictor, inference_state, image_rgb, input_points, input_labels, frame_count)
                person_mask = combine_and_plot_masks(video_segments, frame_count)
                person_mask = person_mask.astype(np.uint8) * 255
            else:
                person_mask = segmenter.get_person_mask(image_rgb, threshold=THRESHOLD)
            
            # 获取面部蒙版
            face_mask = segmenter.get_face_mask(image_rgb)
            if SKIN_DETECT:
                skin_mask = detect_skin_mask(image_rgb)
                skin_mask = remove_small_components(skin_mask, int(width * height / 150))

            if person_mask is None:
                continue
                
            # 对主体和面部蒙版进行膨胀和方块化
            person_mask = apply_dilation_and_squarization(
                person_mask, BODY_DILATION, BODY_SQUARE
            )
            face_mask = apply_dilation_and_squarization(
                face_mask, FACE_DILATION, FACE_SQUARE
            )

            # 清空头的上方区域（不识别）
            if UP_CLEAR:
                face_mask = fill_above_min_y(face_mask)
            # 强制识别下30%的区域
            if DRAW_DOWN:
                person_mask = fill_below_y(person_mask, int(height * 0.7))
            
            # 主体区域 - 面部区域 - 皮肤区域（如果开启）
            body_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(face_mask))
            if SKIN_DETECT:
                body_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(skin_mask))

            # 输出视频帧
            final_output = frame.copy()
            final_output[person_mask == 255] = 0
            # out_person.write(processed_person_mask)
            # out_face.write(face_mask)
            out_body.write(body_mask)
            out_final.write(final_output)

            # 窗口打印
            if display:
                # 创建一个可调整大小的窗口
                cv2.namedWindow('Final Output', cv2.WINDOW_NORMAL)
                
                # 设置窗口的初始大小（可选）
                cv2.resizeWindow('Final Output', width=480, height=832) 
                # 显示图像
                cv2.imshow('Final Output', final_output)
                
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    finally:
        cap.release()
        # out_person.release()
        # out_face.release()
        out_body.release()
        out_final.release()
        segmenter.release()
        cv2.destroyAllWindows()
def process_videos(input_dir, output_root, start_index = 0):
    """
    如果 input_dir 是视频文件 → 处理单个视频
    如果 input_dir 是文件夹 → 遍历处理所有视频
    """
    video_count = start_index
    if os.path.isfile(input_dir):
        if input_dir.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            process_single_video(input_dir, output_root, video_count, True)
        else:
            print(f"输入文件不是视频: {input_dir}")
    elif os.path.isdir(input_dir):
        video_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        for filename in tqdm(video_files, desc="Processing all videos", unit="video"):
            video_path = os.path.join(input_dir, filename)
            process_single_video(video_path, output_root, video_count, False)
            video_count += 1
    else:
        print(f"输入路径不存在: {input_dir}")



positive_points = []
negative_points = []
# 可调节参数
model_cfg = "D:\\AI_Graph\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_b+.yaml"
sam2_checkpoint = "D:\\AI_Graph\\sam2\\checkpoints\\sam2.1_hiera_base_plus.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"

# True False
FLAG_100FRAMS = True
DILATION_KERNEL_SIZE = 0  # 膨胀核大小，可调节膨胀程度
SQUARE_SIZE = 16  # 方块大小，可调节方块化程度
SAM_FLAG = True   # 是否使用sam识别主体
SAM_POINT_CLICK = True  # 是否使用sam点选

DrawPointFrame = 32  # 选择视频的第几帧画点
DRAW_DOWN = False  # 将下1/3区域全部图黑
UP_CLEAR = False    # 将头部上方清空
SKIN_DETECT = False # 去除皮肤部分

FACE_DILATION = 0
FACE_SQUARE = 32
BODY_DILATION = 0
BODY_SQUARE = 0

THRESHOLD = 0.1         # 身体阈值
FACE_THRESHOLD = 0.5    # 面部阈值
# 选择使用的分割器类型
USE_MEDIAPIPE = True  # 设置为False可以切换为其他分割器（如YOLO）
USE_YOLO = False

if SAM_FLAG:
    # 初始化 SAM2 预测器
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

if __name__ == "__main__":
    name = "不许笑 胜利女神新的希望 马斯特 泳装 - 抖音"
    input_dir = f"D:\AI_Graph\视频\输入\原视频_16fps\{name}.mp4"  # 可以是单个视频路径，也可以是文件夹路径
    output_root = r"D:\AI_Graph\视频\输入\输入视频整合"
    
    print("\n\n\n----------------------------------------------------------------------")
    print(f"将{input_dir}生成为遮罩, 输出到{output_root}")
    process_videos(input_dir, output_root, start_index=4)