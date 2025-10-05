import cv2
import os
import shutil
import numpy as np
import json
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
class SamSegmenter:
    def __init__(self, model_cfg, sam2_checkpoint, video_path):
        """
        初始化视频分割器
        
        Args:
            predictor: SAM2预测器对象
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        self.video_path = video_path
        self.positive_points = []
        self.negative_points = []
        self.video_segments = {}
        self.frame = None
        self.scale_factor = 1.0

    def process_video_with_single_point(self):
        """
        使用单点模式处理视频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            video_segments: 包含所有帧分割结果的字典
        """
        # 获取样本点
        if SAM_POINT_CLICK:
            # 窗口点选
            global fram4samPoint_global, positive_points, negative_points
            cap4samPoint = cv2.VideoCapture(self.video_path)
            cap4samPoint.set(cv2.CAP_PROP_POS_FRAMES, StartFrame)
            ret, fram4samPoint_global = cap4samPoint.read()
            cv2.imshow('First Frame', fram4samPoint_global)
            cv2.setMouseCallback('First Frame', click_event)

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()
            self.positive_points = positive_points
            self.negative_points = negative_points
            print(f"正样本点{len(self.positive_points)}个，负样本点{len(self.negative_points)}个")
            print(f"正样本点{self.positive_points}")
            print(f"负样本点{self.negative_points}")
        else:
            # 从JSON文件读取
            self.load_points_from_json()         

        # 初始化预测器状态
        inference_state = self.predictor.init_state(video_path=self.video_path)
        points = np.array(self.positive_points + self.negative_points, dtype=np.float32)
        labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points), dtype=np.int32)
        
        print(f"正样本点{len(self.positive_points)}个，负样本点{len(self.negative_points)}个")
        print(f"正样本点{self.positive_points}")
        print(f"负样本点{self.negative_points}")
        
        _, _, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
        return 

    def process_video_with_dynamic_points(self, json_folder="D:\\AI_Graph\\视频\\输入\\sam坐标"):
        """
        根据JSON文件动态更新跟踪点处理视频
        
        Args:
            video_path: 视频文件路径
            json_folder: JSON文件目录
            
        Returns:
            video_segments: 包含所有帧分割结果的字典
        """
        # 创建临时目录
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 读取视频
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
            
        # 读取JSON文件
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        json_path = os.path.join(json_folder, f"{video_name}.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件未找到: {json_path}")
            
        with open(json_path, 'r') as f:
            points_data = json.load(f)
            
        # 提取所有帧
        frame_dir = os.path.join(temp_dir, "all_frames")
        os.makedirs(frame_dir, exist_ok=True)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(frame_dir, f"{frame_count:06d}.jpg"), frame)
            frame_count += 1
        cap.release()
        
        frame_numbers = sorted([int(k) for k in points_data.keys()])
        
        # 处理每个关键帧区间
        for i, frame_idx in enumerate(frame_numbers):
            # 创建区间目录
            interval_dir = os.path.join(temp_dir, f"interval_{i}")
            os.makedirs(interval_dir, exist_ok=True)
            
            # 确定区间范围
            start_frame = frame_idx
            end_frame = frame_numbers[i+1]-1 if i < len(frame_numbers)-1 else frame_count-1

            
            # 复制帧到区间目录
            for f in range(start_frame, end_frame+1):
                src = os.path.join(frame_dir, f"{f:06d}.jpg")
                dst = os.path.join(interval_dir, f"{f:06d}.jpg")
                shutil.copy(src, dst)
                
            # 加载点坐标
            frame_data = points_data[str(frame_idx)]
            positive_points = [tuple(p) for p in frame_data.get("positive", [])]
            negative_points = [tuple(p) for p in frame_data.get("negative", [])]
            
            points = np.array(positive_points + negative_points, dtype=np.float32)
            labels = np.array([1]*len(positive_points) + [0]*len(negative_points), dtype=np.int32)
            
            # 初始化预测状态
            inference_state = self.predictor.init_state(video_path=interval_dir)

            print(f"处理区间 {i}: 帧 {start_frame} 到 {end_frame}")
            print(f"在帧 {frame_idx} 添加 {len(positive_points)} 个正样本点和 {len(negative_points)} 个负样本点")
            
            _, _, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=i,
                points=points,
                labels=labels,
            )
            
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                actual_frame = start_frame + out_frame_idx
                self.video_segments[actual_frame] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                
        shutil.rmtree(temp_dir)
    def combine_and_plot_masks(self, frame_index):
        """
        将指定帧的所有mask合成并可视化
        :param video_segments: 嵌套字典结构的mask数据
        :param frame_index: 帧索引
        """
        if frame_index not in self.video_segments:
            print(f"Error: Frame index {frame_index} not found.")
            return

        frame_masks = self.video_segments[frame_index]
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

    def load_points_from_json(self, json_folder = "D:\AI_Graph\视频\输入\sam坐标"):
        """
        从 json_folder文件夹内读取同名的json文件, 并从文件中读取正负点数据

        参数:
        json_folder: JSON 文件所在的文件夹路径
        selfvideo_path: 输入视频的路径

        返回:
        positive_points: 正点列表
        negative_points: 负点列表
        """

        # 从视频路径中提取文件名（不含扩展名）
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        json_path = os.path.join(json_folder, f"{video_name}.json")

        # 初始化返回列表
        self.positive_points = []
        self.negative_points = []

        # 检查 JSON 文件是否存在
        if not os.path.exists(json_path):
            print(f"JSON 文件不存在: {json_path}")

        # 读取 JSON 文件内容
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取正负点数据
        self.positive_points = data.get("positive", [])
        self.negative_points = data.get("negative", [])
class MaskProcessor:
    def __init__(self):
        """
        初始化蒙版处理器
        
        Args:

        """
    def fill_above_min_y(self, face_mask):
        """
        清空头的上方区域（不识别）
        
        Args:
            mask: 输入蒙版
            
        Returns:
            处理后的蒙版
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
    def fill_below_y(self, mask, y):
        """
        强制识别下30%的区域

        参数:
        face_mask: 输入的面部遮罩（二值图像，白色为255，黑色为0）

        返回:
        处理后的遮罩
        """

        # 创建一个与原始遮罩相同大小的全白图像
        result_mask = np.ones_like(mask) * 255

        # 将最大y坐标以上的区域恢复为原始遮罩
        result_mask[:y + 1, :] = mask[:y + 1, :]

        return result_mask
    def apply_dilation_and_squarization(self, mask, dilation_kernel_size, square_size):
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
    def detect_skin_mask(self, image):
        """
        输入: image_rgb (RGB格式的numpy数组)
        输出: mask (单通道，皮肤区域为255，其余为0)
        """
        # 将 RGB 转换到 HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 设定肤色范围 (HSV)，可以根据需要调整
        # 常用的人类皮肤色阈值
        lower_hsv = np.array([0, 40, 120], dtype=np.uint8)   # H, S, V 下限
        upper_hsv = np.array([30, 200, 220], dtype=np.uint8) # H, S, V 上限

        # 生成蒙板
        mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

        return mask
    def remove_small_components(self, mask, n = 100):
        """
        移除小面积组件
        
        Args:
            mask: 输入蒙版
            
        Returns:
            处理后的蒙版
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        
        output = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= n:
                output[labels == i] = 255
                
        return output
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

    if SAM_FLAG:
        sam_segmenter = SamSegmenter(model_cfg, sam2_checkpoint, video_path)
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
        # 获取所有帧的sam处理结果
        # sam_segmenter.process_video_with_single_point()
        sam_segmenter.process_video_with_dynamic_points()
    try:
        mask_tool = MaskProcessor()
        # 处理所有帧
        for frame_count in tqdm(range(StartFrame, frames), desc=f"Processing {video_name}", unit="frame"):
            ret, frame = cap.read()

            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if SAM_FLAG:
                # person_mask = combine_and_plot_masks(video_segments, frame_count)
                person_mask = sam_segmenter.combine_and_plot_masks(frame_count)
                person_mask = person_mask.astype(np.uint8) * 255
            else:
                person_mask = segmenter.get_person_mask(image_rgb, threshold=THRESHOLD)
            
            # 获取面部蒙版
            face_mask = segmenter.get_face_mask(image_rgb)
            if SKIN_DETECT:
                skin_mask = mask_tool.detect_skin_mask(image_rgb)
                skin_mask = mask_tool.remove_small_components(skin_mask, int(width * height / 150))

            if person_mask is None:
                continue
                
            # 对主体和面部蒙版进行膨胀和方块化
            person_mask = mask_tool.apply_dilation_and_squarization(
                person_mask, BODY_DILATION, BODY_SQUARE
            )
            face_mask = mask_tool.apply_dilation_and_squarization(
                face_mask, FACE_DILATION, FACE_SQUARE
            )

            # 清空头的上方区域（不识别）
            if UP_CLEAR:
                face_mask = mask_tool.fill_above_min_y(face_mask)
            # 强制识别下30%的区域
            if DRAW_DOWN:
                person_mask = mask_tool.fill_below_y(person_mask, int(height * 0.7))
            
            # 主体区域 - 面部区域 - 皮肤区域（如果开启）
            body_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(face_mask))
            if SKIN_DETECT:
                body_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(skin_mask))

            # 输出视频帧
            final_output = frame.copy()
            final_output[body_mask == 255] = 0
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

# 窗口点选全局变量
positive_points = []
negative_points = []
fram4samPoint_global = None

# 可调节参数
model_cfg = "configs\\sam2.1\\sam2.1_hiera_t.yaml"
sam2_checkpoint = "checkpoints\\sam2.1_hiera_tiny.pt"

# True False
FLAG_100FRAMS = True

SAM_FLAG = True         # 是否使用sam识别主体
SAM_POINT_CLICK = False  # sam窗口点选

StartFrame = 0      # 选择视频的第几帧画点
DRAW_DOWN = False   # 将下1/3区域全部图黑
UP_CLEAR = False    # 将头部上方清空
SKIN_DETECT = False # 去除皮肤部分

FACE_DILATION = 0   # 面部蒙版膨胀
FACE_SQUARE = 32    # 面部蒙版方块化
BODY_DILATION = 32   # 主体蒙版膨胀
BODY_SQUARE = 32     # 主体蒙版方块化

THRESHOLD = 0.1         # 身体阈值
FACE_THRESHOLD = 0.5    # 面部阈值

# 选择使用的分割器类型
USE_MEDIAPIPE = True  # 设置为False可以切换为其他分割器

if __name__ == "__main__":
    name = "好久不见呀 想我了吗 甘雨 cos 原神 fyp - 抖音"
    # input_dir = f"D:\AI_Graph\视频\输入\原视频_16fps"  # 可以是单个视频路径，也可以是文件夹路径
    # input_dir = f"D:\AI_Graph\视频\输入\原视频_16fps\{name}.mp4"  # 可以是单个视频路径，也可以是文件夹路径
    input_dir = "D:\AI_Graph\视频\输入\MultiScene.mp4"
    output_root = r"D:\AI_Graph\视频\输入\输入视频整合"
    
    print("\n\n\n----------------------------------------------------------------------")
    print(f"将{input_dir}生成为遮罩, 输出到{output_root}")
    process_videos(input_dir, output_root, start_index=5)