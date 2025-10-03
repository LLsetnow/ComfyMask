"""
用于输入单个视频 并读取视频的第一帧。通过鼠标左键为正样本，右键为负样本。
通过sam2 生成mask 视频
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor

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

# Global variables to store clicked points
positive_points = []
negative_points = []
frame = None

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        positive_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green for positive points
        cv2.imshow('First Frame', frame)
    elif event == cv2.EVENT_RBUTTONDOWN:
        negative_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red for negative points
        cv2.imshow('First Frame', frame)

# 处理视频的函数
def process_video_with_sam2(
    video_path: str,
    mask_dir: str,
    mask_name: str,
    sam2_checkpoint: str = "D:\\AI_Graph\\sam2\\checkpoints\\sam2.1_hiera_base_plus.pt",
    model_cfg: str = "D:\\AI_Graph\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_b+.yaml"):
    """
    Process a video with SAM2 to generate segmentation masks.
    
    Args:
        video_path: Path to the input video file
        sam2_checkpoint: Path to SAM2 checkpoint file
        model_cfg: Path to SAM2 model config file
    """
    global positive_points, negative_points, frame
    
    # Read video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Display the first frame for point selection
    cv2.imshow('First Frame', frame)
    cv2.setMouseCallback('First Frame', click_event)
    print("Left-click for positive points, Right-click for negative points. Press 'q' to finish.")

    # Wait for user to finish clicking
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

    # Load SAM2 model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    # Prepare input points and labels
    input_points = np.array(positive_points + negative_points, dtype=np.float32)
    input_labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)

    # Process the first frame
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=input_points,
        labels=input_labels,
    )

    # Get the mask for the first frame
    first_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    if first_mask.ndim == 3:
        first_mask = first_mask.squeeze(0)

    # Prepare video writer for the mask video
    bodyMask_name = f"BodyMask{mask_name}.mp4"
    mask_path = os.path.join(mask_dir, bodyMask_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(mask_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Write the first frame mask
    if first_mask.ndim == 2:
        first_mask = (first_mask * 255).astype(np.uint8)
        out_video.write(first_mask)
    else:
        print(f"Error: Invalid mask dimensions. Expected 2D, got {first_mask.ndim}D.")
        cap.release()
        return

    # Process the rest of the frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    frames_processed = 0
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                
                if mask.ndim != 2:
                    print(f"Skipping frame {out_frame_idx}: Invalid mask shape {mask.shape}")
                    continue
                
                if mask.shape != (frame_height, frame_width):
                    mask = cv2.resize(mask, (frame_width, frame_height))
                
                mask_uint8 = (mask * 255).astype(np.uint8)
                if mask_uint8.shape == (frame_height, frame_width):
                    out_video.write(mask_uint8)
                    frames_processed += 1
                else:
                    print(f"Skipping frame {out_frame_idx}: Final shape mismatch {mask_uint8.shape}")
                
                if frames_processed >= max_frames:
                    break
                    
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving partial results...")
            break
        except Exception as e:
            print(f"Critical error at frame {frames_processed}: {str(e)}")
            break

    # Release resources
    cap.release()
    out_video.release()
    print(f"Mask video saved as {mask_path}")

    # 保存原视频（保留音频）
    origin_video_path = os.path.join(mask_dir, f"OriginVideo{mask_name}.mp4")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    origin_video = cv2.VideoWriter(origin_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        origin_video.write(frame)

    cap.release()
    origin_video.release()
    print(f"Original video saved as {origin_video_path}")

    # 生成黑白反转后的视频
    overlay_video_path = os.path.join(mask_dir, f"Background{mask_name}.mp4")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_video = cv2.VideoWriter(overlay_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 确保squared_mask已定义
        if 'squared_mask' not in locals():
            squared_mask = dilated_mask  # 默认使用膨胀后的遮罩
        # 反转mask
        inverted_mask = cv2.bitwise_not(squared_mask)
        # 与原视频进行与运算
        result = cv2.bitwise_and(frame, frame, mask=inverted_mask)
        overlay_video.write(result)

    cap.release()
    overlay_video.release()
    print(f"Overlay video saved as {overlay_video_path}")

    # 横向拼接第二和第三个视频并显示
    cap_mask = cv2.VideoCapture(mask_path)
    cap_overlay = cv2.VideoCapture(overlay_video_path)

    if not cap_mask.isOpened() or not cap_overlay.isOpened():
        print("Error: Failed to open video files for display.")
    else:
        while True:
            ret_mask, frame_mask = cap_mask.read()
            ret_overlay, frame_overlay = cap_overlay.read()
            if not ret_mask or not ret_overlay:
                break

            # 调整尺寸为最大宽960、高720
            frame_mask = cv2.resize(frame_mask, (480, 360))
            frame_overlay = cv2.resize(frame_overlay, (480, 360))
            combined = np.hstack((frame_mask, frame_overlay))
            cv2.imshow("Combined Videos", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap_mask.release()
        cap_overlay.release()
        cv2.destroyAllWindows()


# 处理单帧图片的函数
def process_frame_with_sam2(
    predictor,
    inference_state,
    frame: np.ndarray,
    points: list,
    labels: list,
    frame_idx: int
) -> np.ndarray:
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


if __name__ == "__main__":
    
    video_path = r"D:\\AI_Graph\\sam2output\\3.mp4"
    mask_dir = r"D:\AI_Graph\inputFiles"
    mask_name = "100"
    process_video_with_sam2(video_path, mask_dir, mask_name)