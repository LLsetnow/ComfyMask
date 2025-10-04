"""
遍历文件夹，裁剪面部
"""

import cv2
import mediapipe as mp
from pathlib import Path


def process_images(input_dir, output_dir, target_size=(720, 1080), crop_ratio=0.6):
    """
    遍历文件夹图像 -> 使用Mediapipe分割 -> 裁剪主体 -> 缩放到目标尺寸
    每张图片对应输出 1.png, 1.txt ...

    参数:
        input_dir: 输入图像文件夹
        output_dir: 输出文件夹
        target_size: 输出图像尺寸 (w, h)，默认为 (720, 1080)
        crop_ratio: 裁切比例 (0~1)，数值越小越贴近人脸，越大越多身体
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mediapipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # 遍历输入文件夹
    input_dir = Path(input_dir)
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    for idx, img_path in enumerate(image_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取: {img_path}")
            continue

        # 转换为RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 生成分割掩码
        results = segment.process(rgb)
        mask = (results.segmentation_mask > 0.3).astype("uint8") * 255

        # 找到人物轮廓的最小外接矩形
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"未检测到主体: {img_path}")
            continue
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # 按比例缩小矩形，保持中心不变
        cx, cy = x + w // 2, y + h // 2
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        x1, y1 = max(cx - new_w // 2, 0), max(cy - new_h // 2, 0)
        x2, y2 = min(cx + new_w // 2, img.shape[1]), min(cy + new_h // 2, img.shape[0])

        # 裁剪
        cropped = img[y1:y2, x1:x2]

        # 缩放到目标尺寸
        resized = cv2.resize(cropped, target_size)

        # 输出文件名
        out_img_name = f"{idx}.png"
        out_txt_name = f"{idx}.txt"

        out_img_path = output_dir / out_img_name
        out_txt_path = output_dir / out_txt_name

        # 保存图像
        cv2.imwrite(str(out_img_path), resized)

        # 保存对应的txt文件（原始文件名）
        with open(out_txt_path, "w") as f:
            f.write("")

        print(f"处理完成: {out_img_path}, {out_txt_path}")

    print(f"所有图像处理完成，结果保存在 {output_dir}")


# 使用示例：
# crop_ratio 越小越偏向人脸，越大越包含身体
process_images("D:/AI_Graph/dataset/input", "D:/AI_Graph/dataset/output", crop_ratio=0.7)
