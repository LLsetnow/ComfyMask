import os
import cv2
import numpy as np
import argparse
from glob import glob
import torch
import matplotlib.pyplot as plt


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        初始化深度估计模型
        支持的模型: MiDaS_small, MiDaS, DPT_Hybrid, DPT_Large
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.model_type = model_type
        self.load_model()

    def load_model(self):
        """加载预训练的深度估计模型"""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError("请安装transformers库: pip install transformers")

        # 模型映射
        model_map = {
            "MiDaS_small": "Intel/dpt-small",
            "MiDaS": "Intel/dpt-base",
            "DPT_Hybrid": "Intel/dpt-hybrid-midas",
            "DPT_Large": "Intel/dpt-large"
        }

        if self.model_type not in model_map:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        model_name = model_map[self.model_type]
        print(f"加载模型: {model_name}")

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def estimate_depth(self, image):
        """估计单张图像的深度图"""
        # 预处理图像
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 插值到原始图像大小
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        # 转换为numpy数组并归一化
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
        depth_map = depth_map.astype(np.uint8)

        return depth_map


def process_images(input_folder, output_folder, prefix="depth", start_index=1, model_type="MiDaS_small"):
    """
    处理文件夹中的所有图像，生成深度图并保存

    参数:
        input_folder: 输入图像文件夹路径
        output_folder: 输出深度图文件夹路径
        prefix: 深度图文件前缀
        start_index: 起始序号
        model_type: 深度估计模型类型
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 初始化深度估计器
    depth_estimator = DepthEstimator(model_type)

    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []

    for extension in image_extensions:
        image_paths.extend(glob(os.path.join(input_folder, extension)))
        image_paths.extend(glob(os.path.join(input_folder, extension.upper())))

    if not image_paths:
        print(f"在文件夹 {input_folder} 中未找到图像文件")
        return

    print(f"找到 {len(image_paths)} 张图像")

    # 处理每张图像
    for i, image_path in enumerate(image_paths):
        print(f"处理图像 {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 转换为RGB (模型需要RGB输入)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 估计深度图
        depth_map = depth_estimator.estimate_depth(image_rgb)

        # 生成输出文件名
        output_filename = f"{prefix}{start_index + i}.png"
        output_path = os.path.join(output_folder, output_filename)

        # 保存深度图
        cv2.imwrite(output_path, depth_map)
        print(f"深度图已保存: {output_path}")


def debug_depth_map(image_path, model_type="MiDaS_small"):
    """
    调试函数：显示原始图像和深度图

    参数:
        image_path: 图像文件路径
        model_type: 深度估计模型类型
    """
    # 初始化深度估计器
    depth_estimator = DepthEstimator(model_type)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 估计深度图
    depth_map = depth_estimator.estimate_depth(image_rgb)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 显示原始图像
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('原始图像')
    ax1.axis('off')

    # 显示深度图
    ax2.imshow(depth_map, cmap='inferno')
    ax2.set_title('深度图')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='图像深度图生成器')
    parser.add_argument('--input', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出深度图文件夹路径')
    parser.add_argument('--prefix', type=str, default='depth', help='深度图文件前缀')
    parser.add_argument('--start', type=int, default=1, help='起始序号')
    parser.add_argument('--model', type=str, default='MiDaS_small',
                        choices=['MiDaS_small', 'MiDaS', 'DPT_Hybrid', 'DPT_Large'],
                        help='深度估计模型类型')
    parser.add_argument('--debug', type=str, help='调试模式：指定单张图像路径进行预览')

    args = parser.parse_args()

    if args.debug:
        # 调试模式：显示单张图像的深度图
        debug_depth_map(args.debug, args.model)
    else:
        # 正常模式：处理文件夹中的所有图像
        process_images(args.input, args.output, args.prefix, args.start, args.model)
