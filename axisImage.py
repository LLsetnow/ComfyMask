"""
图像坐标轴与网格标注工具

该模块在图像上绘制灰色网格和坐标轴，用于辅助定位和测量。

主要功能:
- 在图像上绘制灰色网格线（网格间隔100像素）
- 绘制X轴和Y轴坐标轴（黑色粗线）
- 显示坐标刻度值（红色文字）
- 坐标原点位于左上角
- 支持自定义网格间隔和刻度间隔

使用示例:
    from axisImage import axisImage
    from PIL import Image

    # 打开图像并添加坐标轴
    image = Image.open("input.jpg")
    axisImage(image, "output_with_axis.jpg")

网格规格:
    - 网格宽度: 100像素
    - 刻度间隔: 100像素
    - 坐标轴颜色: 黑色
    - 网格线颜色: 灰色 (128, 128, 128)
    - 刻度文字颜色: 红色
"""
import cv2
from PIL import Image, ImageDraw
def axisImage(image, output_path):
    # 获取图像的宽度和高度
    width, height = image.size
    # 创建一个新的图像对象
    new_image = Image.new('RGB', (width, height), (255, 255, 255))
    # 将原图像复制到新图像上
    new_image.paste(image, (0, 0))
    # 创建画笔对象
    draw = ImageDraw.Draw(new_image)
    # 画网格
    for i in range(0, width, 100):
        draw.line((i, 0, i, height), fill=(128, 128, 128))
    for j in range(0, height, 100):
        draw.line((0, j, width, j), fill=(128, 128, 128))
    # 画坐标轴
    draw.line((0, 0, 0, height), fill=(0, 0, 0))
    draw.line((0, 0, width, 0), fill=(0, 0, 0))
    # 画刻度值
    for i in range(0, width, 100):
        draw.text((i, 0), str(i), fill=(255, 0, 0))
    for j in range(0, height, 100):
        draw.text((0, j), str(j), fill=(255, 0, 0))
    
    # 保存新图像
    try:
        new_image.save(output_path)
        print(f"保存成功: {output_path}")
    except Exception as e:
        print(f"保存失败: {e}")

    new_image.save(output_path)     

def main():
    image = Image.open(r'D:\AI_Graph\输入\sam坐标\听不清 根本听不清 蔚蓝档案 cos 萝莉 户外舞蹈 甜妹 - 抖音_0.jpg')
    axisImage(image, r'D:\AI_Graph\输入\sam坐标\test_axis.jpg')

    
if __name__ == '__main__':
    main()