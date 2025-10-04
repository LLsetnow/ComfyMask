# 用于在图像上画上灰色的网格，每个网格宽度为100像素 并添加数字，坐标原点为左上角。坐标轴每500像素一个刻度值
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
    image = Image.open(r'D:\AI_Graph\VideoCutFrame\0.jpg')
    axisImage(image, r'D:\AI_Graph\test_axis.jpg')

    
if __name__ == '__main__':
    main()