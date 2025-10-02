import os
from pathlib import Path

def rename_images(folder):
    folder = Path(folder)
    image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    # 第一步：全部改成临时名，避免冲突
    temp_files = []
    for idx, img_path in enumerate(image_files, start=1):
        temp_name = f"tmp_{idx}{img_path.suffix.lower()}"
        temp_path = folder / temp_name
        img_path.rename(temp_path)
        temp_files.append(temp_path)

    # 第二步：统一改成目标名 1.png, 2.png ...
    for idx, tmp_path in enumerate(sorted(temp_files), start=1):
        new_name = f"{idx}.png"
        text_name = f"{idx}.txt"
        new_path = folder / new_name
        text_path = folder / text_name
        text_path.touch() # 创建txt文件
        tmp_path.rename(new_path)
        print(f"{tmp_path.name} -> {new_name}")

    print(f"重命名完成，共处理 {len(image_files)} 张图像")

# 使用示例
rename_images("D:/AI_Graph/dataset/output")
