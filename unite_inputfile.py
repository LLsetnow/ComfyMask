"""
视频文件整合工具

该模块用于将原始视频文件与对应的背景和遮罩文件整合到统一的目录结构中。

主要功能:
- 遍历指定目录下的视频文件。
- 在目标目录中查找与视频文件同名的文件夹。
- 将原始视频、背景视频 (Background.mp4) 和遮罩视频 (BodyMask.mp4) 复制到统一的输出目录中。

函数说明:
collect_videos(input_video_dir, search_dir, all_input_dir):
    - input_video_dir: 存放原始视频的目录路径。
    - search_dir: 存放与视频文件同名的文件夹的目录路径。
    - all_input_dir: 输出目录路径，用于存放整合后的视频文件。

使用示例:
    from unite_inputfile import collect_videos

    # 定义路径
    input_video_dir = "D:\\AI_Graph\\视频\\输入\\原视频_16fps"
    search_dir = "D:\\AI_Graph\\视频\\遮罩视频"
    all_input_dir = "D:\\AI_Graph\\视频\\输入\\输入视频整合"

    # 执行整合
    collect_videos(input_video_dir, search_dir, all_input_dir)

注意事项:
- 输入视频文件支持 .mp4, .avi, .mov, .mkv 格式。
- 目标目录中必须包含与视频文件同名的文件夹，否则会跳过处理。
- 输出目录会自动创建，无需手动创建。
       
版本: 1.0.0
日期: 2025-10-02
"""
import os
import shutil
from turtle import st

def collect_videos(input_video_dir, search_dir, all_input_dir):
    """
    遍历 input_video_dir 下的视频文件
    在 search_dir 中找到同名文件夹 (name)，并将 Background.mp4、BodyMask.mp4 以及原视频复制到 all_input/name 文件夹下
    """
    # 确保 all_input 目录存在
    os.makedirs(all_input_dir, exist_ok=True)
    i = 0
    for file in os.listdir(input_video_dir):
        if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue  # 只处理视频文件

        name, _ = os.path.splitext(file)
        video_path = os.path.join(input_video_dir, file)

        # 查找 search_dir 下的同名文件夹
        target_folder = os.path.join(search_dir, name)
        if not os.path.isdir(target_folder):
            print(f"⚠️ 未找到对应文件夹: {target_folder}")
            continue

        # all_input/name 目录
        save_dir = os.path.join(all_input_dir, name)
        # os.makedirs(save_dir, exist_ok=True) # 关闭单独文件夹保存

        # 复制原视频
        OriginVideo = f"OriginVideo{i}.mp4"
        # shutil.copy2(video_path, os.path.join(save_dir, OriginVideo)) # 关闭单独文件夹保存
        shutil.copy2(video_path, os.path.join(all_input_dir, OriginVideo))

        # 复制 Background.mp4 和 BodyMask.mp4
        for subfile in ["Background.mp4", "BodyMask.mp4"]:
            src_file = os.path.join(target_folder, subfile)
            if os.path.exists(src_file):
                file_name, _ = os.path.splitext(subfile)
                file_name = file_name + f"{i}.mp4"
                # shutil.copy2(src_file, os.path.join(save_dir, file_name)) # 关闭单独文件夹保存
                shutil.copy2(src_file, os.path.join(all_input_dir, file_name))
            else:
                print(f"⚠️ 文件不存在: {src_file}")

        print(f"✅ 已处理 {file} → {save_dir}")
        i += 1

input_video_dir = "D:\AI_Graph\视频\输入\原视频_16fps"      # 存放原始视频
search_dir = "D:\AI_Graph\视频\遮罩视频"           # 存放 name 文件夹的目录
all_input_dir = "D:\AI_Graph\视频\输入\输入视频整合"          # 输出目录

print("\n\n\n----------------------------------------------------------------------")
print(f"将16pfs视频,采样背景,遮罩 输出到{all_input_dir}")

collect_videos(input_video_dir, search_dir, all_input_dir)

