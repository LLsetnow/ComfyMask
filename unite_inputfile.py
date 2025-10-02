import os
import shutil

def collect_videos(input_video_dir, search_dir, all_input_dir):
    """
    遍历 input_video_dir 下的视频文件
    在 search_dir 中找到同名文件夹 (name)，并将 Background.mp4、BodyMask.mp4 以及原视频复制到 all_input/name 文件夹下
    """
    # 确保 all_input 目录存在
    os.makedirs(all_input_dir, exist_ok=True)

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
        os.makedirs(save_dir, exist_ok=True)

        # 复制原视频
        shutil.copy2(video_path, os.path.join(save_dir, file))

        # 复制 Background.mp4 和 BodyMask.mp4
        for subfile in ["Background.mp4", "BodyMask.mp4"]:
            src_file = os.path.join(target_folder, subfile)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(save_dir, subfile))
            else:
                print(f"⚠️ 文件不存在: {src_file}")

        print(f"✅ 已处理 {file} → {save_dir}")

input_video_dir = "D:\AI_Graph\视频\输入\原视频_16fps"      # 存放原始视频
search_dir = "D:\AI_Graph\视频\遮罩视频"           # 存放 name 文件夹的目录
all_input_dir = "D:\AI_Graph\视频\输入\输入视频整合"          # 输出目录

print("\n\n\n----------------------------------------------------------------------")
print(f"将16pfs视频,采样背景,遮罩 输出到{all_input_dir}")

collect_videos(input_video_dir, search_dir, all_input_dir)