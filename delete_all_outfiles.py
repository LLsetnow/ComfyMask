"""
删除所有输出文件
"""
from pathlib import Path
import shutil
import os

video_path = Path("D:\AI_Graph\视频\输入\原视频_16fps")
mask_path = Path("D:\AI_Graph\视频\遮罩视频")
all_input_path = Path("D:\AI_Graph\视频\输入\输入视频整合")

shutil.rmtree("D:\AI_Graph\视频\输入\原视频_16fps")
shutil.rmtree("D:\AI_Graph\视频\遮罩视频")
shutil.rmtree("D:\AI_Graph\视频\输入\输入视频整合")

video_path.mkdir(exist_ok = True)
mask_path.mkdir(exist_ok = True)
all_input_path.mkdir(exist_ok = True)

print("已清除输入处理文件")