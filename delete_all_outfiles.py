"""
删除所有输出文件
"""
from pathlib import Path
import shutil
import os

video_path = Path("D:\AI_Graph\输入\原视频_16fps")
all_input_path = Path("D:\AI_Graph\输入\输入视频整合")
sam_path = Path("D:\AI_Graph\输入\sam坐标")


shutil.rmtree("D:\AI_Graph\输入\原视频_16fps")
shutil.rmtree("D:\AI_Graph\输入\输入视频整合")
shutil.rmtree("D:\AI_Graph\输入\sam坐标")


video_path.mkdir(exist_ok = True)
all_input_path.mkdir(exist_ok = True)
sam_path.mkdir(exist_ok=True)

print("已清除输入处理文件")