# 读取文件夹内的所有视频，重命名为name_1.mp4, name_2.mp4, ...

import os
from turtle import st
def renameVideos(path, name, start_index=1):
    # 获取文件夹内的所有视频
    videos = [f for f in os.listdir(path) if f.endswith('.mp4')]
    # 重命名视频
    for i, video in enumerate(videos, start=start_index):
        os.rename(os.path.join(path, video), os.path.join(path, f'{name}_{i+1}.mp4'))       
        
if __name__ == '__main__':
    path = r'D:\AI_Graph\视频\视频去衣\102\李梦琳UN\观测视频'
    name = '李梦琳UN_观测'
    renameVideos(path, name, start_index=0)