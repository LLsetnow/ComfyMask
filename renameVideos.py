# 读取文件夹内的所有视频，重命名为name_1.mp4, name_2.mp4, ...

import os
from turtle import st
def renameAudioVideos(path, name, start_index=1, type='观测'):
    # 获取文件夹内的所有视频
    videos = [f for f in os.listdir(path) if f.endswith('.mp4') and type in f and "audio" in f]
    fold_name = os.path.join(path, type)
    os.mkdir(fold_name)

    # 重命名视频
    for i, video in enumerate(videos, start=start_index):
        os.rename(os.path.join(path, video), os.path.join(fold_name, f'{name}{i+1}.mp4'))       
        
if __name__ == '__main__':
    # 对比 观测
    people_name = "李梦琳UN"
    # 输出文件夹
    type = "观测"
    path = f'D:\AI_Graph\图\视频去衣\output'
    start_index = 20


    name_guance = people_name + "_观测"
    name_duibi = people_name + "_对比"
    renameAudioVideos(path, name_guance, start_index, type="观测")
    renameAudioVideos(path, name_duibi, start_index, type="对比")
