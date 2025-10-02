# 读取文件夹内的所有文件，只保留文件名包含”audio“的文件，并将包含”audio“的文件名批量改为以固定前缀name开头，
# 文件名结尾是递增的数字，并且可以设置数字的起始和步长，默认是从1开始，步长是1，
import os   
def rename_files(path, name, start_index=1):
    # 获取文件夹内的所有文件
    files = os.listdir(path)
    # 只保留文件名包含”audio“的文件
    # 删除其他不含”audio“的文件
    for file in files:
        if "audio" not in file:
            os.remove(os.path.join(path, file))

    files = os.listdir(path)
    audio_files = [file for file in files if "audio" in file]
    # 批量改名
    for i, file in enumerate(audio_files, start=start_index):
        new_name = name + str(i) + ".mp4"
        os.rename(os.path.join(path, file), os.path.join(path, new_name))
        
if __name__ == "__main__":
    path = r"D:\AI_Graph\视频\视频去衣\瑶兔叽\观测"
    name = "瑶兔叽_观测"
    rename_files(path, name, start_index=1)
    print("批量改名完成")