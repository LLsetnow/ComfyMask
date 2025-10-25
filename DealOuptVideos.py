"""
读取文件夹内的所有文件,只保留文件名包含”audio“的文件,并将包含”audio“的文件名批量改为以固定前缀name开头,
文件名结尾是递增的数字,并且可以设置数字的起始和步长,默认是从1开始,步长是1,
"""
import os   
import shutil

def rename_files(path, name, style, start_index=1):
    files = os.listdir(path)
    # 只保留文件名包含”audio“的文件
    # 删除其他不含”audio“的文件 但不删除文件夹
    for file in files:
        if "audio" not in file:
            if os.path.isdir(os.path.join(path, file)):
                shutil.rmtree(os.path.join(path, file))
            else:
                os.remove(os.path.join(path, file))
                
    files = os.listdir(path)
    audio_files = [file for file in files if "audio" in file]
    # 批量改名
    for i, file in enumerate(audio_files, start=start_index):
        new_name = name + "_" + style + str(i) + ".mp4"
        os.rename(os.path.join(path, file), os.path.join(path, new_name))
   

def mkdir_and_rename_files(name, path, start_index=1):
    files = os.listdir(path)
    guance_dir = os.path.join(path, "观测")
    duibi_dir = os.path.join(path, "对比")
    os.makedirs(guance_dir, exist_ok=True)
    os.makedirs(duibi_dir, exist_ok=True)
    for file in files:
        if "观测" in file:
            shutil.move(os.path.join(path, file), os.path.join(guance_dir, file))
        elif "对比" in file:
            shutil.move(os.path.join(path, file), os.path.join(duibi_dir, file))
        else:
            print(f"未知文件: {file}")

    rename_files(guance_dir, name, "观测", start_index)
    rename_files(duibi_dir, name, "对比", start_index)
        
if __name__ == "__main__":
    path = r"D:\AI_Graph\图\视频去衣\output"
    name = "小野莉乃"
    mkdir_and_rename_files(name, path, start_index=0)
    print("批量改名完成")