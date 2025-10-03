# 读取视频 并将视频切割成帧 保存到指定文件夹
import cv2
import os

def cutVideo2frames(video_path, output_dir):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频帧率: {fps}")
    
    # 逐帧读取视频
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存帧到文件
        frame_path = os.path.join(output_dir, f"{frame_count}.jpg")
        print(f"输出目录绝对路径: {os.path.abspath(output_dir)}")
        try:
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print(f"保存失败: 无法写入文件 {frame_path}")
            else:
                print(f"已保存帧到: {frame_path}")
                # 验证文件是否实际存在
                if not os.path.exists(frame_path):
                    print(f"警告: 文件未实际创建: {frame_path}")
        except Exception as e:
            print(f"保存失败: {e}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    

def main():
    video_path = "D:\AI_Graph\视频\输入\原视频_16fps\不许笑 胜利女神新的希望 马斯特 泳装 - 抖音.mp4"
    output_dir = "D:\AI_Graph\VideoCutFrame"
    cutVideo2frames(video_path, output_dir)
if __name__ == '__main__':
    main()