"""
为视频的每一帧添加帧号
"""
import cv2
import os
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile


def convert_video_to_16fps_with_audio(input_path, output_path=None):
    """
    将视频转换为16fps，保留音频，并在每帧左上角添加帧号

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径（可选）
    """

    # 如果未指定输出路径，自动生成
    if output_path is None:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_16fps_with_audio{ext}"

    # 创建临时文件用于存储无声视频
    temp_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    try:
        # 第一步：用OpenCV处理视频（不包含音频）
        print("第一步：处理视频帧...")
        success = process_video_frames(input_path, temp_video_path)

        if not success:
            print("视频处理失败")
            return False

        # 第二步：提取原视频音频并合并到新视频
        print("第二步：处理音频...")
        success = add_audio_to_video(input_path, temp_video_path, output_path)

        if success:
            print(f"处理完成！输出文件: {output_path}")
        else:
            print("音频处理失败")

        return success

    finally:
        # 清理临时文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print("临时文件已清理")


def process_video_frames(input_path, output_path):
    """
    处理视频帧：转换为16fps并添加帧号
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {input_path}")
        return False

    # 获取原视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps

    print(f"原视频信息:")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 帧率: {original_fps:.2f} fps")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {duration:.2f} 秒")
    print(f"  - 目标帧率: 16 fps")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 16.0, (width, height))

    if not out.isOpened():
        print("错误：无法创建输出视频文件")
        cap.release()
        return False

    frame_count = 0
    written_frames = 0

    print("开始处理视频帧...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 计算每秒需要保留的帧
        frames_per_second = original_fps
        frames_to_keep_per_second = 16

        # 计算当前秒内的帧序号
        frames_in_current_second = frame_count % frames_per_second

        # 确定是否保留当前帧（每秒均匀选择16帧）
        target_frame_positions = [i * (frames_per_second / frames_to_keep_per_second)
                                  for i in range(frames_to_keep_per_second)]

        keep_frame = False
        for pos in target_frame_positions:
            if abs(frames_in_current_second - pos) < 0.5:  # 允许0.5帧的容差
                keep_frame = True
                break

        if keep_frame:
            # 在帧上添加帧号文本
            text = f"Frame: {written_frames}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # 添加时间信息
            current_second = frame_count / frames_per_second
            time_text = f"Time: {current_second:.2f}s"
            cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # 写入帧
            out.write(frame)
            written_frames += 1

            # 显示进度
            if written_frames % 16 == 0:
                print(f"已处理第 {int(current_second)} 秒，总帧数: {written_frames}")

        frame_count += 1

    # 释放资源
    cap.release()
    out.release()

    print(f"视频帧处理完成!")
    print(f"原视频帧数: {frame_count}")
    print(f"输出视频帧数: {written_frames}")

    return True


def add_audio_to_video(original_video_path, video_without_audio_path, output_path):
    """
    从原视频提取音频并合并到新视频
    """
    try:
        # 加载原视频（包含音频）
        original_clip = VideoFileClip(original_video_path)

        # 加载处理后的视频（无音频）
        video_clip = VideoFileClip(video_without_audio_path)

        # 检查原视频是否有音频
        if original_clip.audio is not None:
            print("检测到音频流，正在合并...")

            # 提取音频并确保长度与视频匹配
            audio_clip = original_clip.audio

            # 确保音频长度不超过视频长度
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)

            # 合并音频和视频
            final_clip = video_clip.set_audio(audio_clip)

            # 写入最终文件
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None  # 禁用详细日志
            )

            print("音频合并成功！")
        else:
            print("原视频无音频，直接复制视频...")
            # 如果原视频没有音频，直接复制处理后的视频
            video_clip.write_videofile(
                output_path,
                codec='libx264',
                verbose=False,
                logger=None
            )

        # 关闭所有clip释放资源
        original_clip.close()
        video_clip.close()
        if original_clip.audio is not None:
            audio_clip.close()

        return True

    except Exception as e:
        print(f"音频处理错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将视频转换为16fps并添加帧号（保留音频）')
    parser.add_argument('input', help='输入视频文件路径')
    parser.add_argument('-o', '--output', help='输出视频文件路径（可选）')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误：输入文件 {args.input} 不存在")
        return

    # 执行转换
    success = convert_video_to_16fps_with_audio(args.input, args.output)

    if success:
        print("转换成功！")
    else:
        print("转换失败！")


if __name__ == "__main__":
    # 安装依赖检查
    try:
        import cv2
        from moviepy.editor import VideoFileClip
    except ImportError as e:
        print("缺少必要的依赖库，请安装：")
        print("pip install opencv-python moviepy")
        exit(1)

    # 示例用法
    input_file = r"D:\桌面中转\视频\输入视频\旅行开启请保持飞行模式 ai ai飞行模式转场 旅行转场 即梦AI ai创作浪潮计划 - 抖音.mp4"  # 替换为你的视频文件路径

    if os.path.exists(input_file):
        convert_video_to_16fps_with_audio(input_file)
    else:
        print("请通过命令行参数指定视频文件路径")
        print("用法: python script.py input_video.mp4 [-o output_video.mp4]")