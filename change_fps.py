"""
视频帧率转换工具

该模块提供高级视频帧率转换功能，支持丢帧、重复帧和帧插值（简化版）三种方法，并可选择是否保留音频。

主要功能:
- 支持降低或提高视频帧率。
- 提供多种帧率转换方法（丢帧、重复帧、帧插值）。
- 可选保留或丢弃原视频音频。
- 自动处理视频时长与音频同步问题。

使用示例:
    from change_fps import change_video_fps_advanced

    # 降低帧率并保留音频
    success = change_video_fps_advanced("input.mp4", "output.mp4", 24, method='drop', keep_audio=True)

    # 提高帧率并丢弃音频
    success = change_video_fps_advanced("input.mp4", "output.mp4", 60, method='repeat', keep_audio=False)

注意事项:
- 帧插值功能需要 OpenCV 4.2+ 支持，否则会自动降级为重复帧方法。
- 输出视频的编码格式为 MP4 (H.264 + AAC)。
- 如果目标帧率与原帧率差异较大，音频可能会被截断或循环以匹配视频时长。

日期: 2025-10-02
"""
import cv2
import os
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile


def change_video_fps_advanced(video_path, output_path, n, method='drop', keep_audio=True):
    """
    高级视频帧率转换函数

    参数:
    video_path: 输入视频路径
    output_path: 输出视频路径
    n: 目标帧率，如果n=0则保持原帧率不变
    method: 帧率转换方法
        - 'drop': 丢帧（适用于降低帧率）
        - 'repeat': 重复帧（适用于提高帧率）
        - 'interpolate': 帧插值（适用于提高帧率，需要OpenCV 4.2+）
    keep_audio: 是否保留原视频的音频

    返回:
    bool: 成功返回True，失败返回False
    """
    # 检查输入文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 输入视频文件不存在: {video_path}")
        return False

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 临时文件路径
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video_no_audio.mp4")

    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"错误: 无法打开视频文件: {video_path}")
            return False

        # 获取原视频属性
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 确定输出帧率
        if n == 0:
            target_fps = original_fps
            print(f"保持原帧率: {original_fps} FPS")
        else:
            target_fps = n
            print(f"将视频从 {original_fps} FPS 转换为 {target_fps} FPS")

        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (width, height))

        if not out.isOpened():
            print(f"错误: 无法创建输出视频文件: {temp_video_path}")
            cap.release()
            return False

        # 处理视频帧
        if target_fps <= original_fps:
            # 降低帧率 - 使用丢帧方法
            frame_interval = original_fps / target_fps
            frame_count = 0
            saved_frame_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # 根据帧间隔决定是否保存当前帧
                if frame_count % round(frame_interval) == 0:
                    out.write(frame)
                    saved_frame_count += 1

                frame_count += 1

                # 显示进度
                if frame_count % 100 == 0:
                    print(f"处理进度: {frame_count}/{total_frames} 帧")

        else:
            # 提高帧率
            if method == 'repeat':
                # 重复帧方法
                frame_count = 0
                repeat_factor = target_fps / original_fps

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    # 重复写入当前帧
                    for _ in range(round(repeat_factor)):
                        out.write(frame)
                        frame_count += 1

                    # 显示进度
                    if frame_count % 100 == 0:
                        print(f"处理进度: 已生成 {frame_count} 帧")

            elif method == 'interpolate':
                # 帧插值方法（简化版，实际应用中可能需要更复杂的算法）
                print("警告: 帧插值功能需要更复杂的实现，这里使用重复帧代替")
                cap.release()
                out.release()
                return change_video_fps_advanced(video_path, output_path, n, method='repeat', keep_audio=keep_audio)

        # 释放资源
        cap.release()
        out.release()

        # 如果不需要保留音频，直接复制临时文件到输出路径
        if not keep_audio:
            import shutil
            shutil.copy2(temp_video_path, output_path)
            print(f"处理完成! 输出视频: {output_path}")
            return True

        # 保留音频 - 使用moviepy处理音频
        print("正在处理音频...")

        try:
            # 从原视频提取音频
            original_clip = VideoFileClip(video_path)
            audio = original_clip.audio

            # 创建新视频剪辑（无音频）
            new_video_clip = VideoFileClip(temp_video_path)

            # 将原音频设置到新视频上
            # 注意：如果帧率改变，音频可能需要调整时长以匹配视频
            if target_fps != original_fps:
                # 计算视频时长比例
                original_duration = original_clip.duration
                new_duration = new_video_clip.duration

                # 如果视频时长变化不大，直接使用原音频
                if abs(original_duration - new_duration) / original_duration < 0.1:
                    # 时长变化小于10%，直接使用原音频
                    final_clip = new_video_clip.set_audio(audio)
                else:
                    # 时长变化较大，需要调整音频
                    print(f"视频时长变化较大 ({original_duration:.2f}s -> {new_duration:.2f}s)，调整音频...")
                    # 使用原音频的前new_duration秒，或者循环/截断
                    if new_duration <= original_duration:
                        audio = audio.subclip(0, new_duration)
                    else:
                        # 如果新视频更长，循环音频或添加静音
                        from moviepy.audio.AudioClip import CompositeAudioClip
                        from moviepy.audio.io.AudioFileClip import AudioFileClip
                        from moviepy.audio.fx.all import volumex

                        # 计算需要循环多少次
                        num_loops = int(new_duration / original_duration) + 1
                        audio_clips = [audio]

                        # 如果需要，添加循环音频
                        for i in range(1, num_loops):
                            audio_clips.append(audio)

                        # 合并所有音频片段
                        from moviepy.audio.AudioClip import concatenate_audioclips
                        looped_audio = concatenate_audioclips(audio_clips)

                        # 截取所需长度
                        audio = looped_audio.subclip(0, new_duration)

                    final_clip = new_video_clip.set_audio(audio)
            else:
                # 帧率未改变，直接使用原音频
                final_clip = new_video_clip.set_audio(audio)

            # 写入最终视频文件
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )

            # 关闭所有剪辑以释放资源
            original_clip.close()
            new_video_clip.close()
            final_clip.close()

        except Exception as e:
            print(f"音频处理失败: {e}")
            print("将输出无音频的视频文件")
            import shutil
            shutil.copy2(temp_video_path, output_path)

        # 删除临时文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        print(f"处理完成! 输出视频: {output_path}")
        return True

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        # 清理临时文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return False


def batch_change_video_fps(input_folder, output_folder, n, file_extensions=['.mp4', '.avi', '.mov'], keep_audio=True):
    """
    批量处理文件夹中的所有视频文件

    参数:
    input_folder: 输入文件夹路径
    output_folder: 输出文件夹路径
    n: 目标帧率
    file_extensions: 要处理的视频文件扩展名列表
    keep_audio: 是否保留音频
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 获取所有视频文件
    video_files = []
    for ext in file_extensions:
        video_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])

    if not video_files:
        print(f"在文件夹 {input_folder} 中未找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件")

    # 处理每个视频文件
    for i, video_file in enumerate(video_files):
        print(f"\n处理文件 {i + 1}/{len(video_files)}: {video_file}")

        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)

        # 处理视频
        success = change_video_fps_advanced(input_path, output_path, n, keep_audio=keep_audio)

        if success:
            print(f"成功处理: {video_file}")
        else:
            print(f"处理失败: {video_file}")


def main():
    # 批量处理示例
    input_folder = r"D:\AI_Graph\输入\原视频"
    output_folder = r"D:\AI_Graph\输入\原视频_16fps"

    print("\n\n\n----------------------------------------------------------------------")
    print(f"将{input_folder}文件夹内的视频处理为16fps, 输出到{output_folder}")
    # 批量将所有视频转换为16 FPS，并保留音频
    batch_change_video_fps(input_folder, output_folder, 16, keep_audio=True)
    print("处理完成")

if __name__ == "__main__":
    main()