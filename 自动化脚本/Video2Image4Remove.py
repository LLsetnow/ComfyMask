import cv2
import os
import sys
import subprocess
import shutil
import random

# 设置标准输出为UTF-8编码，避免Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def generate_random_15digit():
    """
    生成15位随机数
    """
    return str(random.randint(10**14, 10**15 - 1))


def convert_video(video_path, output_folder, random_id, ffmpeg_path=None):
    """
    转换视频：重命名、调整分辨率(720x1080)、帧率(16fps)、保留音频
    :param video_path: 原视频路径
    :param output_folder: 输出文件夹
    :param random_id: 15位随机ID
    :param ffmpeg_path: ffmpeg可执行文件路径
    :return: 转换后的视频路径，失败返回None
    """
    # 设置ffmpeg路径
    if not ffmpeg_path:
        ffmpeg_path = r"E:\Program Files\ffmpeg-6.1\bin\ffmpeg.exe"

    # 检查ffmpeg是否存在
    if not os.path.exists(ffmpeg_path):
        print(f"    错误: 未找到ffmpeg: {ffmpeg_path}")
        return None

    # 构造输出路径
    output_path = os.path.join(output_folder, f"{random_id}.mp4")

    print(f"    转换视频: {os.path.basename(video_path)} -> {random_id}.mp4")
    print(f"    目标分辨率: 720x1080, 帧率: 16fps")

    # 使用ffmpeg转换视频
    # -i: 输入文件
    # -vf scale=720:1080: 调整分辨率
    # -r 16: 设置帧率为16fps
    # -c:a copy: 复制音频流（不重新编码）
    cmd = [
        ffmpeg_path,
        '-i', video_path,
        '-vf', 'scale=720:1080',
        '-r', '16',
        '-c:a', 'copy',
        '-y',  # 覆盖输出文件
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode == 0:
            print(f"    转换成功: {random_id}.mp4")
            return output_path
        else:
            print(f"    转换失败: ffmpeg返回码 {result.returncode}")
            if result.stderr:
                print(f"    错误信息: {result.stderr[:200]}")
            return None

    except Exception as e:
        print(f"    转换异常: {e}")
        return None


def extract_frame_20(video_path, output_folder, use_random_id=False, random_id=None):
    """
    从视频中提取第20帧图像
    :param video_path: 视频文件路径
    :param output_folder: 输出文件夹路径
    :param use_random_id: 是否使用随机ID作为文件名
    :param random_id: 随机ID
    :return: 提取的图像路径，失败返回None
    """
    print(f"    尝试打开视频: {video_path}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"    错误: 无法打开视频文件: {os.path.basename(video_path)}")
        print(f"    视频路径: {video_path}")
        print(f"    文件是否存在: {os.path.exists(video_path)}")
        cap.release()
        return None

    print(f"    视频已成功打开")

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    视频总帧数: {total_frames}")

    # 检查是否有足够的帧数
    if total_frames < 20:
        print(f"    警告: 视频帧数不足20帧（共{total_frames}帧），使用最后一帧")
        target_frame = total_frames - 1
    else:
        target_frame = 19  # 第20帧（从0开始计数）

    # 跳转到目标帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # 读取帧
    ret, frame = cap.read()

    if not ret:
        print(f"    错误: 无法读取第{target_frame + 1}帧")
        cap.release()
        return None

    # 构造输出文件路径
    if use_random_id and random_id:
        video_filename = random_id
    else:
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        # 处理可能包含特殊字符的文件名
        video_filename = video_filename.replace('～', '~')
        video_filename = video_filename.replace('（', '(')
        video_filename = video_filename.replace('）', ')')
        video_filename = video_filename.replace('，', ',')
        video_filename = video_filename.replace('。', '.')

    output_path = os.path.join(output_folder, f"{video_filename}.png")

    print(f"    保存图像到: {output_path}")

    # 保存图像
    try:
        success = cv2.imwrite(output_path, frame)
        if not success:
            # 尝试使用imencode方式保存（更兼容）
            import numpy as np
            ret, buf = cv2.imencode('.png', frame)
            if ret:
                with open(output_path, 'wb') as f:
                    f.write(buf)
                success = True
                print(f"    使用imencode方式保存成功")
    except Exception as e:
        print(f"    保存图像时发生异常: {e}")
        success = False

    if not success:
        print(f"    错误: 图像保存失败")
        cap.release()
        return None

    cap.release()

    print(f"    成功提取第{target_frame + 1}帧: {os.path.basename(output_path)}")
    return output_path


def get_video_files(video_folder):
    """
    获取文件夹中所有视频文件
    :param video_folder: 视频文件夹路径
    :return: 视频文件路径列表
    """
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm',
                        '.MP4', '.AVI', '.MOV', '.MKV', '.FLV', '.WMV', '.WEBM']

    video_files = []

    if not os.path.exists(video_folder):
        print(f"错误: 视频文件夹不存在: {video_folder}")
        return video_files

    for filename in os.listdir(video_folder):
        filepath = os.path.join(video_folder, filename)
        if os.path.isfile(filepath):
            # 检查文件扩展名是否为视频格式
            for ext in video_extensions:
                if filename.endswith(ext):
                    video_files.append(filepath)
                    break

    return sorted(video_files)


def copy_output_images(random_ids, output_prefixes, source_folder, target_folder):
    """
    复制输出图像到目标文件夹并重命名为随机ID
    :param random_ids: 随机ID列表
    :param output_prefixes: 输出文件前缀列表（output_prefix）
    :param source_folder: ComfyUI输出文件夹
    :param target_folder: 目标文件夹
    :return: 成功复制的文件数量
    """
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    copied_count = 0

    print(f"\n开始复制处理后的图像...")

    for i, (random_id, output_prefix) in enumerate(zip(random_ids, output_prefixes)):
        print(f"\n[{i+1}/{len(random_ids)}] 处理ID: {random_id}")

        # 在source_folder中查找匹配的文件
        # output_prefix格式: AutoOutput\{image_basename}_A
        # 实际文件名格式: {image_basename}_A_00001.png
        pattern = output_prefix.replace("AutoOutput\\", "") + "_"

        found_files = []
        for filename in os.listdir(source_folder):
            if filename.startswith(pattern) and (filename.endswith('.png') or filename.endswith('.jpg')):
                found_files.append(filename)

        if not found_files:
            print(f"    警告: 未找到匹配的输出文件 (pattern: {pattern})")
            continue

        # 复制找到的文件
        for src_filename in found_files:
            src_path = os.path.join(source_folder, src_filename)

            # 构造目标文件名：随机ID.png
            # 如果有多个输出文件（多次迭代），添加序号后缀
            if len(found_files) > 1:
                # 提取后缀字母 (A, B, C, etc.)
                suffix = src_filename.split('_')[-2]
                target_filename = f"{random_id}_{suffix}.png"
            else:
                target_filename = f"{random_id}.png"

            target_path = os.path.join(target_folder, target_filename)

            try:
                shutil.copy2(src_path, target_path)
                print(f"    已复制: {src_filename} -> {target_filename}")
                copied_count += 1
            except Exception as e:
                print(f"    错误: 复制失败 - {e}")

    return copied_count


def main():
    """
    主函数：批量处理视频
    1. 从D:\AI_Graph\输入\原视频读取所有视频
    2. 转换视频（重命名为15位随机数、分辨率720x1080、16fps、保留音频）到D:\AI_Graph\输入\原视频_16fps
    3. 提取第20帧保存到D:\AI_Graph\输入\输入首帧（使用随机ID命名）
    4. 调用QwenRemoveParaV2.py处理提取的图像
    5. 将处理后的图像复制到D:\AI_Graph\输入\输入视频整合并重命名为随机ID
    """
    # 配置路径
    video_folder = r"D:\AI_Graph\输入\原视频"
    video_16fps_folder = r"D:\AI_Graph\输入\原视频_16fps"
    output_folder = r"D:\AI_Graph\输入\输入首帧"
    target_folder = r"D:\AI_Graph\输入\输入视频整合"
    comfyui_output_folder = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\output\AutoOutput"

    # 获取QwenRemoveParaV2.py的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qwen_script = os.path.join(script_dir, "QwenRemoveParaV2.py")

    # 确保输出文件夹存在
    os.makedirs(video_16fps_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)

    print("=" * 50)
    print("视频批量转换并处理")
    print("=" * 50)

    # 获取所有视频文件
    print(f"\n正在扫描视频文件夹: {video_folder}")
    video_files = get_video_files(video_folder)

    if not video_files:
        print(f"未找到视频文件")
        return

    print(f"共找到 {len(video_files)} 个视频文件")

    # 为每个视频生成随机ID
    random_ids = []
    for video_path in video_files:
        random_id = generate_random_15digit()
        random_ids.append(random_id)

    # 转换视频
    converted_videos = []
    print(f"\n========== 开始转换视频 ==========")
    for i, (video_path, random_id) in enumerate(zip(video_files, random_ids)):
        print(f"\n[{i+1}/{len(video_files)}] 处理视频: {os.path.basename(video_path)}")

        # 转换视频
        converted_path = convert_video(video_path, video_16fps_folder, random_id)
        if converted_path:
            converted_videos.append(converted_path)
            print(f"    转换完成: {random_id}.mp4")

    if not converted_videos:
        print("\n错误: 未能转换任何视频")
        return

    print(f"\n成功转换 {len(converted_videos)} 个视频")

    # 提取每段视频的第20帧
    print(f"\n========== 开始提取首帧 ==========")
    extracted_images = []
    for i, (converted_path, random_id) in enumerate(zip(converted_videos, random_ids)):
        print(f"\n[{i+1}/{len(converted_videos)}] 提取帧: {random_id}.mp4")

        # 提取第20帧（使用随机ID命名）
        image_path = extract_frame_20(converted_path, output_folder, use_random_id=True, random_id=random_id)
        if image_path:
            extracted_images.append(image_path)

    if not extracted_images:
        print("\n错误: 未能提取任何图像")
        return

    print(f"\n成功提取 {len(extracted_images)} 张图像")

    # 调用QwenRemoveParaV2.py处理提取的图像
    print(f"\n========== 开始处理图像 ==========")
    print(f"\n正在调用 QwenRemoveParaV2.py 处理图像...")

    if not os.path.exists(qwen_script):
        print(f"错误: 找不到 QwenRemoveParaV2.py: {qwen_script}")
        return

    # 构造命令
    cmd = [sys.executable, qwen_script] + extracted_images

    # 存储QwenRemoveParaV2.py输出的文件名
    output_prefixes = []
    capture_output = False  # 标记是否开始捕获输出文件名

    try:
        # 执行QwenRemoveParaV2.py并捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False  # 禁用自动解码
        )

        # 实时输出并捕获文件名
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                try:
                    # 尝试用UTF-8解码
                    decoded_output = output.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        # 如果UTF-8失败，尝试GBK
                        decoded_output = output.decode('gbk')
                    except UnicodeDecodeError:
                        # 如果都失败，使用ignore模式
                        decoded_output = output.decode('utf-8', errors='ignore')

                print(decoded_output.strip())

                # 捕获输出文件名
                if "========== 输出文件名列表 ==========" in decoded_output:
                    capture_output = True
                elif capture_output and decoded_output.strip() and not decoded_output.strip().startswith("=") and not decoded_output.strip().startswith("["):
                    # 提取文件名（去除序号和点号）
                    line = decoded_output.strip()
                    # 格式: "1. AutoOutput\{image_basename}_A"
                    if ". " in line:
                        filename = line.split(". ", 1)[1]
                        output_prefixes.append(filename)

        return_code = process.poll()

        if return_code != 0:
            print(f"\n警告: QwenRemoveParaV2.py 返回非零代码: {return_code}")

    except Exception as e:
        print(f"\n错误: 执行过程中发生异常: {e}")
        return

    # 复制处理后的图像到目标文件夹
    if output_prefixes:
        print(f"\n========== 开始复制处理结果 ==========")
        copied_count = copy_output_images(
            random_ids,
            output_prefixes,
            comfyui_output_folder,
            target_folder
        )
        print(f"\n成功复制 {copied_count} 张图像到: {target_folder}")
    else:
        print(f"\n警告: 未获取到输出文件名，跳过复制步骤")

    print(f"\n========== 全部完成 ==========")
    print(f"转换视频: {len(converted_videos)} 个 -> {video_16fps_folder}")
    print(f"提取首帧: {len(extracted_images)} 张 -> {output_folder}")
    print(f"处理图像: {len(output_prefixes)} 张 -> {target_folder}")


if __name__ == "__main__":
    main()
