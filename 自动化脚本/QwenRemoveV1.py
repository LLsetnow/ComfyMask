import json
from urllib import request, error
import time
import os
import shutil
import subprocess
import random


def generate_random_seed():
    """
    生成15位随机数种子，如：297364725394981
    """
    return random.randint(10**14, 10**15 - 1)

def Seed(prompt_workflow, node_id):
    """
    为指定节点设置随机种子值
    :param prompt_workflow: 工作流的JSON字典
    :param node_id: 节点编号（字符串格式），如"41"
    :return: 设置的种子值
    """
    seed_value = generate_random_seed()
    prompt_workflow[node_id]["inputs"]["seed"] = int(seed_value)
    return seed_value


def check_comfyui_server(max_attempts=3, check_delay=2):
    """
    检查ComfyUI服务器是否可访问
    """
    for attempt in range(max_attempts):
        try:
            response = request.urlopen("http://127.0.0.1:8188", timeout=3)
            return True
        except error.URLError:
            if attempt < max_attempts - 1:
                time.sleep(check_delay)
            else:
                return False


def start_comfyui():
    """
    启动ComfyUI服务器
    """
    python_exe = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\python\python.exe"
    main_py = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\main.py"
    working_dir = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1"


    print("ComfyUI服务器未运行，正在启动...")
    subprocess.Popen(
        "D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\A绘世启动器.exe",
        cwd=working_dir,  # 设置工作目录
        # [python_exe, main_py],
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    print("ComfyUI服务器已启动，初始化服务器，等待30秒...")
    time.sleep(30)


def queue_prompt(prompt_workflow, max_retries=3, retry_delay=2):
    """
    将prompt workflow发送到ComfyUI服务器并排队执行
    http://127.0.0.1:8188/prompt
    """
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)

    for attempt in range(max_retries):
        try:
            request.urlopen(req)
            return True
        except Exception as e:
            print(f"    发送失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

def main():
    # 检查ComfyUI服务器是否运行
    print("检查ComfyUI服务器状态...")
    if not check_comfyui_server():
        start_comfyui()
    else:
        print("ComfyUI服务器已运行")

    # 配置参数
    time_remove = 20    # remove工作流处理时间（秒）
    remove_iterations = 3  # remove工作流处理次数（1-20）

    # 输入配置
    image_folder = "D:\桌面中转\input"
    image_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # ComfyUI根目录
    comfyui_folder = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1"
    comfyui_input_folder = os.path.join(comfyui_folder, "input")
    comfyui_output_folder = os.path.join(comfyui_folder, "output\AutoOutput")

    # 验证参数
    if not 1 <= remove_iterations <= 20:
        print("错误：remove_iterations必须在1-20之间")
        return

    print(f"\n========== 使用Qwen_remove.json处理图像 {remove_iterations}次 ==========")

    # 加载Qwen_remove工作流
    workflow_path = 'Qwen_remove.json'
    original_workflow = json.load(open(workflow_path, 'r', encoding='utf-8'))

    # 对每张图像进行多次remove处理
    for i, image_path in enumerate(image_list):
        # 提取文件信息
        if os.path.dirname(image_path):
            full_image_path = image_path
        else:
            full_image_path = os.path.join(image_folder, image_path)
            

        image_filename = os.path.basename(full_image_path)
        image_basename = os.path.splitext(image_filename)[0]

        print(f"\n[{i+1}/{len(image_list)}] 处理图像: {image_filename}")
        print(f"    输入: {full_image_path}")

        # 复制输入图片到ComfyUI input文件夹（仅复制一次，所有迭代共用）
        comfyui_image_path = os.path.join(comfyui_input_folder, image_filename)
        if not os.path.exists(comfyui_image_path):
            print(f"    复制图片到 ComfyUI input 文件夹...")
            shutil.copy2(full_image_path, comfyui_image_path)
        else:
            # 图片已存在，使用随机种子重命名后复制
            random_seed = generate_random_seed()
            file_ext = os.path.splitext(image_filename)[1]
            new_filename = f"{random_seed}{file_ext}"
            new_comfyui_image_path = os.path.join(comfyui_input_folder, new_filename)

            # 检查新文件名是否也存在，如果存在则继续生成
            while os.path.exists(new_comfyui_image_path):
                random_seed = generate_random_seed()
                new_filename = f"{random_seed}{file_ext}"
                new_comfyui_image_path = os.path.join(comfyui_input_folder, new_filename)

            print(f"    图片已存在于 ComfyUI input 文件夹，重命名为 {new_filename} 后复制...")
            shutil.copy2(full_image_path, new_comfyui_image_path)
            image_filename = new_filename  # 更新为新的文件名，供后续使用
            image_basename = os.path.splitext(new_filename)[0]

        # 复制输入图像到output文件夹（保存原始图像）
        original_output_path = os.path.join(comfyui_output_folder, f"{image_basename}_original{os.path.splitext(image_filename)[1]}")
        if not os.path.exists(original_output_path):
            print(f"    复制原始图像到 output 文件夹...")
            shutil.copy2(full_image_path, original_output_path)

        # 进行多次remove处理（每次都使用原始图像作为输入）
        for iteration in range(remove_iterations):
            prompt_workflow = json.loads(json.dumps(original_workflow))
            Seed(prompt_workflow, "65")  # 重新生成随机数

            suffixes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
            current_suffix = suffixes[iteration]
            output_prefix = f"AutoOutput\\{image_basename}_{current_suffix}"

            # 修改工作流参数
            prompt_workflow["41"]["inputs"]["image"] = image_filename
            prompt_workflow["181"]["inputs"]["filename_prefix"] = output_prefix

            queue_prompt(prompt_workflow)
            print(f"    迭代 {iteration+1}/{remove_iterations}: Qwen_remove工作流已发送 -> {current_suffix}")

            # 等待处理完成
            if iteration < remove_iterations - 1 or i < len(image_list) - 1:
                time.sleep(time_remove)

    print(f"\n========== 全部完成 ==========")
    print(f"共处理 {len(image_list)} 张图像")
    print(f"每张图像经过 {remove_iterations} 次 remove 处理")
    print(f"最终输出图像将保存到ComfyUI的output\AutoOutput 文件夹")

if __name__ == "__main__":
    main()


