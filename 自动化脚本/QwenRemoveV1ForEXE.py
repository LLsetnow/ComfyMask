import json
from urllib import request, error
import time
import os
import shutil
import subprocess
import random
import sys


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

def get_image_files_from_args():
    """
    从命令行参数获取图像文件
    支持直接拖入文件或文件夹
    """
    image_files = []

    # 如果没有参数，打印使用说明
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  1. 直接拖入一个或多个图像文件到exe上")
        print("  2. 拖入文件夹，将处理该文件夹内的所有图像")
        print("\n支持的图像格式: .png, .jpg, .jpeg")
        return image_files

    # 遍历所有参数（第一个参数是脚本本身，跳过）
    for arg in sys.argv[1:]:
        path = os.path.abspath(arg)

        if os.path.isfile(path):
            # 如果是文件，检查是否为图像文件
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                image_files.append(path)
                print(f"已添加图像: {os.path.basename(path)}")
            else:
                print(f"警告: 跳过非图像文件: {os.path.basename(path)}")
        elif os.path.isdir(path):
            # 如果是文件夹，遍历其中的图像文件
            print(f"扫描文件夹: {path}")
            for f in os.listdir(path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                    full_path = os.path.join(path, f)
                    image_files.append(full_path)
                    print(f"  已添加图像: {f}")
        else:
            print(f"警告: 路径不存在: {arg}")

    return image_files

def main():
    # 从命令行参数获取图像文件
    image_list = get_image_files_from_args()

    if not image_list:
        print("\n错误: 未找到有效的图像文件")
        print("请拖入图像文件或包含图像的文件夹到exe上")
        input("\n按回车键退出...")
        return

    print(f"\n共找到 {len(image_list)} 张图像文件")

    # 检查ComfyUI服务器是否运行
    print("\n检查ComfyUI服务器状态...")
    if not check_comfyui_server():
        start_comfyui()
    else:
        print("ComfyUI服务器已运行")

    # 配置参数
    time_remove = 20    # remove工作流处理时间（秒）
    remove_iterations = 3  # remove工作流处理次数（1-20）

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
    # 打包后的资源路径处理
    if getattr(sys, 'frozen', False):
        # 如果是打包后的exe
        try:
            # 尝试从临时目录读取打包的文件
            resource_dir = sys._MEIPASS
            workflow_path = os.path.join(resource_dir, 'Qwen_remove.json')
            print(f"从打包资源加载工作流: {workflow_path}")
            with open(workflow_path, 'r', encoding='utf-8') as f:
                original_workflow = json.load(f)
        except:
            # 如果打包资源读取失败，尝试从exe所在目录读取
            script_dir = os.path.dirname(sys.executable)
            workflow_path = os.path.join(script_dir, 'Qwen_remove.json')
            print(f"从exe目录加载工作流: {workflow_path}")
            if not os.path.exists(workflow_path):
                print(f"错误: 找不到工作流文件: {workflow_path}")
                print(f"请确保 Qwen_remove.json 文件与 exe 在同一目录下")
                input("\n按回车键退出...")
                return
            with open(workflow_path, 'r', encoding='utf-8') as f:
                original_workflow = json.load(f)
    else:
        # 如果是源码运行，使用__file__
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_path = os.path.join(script_dir, 'Qwen_remove.json')
        print(f"从源码目录加载工作流: {workflow_path}")
        if not os.path.exists(workflow_path):
            print(f"错误: 找不到工作流文件: {workflow_path}")
            input("\n按回车键退出...")
            return
        with open(workflow_path, 'r', encoding='utf-8') as f:
            original_workflow = json.load(f)

    # 对每张图像进行多次remove处理
    for i, full_image_path in enumerate(image_list):
        # 提取文件信息
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
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
