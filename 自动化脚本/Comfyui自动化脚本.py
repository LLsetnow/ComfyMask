import json
from urllib import request, error
import time
import os
import shutil
import subprocess


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

    # 输入配置

    # 配置输入图像列表（修改这里）
    # image_list = [
    #     "D:\桌面中转\纸悦Etsu_ko 水手服兔女郎 [63P-365MB]\Telegram@ciyuanb@- (32).jpg",
    #     "D:\桌面中转\纸悦Etsu_ko 水手服兔女郎 [63P-365MB]\Telegram@ciyuanb@- (33).jpg",
    #     "D:\桌面中转\纸悦Etsu_ko 水手服兔女郎 [63P-365MB]\Telegram@ciyuanb@- (34).jpg",
    # ]

    # 或者从文件夹批量加载图像
    image_folder = "D:\桌面中转\纸悦Etsu_ko 水手服兔女郎 [63P-365MB]"
    image_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # ComfyUI input 文件夹路径
    comfyui_input_folder = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\input"

    # 加载工作流JSON文件
    workflow_path = 'Qwen 单图编辑.json'
    original_workflow = json.load(open(workflow_path, 'r', encoding='utf-8'))

    # 批量执行工作流
    for i, image_path in enumerate(image_list):
        # 每次循环创建工作流的副本，避免修改原始对象
        prompt_workflow = json.loads(json.dumps(original_workflow))

        # 获取完整的输入路径
        if os.path.dirname(image_path):
            # 如果包含路径，直接使用
            full_image_path = image_path
        else:
            # 如果只是文件名，从文件夹获取完整路径
            full_image_path = os.path.join(image_folder, image_path)

        # 提取文件名（不带扩展名）作为输出前缀
        image_filename = os.path.basename(full_image_path)
        image_basename = os.path.splitext(image_filename)[0]

        print(f"[{i+1}/{len(image_list)}] 正在处理图像: {image_filename}")
        print(f"    输入: {full_image_path}")
        print(f"    输出前缀: result_{image_basename}")

        # 复制图片到 ComfyUI input 文件夹
        comfyui_image_path = os.path.join(comfyui_input_folder, image_filename)
        if not os.path.exists(comfyui_image_path):
            print(f"    复制图片到 ComfyUI input 文件夹...")
            shutil.copy2(full_image_path, comfyui_image_path)
        else:
            print(f"    图片已存在于 ComfyUI input 文件夹")

        # 不同工作流只需修改此处的输入图像 节点编号

        # 提取文件名（只传文件名给ComfyUI）
        prompt_workflow["41"]["inputs"]["image"] = image_filename

        # 修改输出文件名前缀
        prompt_workflow["173"]["inputs"]["filename_prefix"] = f"AutoOutput\\A_{image_basename}"

        # 发送工作流到ComfyUI执行
        queue_prompt(prompt_workflow)
        print("    工作流已发送，正在处理中...")

        # 添加延迟，避免内存溢出
        if i < len(image_list) - 1:
            print("    等待45秒后处理下一张图像...\n")
            time.sleep(45)

    print(f"\n完成！共处理 {len(image_list)} 张图像")
    print("输出图像将保存到ComfyUI的output\AutoOutput 文件夹")

if __name__ == "__main__":
    main()


