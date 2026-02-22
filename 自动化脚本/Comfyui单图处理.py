import json
import sys
import os
import time
import shutil
from urllib import request, error


def check_comfyui_server(max_attempts=3, check_delay=2):
    """检查ComfyUI服务器是否可访问"""
    for attempt in range(max_attempts):
        try:
            response = request.urlopen("http://127.0.0.1:8188", timeout=3)
            return True
        except error.URLError:
            if attempt < max_attempts - 1:
                time.sleep(check_delay)
            else:
                return False


def queue_prompt(prompt_workflow, max_retries=3, retry_delay=2):
    """将prompt workflow发送到ComfyUI服务器并排队执行"""
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


def process_image(image_path):
    """处理单张图像"""
    
    # 固定工作目录
    working_dir = r"D:\AI_Graph\tools\自动化脚本"
    os.chdir(working_dir)
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        return False
    
    # 检查是否为图片文件
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
        print(f"错误: 不支持的文件格式 - {ext}")
        return False
    
    # 检查ComfyUI服务器
    print("检查ComfyUI服务器状态...")
    if not check_comfyui_server():
        print("错误: ComfyUI服务器未运行，请先启动ComfyUI")
        return False
    else:
        print("ComfyUI服务器已运行")
    
    # 获取文件信息
    image_filename = os.path.basename(image_path)
    image_basename = os.path.splitext(image_filename)[0]
    
    # ComfyUI相关路径
    comfyui_input_folder = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\input"
    
    # 获取工作流JSON路径（支持打包后的exe）
    if getattr(sys, 'frozen', False):
        # 打包为exe后的路径
        base_path = sys._MEIPASS
    else:
        # 普通Python脚本路径
        base_path = working_dir
    
    workflow_path = os.path.join(base_path, 'Qwen 单图编辑.json')
    
    # 检查工作流文件
    if not os.path.exists(workflow_path):
        print(f"错误: 工作流文件不存在 - {workflow_path}")
        return False
    
    print(f"\n处理图像: {image_filename}")
    print(f"输入路径: {image_path}")
    
    # 复制图片到 ComfyUI input 文件夹
    comfyui_image_path = os.path.join(comfyui_input_folder, image_filename)
    if not os.path.exists(comfyui_image_path):
        print("复制图片到 ComfyUI input 文件夹...")
        shutil.copy2(image_path, comfyui_image_path)
    else:
        print("图片已存在于 ComfyUI input 文件夹")
    
    # 加载工作流
    print("加载工作流...")
    prompt_workflow = json.load(open(workflow_path, 'r', encoding='utf-8'))
    
    # 设置输入图像节点（根据你的工作流修改节点编号）
    prompt_workflow["41"]["inputs"]["image"] = image_filename
    
    # 设置输出文件名前缀
    prompt_workflow["173"]["inputs"]["filename_prefix"] = f"AutoOutput\\A_{image_basename}"
    
    # 发送工作流
    print("发送工作流到ComfyUI...")
    queue_prompt(prompt_workflow)
    print("工作流已发送，正在处理中...")
    print(f"输出将保存到: ComfyUI的output\\AutoOutput 文件夹")
    
    return True


def main():
    """主函数"""
    # 固定工作目录
    working_dir = r"D:\AI_Graph\tools\自动化脚本"
    os.chdir(working_dir)
    
    print("=" * 50)
    print("ComfyUI 单图处理工具")
    print(f"工作目录: {working_dir}")
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("1. 直接拖入图片到本程序")
        print("2. 或在命令行执行: 程序名 图片路径")
        print("\n按任意键退出...")
        input()
        return
    
    # 获取图片路径（支持多文件拖入）
    image_paths = sys.argv[1:]
    
    # 处理每张图片
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}]")
        process_image(image_path)
        if i < len(image_paths) - 1:
            print("\n等待30秒后处理下一张...")
            time.sleep(30)
    
    print("\n" + "=" * 50)
    print("处理完成！")
    print("=" * 50)
    
    if len(image_paths) == 1:
        print("按任意键退出...")
        input()


if __name__ == "__main__":
    main()
