"""
将Comfyui单图处理.py编译为exe文件
使用前请确保已安装 pyinstaller: pip install pyinstaller
"""

import os
import subprocess

def build_exe():
    """编译为exe文件"""
    
    # 脚本路径
    script_path = "Comfyui单图处理.py"
    output_dir = "dist"
    
    # 检查脚本是否存在
    if not os.path.exists(script_path):
        print(f"错误: 找不到脚本文件 - {script_path}")
        return
    
    print("=" * 50)
    print("开始编译 Comfyui单图处理.py")
    print("=" * 50)
    
    # PyInstaller 命令
    # --onefile: 打包成单个exe
    # --noconsole: 不显示控制台窗口（如果需要看日志，去掉此参数）
    # --name: 指定exe名称
    # --icon: 指定图标（可选）
    # --add-data: 添加额外文件（Windows使用分号;分隔源和目标）
    
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name=Comfyui单图处理",
        "--add-data",
        "Qwen 单图编辑.json;.",
        script_path
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}\n")
    
    try:
        # 执行编译
        subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 50)
        print("编译成功！")
        print("=" * 50)
        print(f"exe文件位置: {os.path.join(output_dir, 'Comfyui单图处理.exe')}")
        print("\n使用说明:")
        print("1. 将exe文件拖入图片即可处理")
        print("2. 或者右键图片 -> 打开方式 -> 选择此exe")
        
        # 提醒工作流文件
        print("\n说明:")
        print("工作流JSON文件已打包到exe中，无需额外配置")
        
    except subprocess.CalledProcessError as e:
        print(f"\n编译失败: {e}")
        print("\n请确保已安装 pyinstaller:")
        print("pip install pyinstaller")


if __name__ == "__main__":
    build_exe()
