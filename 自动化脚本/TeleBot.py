import requests
import os
import subprocess
import sys
import time
import io
from typing import Optional

# 设置标准输出为UTF-8编码，避免Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# 代理配置
PROXY_SETTINGS = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

# 设置环境变量代理
os.environ['HTTP_PROXY'] = PROXY_SETTINGS["http"]
os.environ['HTTPS_PROXY'] = PROXY_SETTINGS["https"]

# Telegram配置
TELEGRAM_BOT_TOKEN = "8413449344:AAE3r29-jiHjDpmFm4AMZYWH78iwwczq0QM"
AUTHORIZED_USER_IDS = [5468961835]  # 设置授权用户的Telegram ID列表，例如：[123456789, 987654321]，为空则允许所有用户

# ComfyUI配置
COMFYUI_INPUT_FOLDER = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\input"
COMFYUI_OUTPUT_FOLDER = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1\output"

# 脚本路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN_REMOVE_PARA_V1 = os.path.join(SCRIPT_DIR, 'QwenRemoveParaV1.py')
PYTHON_EXE = sys.executable  # 使用当前Python环境


def send_message(chat_id: str, text: str):
    """发送文本消息到Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text
    }
    try:
        response = requests.post(url, json=data, timeout=10, proxies=PROXY_SETTINGS)
        return response.json()
    except Exception as e:
        print(f"发送消息失败: {e}")
        return None


def send_photo(chat_id: str, photo_path: str, caption: Optional[str] = None):
    """发送图片到Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption

    try:
        with open(photo_path, 'rb') as photo_file:
            files = {"photo": photo_file}
            response = requests.post(url, data=data, files=files, timeout=30, proxies=PROXY_SETTINGS)
        return response.json()
    except Exception as e:
        print(f"发送图片失败: {e}")
        return None


def download_telegram_photo(file_id: str):
    """从Telegram下载图片并返回保存路径"""
    try:
        # 获取文件信息
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"
        response = requests.get(url, params={"file_id": file_id}, timeout=10, proxies=PROXY_SETTINGS)
        file_info = response.json()

        if not file_info.get("ok"):
            print(f"获取文件信息失败: {file_info}")
            return None

        file_path = file_info["result"]["file_path"]
        download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

        # 下载文件
        response = requests.get(download_url, timeout=30, proxies=PROXY_SETTINGS)

        # 保存文件
        original_filename = os.path.basename(file_path)
        temp_path = os.path.join(COMFYUI_INPUT_FOLDER, f"temp_{original_filename}")

        with open(temp_path, 'wb') as f:
            f.write(response.content)

        return temp_path

    except Exception as e:
        print(f"下载Telegram图片失败: {e}")
        return None


def call_qwen_remove_para_v1(image_path):
    """
    调用QwenRemoveParaV1.py处理图像
    :param image_path: 图像文件路径
    :return: 输出文件列表
    """
    try:
        print(f"\n调用 QwenRemoveParaV1.py 处理图像: {os.path.basename(image_path)}")

        # 使用subprocess调用脚本，不使用text和encoding避免编码问题
        result = subprocess.run(
            [PYTHON_EXE, QWEN_REMOVE_PARA_V1, image_path],
            capture_output=True,
            timeout=600  # 10分钟超时
        )

        if result.returncode != 0:
            # 解码stderr输出
            stderr_text = result.stderr.decode('utf-8', errors='ignore')
            print(f"QwenRemoveParaV1.py 执行失败:")
            print(stderr_text)
            return []

        # 解码stdout输出，忽略编码错误
        stdout_text = result.stdout.decode('utf-8', errors='ignore')

        # 解析输出，提取生成的文件
        output_files = []
        for line in stdout_text.split('\n'):
            if '已生成输出:' in line:
                file_info = line.split('已生成输出:')[-1].strip()
                # 搜索output文件夹
                for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
                    if file_info in files:
                        output_files.append(os.path.join(root, file_info))
                        break

        print(f"QwenRemoveParaV1.py 处理完成，共生成 {len(output_files)} 张结果")
        return output_files

    except subprocess.TimeoutExpired:
        print("QwenRemoveParaV1.py 执行超时")
        return []
    except Exception as e:
        print(f"调用 QwenRemoveParaV1.py 时出错: {e}")
        return []


def process_image(image_path, chat_id: str):
    """
    处理图像并发送结果到Telegram
    :param image_path: 图像文件路径
    :param chat_id: Telegram聊天ID
    """
    try:
        send_message(chat_id, f"开始处理图像...")

        # 调用QwenRemoveParaV1.py处理图像
        output_files = call_qwen_remove_para_v1(image_path)

        if not output_files:
            send_message(chat_id, "处理失败或未找到输出文件")
            return

        # 发送处理后的图像
        send_message(chat_id, f"处理完成，正在发送 {len(output_files)} 张结果...")

        for i, output_file in enumerate(output_files, 1):
            if os.path.exists(output_file):
                send_photo(chat_id, output_file, f"处理结果 {i}/{len(output_files)}")
                time.sleep(1)  # 避免发送过快

        send_message(chat_id, f"✅ 处理完成！共发送 {len(output_files)} 张处理结果")

        # 清理临时文件
        if "temp_" in os.path.basename(image_path):
            try:
                os.remove(image_path)
            except:
                pass

    except Exception as e:
        error_msg = f"处理图像时出错: {str(e)}"
        print(error_msg)
        send_message(chat_id, error_msg)


def get_updates(offset: Optional[int] = None, timeout: int = 100):
    """获取Telegram更新"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": timeout}
    if offset:
        params["offset"] = offset

    try:
        response = requests.get(url, params=params, timeout=timeout + 10, proxies=PROXY_SETTINGS)
        return response.json()
    except Exception as e:
        print(f"获取更新失败: {e}")
        return {"ok": False, "result": []}


def is_authorized(user_id: int) -> bool:
    """检查用户是否授权"""
    if not AUTHORIZED_USER_IDS:  # 列表为空，允许所有用户
        return True
    return user_id in AUTHORIZED_USER_IDS


def check_comfyui_server():
    """检查ComfyUI服务器是否运行"""
    from urllib import request, error
    try:
        response = request.urlopen("http://127.0.0.1:8188", timeout=3)
        return True
    except error.URLError:
        return False


def main():
    print("========== Telegram机器人启动 ==========")
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")

    if AUTHORIZED_USER_IDS:
        print(f"授权用户ID: {AUTHORIZED_USER_IDS}")
    else:
        print(f"授权用户ID: 所有用户（测试模式）")
        print("⚠️  警告: 未设置授权用户ID列表，允许所有用户访问（建议设置AUTHORIZED_USER_IDS）")
        print("   如何获取用户ID: 发送消息给 @userinfobot")

    print(f"QwenRemoveParaV1.py: {QWEN_REMOVE_PARA_V1}")

    # 检查QwenRemoveParaV1.py是否存在
    if not os.path.exists(QWEN_REMOVE_PARA_V1):
        print(f"错误: 找不到 QwenRemoveParaV1.py")
        print(f"请确保该文件位于: {QWEN_REMOVE_PARA_V1}")
        return

    # 检查ComfyUI服务器
    print("\n检查ComfyUI服务器状态...")
    if not check_comfyui_server():
        print("错误: ComfyUI服务器未运行，请先启动ComfyUI")
        return
    else:
        print("ComfyUI服务器已运行")

    offset = 0

    print("\n========== 开始监听消息 ==========")

    while True:
        try:
            updates = get_updates(offset=offset, timeout=30)

            if not updates.get("ok"):
                time.sleep(5)
                continue

            for update in updates["result"]:
                offset = update["update_id"] + 1

                # 检查消息
                if "message" not in update:
                    continue

                message = update["message"]
                user_id = message.get("from", {}).get("id")
                chat_id = message.get("chat", {}).get("id")
                username = message.get("from", {}).get("username", "Unknown")

                print(f"\n收到消息 - 用户ID: {user_id} (@{username}), 聊天ID: {chat_id}")

                # 检查授权
                if not is_authorized(user_id):
                    print(f"用户 {user_id} 未授权，忽略消息")
                    send_message(chat_id, "❌ 您没有权限使用此机器人")
                    continue

                # 处理图片消息
                if "photo" in message:
                    print(f"收到图片消息")

                    # 获取最大尺寸的照片
                    photos = message["photo"]
                    largest_photo = max(photos, key=lambda p: p.get("file_size", 0))
                    file_id = largest_photo["file_id"]

                    # 下载图片
                    send_message(chat_id, "收到图片，正在处理...")
                    image_path = download_telegram_photo(file_id)

                    if image_path:
                        # 处理图像
                        process_image(image_path, chat_id)
                    else:
                        send_message(chat_id, "❌ 下载图片失败")

                # 处理文档消息（可能是图片文件）
                elif "document" in message:
                    document = message["document"]
                    mime_type = document.get("mime_type", "")

                    if mime_type and mime_type.startswith("image/"):
                        print(f"收到图片文档消息")

                        file_id = document["file_id"]
                        send_message(chat_id, "收到图片文档，正在处理...")
                        image_path = download_telegram_photo(file_id)

                        if image_path:
                            process_image(image_path, chat_id)
                        else:
                            send_message(chat_id, "❌ 下载图片失败")
                    else:
                        send_message(chat_id, f"❌ 不支持的文件类型: {mime_type}")

                # 处理文本消息
                elif "text" in message:
                    text = message["text"].strip()

                    if text == "/start":
                        send_message(chat_id,
                                   "🤖 欢迎使用 Qwen Remove 机器人！\n\n"
                                   "发送图片给我，我会使用 ComfyUI 处理图片并返回结果。\n\n"
                                   f"支持格式: JPG, PNG, JPEG")

                    elif text == "/help":
                        send_message(chat_id,
                                   "📖 使用说明:\n\n"
                                   "1. 发送图片给我\n"
                                   "2. 等待处理完成\n"
                                   "3. 接收处理结果\n\n"
                                   "命令:\n"
                                   "/start - 开始\n"
                                   "/help - 帮助")

                    else:
                        send_message(chat_id, "请发送图片给我，我会处理并返回结果")

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"运行出错: {e}")
            time.sleep(5)

    print("程序结束")


if __name__ == "__main__":
    main()
