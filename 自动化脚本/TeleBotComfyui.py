import json
from urllib import request, error
import time
import os
import shutil
import subprocess
import random
import requests
from typing import Optional


class ComfyUIWorkflow:
    """ComfyUI工作流处理类"""

    def __init__(self, seed_id=65, input_image_id=41, output_image_id=181, workflow='Qwen_remove.json'):
        """
        初始化工作流处理器
        :param seed_id: 种子节点ID
        :param input_image_id: 输入图像节点ID
        :param output_image_id: 输出图像节点ID
        :param workflow: 工作流JSON文件名
        """
        self.seed_id = str(seed_id)
        self.input_image_id = str(input_image_id)
        self.output_image_id = str(output_image_id)
        self.workflow_file = workflow
        self.original_workflow = None

    def load_workflow(self):
        """加载工作流JSON文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_path = os.path.join(script_dir, self.workflow_file)

        if not os.path.exists(workflow_path):
            raise FileNotFoundError(f"找不到工作流文件: {workflow_path}")

        with open(workflow_path, 'r', encoding='utf-8') as f:
            self.original_workflow = json.load(f)

        return True

    def set_seed(self, seed_value):
        """
        设置随机种子
        :param seed_value: 种子值
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        self.original_workflow[self.seed_id]["inputs"]["seed"] = int(seed_value)

    def set_input_image(self, image_filename):
        """
        设置输入图像
        :param image_filename: 图像文件名
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        self.original_workflow[self.input_image_id]["inputs"]["image"] = image_filename

    def set_output_prefix(self, output_prefix):
        """
        设置输出文件前缀
        :param output_prefix: 输出文件前缀
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        self.original_workflow[self.output_image_id]["inputs"]["filename_prefix"] = output_prefix

    def get_workflow(self):
        """获取当前工作流配置"""
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        return self.original_workflow

    def create_workflow_copy(self):
        """
        创建工作流的深拷贝
        :return: 工作流的副本
        """
        if not self.original_workflow:
            raise RuntimeError("工作流未加载，请先调用 load_workflow()")

        return json.loads(json.dumps(self.original_workflow))


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
AUTHORIZED_USER_IDS = [5468961835]  # 设置授权用户的Telegram ID列表, 为空则允许所有用户

# ComfyUI配置
COMFYUI_FOLDER = r"D:\AI_Graph\ConfyUI-aki\ComfyUI-aki-v1"
COMFYUI_INPUT_FOLDER = os.path.join(COMFYUI_FOLDER, "input")
COMFYUI_OUTPUT_FOLDER = os.path.join(COMFYUI_FOLDER, "output")

# 工作流配置
WORKFLOW_CONFIGS = {
    "面部重绘": {
        "seed_id": 9,
        "input_image_id": 27,
        "output_image_id": 72,
        "workflow": "FaceFix.json",
        "remove_iterations": 3  # 面部重绘只处理1次
    },
    "去除背景杂物": {
        "seed_id": 65,
        "input_image_id": 41,
        "output_image_id": 224,
        "workflow": "BackgroundRemove.json",
        "remove_iterations": 1  # 去除背景杂物只处理1次
    },
    "RC": {
        "seed_id": 65,
        "input_image_id": 41,
        "output_image_id": 181,
        "workflow": "Qwen_remove.json",
        "remove_iterations": 1  # RC默认处理1次
    },
    "BF": {
        "seed_id": 137,
        "input_image_id": 41,
        "output_image_id": 181,
        "workflow": "boobs_fix.json",
        "remove_iterations": 1  # BF只处理1次
    }
}

# 默认工作流配置
DEFAULT_WORKFLOW = "RC"  # 可选值: "面部重绘", "去除背景杂物", "RC", "BF"

# 用户自定义工作流配置
user_workflows = {}  # {chat_id: workflow_name}


def generate_random_seed():
    """
    生成15位随机数种子，如：297364725394981
    """
    return random.randint(10**14, 10**15 - 1)


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


# ComfyUI 服务器状态监控
comfyui_running = True  # 服务器运行状态


def queue_prompt(prompt_workflow, max_retries=3, retry_delay=2):
    """
    将prompt workflow发送到ComfyUI服务器并排队执行
    返回 prompt_id，如果失败返回 None
    http://127.0.0.1:8188/prompt
    """
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)

    for attempt in range(max_retries):
        try:
            response = request.urlopen(req)
            result = json.loads(response.read().decode('utf-8'))
            prompt_id = result.get('prompt_id')
            print(f"    工作流已提交，prompt_id: {prompt_id}")
            return prompt_id
        except Exception as e:
            print(f"    发送失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None


def wait_for_completion(prompt_id, check_interval=2, timeout=300):
    """
    轮询检查任务完成状态
    :param prompt_id: 工作流的唯一ID
    :param check_interval: 检查间隔（秒）
    :param timeout: 超时时间（秒）
    :return: True表示完成，False表示超时或错误
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            req = request.Request(f"http://127.0.0.1:8188/history/{prompt_id}")
            response = request.urlopen(req, timeout=3)
            result = json.loads(response.read().decode('utf-8'))

            if prompt_id in result:
                history_data = result[prompt_id]
                status = history_data.get('status', {}).get('completed', False)
                if status:
                    print(f"    任务已完成 (耗时: {int(time.time() - start_time)}秒)")
                    return True

                if history_data.get('status', {}).get('exec_info', None):
                    exec_info = history_data['status'].get('exec_info')
                    if exec_info and 'error' in str(exec_info).lower():
                        print(f"    任务执行出错: {exec_info}")
                        return False

        except error.HTTPError as e:
            if e.code == 404:
                pass
            else:
                print(f"    检查状态时出错: {e}")
        except Exception as e:
            print(f"    检查状态时出错: {e}")

        time.sleep(check_interval)

    print(f"    等待超时 (超过 {timeout} 秒)")
    return False


def save_image_with_unique_name(source_path, target_folder):
    """
    保存图像文件到指定文件夹，如果文件名重复则使用随机种子重命名
    :param source_path: 源文件路径
    :param target_folder: 目标文件夹
    :return: 保存后的文件名
    """
    original_filename = os.path.basename(source_path)
    file_ext = os.path.splitext(original_filename)[1]
    image_basename = os.path.splitext(original_filename)[0]

    # 尝试使用原始文件名
    target_path = os.path.join(target_folder, original_filename)

    if not os.path.exists(target_path):
        shutil.copy2(source_path, target_path)
        return original_filename
    else:
        # 文件已存在，使用随机种子重命名
        random_seed = generate_random_seed()
        new_filename = f"{random_seed}{file_ext}"
        target_path = os.path.join(target_folder, new_filename)

        # 检查新文件名是否也存在，如果存在则继续生成
        while os.path.exists(target_path):
            random_seed = generate_random_seed()
            new_filename = f"{random_seed}{file_ext}"
            target_path = os.path.join(target_folder, new_filename)

        shutil.copy2(source_path, target_path)
        return new_filename


# Telegram API函数
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


def process_image(image_path, chat_id: str, workflow_name: str):
    """
    处理图像并发送结果到Telegram
    :param image_path: 图像文件路径
    :param chat_id: Telegram聊天ID
    :param workflow_name: 工作流名称（从WORKFLOW_CONFIGS中获取）
    """

    # 检查 ComfyUI 服务器状态
    if not comfyui_running:
        send_message(chat_id, "❌ ComfyUI 服务器未运行，无法处理图片")
        return

    try:
        # 获取工作流配置
        if workflow_name not in WORKFLOW_CONFIGS:
            send_message(chat_id, f"错误: 未知的工作流 {workflow_name}")
            return

        config = WORKFLOW_CONFIGS[workflow_name]
        remove_iterations = config.get("remove_iterations", 1)

        send_message(chat_id, f"开始{workflow_name}处理... (将进行{remove_iterations}次迭代)")

        # 保存图像到ComfyUI input文件夹
        print(f"保存图像到 input 文件夹...")
        image_filename = save_image_with_unique_name(image_path, COMFYUI_INPUT_FOLDER)
        image_basename = os.path.splitext(image_filename)[0]

        print(f"图像文件名: {image_filename}")

        # 初始化工作流处理器
        workflow_handler = ComfyUIWorkflow(
            seed_id=config["seed_id"],
            input_image_id=config["input_image_id"],
            output_image_id=config["output_image_id"],
            workflow=config["workflow"]
        )

        # 加载工作流
        try:
            workflow_handler.load_workflow()
        except FileNotFoundError as e:
            send_message(chat_id, str(e))
            return

        # 进行多次处理
        suffixes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

        for iteration in range(remove_iterations):
            # 检查服务器状态
            if not comfyui_running:
                send_message(chat_id, f"❌ ComfyUI 服务器已关闭，处理中止（已完成 {iteration}/{remove_iterations}）")
                return

            # 创建工作流副本
            prompt_workflow = workflow_handler.create_workflow_copy()

            # 设置随机种子
            seed_value = generate_random_seed()
            prompt_workflow[workflow_handler.seed_id]["inputs"]["seed"] = int(seed_value)

            current_suffix = suffixes[iteration]
            output_prefix = f"AutoOutput\\{image_basename}_{current_suffix}"

            # 修改工作流参数
            prompt_workflow[workflow_handler.input_image_id]["inputs"]["image"] = image_filename
            prompt_workflow[workflow_handler.output_image_id]["inputs"]["filename_prefix"] = output_prefix

            print(f"\n迭代 {iteration+1}/{remove_iterations}: {current_suffix}")
            send_message(chat_id, f"处理中... {iteration+1}/{remove_iterations} ({current_suffix})")

            # 提交工作流
            prompt_id = queue_prompt(prompt_workflow)

            if not prompt_id:
                send_message(chat_id, f"提交工作流失败，跳过此迭代")
                continue

            # 等待任务完成
            if not wait_for_completion(prompt_id, check_interval=2, timeout=300):
                send_message(chat_id, f"任务未完成，继续下一个迭代")
                continue

            # 获取并发送当前迭代的结果
            search_pattern = f"{image_basename}_{current_suffix}"
            output_file = None

            # 在output文件夹中查找匹配的文件
            for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
                for file in files:
                    if file.startswith(search_pattern):
                        output_file = os.path.join(root, file)
                        break
                if output_file:
                    break

            if output_file and os.path.exists(output_file):
                send_photo(chat_id, output_file, f"{workflow_name} - 处理结果 {iteration+1}/{remove_iterations}")
                time.sleep(1)  # 避免发送过快
            else:
                send_message(chat_id, f"未找到迭代 {iteration+1} 的输出文件")

        send_message(chat_id, f"✅ {workflow_name}处理完成！共发送 {remove_iterations} 张处理结果")

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


def is_authorized(user_id: int) -> bool:
    """检查用户是否授权"""
    if not AUTHORIZED_USER_IDS:  # 列表为空，允许所有用户
        return True
    return user_id in AUTHORIZED_USER_IDS


def get_user_workflow(chat_id: str) -> str:
    """获取用户当前的工作流设置"""
    return user_workflows.get(chat_id, DEFAULT_WORKFLOW)


def set_user_workflow(chat_id: str, workflow_name: str) -> bool:
    """设置用户的工作流"""
    if workflow_name in WORKFLOW_CONFIGS:
        user_workflows[chat_id] = workflow_name
        return True
    return False


def monitor_comfyui_server():
    """
    持续监控 ComfyUI 服务器状态
    如果服务器关闭，更新全局状态
    """
    global comfyui_running
    while comfyui_running:
        try:
            response = request.urlopen("http://127.0.0.1:8188", timeout=2)
            # 服务器正常运行
            time.sleep(5)  # 每5秒检查一次
        except error.URLError:
            print("⚠️ ComfyUI 服务器已关闭！")
            comfyui_running = False
            break
        except Exception as e:
            print(f"⚠️ 检测 ComfyUI 服务器时出错: {e}")
            comfyui_running = False
            break


def notify_users_comfyui_down():
    """
    通知所有等待中的用户 ComfyUI 服务器已关闭
    """
    for chat_id in user_workflows.keys():
        try:
            send_message(chat_id, "⚠️ ComfyUI 服务器已关闭，无法处理图片。请检查服务器状态。")
        except:
            pass


def main():
    global comfyui_running
    print("========== Telegram机器人启动 ==========")
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")

    if AUTHORIZED_USER_IDS:
        print(f"授权用户ID: {AUTHORIZED_USER_IDS}")
    else:
        print(f"授权用户ID: 所有用户（测试模式）")
        print("⚠️  警告: 未设置授权用户ID列表，允许所有用户访问（建议设置AUTHORIZED_USER_IDS）")
        print("   如何获取用户ID: 发送消息给 @userinfobot")

    print(f"默认工作流: {DEFAULT_WORKFLOW}")
    print(f"可用工作流: {list(WORKFLOW_CONFIGS.keys())}")

    # 检查ComfyUI服务器
    print("\n检查ComfyUI服务器状态...")
    if not check_comfyui_server():
        print("ComfyUI未运行，是否启动？(y/n): ", end="")
        # 注意: 在实际运行时可能需要自动启动
        # user_input = input().strip().lower()
        # if user_input == 'y':
        #     start_comfyui()
        # else:
        #     print("请先启动ComfyUI")
        #     return
        print("请先启动ComfyUI服务器")
        return
    else:
        print("ComfyUI服务器已运行")

    # 启动 ComfyUI 服务器监控线程
    print("\n启动 ComfyUI 服务器监控...")
    import threading
    monitor_thread = threading.Thread(target=monitor_comfyui_server, daemon=True)
    monitor_thread.start()

    offset = 0

    print("\n========== 开始监听消息 ==========")

    while True:
        try:
            # 检查 ComfyUI 服务器状态
            if not comfyui_running:
                print("⚠️ ComfyUI 服务器未运行，暂停处理新请求...")
                time.sleep(5)
                continue

            updates = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"timeout": 30, "offset": offset if offset else None},
                timeout=40,
                proxies=PROXY_SETTINGS
            ).json()

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

                # 检查 ComfyUI 服务器状态
                if not comfyui_running:
                    send_message(chat_id, "⚠️ ComfyUI 服务器未运行，无法处理图片。请检查服务器状态。")
                    continue

                # 处理图片消息
                if "photo" in message:
                    print(f"收到图片消息")

                    # 获取最大尺寸的照片
                    photos = message["photo"]
                    largest_photo = max(photos, key=lambda p: p.get("file_size", 0))
                    file_id = largest_photo["file_id"]
                    print(f"  file_id: {file_id}")

                    # 下载图片
                    workflow_name = get_user_workflow(chat_id)
                    send_message(chat_id, f"收到图片，正在使用「{workflow_name}」处理...")
                    print("  开始下载图片...")
                    image_path = download_telegram_photo(file_id)

                    if image_path:
                        print(f"  图片已下载: {image_path}")
                        # 直接处理图像（在新线程中执行，避免阻塞）
                        try:
                            import threading
                            thread = threading.Thread(
                                target=process_image,
                                args=(image_path, chat_id, workflow_name)
                            )
                            thread.daemon = True
                            thread.start()
                        except Exception as e:
                            print(f"启动处理线程失败: {e}")
                            send_message(chat_id, f"❌ 启动处理失败: {e}")
                    else:
                        print("  下载图片失败")
                        send_message(chat_id, "❌ 下载图片失败")

                # 处理文档消息（可能是图片文件）
                elif "document" in message:
                    document = message["document"]
                    mime_type = document.get("mime_type", "")

                    if mime_type and mime_type.startswith("image/"):
                        print(f"收到图片文档消息")

                        file_id = document["file_id"]
                        workflow_name = get_user_workflow(chat_id)
                        send_message(chat_id, f"收到图片文档，正在使用「{workflow_name}」处理...")
                        image_path = download_telegram_photo(file_id)

                        if image_path:
                            # 直接处理图像（在新线程中执行，避免阻塞）
                            try:
                                import threading
                                thread = threading.Thread(
                                    target=process_image,
                                    args=(image_path, chat_id, workflow_name)
                                )
                                thread.daemon = True
                                thread.start()
                            except Exception as e:
                                print(f"启动处理线程失败: {e}")
                                send_message(chat_id, f"❌ 启动处理失败: {e}")
                        else:
                            send_message(chat_id, "❌ 下载图片失败")
                    else:
                        send_message(chat_id, f"❌ 不支持的文件类型: {mime_type}")

                # 处理文本消息
                elif "text" in message:
                    text = message["text"].strip()

                    if text == "/start":
                        current_workflow = get_user_workflow(chat_id)
                        send_message(chat_id,
                                   "🤖 欢迎使用 ComfyUI 图像处理机器人！\n\n"
                                   f"发送图片给我，将使用「{current_workflow}」处理方式。\n\n"
                                   f"支持格式: JPG, PNG, JPEG\n\n"
                                   f"当前处理方式: {current_workflow}\n"
                                   "可用的处理方式:\n"
                                   "• 面部重绘\n"
                                   "• 去除背景杂物\n"
                                   "• RC (多次迭代)\n"
                                   "• BF\n\n"
                                   "发送「面部重绘」、「去除背景杂物」、「RC」、「BF」来切换处理方式")

                    elif text == "/help":
                        current_workflow = get_user_workflow(chat_id)
                        send_message(chat_id,
                                   "📖 使用说明:\n\n"
                                   "1. 发送图片给我\n"
                                   "2. 图片将自动使用当前处理方式处理\n"
                                   "3. 等待处理完成\n"
                                   "4. 接收处理结果\n\n"
                                   "命令:\n"
                                   "/start - 开始\n"
                                   "/help - 帮助\n"
                                   "切换处理方式: 发送「面部重绘」、「去除背景杂物」、「RC」、「BF」\n\n"
                                   f"当前处理方式: {current_workflow}")

                    # 切换工作流
                    elif text in ["面部重绘", "去除背景杂物", "RC", "BF"]:
                        if set_user_workflow(chat_id, text):
                            send_message(chat_id, f"✅ 已切换到「{text}」处理方式")
                        else:
                            send_message(chat_id, f"❌ 切换失败")

                    else:
                        send_message(chat_id, "请发送图片给我，我会处理并返回结果\n\n发送「面部重绘」、「去除背景杂物」、「RC」、「BF」来切换处理方式")

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            comfyui_running = False  # 停止监控线程
            break
        except Exception as e:
            print(f"运行出错: {e}")
            time.sleep(5)

    print("程序结束")


if __name__ == "__main__":
    main()
