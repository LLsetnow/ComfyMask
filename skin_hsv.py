import cv2
import numpy as np

def nothing(x):
    pass

def main(video_path):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(32):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("视频为空或读取失败")
        return

    # 创建窗口
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    # 添加滑条
    cv2.createTrackbar("FrameIdx", "Controls", 0, len(frames)-1, nothing)

    cv2.createTrackbar("H_min", "Controls", 0, 179, nothing)
    cv2.createTrackbar("H_max", "Controls", 25, 179, nothing)
    cv2.createTrackbar("S_min", "Controls", 40, 255, nothing)
    cv2.createTrackbar("S_max", "Controls", 255, 255, nothing)
    cv2.createTrackbar("V_min", "Controls", 60, 255, nothing)
    cv2.createTrackbar("V_max", "Controls", 255, 255, nothing)

    while True:
        # 获取滑条值
        idx = cv2.getTrackbarPos("FrameIdx", "Controls")
        h_min = cv2.getTrackbarPos("H_min", "Controls")
        h_max = cv2.getTrackbarPos("H_max", "Controls")
        s_min = cv2.getTrackbarPos("S_min", "Controls")
        s_max = cv2.getTrackbarPos("S_max", "Controls")
        v_min = cv2.getTrackbarPos("V_min", "Controls")
        v_max = cv2.getTrackbarPos("V_max", "Controls")

        frame = frames[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 显示
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        # 按 q 退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(r"D:\AI_Graph\视频\输入\原视频_16fps\333.mp4")  # 替换为你的视频路径
