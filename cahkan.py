import cv2

cap = cv2.VideoCapture("rtsp://192.168.50.234/live/test2")  # 打开RTSP流
if not cap.isOpened():
    print("无法打开视频流")
    exit()

while True:
    ret, frame = cap.read()  # 读取一帧视频
    if not ret:
        print("无法读取帧或流已结束")
        break

    cv2.imshow("frame", frame)  # 显示当前帧

    if cv2.waitKey(25) & 0xFF == ord('q'):  # 按下 'q' 键退出
        break

cv2.destroyAllWindows()  # 关闭所有显示的窗口
cap.release()  # 释放视频捕获对象
