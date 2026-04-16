from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.RTSPPush import RTSPPush
from utils.rknnpool import rknnPoolExecutor
import cv2
import time
from utils.func import myFunc

def imgStreamInfer(model_path, rtsp_url, waring_url, res_rtsp_url, event):
    #线程，提高线程可以提高帧数
    TPEs = 3
    ###初始化rknn线程池
    pool = rknnPoolExecutor(
            rknnModel=model_path,
            TPEs=3,
            func=myFunc)
    # 打开RTSP流并设置超时时间
    rtscap = cv2.VideoCapture(rtsp_url)

    # 设置打开超时时间为5000毫秒（5秒）
    rtscap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 20000)

    # 设置读取帧的超时时间为5000毫秒（5秒）
    rtscap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 20000)

    width = int(rtscap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rtscap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    push = None
    if res_rtsp_url != "":
        push = RTSPPush(width, height, res_rtsp_url)
    ##
    # 初始化异步所需要的帧
    if (rtscap.isOpened()):
        for i in range(TPEs + 1):
            ret, frame = rtscap.read()
            if not ret:
                rtscap.release()
                del pool
                exit(-1)
            pool.put(frame)
    frames, loopTime, initTime = 0, time.time(), time.time()
    ##
    while not event.is_set():
        success, image = rtscap.read()
        if not success:
            print("无法读取帧或流已结束")
            rtscap = cv2.VideoCapture(rtsp_url)
            continue
        #单核处理
        #img_with_detections = model.inferData(image)  # 获取处理后的图像
        #3个NPU多线程处理######################
        pool.put(image)
        img_with_detections, flag = pool.get()
        if flag == False:
            break
        #####################################
        if push is not None:
            push.pushData(img_with_detections)  # 只推流处理后的图像
    pool.release()
    rtscap.release()

if __name__ == "__main__":
    onnx_path = './models/yolov8s.rknn'