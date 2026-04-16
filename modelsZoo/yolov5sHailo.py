import subprocess
import time

import cv2
import ffmpeg
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType,
                            InputVStreams, OutputVStreams)
from multiprocessing import Process, Queue, Event

from hailo_platform.pyhailort.pyhailort import HailoRTTimeout

from utils.RTSPPush import RTSPPush


class InferModel:
    def __init__(self, model_path, target):
        self.target = target
        self.hef = HEF(model_path)
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        # Define dataset params
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = []
        self.image_height, self.image_width, self.channels = self.input_vstream_info.shape
        self.modelRes = []
        for i in range(0, len(self.hef.get_output_vstream_infos())):
            output_vstream = self.hef.get_output_vstream_infos()[i]
            self.output_vstream_info.append(output_vstream)
            self.modelRes.append({
                "key": output_vstream.name,
                "shape": output_vstream.shape
            })

        self.anchors = [[116, 90, 156, 198, 373, 326],
                        [30, 61, 62, 45, 59, 119],
                        [10, 13, 16, 30, 33, 23]
                        ]
        self.thed = 0.4
        self.names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                      "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"]

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def getBox(self, res):
        bbox = []
        acx = 2
        resList = []
        # res = res[0]
        if res.shape[0] == 40:
            acx = 1
        if res.shape[0] == 20:
            acx = 0
        res_t = res.reshape(-1, 255)

        for a in range(0, 3):
            slice = res_t[:, 85 * a:85 * (a + 1)]
            res_ids = np.where((slice[:, 4]) > self.thed)[0]

            for res_id in res_ids:
                now = slice[res_id]
                ids = np.argmax(now[5:])
                chosen_row = int(res_id / res.shape[0])
                chosen_col = int(res_id % res.shape[0])
                x, y, w, h = (now[:4])
                x = (x * 2.0 - 0.5 + chosen_col) / res.shape[1]
                y = (y * 2.0 - 0.5 + chosen_row) / res.shape[1]
                w = (2.0 * w) * (2.0 * w) * self.anchors[acx][a * 2] / 640
                h = (2.0 * h) * (2.0 * h) * self.anchors[acx][a * 2 + 1] / 640
                bbox.append((ids, slice[res_id][4], x, y, w, h))
        max_bbox = {}
        for box in bbox:
            if box[0] not in max_bbox.keys() or box[1] > max_bbox[box[0]][1]:
                max_bbox[box[0]] = box
        for keyName in max_bbox:
            resList.append(max_bbox[keyName])
        return resList

    def drawPicture(self, img: np.ndarray, inferResults) -> np.ndarray:
        """
        渲染图片（将解析结果绘制到图片中，主要用于展示demo）
        """
        for box in inferResults:
            if box[1] > 0.5:
                img_h = img.shape[0]
                img_w = img.shape[1]
                x = box[2] * img_w
                y = box[3] * img_h
                w = box[4] * img_w
                h = box[5] * img_h
                # 左上
                pt1 = (int(x - w / 2), int(y - h / 2))
                # 右下
                pt2 = (int(x + w / 2), int(y + h / 2))

                cl = int(box[0].item())
                score = box[1].item()
                cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
                cv2.putText(img, '{0} {1:.2f}'.format(self.names[cl], score),
                            pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        img = np.uint8(img)
        return img

    def startStreamInfer(self, rtspUrl, event):
        def send(configured_network, event):
            configured_network.wait_for_activation(1000)
            vstreams_params = InputVStreamParams.make(configured_network)
            with InputVStreams(configured_network, vstreams_params) as vstreams:
                rtscap = cv2.VideoCapture(rtspUrl)
                try:
                    while not event.is_set():
                        success, image = rtscap.read()
                        if image.shape[-1] != 640:
                            image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
                        image = image[:, :, ::-1]  # BGR2RGB和HWC2CHW
                        image = image.astype(dtype=np.float32)
                        image = np.expand_dims(image, axis=0)
                        vstream_to_buffer = {vstream: image for vstream in vstreams}
                        for vstream, buff in vstream_to_buffer.items():
                            vstream.send(buff)
                except HailoRTTimeout:
                    pass
                finally:
                    rtscap.release()

        def recv(configured_network, vstreams_params, event, nums):
            configured_network.wait_for_activation(1000)
            rtscap = cv2.VideoCapture(rtspUrl)
            width = int(rtscap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(rtscap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rtsp_p = str(nums)
            push = RTSPPush(width, height, rtsp_p)

            with OutputVStreams(configured_network, vstreams_params) as vstreams:
                try:
                    while True:
                        success, image = rtscap.read()
                        for vstream in vstreams:
                            data = vstream.recv()
                            data = data / 255.0
                            boxs = self.getBox(data)
                            if success:
                                rec = self.drawPicture(image, boxs)
                            else:
                                continue
                            push.pushData(rec)
                except HailoRTTimeout:
                    push.relace()
                    rtscap.release()

        vstreams_params_groups = OutputVStreamParams.make_groups(self.network_group)
        recv_procs = []
        for vstreams_params in vstreams_params_groups:
            proc = Process(target=recv, args=(self.network_group, vstreams_params, event, len(recv_procs)))
            proc.start()
            recv_procs.append(proc)
        send_process = Process(target=send, args=(self.network_group, event))
        send_process.start()
        for rec in recv_procs:
            rec.join()
        send_process.join()


def imgStreamInfer(model_path, rtspUrl, event):
    target = VDevice()
    model = InferModel(model_path, target)
    print(rtspUrl)
    with model.network_group.activate(model.network_group_params):
        model.startStreamInfer(rtspUrl, event)

    target.release()


def Manage(sendQ):
    """ Hailo计算卡管理线程(若移植则需要重写此功能)
        Args：
            q(Queue): 消息队列，用于和此线程进行数据交互
    """
    event = Event()
    inferRtsp = None
    while True:
        res = sendQ.get()  # 阻塞等待其他线程传来的数据
        if res["type"] == "startStream":
            if inferRtsp is not None:
                event.set()
                inferRtsp.join()
                inferRtsp = None
                event.clear()
            inferRtsp = Process(target=imgStreamInfer, args=(res["modelName"], res["rstpUrl"], event))
            inferRtsp.start()
        elif res["type"] == "stopStream" and inferRtsp is not None:
            event.set()
            inferRtsp.join()
            inferRtsp = None
            event.clear()


if __name__ == '__main__':
    event = Event()
    imgStreamInfer("/home/hubu/Documents/hefs/yolov5s_sigmoid_actived.hef",
                   "rtsp://127.0.0.1:8554/chan1/sub/av_stream", event)

    # hmq = Queue()
    # flas = Process(target=Manage, args=(hmq,))
    # flas.start()
    # print("sleep......")
    # time.sleep(5)
    # hmq.put({
    #     "modelName": "/home/hubu/Documents/hefs/yolov5s_sigmoid_actived.hef",
    #     "rstpUrl": "rtsp://127.0.0.1:8554/chan1/sub/av_stream",
    #     "type": "startStream"
    # })
    # time.sleep(10)
    # print("stopStreamstopStreamstopStreamstopStreamstopStream")
    # hmq.put({
    #     "modelName": "",
    #     "rstpUrl": "",
    #     "type": "stopStream"
    # })
    # time.sleep(10)
    # print("enddddddddddddddddddd")
    # hmq.put({
    #     "modelName": "/home/hubu/Documents/hefs/yolov5s.hef",
    #     "rstpUrl": "rtsp://127.0.0.1:8554/chan1/sub/av_stream",
    #     "type": "startStream"
    # })
    # time.sleep(10)
    # print("stopStreamstopStreamstopStreamstopStreamstopStream")
    # hmq.put({
    #     "modelName": "",
    #     "rstpUrl": "",
    #     "type": "stopStream"
    # })
