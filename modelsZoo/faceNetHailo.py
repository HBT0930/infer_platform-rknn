import subprocess
import time

import cv2
import ffmpeg
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType,
                            InputVStreams, OutputVStreams)
from multiprocessing import Process, Queue, Event

from hailo_platform.pyhailort.pyhailort import HailoRTTimeout, InferVStreams

from utils.RTSPPush import RTSPPush

myFace = np.load('/home/hubu/Documents/hefs/myFaceF.npy')

class FaceFModel:
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

    def inferData(self, frame):
        image = frame
        img = image.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        with InferVStreams(self.network_group, self.input_vstreams_params,
                           self.output_vstreams_params) as infer_pipeline:
            input_data = {self.input_vstream_info.name: img}
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)[self.modelRes[0]["key"]]
        return infer_results


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
        self.names = ["yes", "no"]

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def getBox(self, reList):
        bbox = []
        acx = 2
        resList = []
        for k in reList:
            res = reList[k]
            res = res[0]
            res = np.array(res)
            if res.shape[0] == 40:
                acx = 1
            if res.shape[0] == 20:
                acx = 0
            res_t = res.reshape(-1, 18)

            for a in range(0, 3):
                slice = res_t[:, 6 * a:6 * (a + 1)]
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
            resList.append(list(max_bbox[keyName]))
        return resList

    def drawPicture(self, img: np.ndarray, inferResults) -> np.ndarray:
        """
        渲染图片（将解析结果绘制到图片中，主要用于展示demo）
        """
        for box in inferResults:
            if box[2] > 0.5:
                cl = box[0]
                cos = box[1]
                img_h = img.shape[0]
                img_w = img.shape[1]
                x = box[3] * img_w
                y = box[4] * img_h
                w = box[5] * img_w
                h = box[6] * img_h
                # 左上
                pt1 = (int(x - w / 2), int(y - h / 2))
                # 右下
                pt2 = (int(x + w / 2), int(y + h / 2))

                score = box[2].item()
                cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
                cv2.putText(img, '{0} face:{1:.2f}, person:{2:.2f}'.format(self.names[cl], score, cos),
                            pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        img = np.uint8(img)
        return img

    def getFaceImg(self, frame, boxs):
        res = []
        for box in boxs:
            if box[1] > 0.5:
                img_h = frame.shape[0]
                img_w = frame.shape[1]
                x = box[2] * img_w
                y = box[3] * img_h
                w = box[4] * img_w
                h = box[5] * img_h
                # 左上
                pt1 = (int(x - w / 2) if int(x - w / 2) > 0 else 0, int(y - h / 2) if int(y - h / 2) > 0 else 0,)
                # 右下
                pt2 = (int(x + w / 2) if int(x + w / 2) < img_h else -1, int(y + h / 2)if int(y + h / 2) < img_w else -1)
                img = frame[pt1[0]:pt2[0], pt1[1]:pt2[1], :]
                if img.shape[-1] != 160:
                    try:
                        img = cv2.resize(img, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
                    except:
                        continue
                res.append(img)
        return np.array(res)

    def numpy_cos(self, a, b):
        dot = a * b  # 对应原始相乘dot.sum(axis=1)得到内积
        a_len = np.linalg.norm(a, axis=0)  # 向量模长
        b_len = np.linalg.norm(b, axis=0)
        cos = dot.sum(axis=0) / (a_len * b_len)
        return cos

    def inferData(self, frame, faceFModel):
        image = frame
        if image.shape[-1] != 640:
            image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img = image[:, :, ::-1]
        img = img.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        with InferVStreams(self.network_group, self.input_vstreams_params,
                           self.output_vstreams_params) as infer_pipeline:
            input_data = {self.input_vstream_info.name: img}
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)
        modelRes = self.getBox(infer_results)
        faceRes = []
        if modelRes:
            res = self.getFaceImg(frame, modelRes)
            f = faceFModel.inferData(res)
            for i, face in enumerate(f):
                np.save("/home/hubu/Documents/hefs/myFaceF.npy", face)
                modelRes[i][0] = 1
                modelRes[i].insert(1, self.numpy_cos(face, myFace))
                if modelRes[i][1] > 0.6:
                    modelRes[i][0] = 0
                faceRes.append(modelRes[i])
        modelRes = self.drawPicture(frame, faceRes)
        return modelRes


def imgStreamInfer(model_path, rtspUrl, event):
    target = VDevice()
    model = InferModel(model_path, target)
    faceFModel = FaceFModel("/home/hubu/Documents/hefs/facenet_mobilenet1.hef", target)
    rtscap = cv2.VideoCapture(rtspUrl)
    width = int(rtscap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rtscap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rtsp_p = str(0)
    push = RTSPPush(width, height, rtsp_p)
    while not event.is_set():
        success, image = rtscap.read()
        if success:
            resImage = model.inferData(image, faceFModel)
            push.pushData(resImage)
    rtscap.release()


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
    imgStreamInfer("/home/hubu/Documents/hefs/best_searchface.hef",
                   "rtsp://admin:123456@192.168.31.68:554/ch01.264", event)

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
