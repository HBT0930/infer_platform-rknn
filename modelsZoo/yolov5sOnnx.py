import onnx
import onnxruntime as ort
import cv2
import numpy as np

from utils.RTSPPush import RTSPPush

class InferModel:
    def __init__(self, model_path):
        onnx_model = onnx.load(model_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
        options = ort.SessionOptions()
        options.enable_profiling = True
        self.onnx_session = ort.InferenceSession(model_path)
        self.input_name = self.get_input_name()
        self.names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                      "bed",
                      "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave",
                      "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                      "hair drier",
                      "toothbrush"]
        self.anchors = [[116, 90, 156, 198, 373, 326],
                        [30, 61, 62, 45, 59, 119],
                        [10, 13, 16, 30, 33, 23]
                        ]
        self.thed = 0.4

    def xywh2xyxy(self, x):
        # [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def nms(self, dets, thresh):
        # dets:x1 y1 x2 y2 score class
        # x[:,n]就是取所有集合的第n个数据
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        # -------------------------------------------------------
        #   计算框的面积
        #	置信度从大到小排序
        # -------------------------------------------------------
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        # print(scores)
        keep = []
        index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
        # [::-1] 从最后一个元素到第一个元素复制一遍。倒序从而从大到小排序

        while index.size > 0:
            i = index[0]
            keep.append(i)
            # -------------------------------------------------------
            #   计算相交面积
            #	1.相交
            #	2.不相交
            # -------------------------------------------------------
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            overlaps = w * h
            # -------------------------------------------------------
            #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
            #	IOU小于thresh的框保留下来
            # -------------------------------------------------------
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep

    def filter_box(self, org_box, conf_thres, iou_thres):  # 过滤掉无用的框
        # -------------------------------------------------------
        #   删除为1的维度
        #	删除置信度小于conf_thres的BOX
        # -------------------------------------------------------
        org_box = np.squeeze(org_box)  # 删除数组形状中单维度条目(shape中为1的维度)
        # (25200, 9)
        # […,4]：代表了取最里边一层的所有第4号元素，…代表了对:,:,:,等所有的的省略。此处生成：25200个第四号元素组成的数组
        conf = org_box[..., 4] > conf_thres  # 0 1 2 3 4 4是置信度，只要置信度 > conf_thres 的
        box = org_box[conf == True]  # 根据objectness score生成(n, 9)，只留下符合要求的框
        # -------------------------------------------------------
        #   通过argmax获取置信度最大的类别
        # -------------------------------------------------------
        cls_cinf = box[..., 5:]  # 左闭右开（5 6 7 8），就只剩下了每个grid cell中各类别的概率
        cls = []
        for i in range(len(cls_cinf)):
            cls.append(int(np.argmax(cls_cinf[i])))  # 剩下的objecctness score比较大的grid cell，分别对应的预测类别列表
        all_cls = list(set(cls))  # 去重，找出图中都有哪些类别
        # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        # -------------------------------------------------------
        #   分别对每个类别进行过滤
        #   1.将第6列元素替换为类别下标
        #	2.xywh2xyxy 坐标转换
        #	3.经过非极大抑制后输出的BOX下标
        #	4.利用下标取出非极大抑制后的BOX
        # -------------------------------------------------------
        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []
            curr_out_box = []
            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])  # 左闭右开，0 1 2 3 4 5

            curr_cls_box = np.array(curr_cls_box)  # 0 1 2 3 4 5 分别是 x y w h score class
            curr_cls_box = self.xywh2xyxy(curr_cls_box)  # 0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
            curr_out_box = self.nms(curr_cls_box, iou_thres)  # 获得nms后，剩下的类别在curr_cls_box中的下标

            for k in curr_out_box:
                output.append(curr_cls_box[k])
        output = np.array(output)
        return output

    def getBox(self, inferResults):
        resList = self.filter_box(inferResults, self.thed, self.thed)
        return resList

    def drawPicture(self, img: np.ndarray, inferResults) -> np.ndarray:
        """
        渲染图片（将解析结果绘制到图片中，主要用于展示demo）
        """
        if inferResults.shape[0] != 0:
            boxes = inferResults[..., :4].astype(np.int32)  # x1 x2 y1 y2
            scores = inferResults[..., 4]
            classes = inferResults[..., 5].astype(np.int32)
            for box, score, cl in zip(boxes, scores, classes):
                top, left, right, bottom = box / 640
                top, left, right, bottom = int(top * img.shape[1]), int(left * img.shape[0]), \
                                           int(right * img.shape[1]), int(bottom * img.shape[0])
                cv2.rectangle(img, (top, left), (right, bottom), (255, 0, 0), 2)
                cv2.putText(img, '{0} {1:.2f}'.format(self.names[cl], score),
                            (top, left), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, image_numpy):
        """获取输入numpy"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy
        return input_feed

    def inferData(self, frame):
        image = frame
        if image.shape[-1] != 640:
            image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img = image[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        img /= 255.0
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        input_feed = self.get_input_feed(img.astype(np.float32))
        modelRes = self.onnx_session.run(None, input_feed)
        modelRes = self.getBox(modelRes)
        modelRes = self.drawPicture(frame, modelRes)
        return modelRes


def imgStreamInfer(model_path, rtsp_url, waring_url, res_rtsp_url, event):
    model = InferModel(model_path)
    rtscap = cv2.VideoCapture(rtsp_url)
    width = int(rtscap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rtscap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    push = None
    if res_rtsp_url != "":
        push = RTSPPush(width, height, res_rtsp_url)
    while not event.is_set():
        success, image = rtscap.read()
        img = model.inferData(image)
        if push is not None:
            push.pushData(img)
    rtscap.release()

if __name__ == "__main__":
    onnx_path = './models/yolov5s_rk3588.rknn'
