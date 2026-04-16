
import onnx
import onnxruntime as ort
import cv2
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import preprocessing
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

    def inferData(self, frame, waringUrl):
        image = frame
        if image.shape[-1] != 160:
            image = cv2.resize(image, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
        img = image[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        img /= 255.0
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        input_feed = self.get_input_feed(img.astype(np.float32))
        modelRes = self.onnx_session.run(None, input_feed)
        if waringUrl != "":
            self.isWaring(modelRes, waringUrl)
        return modelRes

    def isWaring(self, data, waringUrl):
        pass


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
        img = model.inferData(image, waring_url)
        if push is not None:
            push.pushData(img)
    rtscap.release()



if __name__ == "__main__":
    # TODO 完善第一阶段的
    onnx_path = '/home/hubu/Documents/hefs/facenet_mobilenet.onnx'
    model = InferModel(onnx_path)

    img1 = cv2.imread("/home/hubu/Documents/data/yolo_data/bus.jpg")
    outputs1 = model.inferData(img1)[0]

    img2 = cv2.imread("/home/hubu/Documents/data/yolo_data/test.jpg")
    outputs2 = model.inferData(img2)[0]

    l1 = np.linalg.norm(outputs1 - outputs2, axis=1)
    print("l1 %f" % l1)
    cosSim = 1 - pdist(np.vstack([outputs1, outputs2]), 'cosine')
    print("pdist %f" % cosSim)
    outputs1 = preprocessing.normalize(outputs1, norm='l2')
    outputs2 = preprocessing.normalize(outputs2, norm='l2')
    l1 = np.linalg.norm(outputs1 - outputs2, axis=1)
    print("after l2 l1 %f" % l1)
