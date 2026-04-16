import onnx
import onnxruntime as ort
import cv2
import numpy as np
import time
import threading
from utils.RTSPPush import RTSPPush

class InferModel:
    def __init__(self, model_path):
        # 加载ONNX模型
        onnx_model = onnx.load(model_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")

        # 创建ONNX运行会话
        options = ort.SessionOptions()
        options.enable_profiling = True
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # 获取输入输出节点名称
        self.get_input_details()
        self.get_output_details()

        # 类别名称
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

        self.conf_threshold = 0.3
        self.iou_threshold = 0.5
        self.global_fps = 0.0

    def detect_objects(self, image):
        input_tensor, ratio = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        boxes, scores, class_ids = self.process_output(outputs, ratio)
        return boxes, scores, class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img, ratio = self.ratioresize(input_img)
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor, ratio

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output, ratio):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        if len(scores) == 0:
            return [], [], []
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions, ratio)
        indices = self.nms(boxes, scores, self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions, ratio):
        boxes = predictions[:, :4]
        boxes *= ratio
        boxes = self.xywh2xyxy(boxes)
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def ratioresize(self, im, color=114):
        shape = im.shape[:2]
        new_h, new_w = self.input_height, self.input_width
        padded_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * color
        r = min(new_h / shape[0], new_w / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        padded_img[: new_unpad[1], : new_unpad[0]] = im
        padded_img = np.ascontiguousarray(padded_img)
        return padded_img, 1 / r

    def nms(self, boxes, scores, iou_threshold):
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            keep_indices = np.where(ious < iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        return keep_boxes

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.maximum(box[3], boxes[:, 3])
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area
        iou = intersection_area / union_area
        return iou

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def drawPicture(self, img: np.ndarray, inferResults) -> np.ndarray:
        if inferResults.shape[0] != 0:
            boxes = inferResults[..., :4].astype(np.int32)
            scores = inferResults[..., 4]
            classes = inferResults[..., 5].astype(np.int32)
            for box, score, cl in zip(boxes, scores, classes):
                if cl < 0 or cl >= len(self.names):
                    continue  # 跳过不在范围内的类别
                left, top, right, bottom = box
                cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(img, f'{self.names[cl]} {score:.2f}', (left, top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    def infer(self, image: np.ndarray) -> np.ndarray:
        start_time = time.time()
        boxes, scores, class_ids = self.detect_objects(image)
        inference_time = time.time() - start_time
        self.global_fps = (self.global_fps + (1.0 / inference_time)) / 2.0
        if len(boxes) == 0:
            return image
        result = np.concatenate((boxes, scores[:, np.newaxis], class_ids[:, np.newaxis]), axis=1)
        img_with_detections = self.drawPicture(image, result)
        return img_with_detections


# def imgStreamInfer(model_path, rtsp_url, waring_url, res_rtsp_url, event):
#     model = InferModel(model_path)
#     rtscap = cv2.VideoCapture(rtsp_url)
#     width = int(rtscap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(rtscap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     push = None
#     if res_rtsp_url != "":
#         push = RTSPPush(width, height, res_rtsp_url)
#     while not event.is_set():
#         success, image = rtscap.read()
#         if not success:
#             continue
#         img_with_detections = model.infer(image)  # 获取处理后的图像
#         if push is not None:
#             push.pushData(img_with_detections)  # 只推流处理后的图像
#     rtscap.release()

def imgStreamInfer(model_path, rtsp_url, waring_url, res_rtsp_url, event):
    model = InferModel(model_path)

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

    while not event.is_set():
        success, image = rtscap.read()
        if not success:
            print("无法读取帧或流已结束")
            rtscap = cv2.VideoCapture(rtsp_url)
            continue
        img_with_detections = model.infer(image)  # 获取处理后的图像
        if push is not None:
            push.pushData(img_with_detections)  # 只推流处理后的图像

    rtscap.release()


if __name__ == "__main__":
    onnx_path = './models/yolov8s.onnx'
    #onnx_path = './models/yolov5s_rk3588.rknn'
