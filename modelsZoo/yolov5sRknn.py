from multiprocessing import Process, Event
import cv2
from rknnlite.api import RKNNLite
from utils.RTSPPush import RTSPPush
import numpy as np


class InferModel:
    def __init__(self, model_path):
        self.target = RKNNLite()
        self.hef = self.target.load_rknn(model_path)
        self.image_height, self.image_width, self.channels = (640, 640, 3)
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

    def postprocess(self, output):
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = 1
        y_factor = 1

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= 0.5:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)

        resList = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            box.append(score)
            box.append(class_id)
            resList.append(box)
            # Draw the detection on the input image
            # self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return resList

    def drawPicture(self, img: np.ndarray, inferResults, fps: int) -> np.ndarray:
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
        cv2.putText(img, str(fps), (100, 100), cv2.FONT_ITALIC, 2, (0, 255, 0), 5)
        return img

    def inferData(self, frame):
        image = frame
        if image.shape[-1] != 640:
            img = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        image = image.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        image /= 255.0
        image = np.expand_dims(image, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        outputs = self.target.inference(inputs=[image])
        outputs = self.postprocess(outputs)
        rec = self.drawPicture(frame, outputs)
        return rec


def imgStreamInfer(model_path, rtsp_url, waring_url, res_rtsp_url, event):
    # model = InferModel(model_path)
    # rtscap = cv2.VideoCapture(rtsp_url)
    # width = int(rtscap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(rtscap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # rtsp_p = rtspUrl + "/" + str(0)
    # push = RTSPPush(width, height, rtsp_p)
    # while not event.is_set():
    #     success, image = rtscap.read()
    #     img = model.inferData(image)
    #     push.pushData(img)
    # rtscap.release()
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
        img_with_detections = model.inferData(image)  # 获取处理后的图像
        if push is not None:
            push.pushData(img_with_detections)  # 只推流处理后的图像

    rtscap.release()



# def Manage(sendQ):
#     """ Hailo计算卡管理线程(若移植则需要重写此功能)
#         Args：
#             q(Queue): 消息队列，用于和此线程进行数据交互
#     """
#     event = Event()
#     inferRtsp = None
#     while True:
#         res = sendQ.get()  # 阻塞等待其他线程传来的数据
#         if res["type"] == "startStream" and inferRtsp is None:
#             inferRtsp = Process(target=imgStreamInfer, args=(res["modelName"], res["rstpUrl"], event))
#             inferRtsp.start()
#         elif res["type"] == "stopStream" and inferRtsp is not None:
#             event.set()
#             inferRtsp.join()
#             inferRtsp = None
#             event.clear()


if __name__ == "__main__":
    onnx_path = './models/yolov5s_rk3588.rknn'
