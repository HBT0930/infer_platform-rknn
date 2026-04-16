from importlib import import_module
from multiprocessing import Process, Event

import psutil

def ModelPlatformFactory(modelType: str = "onnx"):
    imgStreamInfer = getattr(import_module("modelsZoo." + modelType), 'imgStreamInfer')
    def Manage(sendQ):
        """ Hailo计算卡管理线程
            Args：
                q(Queue): 消息队列，用于和此线程进行数据交互
        """
        event = Event()
        inferRtsp = None
        while True:
            res = sendQ.get()  # 阻塞等待其他线程传来的数据
            if res["type"] == "startStream":
                if inferRtsp is None:
                    inferRtsp = Process(target=imgStreamInfer, args=(res["modelFile"], res["rstpUrl"],
                                                       res["waringUrl"], res["resRtspUrl"], event))
                    inferRtsp.start()
                    sendQ.put("successfully start.")
                else:
                    sendQ.put("faile task is runing.")
            elif res["type"] == "stopStream" and inferRtsp is not None:
                if inferRtsp is not None:
                    event.set()
                    inferRtsp.join()
                    inferRtsp = None
                    event.clear()
                    sendQ.put("successfully stopped.")
                else:
                    sendQ.put("faile task is no runing.")
            elif res["type"] == "state":
                status = "None"
                if inferRtsp is not None:
                    pid = inferRtsp.pid
                    p = psutil.Process(pid)
                    status = p.status()
                sendQ.put(status)
    return Manage
