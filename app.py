import json
import os
from multiprocessing import Process, Queue
from flask import Flask, request
import paramiko  # Add this import for SCP functionality

from utils import systemInfo
from utils.ModelPlatformFactory import ModelPlatformFactory

modelType = "yolov8sRknn"
port = 5000

app = Flask(__name__)

# 消息队列，用于同步flask线程与主线程
hmq = Queue()

# 获得节点信息
@app.route('/device/nodeInfo', methods=['GET'])
def nodeInfo():
    sysInfo = systemInfo.GetCpuConstants()
    sysLoadAverage = systemInfo.GetLoadAverage()
    hmq.put({
        "modelName": "",
        "rstpUrl": "",
        "waringUrl": "",
        "resRtspUrl": "",
        "type": "state"
    })
    res = hmq.get()
    return json.dumps({
        "nodeType": modelType,
        "sysInfo": sysInfo,
        "sysLoadAverage": sysLoadAverage,
        "manageState": res
    })

# 开启视频流任务
@app.route('/device/startStream', methods=['POST'])
def startStream():
    data = request.json
    modelFile = data.get('modelFile')
    rstpUrl = data.get('rtspUrl')
    waringUrl = data.get('waringUrl')
    resRtspUrl = data.get('resRtspUrl')
    modelType = data.get('modelType')
    # modelFile = request.form.get("modelFile")
    # rstpUrl = request.form.get("rstpUrl")
    # waringUrl = request.form.get("waringUrl")
    # resRtspUrl = request.form.get("resRtspUrl")


    # 打印接收到的参数进行调试
    print(f"Received modelFile: {modelFile}")
    print(f"Received rstpUrl: {rstpUrl}")
    print(f"Received waringUrl: {waringUrl}")
    print(f"Received resRtspUrl: {resRtspUrl}")
    print(f"Received resRtspUrl: {modelType}")
    if not modelFile:
        return json.dumps({
            "code": 400,
            "message": "modelFile parameter is missing"
        })

    hmq.put({
        "modelFile": modelFile,
        "rstpUrl": rstpUrl,
        "waringUrl": waringUrl,
        "resRtspUrl": resRtspUrl,
        "type": "startStream"
    })
    res = hmq.get()
    return json.dumps({
        "code": 200,
        "message": res
    })

# 手动停止视频流任务
@app.route('/device/stopStream', methods=['GET'])
def stopStream():
    hmq.put({
        "modelName": "",
        "rstpUrl": "",
        "waringUrl": "",
        "resRtspUrl": "",
        "type": "stopStream"
    })
    res = hmq.get()
    return json.dumps({
        "code": 200,
        "message": res
    })

# 新增接口：通过 SCP 下载 RKNN 模型文件
@app.route('/device/downloadModel', methods=['POST'])
def downloadModel():
    data = request.json
    hostname = data.get('hostname')
    port = data.get('port', 22)  # Default SCP port is 22
    username = data.get('username')
    password = data.get('password')
    remote_path = data.get('remotePath')
    local_path = 'models/' + data.get('localFilename')

    if not all([hostname, username, password, remote_path, local_path]):
        return json.dumps({
            "code": 400,
            "message": "Missing required parameters"
        })

    try:
        # Create an SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port=port, username=username, password=password)

        # Use SCP to download the file
        sftp = ssh.open_sftp()
        sftp.get(remote_path, local_path)
        sftp.close()
        ssh.close()

        return json.dumps({
            "code": 200,
            "message": f"Model downloaded successfully to {local_path}"
        })
    except Exception as e:
        return json.dumps({
            "code": 500,
            "message": f"Failed to download model: {str(e)}"
        })

# 新增接口：返回 models 目录中的所有模型文件
@app.route('/device/listModels', methods=['GET'])
def listModels():
    models_dir = 'models'
    try:
        # List all files in the models directory
        model_files = os.listdir(models_dir)
        return json.dumps({
            "code": 200,
            "models": model_files
        })
    except Exception as e:
        return json.dumps({
            "code": 500,
            "message": f"Failed to list models: {str(e)}"
        })

# 新增接口：删除指定的模型文件
@app.route('/device/deleteModel', methods=['DELETE'])
def deleteModel():
    data = request.json
    filename = data.get('filename')
    models_dir = 'models'
    file_path = os.path.join(models_dir, filename)

    if not filename:
        return json.dumps({
            "code": 400,
            "message": "Filename parameter is missing"
        })

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return json.dumps({
                "code": 200,
                "message": f"Model file '{filename}' deleted successfully"
            })
        else:
            return json.dumps({
                "code": 404,
                "message": f"Model file '{filename}' not found"
            })
    except Exception as e:
        return json.dumps({
            "code": 500,
            "message": f"Failed to delete model file: {str(e)}"
        })

# 新增: 更新 modelType 的 API 端点
@app.route('/device/updateModelType', methods=['POST'])
def updateModelType():
    global modelType  # 访问全局变量

    data = request.json
    new_model_type = data.get('modelType')
    modelType = new_model_type

    if not new_model_type:
        return json.dumps({
            "code": 400,
            "message": "缺少 modelType 参数"
        })

    hmq.put({
        "type": "updateModelType",
        "modelType": new_model_type
    })

    return json.dumps({
        "code": 200,
        "message": f"{new_model_type}"
    })

def start_flask_app(queue):
    global hmq
    hmq = queue
    app.run(host='0.0.0.0', port=port)

Manage = ModelPlatformFactory(modelType)

if __name__ == '__main__':
    # 创建新的进程用于运行 Flask 应用
    flask_process = Process(target=start_flask_app, args=(hmq,))
    flask_process.start()

    # 启动管理线程
    Manage(hmq)

    # 等待 Flask 进程结束
    flask_process.join()

