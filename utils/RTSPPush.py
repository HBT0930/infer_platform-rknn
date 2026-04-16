import subprocess

import numpy


# class RTSPPush:
#     def __init__(self, width, height, pushUrl):
#         rtsp_p = pushUrl
#         command = ['ffmpeg',
#                    '-y', '-an',
#                    '-re',
#                    '-f', 'rawvideo',
#                    '-pix_fmt', 'bgr24',
#                    '-s', str(width) + "x" + str(height),
#                    '-i', '-',
#                    '-c:v', 'libx264',
#                    '-g', '5',
#                    '-maxrate:v', '2M',
#                    '-minrate:v', '2M',
#                    '-bufsize:v', '1M',
#                    '-pix_fmt', 'yuv420p',
#                    '-preset', 'ultrafast',
#                    '-tune', 'animation',
#                    '-f', 'rtsp',
#                    rtsp_p]
#         self.pipe = subprocess.Popen(command
#                                      , shell=False
#                                      , stdin=subprocess.PIPE
#                                      )
#
#     def pushData(self, data: numpy):
#         self.pipe.stdin.write(data.tobytes())
#
#     def relace(self):
#         self.pipe.terminate()

import subprocess
import numpy as np

class RTSPPush:
    def __init__(self, width, height, pushUrl):
        rtsp_p = pushUrl
        command = [
                'ffmpeg',
            '-y', '-an',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',  # 设置视频宽高
            '-i', '-',  # 从标准输入读取视频数据
            '-c:v', 'libx264',  # 使用H.264编码
            '-g', '0',  # 调整关键帧间隔，减小延迟
            '-maxrate:v', '16M',
            '-minrate:v', '16M',
            '-bufsize:v', '8M',  # 调整缓冲区大小
            '-pix_fmt', 'yuv420p',  # 设置像素格式为YUV420P
            '-preset', 'ultrafast',  # 使用最快的编码预设
            '-tune', 'zerolatency',  # 调整编码以减少延迟
            '-f', 'rtsp',  # 输出为RTSP格式
            rtsp_p  # 推流地址
        ]
        # 启动FFmpeg子进程
        self.pipe = subprocess.Popen(command, shell=False,stdin=subprocess.PIPE)

    def pushData(self, data: np.ndarray):
        try:
            # 将numpy数组转为字节流写入FFmpeg
            self.pipe.stdin.write(data.tobytes())
        except BrokenPipeError:
            print("错误：向FFmpeg写入数据时管道断裂。")
        except Exception as e:
            print(f"错误：{e}")

    def release(self):
        # 安全终止FFmpeg进程
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.terminate()
            self.pipe.wait()


