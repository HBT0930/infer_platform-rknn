import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# 导入数据
df = pd.read_csv("S Parameter Plot 1.csv")

# 定义金属片长度的唯一值
w2_values = df['W2 [mm]'].unique()

# 存储结果的列表
qe_values = []

# 循环遍历每个金属片长度
for w2 in w2_values:
    # 提取当前长度的数据
    df_w2 = df[df['W2 [mm]'] == w2]

    # 将频率和 S 参数转换为 NumPy 数组
    freq = df_w2['Freq [GHz]'].to_numpy()
    s11_db = df_w2['dB(S(1,1)) []'].to_numpy()

    # 创建插值函数
    s11_interp = interp1d(freq, s11_db)

    # 找到谐振频率（S11 最小值）
    peaks, _ = find_peaks(-s11_db)  # find_peaks 找最大值，所以要取反
    resonant_freq = freq[peaks[0]]

    # 找到 -3dB 带宽
    s11_resonance = s11_db[peaks[0]]
    lower_bound = resonant_freq
    upper_bound = resonant_freq
    while s11_interp(lower_bound) < s11_resonance + 3 and lower_bound > freq[0]:
        lower_bound -= 0.001  # 步长根据你的数据精度调整
    while s11_interp(upper_bound) < s11_resonance + 3 and upper_bound < freq[-1]:
        upper_bound += 0.001  # 步长根据你的数据精度调整
    bandwidth = upper_bound - lower_bound

    # 计算 Qe
    qe = resonant_freq / bandwidth
    qe_values.append(qe)

# 绘制 Qe 与金属片长度的关系图
plt.plot(w2_values, qe_values, 'o-')
plt.xlabel("W2 [mm]")
plt.ylabel("Qe")
plt.title("Qe vs. W2")
plt.grid(True)
plt.show()

# 打印结果
for w2, qe in zip(w2_values, qe_values):
    print(f"W2 = {w2:.1f} mm, Qe = {qe:.2f}")
