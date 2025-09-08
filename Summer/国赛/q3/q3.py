import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持


def f(file_path, angle_deg, refractive_index,
                                     window_length=33, polyorder=3):
    """
    使用平滑之后求峰值列表，然后算平均，最后计算厚度
    """

    # 读取数据
    df = pd.read_excel(file_path, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
    df = df.astype(float) # 读入的 数据是浮点类型

    wavenumbers = df['wavenumber'].values
    reflectivity_raw = df['reflectivity'].values

    # 平滑
    smooth_re = savgol_filter(reflectivity_raw, window_length, polyorder)

    # 取峰值
    peaks, _ = find_peaks(smooth_re, prominence=0.5)
    maxima_wavenumbers = wavenumbers[peaks]

    # 计算波数差
    delta_v = np.diff(maxima_wavenumbers)
    avg_delta_v = np.mean(delta_v)

    # 厚度公式
    #! 这里要注意求出来的是cm，要转换为微米
    theta_rad = np.deg2rad(angle_deg)
    thickness_cm = 1 / (2 * avg_delta_v * np.sqrt(refractive_index**2 - np.sin(theta_rad)**2))
    thickness_um = thickness_cm * 1e4  # 单位换算

    return thickness_um, maxima_wavenumbers, df, smooth_re, delta_v, avg_delta_v


def main():
    refractive_index = 3.41 # 查阅资料这个
    # 3.41
    data3, angle3 = "附件3.xlsx", 10.0
    data4, angle4 = "附件4.xlsx", 15.0

    # 附件3
    thick3, maxima3, df3, smooth3, delta3, avg_delta3 = f(
        data3, angle3, refractive_index)
    # 附件4
    thick4, maxima4, df4, smooth4, delta4, avg_delta4 = f(
        data4, angle4, refractive_index)

    avg_thick = (thick3 + thick4) / 2
    print(f"附件3厚度: {thick3:.2f} μm")
    print(f"附件4厚度: {thick4:.2f} μm")
    print(f"平均厚度 = {avg_thick:.2f} μm")

    # 输出所有峰值波数
    print("附件3 峰值波数列表:", maxima3.tolist())
    print("附件4 峰值波数列表:", maxima4.tolist())

    # 输出相邻峰值差
    print("附件3 相邻峰值波数差:", delta3.tolist())
    print(f"附件3 平均波数差: {avg_delta3:.4f}")
    print("附件4 相邻峰值波数差:", delta4.tolist())
    print(f"附件4 平均波数差: {avg_delta4:.4f}")


if __name__ == "__main__":
    main()
