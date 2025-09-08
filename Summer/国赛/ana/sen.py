import numpy as np
import math
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
# 折射率公式
def n_i(x):
    # 输入单位 nm，计算要用 μm
    x = x / 1000
    # 碳化硅外延层波长-折射率公式
    n_i = (1 + 0.20075 / (1 + 12.07224 / x**2) +
           5.54861 / (1 - 0.02641 / x**2) +
           35.65066 / (1 - 1268.24708 / x**2))**0.5 + 0.1
    return n_i

# 生成级数 P
def create_P_i(波长列表, lambda_1, m):
    P_list = []
    for lambda_i in 波长列表:
        P_i = m * lambda_1 / (lambda_1 - lambda_i)
        P_list.append(P_i)
        m += 1
    return P_list

# 10° 厚度计算
def create_epi_thickness_10(P_list, 波长列表, incident_angle=10.0):
    theta0 = math.radians(incident_angle)
    sin_theta = math.sin(theta0)

    res = []
    for P, wavelen in zip(P_list, 波长列表):
        折射率 = n_i(wavelen)
        de = math.sqrt(折射率**2 - sin_theta**2)
        T_i = (P + 0.5) * (0.001 * wavelen) / (2 * de)
        res.append(T_i)
    return res

# 15° 厚度计算
def create_epi_thickness_15(P_list, 波长列表, incident_angle=15.0):
    theta0 = math.radians(incident_angle)
    sin_theta = math.sin(theta0)

    res = []
    for P, wavelen in zip(P_list, 波长列表):
        折射率 = n_i(wavelen)
        de = math.sqrt(折射率**2 - sin_theta**2)
        T_i = (P + 0.5) * (0.001 * wavelen) / (2 * de)
        res.append(T_i)
    return res

# 灵敏度分析函数
def analyze_thickness_vs_window(file_path, angle, create_thickness_func,peak_prominence=0.5, polyorder=3,window_sizes=range(11, 81, 2)):
    results_mean = []
    results_std = []

    # 读入数据
    df = pd.read_excel(file_path, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
    df = df.astype(float)
    v = df['wavenumber'].values     # 波数
    R_raw = df['reflectivity'].values
    wavelengths = (10**7) / v   # 转换 nm

    for sg_window in window_sizes:
        try:
            # 平滑
            R_smooth = savgol_filter(R_raw, sg_window, polyorder)
            # 找峰
            peaks, _ = find_peaks(R_smooth, prominence=peak_prominence)
            peak_wavs = wavelengths[peaks]

            if len(peak_wavs) < 2:  # 至少要两个点才有级数
                results_mean.append(np.nan)
                results_std.append(np.nan)
                continue

            lambda_1 = float(peak_wavs[0])
            remaining_wavs = list(peak_wavs[1:])
            m0 = 0
            P_list = create_P_i(remaining_wavs, lambda_1, m0)

            # 计算厚度
            thicknesses = create_thickness_func(P_list, remaining_wavs, incident_angle=angle)
            avg_thickness = np.nanmean(thicknesses)
            std_thickness = np.nanstd(thicknesses)

            results_mean.append(avg_thickness)
            results_std.append(std_thickness)
        except Exception as e:
            results_mean.append(np.nan)
            results_std.append(np.nan)

    return list(window_sizes), results_mean, results_std


if __name__ == "__main__":
    data10 = "附件1.xlsx"   # 10° 数据
    data15 = "附件2.xlsx"   # 15° 数据

    # 灵敏度分析
    win10, thickness10_mean, thickness10_std = analyze_thickness_vs_window(data10, 10.0, create_epi_thickness_10)
    win15, thickness15_mean, thickness15_std = analyze_thickness_vs_window(data15, 15.0, create_epi_thickness_15)
    
    mean_10 = np.nanmean(thickness10_mean)
    mean_15 = np.nanmean(thickness15_mean)
    print(f"10° 所有窗口平均厚度的总体平均值: {mean_10:.4f} μm")
    print(f"15° 所有窗口平均厚度的总体平均值: {mean_15:.4f} μm")
    plt.figure(figsize=(10,6))
    plt.plot(win10, thickness10_mean, marker='o', label="10° 平均厚度")
    plt.plot(win15, thickness15_mean, marker='s', label="15° 平均厚度")
    plt.xlabel("平滑窗口大小")
    plt.ylabel("平均厚度 (μm)")
    plt.title("平滑窗口大小对厚度计算平均值的影响")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(win10, thickness10_std, marker='o', label="10° 厚度标准差")
    plt.plot(win15, thickness15_std, marker='s', label="15° 厚度标准差")
    plt.xlabel("平滑窗口大小")
    plt.ylabel("厚度标准差 (μm)")
    plt.title("平滑窗口大小对厚度计算稳定性的影响")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()
