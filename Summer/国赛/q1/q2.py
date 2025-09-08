import numpy as np
import math
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import os


def n_i(x):
    # 传入的是纳米，计算要用微米
    # 传入波长计算，当前类型的外延层折射率
    x = x / 1000
    # 碳化硅外延层波长-折射率
    n_i = (1+0.20075/(1+12.07224/x**2)+5.54861/(1-0.02641/x**2)+35.65066/(1-1268.24708/x**2))**0.5
    
    
    # n_i = 3.41 # 硅的折射率在温度为25的时候基本稳定在3.41    
    return n_i
    
def create_P_i(波长列表, lambda_1, m):
    """ 求P列表
    波长列表 单位 注意是nm
    lambda_1是参考波长nm
    m -- 级数差
    """
    P_list = []
    for lambda_i in 波长列表:
        P_i = m * lambda_1 / (lambda_1 - lambda_i)
        # print("参考波长 lambda_1 =", lambda_1)
        # print("当前波长 lambda_i =", lambda_i)
        # print("计算的 P_i =", P_i)
        P_list.append(P_i)
        m += 1
    return P_list

def create_epi_thickness_10(P_list, 波长列表, incident_angle=10.0):
    """10
    P_list极值级数列表
    波长列表 nm
    返回：每个波长对应的厚度列表单位 μm
    """
    # 碳化硅折射率在不同波长的情况下不一样
    
    # 计算相关参数
    theta0 = math.radians(incident_angle) # 入射角
    sin_theta = math.sin(theta0) # 换成sin
    
    
    # 厚度计算
    res = []
    for P, wavelen in zip(P_list, 波长列表):
        折射率 = n_i(wavelen)
        de = math.sqrt(折射率**2 - sin_theta**2)
        T_i = (P + 0.5) * (0.001 * wavelen) / (2*de)
        res.append(T_i)
    
    return res

def create_epi_thickness_15(P_list, 波长列表, incident_angle=15.0):
    """15
    返回：每个波长对应的厚度列表（单位 μm）
    """
    # 折射率 = 2.55  # 碳化硅折射率，从参考文献中获取的，之后可以代入一些常见的折射率
    
    # 计算相关参数
    theta0 = math.radians(incident_angle) # 入射角
    sin_theta = math.sin(theta0) # 换成sin
    
    # 厚度计算
    res = []
    for P, wavelen in zip(P_list, 波长列表):
        print("波长:",wavelen / 1000)
        折射率 = n_i(wavelen)
        print(折射率)
        de = math.sqrt(折射率**2 - sin_theta**2)
        T_i = (P + 0.5) * (0.001 * wavelen) / (2*de)
        res.append(T_i)
    
    return res

if __name__ == "__main__":
    file_10 = "附件1.xlsx"   # 10度数据
    file_15 = "附件2.xlsx"   # 15度数据

    # 平滑算法参数
    sg_window = 33   # 窗口长度，必须为奇数，可根据数据点密度调整
    sg_poly = 3      # 多项式阶数

    # 峰值检测参数
    peak_prominence = 0.5

    df10 = pd.read_excel(file_10, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
    df10 = df10.astype(float)
    k10 = df10['wavenumber'].values     # 波数 cm^-1
    R10_raw = df10['reflectivity'].values

    # 转换为波长 nm
    wavelens10 = (10**7) / k10   # nm 

    # 平滑处理
    R10_smooth = savgol_filter(R10_raw, sg_window, sg_poly)

    # 找极大峰
    peaks10, _ = find_peaks(R10_smooth, prominence=peak_prominence)
    
    
    peak_wavelengths10 = wavelens10[peaks10]
    peak_wavenumbers10 = k10[peaks10]
    print("附件1检测到的峰对应波长:", peak_wavelengths10.tolist())
    print("对应的波数:", peak_wavenumbers10.tolist())

    # 取第一个峰作为参考 lambda_1，剩下的波长用来计算 P
    lambda_1_10 = float(peak_wavelengths10[0])
    remaining_wavs10 = list(peak_wavelengths10[1:])  # 保持顺序：第二个峰开始到最后
    remaining_wavenums10 = list(peak_wavenumbers10[1:])
    # 使用原始 m 初始值
    m0 = 0
    P_list10 = create_P_i(remaining_wavs10, lambda_1_10, m0)

    use_wavs10 = remaining_wavs10
    use_wavenums10 = remaining_wavenums10

    # 基于 P_list 计算厚度（调用你原函数）
    thicknesses10 = create_epi_thickness_10(P_list10, use_wavs10, incident_angle=10.0)
    print("附件1厚度结果:", thicknesses10)
    print("平均:", np.nanmean(thicknesses10))
    print("标准差:", np.nanstd(thicknesses10))

    print("-" * 60)


    df15 = pd.read_excel(file_15, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
    df15 = df15.astype(float)
    k15 = df15['wavenumber'].values     # 波数 cm^-1
    R15_raw = df15['reflectivity'].values

    # 转换为波长 nm
    wavelengths15 = (10**7) / k15   # nm

    # 平滑
    R15_smooth = savgol_filter(R15_raw, sg_window, sg_poly)

    # 找峰
    peaks15, _ = find_peaks(R15_smooth, prominence=peak_prominence)
    peak_wavelengths15 = wavelengths15[peaks15]
    peak_wavenumbers15 = k15[peaks15]
    print("附件2 (15°) 检测到的峰对应波长（nm）:", peak_wavelengths15.tolist())
    print("附件2 (15°) 对应的波数（cm⁻¹）:", peak_wavenumbers15.tolist())


    lambda_1_15 = float(peak_wavelengths15[0])
    remaining_wavs15 = list(peak_wavelengths15[1:])
    remaining_wavenums15 = list(peak_wavenumbers15[1:])
    m0 = 0
    P_list15 = create_P_i(remaining_wavs15, lambda_1_15, m0)

    use_wavs15 = remaining_wavs15
    use_wavenums15 = remaining_wavenums15

    thicknesses15 = create_epi_thickness_15(P_list15, use_wavs15, incident_angle=15.0)
    print("附件2（15°）厚度结果 (μm):", thicknesses15)
    if len(thicknesses15) > 0:
        print("平均:", np.nanmean(thicknesses15))
        print("标准差:", np.nanstd(thicknesses15))
    print("-" * 60)
    
    results = []

    # ?结果保存到Excel
    if 'use_wavs10' in locals():
        for wav, wn, P, t in zip(use_wavs10, use_wavenums10, P_list10, thicknesses10):
            results.append({
                "角度": "10°",
                "波长 (nm)": wav,
                "波数 (cm⁻¹)": wn,
                "级数 P": P,
                "厚度 (μm)": t
            })

    if 'use_wavs15' in locals():
        for wav, wn, P, t in zip(use_wavs15, use_wavenums15, P_list15, thicknesses15):
            results.append({
                "角度": "15°",
                "波长 (nm)": wav,
                "波数 (cm⁻¹)": wn,
                "级数 P": P,
                "厚度 (μm)": t
            })

    df_res = pd.DataFrame(results)
    df_res.to_excel("res/res.xlsx", index=False)
