import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import os

def calculate_thickness_peaks_smooth(file_path, angle_deg, refractive_index,
                                     window_length=11, polyorder=3):
    """
    使用 Savitzky-Golay 平滑 + 峰值 (极大值) 计算厚度
    """


    # 读取数据
    df = pd.read_excel(file_path, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
    df = df.astype(float)

    wavenumbers = df['wavenumber'].values
    reflectivity_raw = df['reflectivity'].values

    # === Savitzky-Golay 平滑 ===
    reflectivity_smooth = savgol_filter(reflectivity_raw, window_length, polyorder)

    # === 峰值检测（极大值） ===
    peaks, _ = find_peaks(reflectivity_smooth, prominence=0.5)
    maxima_wavenumbers = wavenumbers[peaks]

    if len(maxima_wavenumbers) < 2:
        print(f"{file_path} 中极大值点不足")
        return None, None, df, reflectivity_smooth

    # 计算 Δk
    delta_k = np.diff(maxima_wavenumbers)
    avg_delta_k = np.mean(delta_k)

    # 厚度公式
    theta_rad = np.deg2rad(angle_deg)
    thickness_cm = 1 / (2 * avg_delta_k * np.sqrt(refractive_index**2 - np.sin(theta_rad)**2))
    thickness_um = thickness_cm * 1e4

    return thickness_um, maxima_wavenumbers, df, reflectivity_smooth


def plot_peaks_with_smoothing(df, reflectivity_smooth, maxima_w, output_filename):
    """绘制原始数据 vs 平滑后光谱，并标记峰值"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['wavenumber'], df['reflectivity'], label="原始光谱", alpha=0.6)
    plt.plot(df['wavenumber'], reflectivity_smooth, label="平滑后光谱", linewidth=2)

    if maxima_w is not None:
        maxima_df = pd.DataFrame({'wavenumber': maxima_w})
        maxima_points = pd.merge(maxima_df, df, on='wavenumber', how='inner')
        plt.scatter(maxima_points['wavenumber'], maxima_points['reflectivity'],
                    color='blue', marker='^', s=60, label="检测到的峰值")

    plt.xlabel("波数 (cm⁻¹)")
    plt.ylabel("反射率 (%)")
    plt.title("平滑 + 峰值检测")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"图已保存为 {output_filename}")


def main():
    refractive_index = 3.41
    # 碳化硅外延层折射率 -- 2.55
    path3, angle3 = "附件3.xlsx", 10.0
    path4, angle4 = "附件4.xlsx", 15.0

    # 附件3
    thick3, maxima3, df3, smooth3 = calculate_thickness_peaks_smooth(path3, angle3, refractive_index)
    # 附件4
    thick4, maxima4, df4, smooth4 = calculate_thickness_peaks_smooth(path4, angle4, refractive_index)

    if thick3 and thick4:
        avg_thick = (thick3 + thick4) / 2
        print(f"基于附件3 (10°) 平滑+峰值法: 厚度 = {thick3:.2f} μm")
        print(f"基于附件4 (15°) 平滑+峰值法: 厚度 = {thick4:.2f} μm")
        print(f"-> 平均厚度 = {avg_thick:.2f} μm")

        plot_peaks_with_smoothing(df3, smooth3, maxima3, "peaks_smooth_附件3.png")
        plot_peaks_with_smoothing(df4, smooth4, maxima4, "peaks_smooth_附件4.png")
    else:
        print("未能成功计算厚度")

if __name__ == "__main__":
    main()
