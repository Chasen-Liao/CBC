import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_thickness(file_path, angle_deg, refractive_index, mode="minima"):
    """
    根据反射率光谱计算外延层厚度
    mode="minima" -> 用极小值
    mode="maxima" -> 用极大值
    """
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return None, None, None

    df = pd.read_excel(file_path, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
    df = df.astype(float)

    wavenumbers = df['wavenumber'].values
    reflectivity = df['reflectivity'].values

    if mode == "minima":
        # 极小值 = 找反射率的负峰
        peaks, _ = find_peaks(-reflectivity, prominence=0.5)
        selected_wavenumbers = wavenumbers[peaks]
    else:
        # 极大值 = 直接找正峰
        peaks, _ = find_peaks(reflectivity, prominence=0.5)
        selected_wavenumbers = wavenumbers[peaks]

    if len(selected_wavenumbers) < 2:
        print(f"{file_path} 中 {mode} 点不足")
        return None, None, df

    delta_k = np.diff(selected_wavenumbers)
    avg_delta_k = np.mean(delta_k)

    theta_rad = np.deg2rad(angle_deg)
    thickness_cm = 1 / (2 * avg_delta_k * np.sqrt(refractive_index**2 - np.sin(theta_rad)**2))
    thickness_um = thickness_cm * 1e4

    return thickness_um, selected_wavenumbers, df


def plot_with_minima_maxima(df, minima_w, maxima_w, output_filename):
    """绘制反射率曲线并标记极大值与极小值"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['wavenumber'], df['reflectivity'], label="反射率光谱")

    if minima_w is not None:
        minima_df = pd.DataFrame({'wavenumber': minima_w})
        minima_points = pd.merge(minima_df, df, on='wavenumber', how='inner')
        plt.scatter(minima_points['wavenumber'], minima_points['reflectivity'],
                    color='red', marker='v', s=60, label="极小值")

    if maxima_w is not None:
        maxima_df = pd.DataFrame({'wavenumber': maxima_w})
        maxima_points = pd.merge(maxima_df, df, on='wavenumber', how='inner')
        plt.scatter(maxima_points['wavenumber'], maxima_points['reflectivity'],
                    color='blue', marker='^', s=60, label="极大值")

    plt.xlabel("波数 (cm⁻¹)")
    plt.ylabel("反射率 (%)")
    plt.title("反射率光谱（极大值 vs 极小值）")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"对比图已保存为 {output_filename}")


def main():
    refractive_index = 3.42
    path3, angle3 = "附件/附件3.xlsx", 10.0

    # 极小值法
    thick_min, minima_w, df = calculate_thickness(path3, angle3, refractive_index, mode="minima")
    # 极大值法
    thick_max, maxima_w, _ = calculate_thickness(path3, angle3, refractive_index, mode="maxima")

    if thick_min and thick_max:
        print(f"基于极小值计算厚度: {thick_min:.2f} μm")
        print(f"基于极大值计算厚度: {thick_max:.2f} μm")
        print(f"差异: {abs(thick_min - thick_max):.2f} μm")

        plot_with_minima_maxima(df, minima_w, maxima_w, "compare_minima_maxima.png")
    else:
        print("无法完成厚度对比")

if __name__ == "__main__":
    main()
