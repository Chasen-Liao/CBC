import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

def setup_matplotlib_for_chinese():
    """配置Matplotlib以支持中文显示。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("Matplotlib 中文环境配置成功。")
    except Exception as e:
        print(f"配置中文环境失败: {e}。图表标题可能无法正确显示中文。")

def visualize_dual_beam_interference():
    """图表一：可视化双光束干涉原理"""
    x = np.linspace(0, 4 * np.pi, 200)
    # 光束1
    wave1 = np.sin(x)
    # 光束2，假设有 pi/2 的相位差
    phase_shift = np.pi / 2
    wave2 = np.sin(x - phase_shift)
    # 干涉结果
    interference = wave1 + wave2

    plt.figure(figsize=(12, 7))
    plt.plot(x, wave1, label='光束 1 (上表面反射)', linestyle='--', alpha=0.7)
    plt.plot(x, wave2, label=f'光束 2 (下表面反射, 相位差 {phase_shift/np.pi:.2f}π)', linestyle=':', alpha=0.7)
    plt.plot(x, interference, label='干涉结果 (两束光叠加)', linewidth=2.5, color='green')
    
    plt.title('原理一：双光束干涉模拟', fontsize=16)
    plt.xlabel('空间位置 / 时间', fontsize=12)
    plt.ylabel('振幅', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    filename = "principle_1_dual_beam_interference.png"
    plt.savefig(filename, dpi=300)
    print(f"图表一已保存为 '{filename}'")

def visualize_algorithm_steps(file_path='附件/附件1.xlsx'):
    """图表二：可视化厚度计算算法的关键步骤"""
    if not os.path.exists(file_path):
        print(f"错误：数据文件 {file_path} 未找到，无法生成算法可视化图。")
        return

    df = pd.read_excel(file_path, header=None, skiprows=1, names=['wavenumber', 'reflectivity']).astype(float)
    
    # 步骤1: 寻找极小值
    reflectivity_inv = -df['reflectivity'].values
    wavenumbers = df['wavenumber'].values
    peaks, _ = find_peaks(reflectivity_inv, prominence=0.5)
    
    # 步骤2: 计算Δk
    minima_wavenumbers = wavenumbers[peaks]
    delta_k = np.diff(minima_wavenumbers)
    mid_wavenumbers = (minima_wavenumbers[:-1] + minima_wavenumbers[1:]) / 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('原理二：厚度计算算法步骤可视化 (基于附件1数据)', fontsize=16)

    # 子图1: 寻找极小值
    axes[0].plot(wavenumbers, reflectivity_inv, label='反转的反射率谱线')
    axes[0].plot(minima_wavenumbers, reflectivity_inv[peaks], "x", color='red', markersize=10, label='算法识别的极小值点 (峰)')
    axes[0].set_title('步骤1: 在反转光谱上识别干涉谷值', fontsize=14)
    axes[0].set_xlabel('波数 (cm^-1)', fontsize=12)
    axes[0].set_ylabel('- 反射率', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 子图2: Δk一致性
    axes[1].plot(mid_wavenumbers, delta_k, 'o-', label='计算出的Δk值')
    mean_dk = np.mean(delta_k)
    axes[1].axhline(y=mean_dk, color='red', linestyle='--', label=f'Δk 平均值: {mean_dk:.2f} cm⁻¹')
    axes[1].set_title('步骤2: 检验相邻谷值波数差 (Δk) 的一致性', fontsize=14)
    axes[1].set_xlabel('波数中点 (cm^-1)', fontsize=12)
    axes[1].set_ylabel('Δk (cm^-1)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = "principle_2_algorithm_visualization.png"
    plt.savefig(filename, dpi=300)
    print(f"图表二已保存为 '{filename}'")

def visualize_multibeam_effect():
    """图表三：可视化多光束干涉效应"""
    delta = np.linspace(0, 4 * np.pi, 1000)
    
    def airy_reflectivity(R, delta):
        # R = r^2 是强度反射率
        return (4 * R * np.sin(delta / 2)**2) / ((1 - R)**2 + 4 * R * np.sin(delta / 2)**2)

    # 模拟不同强度反射率下的干涉条纹
    R1 = 0.1  # 低反射率, 接近双光束
    R2 = 0.6  # 高反射率, 显著多光束
    
    reflectivity1 = airy_reflectivity(R1, delta)
    reflectivity2 = airy_reflectivity(R2, delta)

    plt.figure(figsize=(12, 7))
    plt.plot(delta / (2*np.pi), reflectivity1, label=f'低反射率 (R={R1}), 接近双光束', linewidth=2)
    plt.plot(delta / (2*np.pi), reflectivity2, label=f'高反射率 (R={R2}), 显著多光束', linewidth=2, linestyle='--')
    
    # 标记极小值点的位置
    minima_x = [0, 1, 2]
    minima_y = [0, 0, 0]
    plt.scatter(minima_x, minima_y, color='red', marker='v', s=100, zorder=5, label='干涉极小值点位置')

    plt.title('原理三：多光束干涉效应 (艾里公式)', fontsize=16)
    plt.xlabel('相位差 δ / 2π (等效于干涉级次 m)', fontsize=12)
    plt.ylabel('归一化反射率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    filename = "principle_3_multibeam_effect.png"
    plt.savefig(filename, dpi=300)
    print(f"图表三已保存为 '{filename}'")

if __name__ == '__main__':
    setup_matplotlib_for_chinese()
    print("\n--- 正在生成原理一的可视化图表 ---")
    visualize_dual_beam_interference()
    print("\n--- 正在生成原理二的可视化图表 ---")
    visualize_algorithm_steps()
    print("\n--- 正在生成原理三的可视化图表 ---")
    visualize_multibeam_effect()
    print("\n所有原理可视化图表已生成完毕。")
