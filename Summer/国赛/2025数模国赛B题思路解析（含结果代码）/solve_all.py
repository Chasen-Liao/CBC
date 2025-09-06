import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import os

def setup_matplotlib_for_chinese():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("Matplotlib 中文环境配置成功。")
    except Exception as e:
        print(f"配置中文环境失败: {e}。图表标题可能无法正确显示中文。")

def plot_spectrum_with_minima(df1, df2, angle1, angle2, material_name, output_filename, minima1_w=None, minima2_w=None):
    """
    绘制并保存光谱图，并可选地在图上标记极小值点。
    """
    try:
        plt.figure(figsize=(14, 8))
        plt.plot(df1['wavenumber'], df1['reflectivity'], label=f'入射角 {angle1}°', alpha=0.8)
        plt.plot(df2['wavenumber'], df2['reflectivity'], label=f'入射角 {angle2}°', alpha=0.8)

        title = f'{material_name} 晶圆片反射光谱'
        
        if minima1_w is not None and minima2_w is not None:
            minima1_df = pd.DataFrame({'wavenumber': minima1_w})
            minima1_points = pd.merge(minima1_df, df1, on='wavenumber', how='inner')
            
            minima2_df = pd.DataFrame({'wavenumber': minima2_w})
            minima2_points = pd.merge(minima2_df, df2, on='wavenumber', how='inner')

            plt.scatter(minima1_points['wavenumber'], minima1_points['reflectivity'], 
                        color='red', marker='v', s=60, zorder=5, label=f'{angle1}° 极小值点')
            plt.scatter(minima2_points['wavenumber'], minima2_points['reflectivity'], 
                        color='black', marker='x', s=60, zorder=5, label=f'{angle2}° 极小值点')
            title += ' (及检测到的极小值点)'

        plt.xlabel('波数 (cm^-1)', fontsize=12)
        plt.ylabel('反射率 (%)', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(output_filename, dpi=300)
        print(f"光谱图已保存为 '{output_filename}'")
    except Exception as e:
        print(f"绘制光谱图时出错: {e}")

def plot_advanced_analysis(delta_k1, delta_k2, mid_wavenumbers1, mid_wavenumbers2, angle1, angle2, material_name, output_filename):
    """
    绘制高级分析图表：Δk一致性图和统计分布图。
    """
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle(f'{material_name} - 高级可靠性分析', fontsize=18)

        # --- 子图1: Δk 一致性图 ---
        axes[0].plot(mid_wavenumbers1, delta_k1, 'o-', label=f'入射角 {angle1}°')
        axes[0].plot(mid_wavenumbers2, delta_k2, 'x-', label=f'入射角 {angle2}°')
        
        mean_dk1 = np.mean(delta_k1)
        mean_dk2 = np.mean(delta_k2)
        axes[0].axhline(y=mean_dk1, color='blue', linestyle='--', alpha=0.5, label=f'{angle1}° 平均值: {mean_dk1:.2f}')
        axes[0].axhline(y=mean_dk2, color='orange', linestyle='--', alpha=0.5, label=f'{angle2}° 平均值: {mean_dk2:.2f}')
        
        axes[0].set_title('相邻极小值点波数差 (Δk) 的一致性', fontsize=14)
        axes[0].set_xlabel('波数中点 (cm^-1)', fontsize=12)
        axes[0].set_ylabel('Δk (cm^-1)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # --- 子图2: Δk 统计分布图 (箱形图) ---
        axes[1].boxplot([delta_k1, delta_k2], labels=[f'入射角 {angle1}°', f'入射角 {angle2}°'])
        axes[1].set_title('Δk 值的统计分布', fontsize=14)
        axes[1].set_ylabel('Δk (cm^-1)', fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 调整布局以适应主标题
        plt.savefig(output_filename, dpi=300)
        print(f"高级分析图已保存为 '{output_filename}'")

    except Exception as e:
        print(f"绘制高级分析图时出错: {e}")


def calculate_thickness(file_path, angle_deg, refractive_index):
    """
    根据光谱数据计算外延层厚度。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件未找到 -> {file_path}")
        return None, None, None, None

    try:
        df = pd.read_excel(file_path, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
        df = df.astype(float)
        
        reflectivity_inv = -df['reflectivity'].values
        wavenumbers = df['wavenumber'].values
        
        peaks, _ = find_peaks(reflectivity_inv, prominence=0.5)
        minima_wavenumbers = wavenumbers[peaks]

        if len(minima_wavenumbers) < 2:
            print(f"在文件 {file_path} 中未找到足够的极小值点。")
            return None, None, None, None

        delta_k = np.diff(minima_wavenumbers)
        avg_delta_k = np.mean(delta_k)
        
        # 计算波数中点用于绘图
        mid_wavenumbers = (minima_wavenumbers[:-1] + minima_wavenumbers[1:]) / 2
        
        theta_rad = np.deg2rad(angle_deg)
        thickness_cm = 1 / (2 * avg_delta_k * np.sqrt(refractive_index**2 - np.sin(theta_rad)**2))
        thickness_um = thickness_cm * 10000
        
        return thickness_um, minima_wavenumbers, df, (delta_k, mid_wavenumbers)

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, None, None, None

def main():
    """
    主函数，执行问题一、二、三的完整计算流程。
    """
    setup_matplotlib_for_chinese()

    # --- 材料与实验参数设定 ---
    MATERIALS = {
        'Si': {
            'name': '硅 (Si)',
            'n': 3.42,
            'files': {'path1': '附件/附件3.xlsx', 'angle1': 10.0,
                      'path2': '附件/附件4.xlsx', 'angle2': 15.0}
        }
    }
    
    results = {}

    print("\n" + "="*60)
    print("      开始处理所有附件，执行问题二和问题三的计算分析")
    print("="*60 + "\n")

    # --- 循环处理每种材料 ---
    for key, params in MATERIALS.items():
        print(f"--- 正在分析: {params['name']} ---")
        
        path1, angle1 = params['files']['path1'], params['files']['angle1']
        path2, angle2 = params['files']['path2'], params['files']['angle2']
        
        thick1, minima1, df1, analysis_data1 = calculate_thickness(path1, angle1, params['n'])
        thick2, minima2, df2, analysis_data2 = calculate_thickness(path2, angle2, params['n'])

        if thick1 is not None and thick2 is not None:
            avg_thick = (thick1 + thick2) / 2
            results[key] = avg_thick
            print(f"  基于 {os.path.basename(path1)} (入射角 {angle1}°), 计算厚度: {thick1:.2f} μm")
            print(f"  基于 {os.path.basename(path2)} (入射角 {angle2}°), 计算厚度: {thick2:.2f} μm")
            print(f"  -> {params['name']} 的平均厚度为: {avg_thick:.2f} μm\n")
            
            # --- 数据可视化 ---
            plot_filename = f"reflectivity_spectrum_{key}.png"
            plot_spectrum_with_minima(df1, df2, angle1, angle2, params['name'], plot_filename, minima1, minima2)

            # --- 高级可视化 ---
            delta_k1, mid_w1 = analysis_data1
            delta_k2, mid_w2 = analysis_data2
            advanced_plot_filename = f"advanced_analysis_{key}.png"
            plot_advanced_analysis(delta_k1, delta_k2, mid_w1, mid_w2, angle1, angle2, params['name'], advanced_plot_filename)
            print("-" * 50)
        else:
            print(f"  !! 未能成功计算 {params['name']} 的厚度。\n")

    # --- 问题三的结论部分 ---
    print("\n" + "="*60)
    print("      问题三：")
    print("="*60 + "\n")
    if 'SiC' in results and 'Si' in results:
        print(f"   - SiC 外延层厚度计算结果为: {results['SiC']:.2f} μm")
        print(f"   - Si 外延层厚度计算结果为: {results['Si']:.2f} μm")

    



if __name__ == '__main__':
    main()
