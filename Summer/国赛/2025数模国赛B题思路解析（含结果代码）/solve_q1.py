import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

def plot_spectrum_with_minima(df1, df2, output_filename, minima1_w=None, minima2_w=None):
    """
    绘制并保存光谱图，并可选地在图上标记极小值点。
    """
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(df1['wavenumber'], df1['reflectivity'], label='入射角 10° (附件1)', alpha=0.8)
        plt.plot(df2['wavenumber'], df2['reflectivity'], label='入射角 15° (附件2)', alpha=0.8)

        title = '碳化硅晶圆片反射光谱'
        
        # 如果传入了极小值点数据，则在图上标记出来
        if minima1_w is not None and minima2_w is not None:
            # 使用merge安全地获取极小值点对应的反射率
            minima1_df = pd.DataFrame({'wavenumber': minima1_w})
            minima1_points = pd.merge(minima1_df, df1, on='wavenumber', how='inner')
            
            minima2_df = pd.DataFrame({'wavenumber': minima2_w})
            minima2_points = pd.merge(minima2_df, df2, on='wavenumber', how='inner')

            # 使用散点图标记极小值点
            plt.scatter(minima1_points['wavenumber'], minima1_points['reflectivity'], 
                        color='red', marker='v', s=50, zorder=5, label='附件1 极小值点')
            plt.scatter(minima2_points['wavenumber'], minima2_points['reflectivity'], 
                        color='black', marker='x', s=50, zorder=5, label='附件2 极小值点')
            title += ' (及检测到的极小值点)'

        plt.xlabel('波数 (cm^-1)', fontsize=12)
        plt.ylabel('反射率 (%)', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(output_filename, dpi=300) # 提高图像分辨率
        print(f"带有极小值标记的光谱图已保存为 '{output_filename}'")
    except Exception as e:
        print(f"绘制光谱图时出错: {e}")


def calculate_thickness(file_path, angle_deg, refractive_index):
    """
    根据光谱数据计算外延层厚度。
    """
    try:
        df = pd.read_excel(file_path, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
        df = df.astype(float)
        reflectivity_inv = -df['reflectivity'].values
        wavenumbers = df['wavenumber'].values
        peaks, _ = find_peaks(reflectivity_inv, prominence=0.5)
        minima_wavenumbers = wavenumbers[peaks]
        if len(minima_wavenumbers) < 2:
            print(f"在文件 {file_path} 中未找到足够的极小值点。")
            return None, None
        delta_k = np.diff(minima_wavenumbers)
        avg_delta_k = np.mean(delta_k)
        theta_rad = np.deg2rad(angle_deg)
        thickness_cm = 1 / (2 * avg_delta_k * np.sqrt(refractive_index**2 - np.sin(theta_rad)**2))
        thickness_um = thickness_cm * 10000
        return thickness_um, minima_wavenumbers
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, None

def main():
    """
    主函数，执行问题一的完整计算流程。
    """
    # --- 1. 参数设定 ---
    N_SIC = 2.55 
    file1 = '附件/附件1.xlsx'
    angle1 = 10.0
    file2 = '附件/附件2.xlsx'
    angle2 = 15.0

    print("--- 问题一：碳化硅外延层厚度计算 ---")
    
    # --- 2. 计算厚度并获取极小值点 ---
    print("\n--- 开始计算厚度 ---")
    thickness1, minima1 = calculate_thickness(file1, angle1, N_SIC)
    if minima1 is not None:
        print(f"\n文件: {file1} (入射角: {angle1}°)")
        print(f"找到 {len(minima1)} 个反射率极小值点。")
        print(f"计算出的厚度为: {thickness1:.2f} μm")
        print("-" * 40)

    thickness2, minima2 = calculate_thickness(file2, angle2, N_SIC)
    if minima2 is not None:
        print(f"文件: {file2} (入射角: {angle2}°)")
        print(f"找到 {len(minima2)} 个反射率极小值点。")
        print(f"计算出的厚度为: {thickness2:.2f} μm")
        print("-" * 40)

    # --- 3. 数据可视化 ---
    if minima1 is not None and minima2 is not None:
        print("\n--- 正在生成可视化图表 ---")
        try:
            df1_vis = pd.read_excel(file1, header=None, skiprows=1, names=['wavenumber', 'reflectivity']).astype(float)
            df2_vis = pd.read_excel(file2, header=None, skiprows=1, names=['wavenumber', 'reflectivity']).astype(float)
            plot_spectrum_with_minima(df1_vis, df2_vis, 'reflectivity_spectrum_with_minima.png', minima1, minima2)
        except Exception as e:
            print(f"加载数据用于绘图时出错: {e}")

    # --- 4. 结果总结 ---
    if thickness1 is not None and thickness2 is not None:
        avg_thickness = (thickness1 + thickness2) / 2
        print(f"\n[最终结果]")
        print(f"综合两次测量的平均厚度为: {avg_thickness:.2f} μm")

if __name__ == '__main__':
    main()
