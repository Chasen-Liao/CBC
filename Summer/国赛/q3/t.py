import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

def detect_multibeam_interference(file_path, angle_deg):
    """
    判定附件数据是否存在多光束干涉
    :param file_path: 数据文件路径（附件3或4）
    :param angle_deg: 入射角度（10或15）
    :return: (是否多光束干涉, 反射率对比度)
    """
    # 读取数据
    data = pd.read_excel(file_path)
    wave_number = data.iloc[:, 0].values  # 波数(cm⁻¹)
    reflectance = data.iloc[:, 1].values  # 反射率(%)
    
    # 转换为波长(μm)
    wavelength = 1e4 / wave_number
    
    # 计算反射率对比度
    R_min = np.min(reflectance)/100
    R_max = np.max(reflectance)/100
    contrast = (R_max - R_min) / (R_max + R_min)
    
    # 判定条件：反射率对比度>0.3且存在明显振荡
    is_multibeam = contrast > 0.3 and len(np.where(np.diff(np.sign(np.diff(reflectance))) < 0)[0]) > 3
    
    # 可视化判定
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, reflectance, 'b-', label='实测数据')
    plt.axhline(y=R_min*100, color='r', linestyle='--', label='最小反射率')
    plt.axhline(y=R_max*100, color='g', linestyle='--', label='最大反射率')
    plt.xlabel('波长 (μm)')
    plt.ylabel('反射率 (%)')
    plt.title(f'入射角{angle_deg}° - 反射率对比度: {contrast:.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return is_multibeam, contrast

# 示例使用（附件3和附件4）
file3 = '附件3.xlsx'
file4 = '附件4.xlsx'

multibeam_10, contrast_10 = detect_multibeam_interference(file3, 10)
multibeam_15, contrast_15 = detect_multibeam_interference(file4, 15)

print(f"10°入射角是否多光束干涉: {multibeam_10} (对比度: {contrast_10:.3f})")
print(f"15°入射角是否多光束干涉: {multibeam_15} (对比度: {contrast_15:.3f})")




def silicon_refractive_index(wavelength):
    """硅折射率Sellmeier方程 (波长单位: μm)"""
    # 硅的Sellmeier系数 (参考文献值)
    B1, B2, B3 = 10.668429, 0.003043, 1.541334
    C1, C2, C3 = 0.301516, 1.134751, 1104.0
    
    w2 = wavelength**2
    n_sq = 1 + B1*w2/(w2-C1) + B2*w2/(w2-C2) + B3*w2/(w2-C3)
    return np.sqrt(n_sq)

def fabry_perot_model(wavelength, d, R2, angle_deg):
    """
    法布里-珀罗干涉模型
    :param wavelength: 波长(μm)
    :param d: 外延层厚度(μm)
    :param R2: 衬底界面反射率
    :param angle_deg: 入射角(度)
    :return: 理论反射率
    """
    theta_i = np.deg2rad(angle_deg)
    n = silicon_refractive_index(wavelength)
    
    # 计算折射角
    theta_t = np.arcsin(np.sin(theta_i)/n)
    cos_theta_t = np.cos(theta_t)
    
    # 空气-外延层反射率
    R1 = ((1 - n)/(1 + n))**2
    
    # 相位差
    delta = 4 * np.pi * n * d * cos_theta_t / wavelength
    
    # 多光束干涉公式
    numerator = R1 + R2 + 2*np.sqrt(R1*R2)*np.cos(delta)
    # denominator = 1 + R1*R2 + 2*np.sqrt(R1*R2)*np.cos(delta)
    # 近似求解
    
    return numerator

def fit_thickness(file_path, angle_deg, initial_guess=[10, 0.1]):
    """
    拟合外延层厚度
    :param file_path: 数据文件路径
    :param angle_deg: 入射角度
    :param initial_guess: 初始猜测[d(μm), R2]
    :return: 拟合厚度(μm), 拟合参数
    """
    # 读取数据
    data = pd.read_excel(file_path)
    wave_number = data.iloc[:, 0].values
    reflectance = data.iloc[:, 1].values/100  # 转换为小数
    
    # 转换为波长(μm)
    wavelength = 1e4 / wave_number
    
    # 定义拟合函数
    def model_func(w, d, R2):
        return fabry_perot_model(w, d, R2, angle_deg)
    
    # 执行拟合
    params, cov = curve_fit(model_func, wavelength, reflectance, 
                           p0=initial_guess, bounds=([0.1, 0.001], [100, 0.5]))
    
    # 计算拟合残差
    fitted_reflectance = model_func(wavelength, *params)
    residual = np.sum((reflectance - fitted_reflectance)**2)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.plot(wavelength, reflectance*100, 'bo', markersize=4, label='实测数据')
    plt.plot(wavelength, fitted_reflectance*100, 'r-', linewidth=2, label=f'拟合曲线(d={params[0]:.2f}μm)')
    plt.xlabel('波长 (μm)')
    plt.ylabel('反射率 (%)')
    plt.title(f'入射角{angle_deg}° - 多光束干涉模型拟合')
    plt.legend()
    plt.grid(True)
    
    # 残差分析
    plt.figure(figsize=(12, 4))
    plt.plot(wavelength, (reflectance - fitted_reflectance)*100, 'g-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'拟合残差 (MSE={residual:.6f})')
    plt.xlabel('波长 (μm)')
    plt.ylabel('残差 (%)')
    plt.grid(True)
    plt.show()
    
    return params[0], params, residual

# 对附件3和附件4进行计算
d_10, params_10, res_10 = fit_thickness('附件3.xlsx', 10, [15, 0.15])
d_15, params_15, res_15 = fit_thickness('附件4.xlsx', 15, [15, 0.15])

print(f"10°入射角拟合厚度: {d_10:.2f} μm (残差: {res_10:.6f})")
print(f"15°入射角拟合厚度: {d_15:.2f} μm (残差: {res_15:.6f})")
print(f"厚度一致性: {abs(d_10 - d_15):.4f} μm")


def sic_refractive_index(wavelength):
    """碳化硅折射率模型 (波长单位: μm)"""
    # 4H-SiC近似模型 (参考文献)
    return 2.55 + 34000/wavelength**2

def correct_sic_thickness(file_path, angle_deg):
    """
    修正碳化硅数据中的多光束干涉影响
    :param file_path: 附件1或2路径
    :param angle_deg: 入射角度
    """
    # 首先检测是否有多光束干涉
    is_multibeam, _ = detect_multibeam_interference(file_path, angle_deg)
    
    if is_multibeam:
        print("检测到多光束干涉影响，使用法布里-珀罗模型修正")
        
        # 定义碳化硅专用模型
        def sic_model(wavelength, d, R2):
            theta_i = np.deg2rad(angle_deg)
            n = sic_refractive_index(wavelength)
            theta_t = np.arcsin(np.sin(theta_i)/n)
            cos_theta_t = np.cos(theta_t)
            R1 = ((1 - n)/(1 + n))**2
            delta = 4 * np.pi * n * d * cos_theta_t / wavelength
            numerator = R1 + R2 + 2*np.sqrt(R1*R2)*np.cos(delta)
            denominator = 1 + R1*R2 + 2*np.sqrt(R1*R2)*np.cos(delta)
            return numerator / denominator
        
        # 执行拟合
        data = pd.read_excel(file_path)
        wave_number = data.iloc[:, 0].values
        reflectance = data.iloc[:, 1].values/100
        wavelength = 1e4 / wave_number
        
        params, cov = curve_fit(sic_model, wavelength, reflectance, 
                               p0=[10, 0.1], bounds=([1, 0.01], [50, 0.3]))
        
        return params[0]
    
    else:
        print("未检测到显著多光束干涉，使用双光束模型")
        # 此处可添加问题1的双光束模型计算
        return None

# 对附件1和附件2进行修正
sic_thickness_10 = correct_sic_thickness('附件1.xlsx', 10)
sic_thickness_15 = correct_sic_thickness('附件2.xlsx', 15)

print(f"10°入射角修正厚度: {sic_thickness_10:.2f} μm")
print(f"15°入射角修正厚度: {sic_thickness_15:.2f} μm")




def evaluate_reliability(d1, d2, res1, res2, angle_diff=5):
    """
    评估厚度计算结果可靠性
    :param d1, d2: 不同入射角计算的厚度
    :param res1, res2: 拟合残差
    :param angle_diff: 入射角差异(度)
    """
    # 一致性分析
    diff_percent = abs(d1 - d2)/max(d1, d2)*100
    
    # 残差分析
    avg_res = (res1 + res2)/2
    
    # 可靠性评级
    if diff_percent < 1 and avg_res < 0.01:
        reliability = "优秀 (误差<1%)"
    elif diff_percent < 3 and avg_res < 0.05:
        reliability = "良好 (误差<3%)"
    elif diff_percent < 5:
        reliability = "可接受 (误差<5%)"
    else:
        reliability = "不可靠 (建议检查模型)"
    
    # 生成报告
    report = f"""
    =============== 可靠性分析报告 ===============
    厚度计算结果:
      10°入射角: {d1:.4f} μm
      15°入射角: {d2:.4f} μm
      差异: {diff_percent:.2f}%
    
    模型拟合质量:
      10°残差: {res1:.6f}
      15°残差: {res2:.6f}
      平均残差: {avg_res:.6f}
    
    综合评价: {reliability}
    =============================================
    """
    return report

# 生成硅片分析报告
sic_report = evaluate_reliability(sic_thickness_10, sic_thickness_15, res_10, res_15)
print(sic_report)



if __name__ == "__main__":
    # 步骤1: 硅片多光束干涉分析
    print("="*50)
    print("硅晶圆片(附件3&4)分析")
    print("="*50)
    d_si_10, _, res_si_10 = fit_thickness('附件3.xlsx', 10)
    d_si_15, _, res_si_15 = fit_thickness('附件4.xlsx', 15)
    
    # 步骤2: 碳化硅多光束干涉修正
    print("\n" + "="*50)
    print("碳化硅晶圆片(附件1&2)修正")
    print("="*50)
    d_sic_10 = correct_sic_thickness('附件1.xlsx', 10)
    d_sic_15 = correct_sic_thickness('附件2.xlsx', 15)
    
    # 步骤3: 生成最终报告
    print("\n" + "="*50)
    print("最终计算结果")
    print("="*50)
    print(f"硅外延层厚度: {np.mean([d_si_10, d_si_15]):.2f} μm")
    print(f"碳化硅外延层修正厚度: {np.mean([d_sic_10, d_sic_15]):.2f} μm")
    
    # 可靠性评估
    si_report = evaluate_reliability(d_si_10, d_si_15, res_si_10, res_si_15)
    print(si_report)

