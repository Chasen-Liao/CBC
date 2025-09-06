import numpy as np
import math

def calculate_epi_thickness(P_list, wavenumber_list, ru=15.0):
    """
    计算碳化硅外延层厚度 -- 通过在图像中选择极点
    P_list -- 极值级数列表
    wavenumber_list -- 对应波数列表
    ru -- 入射角度
    """
    
    # 常量设置 
    折射 = 2.55  # 碳化硅折射率
    
    # 计算相关参数
    入射角 = math.radians(ru)  # 角度转弧度
    sin_theta = math.sin(入射角)
    分母 = math.sqrt(折射**2 - sin_theta**2)  # 公式分母部分
    
    # 波数转波长 nm)
    wavelen_list = [10**7 / w for w in wavenumber_list]
    
    # 厚度计算
    res = []
    for P, wavelen in zip(P_list, wavelen_list):
        # 0.001是nm→μm的单位转换因子
        T_i = (P-0.5) * (0.001 * wavelen) / (分母)
        res.append(T_i)
    
    return res

# 测试案例 (使用您提供的示例数据)
if __name__ == "__main__":
    # 示例输入数据 (波数单位: cm⁻¹)
    waves = [403 ,518 ,636.27, 702.81, 821.72]  # 极值点波数
    waves = sorted(waves)
    P_values = [1, 2, 3, 4, 5]  # 对应级数
    
    # 计算厚度
    thicknesses = calculate_epi_thickness(P_values, waves)
    
    # 输出结果
    print("计算厚度结果 (μm):")
    for i, (P, wavenumber, thickness) in enumerate(zip(P_values, waves, thicknesses)):
        print(f"极值点 {i+1}: P={P}, 波数={wavenumber} cm⁻¹ → 厚度={thickness:.2f} μm")
    
    # 平均厚度计算
    avg_thickness = np.mean(thicknesses)
    print(f"\n平均厚度: {avg_thickness:.2f} μm")