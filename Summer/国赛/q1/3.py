import numpy as np

def calc_thickness(m, n, wavenumber):
    """
    计算单个极值对应的碳化硅外延层厚度
    
    参数:
    m : int
        干涉级数 (第几个极值)
    n : float
        外延层折射率
    wavenumber : float
        极值对应的波数 (cm^-1)
    
    返回:
    厚度 d (μm)
    """
    d_cm = m / (2 * n * wavenumber)
    return d_cm * 1e4   # cm → μm


def batch_calc_thickness(ms, n, wavenumbers):
    """
    批量计算厚度并返回平均值
    
    参数:
    ms : list[int]
        干涉级数列表
    n : float
        外延层折射率
    wavenumbers : list[float]
        极值对应的波数列表 (cm^-1)
    
    返回:
    (厚度列表 μm, 平均厚度 μm)
    """
    thicknesses = [calc_thickness(m, n, wn) for m, wn in zip(ms, wavenumbers)]
    avg_thickness = np.mean(thicknesses)
    return thicknesses, avg_thickness


# ================= 使用示例 =================
# 假设从图上读到第3、4、5个极值分别在 1200, 1000, 850 cm^-1
ms = [1, 2, 3]  
wavenumbers = [821.72, 702.81 ,636.27]
n = 2.55  # 折射率

thicknesses, avg_thickness = batch_calc_thickness(ms, n, wavenumbers)

print("各极值对应的厚度 (μm):")
for m, wn, d in zip(ms, wavenumbers, thicknesses):
    print(f"  第 {m} 个极值 (波数 {wn} cm^-1): {d:.3f} μm")

print(f"\n平均厚度 ≈ {avg_thickness:.3f} μm")
