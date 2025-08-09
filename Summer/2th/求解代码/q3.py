import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题


# 常量定义
L0 = 22.05        # 锚链长度(m)
m0 = 7.0          # 锚链每米质量(kg/m)
m = 1200.0        # 重物球质量(kg)
m1 = 100.0        # 钢桶质量(kg)
m2 = 10.0         # 每节钢管质量(kg)
H_water = 18.0    # 海水深度(m)
v_wind = 12.0     # 风速(m/s)
v_water = 1.5     # 水速(m/s)
rho_steel = 7850  # 钢材密度(kg/m³)
rho_water = 1020  # 海水密度(kg/m³)
g = 9.8           # 重力加速度(m/s²)

# =====================
# 悬链线方程函数
# =====================
def catenary_equation(x, a, x0=0, y0=0, theta0=0):
    """
    通用悬链线方程，支持最低点不在原点的情况
    
    参数:
        x: x坐标
        a: 悬链线参数 a = H/ω
        x0: 参考点x坐标
        y0: 参考点y坐标
        theta0: 参考点处切线与x轴的夹角(弧度)
        
    返回:
        y坐标值
    """
    # 论文中公式(3)
    if theta0 == 0:  # 最低点在原点的情况
        return a * (np.cosh(x/a) - 1)
    else:
        C = np.log((1 + np.sin(theta0)) / np.cos(theta0))
        # print('a', a)
        # print('C    x_0    y_0    a/cos()')
        # print(C, x0, y0, a / np.cos(theta0))
        return a * np.cosh((x - x0)/a + C) + y0 - a / np.cos(theta0)
    
# =====================
# 浮力计算函数
# =====================
def calculate_buoyant_weight(mass, volume=None):
    """计算物体在水中的有效重量（考虑浮力）"""
    if volume is None:
        volume = mass / rho_steel  # 默认钢材体积
        print(f'V:{volume:.2f}')
    buoyant_force = rho_water * g * volume  # 浮力(N)
    weight_in_air = mass * g               # 空气中重量(N)
    return weight_in_air - buoyant_force   # 水中有效重量(N)