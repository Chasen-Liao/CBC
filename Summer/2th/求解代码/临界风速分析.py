import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# =====================
# 1. 基本参数定义
# =====================
# 常量定义
L0 = 22.05        # 锚链长度(m)
m0 = 7.0          # 锚链每米质量(kg/m)
m = 1200.0        # 重物球质量(kg)
m1 = 100.0        # 钢桶质量(kg)
m2 = 10.0         # 每节钢管质量(kg)
H_water = 18.0    # 海水深度(m)
rho_steel = 7850  # 钢材密度(kg/m³)
rho_water = 1025  # 海水密度(kg/m³)
g = 9.8           # 重力加速度(m/s²)

# =====================
# 2. 浮力计算函数
# =====================
def calculate_buoyant_weight(mass, volume=None):
    """计算物体在水中的有效重量（考虑浮力）"""
    if volume is None:
        volume = mass / rho_steel  # 默认钢材体积
    buoyant_force = rho_water * g * volume  # 浮力(N)
    weight_in_air = mass * g               # 空气中重量(N)
    return weight_in_air - buoyant_force   # 水中有效重量(N)

# 计算各部件在水中的有效重量
w0 = calculate_buoyant_weight(m0, volume=m0/rho_steel)  # 锚链每米有效重量(N/m)
w_ball = calculate_buoyant_weight(m)                    # 重物球有效重量(N)
w_bucket = calculate_buoyant_weight(m1)                # 钢桶有效重量(N)
w_pipe = calculate_buoyant_weight(m2)                   # 每节钢管有效重量(N)

# =====================
# 3. 悬链线方程函数
# =====================
def catenary_equation(x, a, x0=0, y0=0, theta0=0):
    """通用悬链线方程，支持最低点不在原点的情况"""
    if theta0 == 0:  # 最低点在原点的情况
        return a * (np.cosh(x/a) - 1)
    else:
        C = np.log((1 + np.sin(theta0)) / np.cos(theta0))
        return a * np.cosh((x - x0)/a + C) + y0 - a / np.cos(theta0)

# =====================
# 4. 求解函数（任意风速）
# =====================
def solve_for_v(v, x_guess=None):
    """
    求解任意风速下的系泊系统状态
    
    参数:
        v: 风速(m/s)
        x_guess: 初始猜测值[x0, x1, x5]
        
    返回:
        (H_w, drag_length, x5, [x0, x1, x5])
    """
    # 设置初始猜测值
    if x_guess is None:
        x_guess = [8.0, 8.5, 9.0]
    
    # 初始风力估计
    F = 0.625 * 2 * 2 * v**2
    
    # 浮力系数
    buoyancy_coeff = rho_water * np.pi * g
    
    # 迭代求解吃水深度
    H_w = 0.7  # 初始吃水深度估计
    for _ in range(10):  # 迭代10次
        # 更新风力（考虑实际吃水深度）
        F = 0.625 * 2 * (2 - H_w) * v**2
        
        # 计算悬链线参数
        a1 = F / w0  # 锚链段参数
        a2 = F / (w_bucket / 1.0)  # 钢桶段参数
        a3 = F / (w_pipe / 1.0)    # 钢管段参数
        
        # 计算角度参数
        W_m0m = L0 * w0 + w_ball  # 锚链+重物球水中重量
        theta1 = np.arctan(W_m0m / F)  # 钢桶下端角度
        W_m0m_m1 = W_m0m + w_bucket  # 锚链+重物球+钢桶水中重量
        theta2 = np.arctan(W_m0m_m1 / F)  # 钢管下端角度
        
        # 定义方程组求解连接点坐标
        def equations(vars):
            x0, x1, x5 = vars
            # 第一段悬链线末端坐标
            y0 = catenary_equation(x0, a1)
            
            # 第二段悬链线方程
            y1 = catenary_equation(x1, a2, x0, y0, theta1)
            
            # 第三段悬链线方程
            y5 = catenary_equation(x5, a3, x1, y1, theta2)
            
            # 约束条件
            eq1 = y5 - (H_water - H_w)  # 浮标底部位置
            eq2 = a2 * (np.sinh((x1 - x0)/a2 + np.log((1+np.sin(theta1))/np.cos(theta1)))) - \
                   a2 * np.sinh(np.log((1+np.sin(theta1))/np.cos(theta1))) - 1.0  # 钢桶长度
            eq3 = a3 * (np.sinh((x5 - x1)/a3 + np.log((1+np.sin(theta2))/np.cos(theta2)))) - \
                   a3 * np.sinh(np.log((1+np.sin(theta2))/np.cos(theta2))) - 4.0  # 钢管总长度
            return [eq1, eq2, eq3]
        
        # 求解方程组
        x0, x1, x5 = fsolve(equations, x_guess)
        
        # 计算实际被拉起的锚链长度
        s_lifted = a1 * np.sinh(x0 / a1)
        drag_length = L0 - s_lifted
        
        # 更新总重量（只考虑被拉起的锚链）
        total_weight = 1000*g + s_lifted*w0 + w_ball + w_bucket + 4*w_pipe
        H_w = total_weight / buoyancy_coeff  # 更新吃水深度
        
        # 确保拖地长度非负
        drag_length = max(0, drag_length)
    
    return H_w, drag_length, x5, [x0, x1, x5]

# =====================
# 5. 临界风速求解函数
# =====================
def find_critical_wind():
    """寻找锚链拖地长度为0的临界风速"""
    # 风速范围和步长
    wind_speeds = np.arange(12.0, 24.1, 0.01)
    drag_lengths = [] # 拖地长度
    valid_wind_speeds = []  # 存储有效风速
    solutions = []
    
    # 初始猜测值
    x_guess = [8.0, 8.5, 9.0]
    
    print("开始计算风速从12m/s到24m/s的过程...")
    for v in wind_speeds:
        try:
            H_w, drag_length, x5, x_guess = solve_for_v(v, x_guess)
            drag_lengths.append(drag_length)
            valid_wind_speeds.append(v)  # 只记录成功计算的风速
            solutions.append(x_guess)
            
            print(f"风速: {v:.2f} m/s, 拖地长度: {drag_length:.2f} m")
            
            if drag_length <= 0.01:  # 临界状态判断
                critical_v = v
                print(f"\n>>> 达到临界状态! 风速: {critical_v:.2f} m/s")
                return critical_v, valid_wind_speeds, drag_lengths, solutions
                
        except Exception as e:
            print(f"风速 {v:.2f} m/s 计算失败: {str(e)}")
            continue  # 跳过当前风速继续计算
    
    # 如果没有找到临界风速
    print("在24m/s内未找到临界风速")
    return None, valid_wind_speeds, drag_lengths, solutions

# =====================
# 6. 可视化函数
# =====================
def plot_results(wind_speeds, drag_lengths, critical_v=None):
    # 确保两个数组维度一致
    if len(wind_speeds) != len(drag_lengths):
        print(f"警告：风速点({len(wind_speeds)})和拖地长度({len(drag_lengths)})数量不一致，使用有效数据点绘图")
        min_len = min(len(wind_speeds), len(drag_lengths))
        wind_speeds = wind_speeds[:min_len]
        drag_lengths = drag_lengths[:min_len]
    
    plt.plot(wind_speeds, drag_lengths, 'b-o', linewidth=2, markersize=5)
    plt.xlabel('风速 (m/s)')
    plt.ylabel('锚链拖地长度 (m)')
    plt.title('锚链拖地长度随风速变化')
    plt.grid(True)
    
    # 标记临界点
    if critical_v is not None:
        plt.axvline(x=critical_v, color='r', linestyle='--', label=f'临界风速: {critical_v:.2f} m/s')
        plt.axhline(y=0, color='g', linestyle='--')
        plt.legend()
    
    plt.tight_layout()
    # plt.savefig('drag_length_vs_wind.png', dpi=300)  # 保存图像[3,5](@ref)
    plt.show()

# =====================
# 7. 主程序
# =====================
if __name__ == "__main__":
    critical_v, valid_wind_speeds, drag_lengths, solutions = find_critical_wind()
    plot_results(valid_wind_speeds, drag_lengths, critical_v)
    
    if critical_v is not None:
        H_w, drag_length, x5, _ = solve_for_v(critical_v)
        print("\n>>> 临界状态系统参数:")
        print(f"临界风速: {critical_v:.2f} m/s")
        print(f"浮标吃水深度: {H_w:.3f} m")
        print(f"锚链拖地长度: {drag_length:.3f} m")