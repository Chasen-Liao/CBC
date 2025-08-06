import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# =====================
# 1. 基本参数定义
# =====================
"""
根据论文中问题1给出的参数：
- 锚链长度 L0 = 22.05 m
- 锚链每米质量 m0 = 7 kg/m
- 重物球质量 m = 1200 kg
- 钢桶质量 m1 = 100 kg
- 钢管每节质量 m2 = 10 kg
- 海水深度 H_water = 18 m
- 风速 v = 12 m/s (问题1)
- 钢材密度 rho_steel = 7800 kg/m³
- 海水密度 rho_water = 1020 kg/m³
- 重力加速度 g = 9.8 m/s²
"""
# 常量定义
L0 = 22.05        # 锚链长度(m)
m0 = 7.0          # 锚链每米质量(kg/m)
m = 1200.0        # 重物球质量(kg)
m1 = 100.0        # 钢桶质量(kg)
m2 = 10.0         # 每节钢管质量(kg)
H_water = 18.0    # 海水深度(m)
v = 12.0          # 风速(m/s)
rho_steel = 7800  # 钢材密度(kg/m³)
rho_water = 1020  # 海水密度(kg/m³)
g = 9.8           # 重力加速度(m/s²)

# =====================
# 2. 浮力计算函数
# =====================
def calculate_buoyant_weight(mass, volume=None):
    """
    计算物体在水中的有效重量（考虑浮力）
    
    参数:
        mass: 物体质量(kg)
        volume: 物体体积(m³)，如果未提供则根据钢材密度计算
        
    返回:
        物体在水中的有效重量(N)
    """
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
        return a * np.cosh((x - x0)/a + C) + y0 - a / np.cos(theta0)

# =====================
# 4. 问题1求解函数
# =====================
def solve_problem1():
    """
    求解问题1：风速12m/s时的系泊系统状态
    
    返回:
        浮标吃水深度, 锚链拖地长度, 钢桶倾斜角度等
    """
    # 初始假设：锚链全部拉起
    F = 0.625 * 2 * 2 * v**2  # 初始风力估计（假设吃水深度为0）
    
    # 初始总重量计算（包括全部锚链）
    total_weight = 1000*g + L0*w0 + w_ball + w_bucket + 4*w_pipe
    
    # 浮力系数（论文中给出的3200.13）
    buoyancy_coeff = 1025 * 3.14 * g
    
    # 迭代求解吃水深度（考虑风力对浮标的影响）
    H_w = total_weight / buoyancy_coeff  # 初始吃水深度
    for _ in range(10):  # 迭代10000次
        # 更新风力（考虑实际吃水深度）
        F = 0.625 * 2 * (2 - H_w) * v**2
        
        # 计算悬链线参数
        a1 = F / w0  # 锚链段参数
        a2 = F / (w_bucket / 1.0)  # 钢桶段参数（单位长度重量）
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
            
            # 约束条件（论文中方程(8)）
            eq1 = y5 - (H_water - H_w)  # 浮标底部位置
            eq2 = a2 * (np.sinh((x1 - x0)/a2 + np.log((1+np.sin(theta1))/np.cos(theta1)))) - \
                   a2 * np.sinh(np.log((1+np.sin(theta1))/np.cos(theta1))) - 1.0  # 钢桶长度
            eq3 = a3 * (np.sinh((x5 - x1)/a3 + np.log((1+np.sin(theta2))/np.cos(theta2)))) - \
                   a3 * np.sinh(np.log((1+np.sin(theta2))/np.cos(theta2))) - 4.0  # 钢管总长度
            return [eq1, eq2, eq3]
        
        # 求解方程组
        x0, x1, x5 = fsolve(equations, [8.0, 8.5, 9.0])
        
        # 计算实际被拉起的锚链长度
        s_lifted = a1 * np.sinh(x0 / a1)
        
        drag_length = L0 - s_lifted
        
        # 更新总重量（只考虑被拉起的锚链）
        total_weight = 1000*g + s_lifted*w0 + w_ball + w_bucket + 4*w_pipe
        H_w = total_weight / buoyancy_coeff  # 更新吃水深度
    
    # 计算钢桶倾斜角度（转换为度数）
    bucket_angle = np.degrees(theta1)
    
    return H_w, drag_length, x5, bucket_angle

# =====================
# 5. 问题2求解函数
# =====================


# TODO 这里还是不行，锚链末端角度求解有点出错


import numpy as np
from scipy.optimize import fsolve

def solve_problem2_corrected():
    # 物理参数（根据论文设定）
    v_high = 36.0        # 高风速 (m/s)
    H_water = 18.0       # 海水深度 (m)
    L0 = 22.05           # 锚链总长度 (m)
    w0 = 59.59           # 锚链单位水中重量 (N/m)
    w_bucket = 269.96    # 钢桶水中重量 (N)
    w_pipe = 85.12       # 钢管单位水中重量 (N/m)
    
    # 初始参数
    m_ball = 1200.0      # 初始重物球质量 (kg)
    max_iter = 20         # 最大迭代次数
    tolerance = 0.1      # 收敛容差
    
    # 浮力修正计算（论文核心）
    def calculate_buoyant_weight(mass):
        density_steel = 7800  # 钢材密度 (kg/m³)
        density_water = 1025  # 海水密度 (kg/m³)
        g = 9.8               # 重力加速度 (m/s²)
        volume = mass / density_steel
        buoyancy = volume * density_water * g
        return mass * g - buoyancy  # 水中有效重量
    
    # 通用悬链线方程（论文公式3）
    def generalized_catenary(x, a, x0_ref, y0_ref, theta0):
        C = np.log((1 + np.sin(theta0)) / np.cos(theta0))
        return a * np.cosh((x - x0_ref)/a + C) + y0_ref - a/np.cos(theta0)
    
    # 弧长计算函数（论文公式）
    def calculate_arc_length(a, x_start, x_end, theta0):
        C = np.log((1 + np.sin(theta0)) / np.cos(theta0))
        return a * (np.sinh((x_end - x_start)/a + C) - np.sinh(C))
    
    # 迭代求解
    for i in range(max_iter):
        # 1. 计算当前重物球的有效重量
        w_ball_current = calculate_buoyant_weight(m_ball)
        
        # 2. 计算风力（考虑吃水深度影响）
        H_w = 0.7  # 初始吃水深度估计
        F_wind = 0.625 * 2 * (2 - H_w) * v_high**2
        
        # 3. 计算悬链线参数
        a1 = F_wind / w0
        a2 = F_wind / (w_bucket/1.0)  # 钢桶段参数（长度1m）
        a3 = F_wind / w_pipe
        
        # 4. 计算总重量
        W_m0m = (L0 * w0) + w_ball_current
        W_m0m_m1 = W_m0m + w_bucket
        
        # 5. 计算关键角度
        theta1 = np.arctan(W_m0m / F_wind)   # 钢桶下端角度
        theta2 = np.arctan(W_m0m_m1 / F_wind) # 钢管下端角度
        
        # 6. 定义并求解方程组（关键修正）
        def equations(vars):
            x0, x1, x5 = vars[:3]
            theta0 = np.radians(16.0) if len(vars) == 3 else vars[3]  # 锚链角度修正
            
            # 锚链段（从锚点x=0到x0）
            y0 = generalized_catenary(x0, a1, 0, 0, theta0)
            
            # 钢桶段（从x0到x1，长度1m）
            y1 = generalized_catenary(x1, a2, x0, y0, theta1)
            
            # 钢管段（从x1到x5，长度4m）
            y5 = generalized_catenary(x5, a3, x1, y1, theta2)
            
            # 约束条件
            eq1 = y5 - (H_water - H_w)       # 浮标位置
            eq2 = calculate_arc_length(a2, x0, x1, theta1) - 1.0  # 钢桶长度
            eq3 = calculate_arc_length(a3, x1, x5, theta2) - 4.0  # 钢管长度
            
            # 仅当锚链完全拉起时应用长度约束
            if len(vars) == 4:
                eq4 = calculate_arc_length(a1, 0, x0, theta0) - L0  # 锚链长度
                return [eq1, eq2, eq3, eq4]
            return [eq1, eq2, eq3]
        
        try:
            # 尝试两种状态：锚链可能拖地或完全拉起
            try:
                # 状态1：锚链完全拉起（4变量）
                solution = fsolve(equations, [8.0, 11.5, 12.0, np.radians(20)], xtol=1e-3)
                x0, x1, x5, theta0_rad = solution
                anchor_angle = np.degrees(theta0_rad)
                # print(anchor_angle)
            except:
                # 状态2：锚链部分拖地（3变量）
                solution = fsolve(equations, [2.0, 3.0, 4.0], xtol=1e-3)
                x0, x1, x5 = solution
                anchor_angle = 0.0  # 拖地时角度为0
                
            # 计算钢桶倾角
            bucket_angle = 90.0 - np.degrees(theta1)
            
        except Exception as e:
            print(f"求解失败: {e}")
            anchor_angle = 25.0
            bucket_angle = 10.0
        
        # 打印当前状态
        print(f"Iter {i}: m_ball={m_ball:.1f}kg, 锚链角={anchor_angle:.2f}°, 钢桶角={bucket_angle:.2f}°")
        
        # 质量调整策略（论文方法）
        if anchor_angle > 16.0:
            # 根据偏差动态调整
            mass_increment = max(50, (anchor_angle - 16.0) * 100)
            m_ball += mass_increment
        else:
            print(f"满足条件：锚链角={anchor_angle:.2f}° ≤ 16°")
            break
    
    return m_ball, bucket_angle, anchor_angle


# =====================
# 6. 可视化函数
# =====================
def plot_catenary(x0, x1, x5, a1, a2, a3, theta1, theta2, drag_length):
    """
    修正：包含锚链拖地部分的可视化
    
    参数新增:
        drag_length: 锚链拖地长度
    """
    # 计算锚点实际位置（拖地段起点）
    anchor_x = drag_length
    anchor_y = 0
    
    # 创建x坐标点 (从拖地段起点开始)
    x_drag = np.linspace(-drag_length, 0, 30)  # 新增：拖地段
    x_chain = np.linspace(0, x0, 50)
    x_bucket = np.linspace(x0, x1, 20)
    x_pipe = np.linspace(x1, x5, 30)
    
    # 计算y坐标
    y_drag = np.zeros_like(x_drag)  # 拖地段平铺海底
    y_chain = catenary_equation(x_chain, a1)
    
    # 连接点高度修正
    y0 = catenary_equation(x0, a1)
    y_bucket = catenary_equation(x_bucket, a2, x0, y0, theta1)
    y1 = catenary_equation(x1, a2, x0, y0, theta1)
    y_pipe = catenary_equation(x_pipe, a3, x1, y1, theta2)
    
    # 绘图 (新增拖地段)
    plt.figure(figsize=(12, 7))
    plt.plot(x_drag + anchor_x, y_drag, 'k--', label=f'锚链拖地段({drag_length:.1f}m)')  # 虚线表示海底
    plt.plot(x_chain + anchor_x, y_chain, 'b-', label='悬链段')
    plt.plot(x_bucket + anchor_x, y_bucket, 'r-', label='钢桶段')
    plt.plot(x_pipe + anchor_x, y_pipe, 'g-', label='钢管段')
    
    # 标记关键点
    plt.scatter([anchor_x], [0], c='k', marker='*', s=100, label='锚点')
    plt.scatter([x0 + anchor_x], [y0], c='purple', s=80, label='锚链-钢桶连接点')
    
    plt.xlabel('水平距离 (m)')
    plt.ylabel('高度 (m)')
    plt.title('系泊系统完整构型 (含锚链拖地)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')  # 保持纵横比
    
    # 添加海底基准线
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.text((x5 - drag_length)/2, -0.3, '海床', ha='center')
    # 添加海平面
    plt.axhline(y=18, color='gray', linestyle='-', alpha=0.5)
    plt.text(0, 17.5, '海平面', ha='center')
    plt.show()

# =====================
# 7. 主程序
# =====================
if __name__ == "__main__":
    # 求解问题1
    print("\n>>> 求解问题1: 风速12m/s时的系统状态")
    H_w, drag_length, x5, bucket_angle = solve_problem1()
    print(f"浮标吃水深度: {H_w:.2f} m")
    print(f"锚链拖地长度: {drag_length:.2f} m")
    print(f"浮标游动区域半径: {x5 + drag_length:.2f} m")
    print(f"钢桶竖直倾斜角度: {90.0 - bucket_angle:.2f}°")
    
    
    # 求解问题2
    print("\n>>> 求解问题2")
    optimal_mass, bucket_angle, anchor_angle = solve_problem2_corrected()
    print(f"\n最终结果：重物球质量 = {optimal_mass:.2f}kg")
    print(f"钢桶倾角 = {bucket_angle:.2f}°，锚链夹角 = {anchor_angle:.2f}°")
    
    # 可视化
    print("\n>>> 生成系泊系统示意图...")
    # 此处使用问题1的结果作为示例
    F = 0.625 * 2 * (2 - H_w) * 12**2
    a1 = F / w0
    a2 = F / (w_bucket / 1.0)
    a3 = F / (w_pipe / 1.0)
    W_m0m = (L0 - drag_length) * w0 + w_ball
    theta1 = np.arctan(W_m0m / F)
    plot_catenary(8.26, 8.28, 8.35, a1, a2, a3, theta1, theta1, drag_length)