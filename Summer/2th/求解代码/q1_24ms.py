import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# =====================
# 1. 基本参数定义 (24m/s)
# =====================
L0 = 22.05        # 锚链长度(m)
m0 = 7.0          # 锚链每米质量(kg/m)
m = 1200.0        # 重物球质量(kg)
m1 = 100.0        # 钢桶质量(kg)
m2 = 10.0         # 每节钢管质量(kg)
H_water = 18.0    # 海水深度(m)
v = 24.0          # 风速(m/s)
rho_steel = 7850  # 钢材密度(kg/m³)
rho_water = 1070  # 海水密度(kg/m³)
g = 9.8           # 重力加速度(m/s²)

# 浮标参数
diameter_float = 2.0  # 浮标直径(m)
height_float = 2.0    # 浮标高度(m)

# =====================
# 2. 浮力计算函数
# =====================
def calculate_buoyant_weight(mass, volume=None):
    """计算物体在水中的有效重量（考虑浮力）"""
    if volume is None:
        volume = mass / rho_steel
    buoyant_force = rho_water * g * volume
    weight_in_air = mass * g
    return weight_in_air - buoyant_force

# 计算各部件在水中的有效重量
w0 = calculate_buoyant_weight(m0, volume=m0/rho_steel)  # 锚链每米有效重量(N/m)
w_ball = calculate_buoyant_weight(m)                      # 重物球有效重量(N)
w_bucket = calculate_buoyant_weight(m1)                  # 钢桶有效重量(N)
w_pipe = calculate_buoyant_weight(m2)                    # 每节钢管有效重量(N)

# =====================
# 3. 悬链线方程函数 (公式9)
# =====================
def catenary_equation(x, a, x0=0, y0=0, theta0=0):
    """严格遵循论文公式(9)的悬链线方程"""
    C = np.log((1 + np.sin(theta0)) / np.cos(theta0))
    return a * np.cosh((x - x0)/a + C) + y0 - a / np.cos(theta0)

# =====================
# 4. 悬链线弧长计算函数
# =====================
def catenary_length(a, x_start, x_end, C):
    """计算悬链线从x_start到x_end的弧长"""
    return a * (np.sinh((x_end - x_start)/a + C) - np.sinh(C))

# =====================
# 5. 钢管竖直夹角计算函数
# =====================
def calculate_pipe_angles(a3, x1, x5, theta2):
    """
    计算4节钢管中点处的竖直方向夹角
    参数:
        a3: 钢管段悬链线参数
        x1: 钢管段起点x坐标
        x5: 钢管段终点x坐标
        theta2: 钢管段起点切线与x轴夹角(弧度)
    """
    # 钢管总长4m（4节，每节1m）
    total_length = 4.0
    pipe_angles = []
    
    # 钢管段悬链线参数
    C = np.log((1 + np.sin(theta2)) / np.cos(theta2))
    
    # 计算每节钢管中点弧长位置（s=0.5, 1.5, 2.5, 3.5m）
    for i in range(4):
        s_mid = 0.5 + i * 1.0  # 每节钢管中点弧长
        
        # 通过弧长反求中点x坐标（数值求解）
        def equation(x):
            # 弧长积分公式
            ds = a3 * np.sinh((x - x1)/a3 + C) - a3 * np.sinh(C)
            return ds - s_mid
        
        # 初始猜测值：按线性比例估计中点位置
        x_guess = x1 + (x5 - x1) * s_mid / total_length
        x_mid = fsolve(equation, x_guess)[0]
        
        # 计算中点导数 dy/dx
        dy_dx = np.sinh((x_mid - x1)/a3 + C)
        
        # 计算竖直方向夹角（单位：度）
        angle_x_axis = np.arctan(dy_dx)        # 切线与x轴夹角（弧度）
        angle_vertical = 90 - np.degrees(angle_x_axis)  # 与竖直方向夹角
        pipe_angles.append(angle_vertical)
    
    return pipe_angles

# =====================
# 6. 24m/s求解函数 (修正版)
# =====================
def solve_problem24_corrected():
    """严格遵循论文的24m/s求解函数"""
    # 初始猜测值
    theta0_deg_guess = 4  # 锚链与海床夹角初始猜测值(度)
    H_w_guess = 0.6       # 浮标吃水深度初始猜测值(m)
    
    # 计算浮力系数
    buoyancy_coeff = 1025 * np.pi * 9.8
    
    # 定义求解方程组
    def equations(vars):
        theta0_deg, H_w = vars
        
        # 转换为弧度
        theta0_rad = np.radians(theta0_deg)
        
        # 计算风力 (论文中风力公式)
        S_float = diameter_float * (height_float - H_w)
        F = 0.625 * S_float * v**2
        
        # 计算悬链线参数
        a1 = F / w0
        a2 = F / (w_bucket / 1.0)  # 钢桶段参数
        a3 = F / (w_pipe / 1.0)    # 钢管段参数
        
        # 计算角度参数 (论文中受力分析)
        W_m0m = L0 * w0 + w_ball
        theta1 = np.arctan(W_m0m / F)  # 钢桶下端角度
        
        W_m0m_m1 = W_m0m + w_bucket
        theta2 = np.arctan(W_m0m_m1 / F)  # 钢管下端角度
        
        # 锚链段悬链线常数C0
        C0 = np.log((1 + np.sin(theta0_rad)) / np.cos(theta0_rad))
        
        # 计算锚链末端x0 (从锚点到钢桶连接点)
        def anchor_equation(x0):
            # 锚链段弧长应等于L0
            s1 = catenary_length(a1, 0, x0, C0)
            return s1 - L0
        
        x0 = fsolve(anchor_equation, 15.0)[0]
        y0 = catenary_equation(x0, a1, theta0=theta0_rad)
        
        # 钢桶段悬链线常数C1
        C1 = np.log((1 + np.sin(theta1)) / np.cos(theta1))
        
        # 计算钢桶末端x1 (从钢桶连接到钢管连接点)
        def bucket_equation(x1):
            # 钢桶段弧长应等于1m
            s2 = catenary_length(a2, x0, x1, C1)
            return s2 - 1.0
        
        x1 = fsolve(bucket_equation, x0 + 0.1)[0]
        y1 = catenary_equation(x1, a2, x0, y0, theta1)
        
        # 钢管段悬链线常数C2
        C2 = np.log((1 + np.sin(theta2)) / np.cos(theta2))
        
        # 计算钢管末端x5 (从钢管连接到浮标)
        def pipe_equation(x5):
            # 钢管段弧长应等于4m
            s3 = catenary_length(a3, x1, x5, C2)
            return s3 - 4.0
        
        x5 = fsolve(pipe_equation, x1 + 1.0)[0]
        y5 = catenary_equation(x5, a3, x1, y1, theta2)
        
        # 浮标底部应位于海平面下(18 - H_w)处
        eq1 = y5 - (H_water - H_w)
        
        # 浮标吃水深度应满足公式(10)
        total_weight = 1000*g + L0*w0 + w_ball + w_bucket + 4*w_pipe
        eq2 = H_w - (total_weight + F * np.tan(theta0_rad)) / buoyancy_coeff
        
        return [eq1, eq2]
    
    # 求解方程组
    theta0_deg, H_w = fsolve(equations, [theta0_deg_guess, H_w_guess])
    
    # 计算最终结果
    theta0_rad = np.radians(theta0_deg)
    
    # 重新计算所有参数
    S_float = diameter_float * (height_float - H_w)
    F = 0.625 * S_float * v**2
    a1 = F / w0
    a2 = F / (w_bucket / 1.0)
    a3 = F / (w_pipe / 1.0)
    W_m0m = L0 * w0 + w_ball
    theta1 = np.arctan(W_m0m / F)
    W_m0m_m1 = W_m0m + w_bucket
    theta2 = np.arctan(W_m0m_m1 / F)
    
    # 计算连接点坐标
    C0 = np.log((1 + np.sin(theta0_rad)) / np.cos(theta0_rad))
    x0 = fsolve(lambda x0: catenary_length(a1, 0, x0, C0) - L0, 15.0)[0]
    y0 = catenary_equation(x0, a1, theta0=theta0_rad)
    
    C1 = np.log((1 + np.sin(theta1)) / np.cos(theta1))
    x1 = fsolve(lambda x1: catenary_length(a2, x0, x1, C1) - 1.0, x0 + 0.1)[0]
    y1 = catenary_equation(x1, a2, x0, y0, theta1)
    
    C2 = np.log((1 + np.sin(theta2)) / np.cos(theta2))
    x5 = fsolve(lambda x5: catenary_length(a3, x1, x5, C2) - 4.0, x1 + 1.0)[0]
    y5 = catenary_equation(x5, a3, x1, y1, theta2)
    
    # 输出方程组的参数
    print('这是y1的参数：')
    print(f'a1:{a1:.2f}    C0:{C0:.2f}     cos(theta0):{np.cos(theta0_rad):.2f}')
    print('='*30)
    print('这是y2的参数：')
    print(f'a1:{a2:.2f}    C0:{C1:.2f}     a2/cos(theta1):{(a2/np.cos(theta1)):.2f}     y10:{y1:.2f}')
    print('='*30)
    print('这是y1的参数：')
    print(f'a2:{a3:.2f}    C0:{C2:.2f}     a3/cos(theta2):{(a3/np.cos(theta2)):.2f}     y21:{y5:.2f}')
    print('='*30)
    
    # x的取值
    print(f'x0:{x0:.2f}')
    print(f'x1:{x1:.2f}')
    print(f'x5:{x5:.2f}')
    print('='*30)
    
    # 计算钢管竖直夹角
    pipe_angles = calculate_pipe_angles(a3, x1, x5, theta2)
    
    # 钢桶竖直倾斜角度
    bucket_angle_deg = 90 - np.degrees(theta1)
    
    return H_w, x5, bucket_angle_deg, theta0_deg, a1, a2, a3, theta1, theta2, x0, x1, x5, pipe_angles

# =====================
# 7. 可视化函数
# =====================
def plot_catenary_corrected(theta0_deg, H_w, x0, x1, x5, a1, a2, a3, theta1, theta2, pipe_angles):
    """无拖地部分的系泊系统可视化"""
    # 转换为弧度
    theta0_rad = np.radians(theta0_deg)
    
    # 创建x坐标点 (从锚点开始)
    x_chain = np.linspace(0, x0, 50)    # 锚链段
    x_bucket = np.linspace(x0, x1, 20)  # 钢桶段
    x_pipe = np.linspace(x1, x5, 30)    # 钢管段
    
    # 计算y坐标 (公式9)
    y_chain = catenary_equation(x_chain, a1, theta0=theta0_rad)
    y0 = catenary_equation(x0, a1, theta0=theta0_rad)
    y_bucket = catenary_equation(x_bucket, a2, x0, y0, theta1)
    y1_val = catenary_equation(x1, a2, x0, y0, theta1)
    y_pipe = catenary_equation(x_pipe, a3, x1, y1_val, theta2)
    
    # 绘图
    plt.figure(figsize=(12, 7))
    plt.plot(x_chain, y_chain, 'b-', label='锚链段')
    plt.plot(x_bucket, y_bucket, 'r-', label='钢桶段')
    plt.plot(x_pipe, y_pipe, 'g-', label='钢管段')
    
    # 标记关键点
    plt.scatter([0], [0], c='k', marker='*', s=100, label='锚点')
    plt.scatter([x0], [y0], c='purple', s=80, label='锚链-钢桶连接点')
    plt.scatter([x1], [y1_val], c='orange', s=80, label='钢桶-钢管连接点')
    plt.scatter([x5], [y_pipe[-1]], c='blue', s=100, label='浮标位置')
    
    # 添加角度标注
    plt.annotate('', xy=(0, 0), xytext=(1, 0.1),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    plt.text(0.5, 0.2, f'θ={theta0_deg:.2f}°', fontsize=12)
    
    # 标注钢管角度
    C = np.log((1 + np.sin(theta2)) / np.cos(theta2))
    for i, angle in enumerate(pipe_angles):
        s_mid = 0.5 + i * 1.0
        def equation(x):
            ds = a3 * np.sinh((x - x1)/a3 + C) - a3 * np.sinh(C)
            return ds - s_mid
        x_mid = fsolve(equation, x1 + (x5 - x1)*s_mid/4.0)[0]
        y_mid = catenary_equation(x_mid, a3, x1, y1_val, theta2)
        plt.text(x_mid, y_mid, f"{angle:.2f}°", 
                 fontsize=10, ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('水平距离 (m)')
    plt.ylabel('高度 (m)')
    plt.title(f'24m/s风速系泊系统构型 (θ0={theta0_deg:.2f}°, H_w={H_w:.2f}m)')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 添加海底和海平面
    plt.axhline(y=0, color='brown', linestyle='-', alpha=0.7, label='海床')
    plt.axhline(y=H_water, color='blue', linestyle='-', alpha=0.5, label='海平面')
    
    plt.axis('equal')
    plt.show()

# =====================
# 8. 主程序
# =====================
if __name__ == "__main__":
    print("\n>>> 严格求解24m/s风速时的系统状态")
    results = solve_problem24_corrected()
    H_w, x5, bucket_angle, theta0_deg, a1, a2, a3, theta1, theta2, x0, x1, x5_val, pipe_angles = results
    
    print("\n=== 最终结果 ===")
    print(f"浮标吃水深度: {H_w:.2f} m")
    print(f"浮标游动区域半径: {x5:.2f} m")
    print(f"钢桶竖直倾斜角度: {bucket_angle:.2f}°")
    print(f"锚链与海床夹角: {theta0_deg:.2f}°")
    
    # 输出钢管竖直夹角
    print("\n>>> 钢管与竖直方向夹角：")
    for i, angle in enumerate(pipe_angles):
        print(f"第{i+1}节钢管: {angle:.2f}°")
    
    # 可视化
    print("\n>>> 生成系泊系统示意图...")
    plot_catenary_corrected(theta0_deg, H_w, x0, x1, x5_val, a1, a2, a3, theta1, theta2, pipe_angles)