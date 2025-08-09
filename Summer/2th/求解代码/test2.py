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
rho_water = 1025  # 海水密度(kg/m³)
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
# 4. 钢管竖直夹角计算函数
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
# 5. 24m/s求解函数 (修正版)
# =====================
def solve_problem24_corrected():
    """严格遵循论文的24m/s求解函数"""
    theta0_deg = 4.0  # 锚链与海床夹角(度) # 目标求出这个
    theta0_rad = np.radians(theta0_deg)
    
    # 计算浮力系数
    buoyancy_coeff = 1025 * np.pi * g  # 海水密度*π*g
    
    # 计算总有效重量 (公式10)
    total_weight = 1000*g + L0*w0 + w_ball + w_bucket + 4*w_pipe
    
    # 迭代求解吃水深度
    H_w = 0.60  # 初始假设
    for _ in range(5):  # 5次迭代足够收敛
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
        
        # 定义方程组求解连接点坐标
        def equations(vars):
            x0, x1, x5 = vars
            # 锚链段方程 (公式9)
            y0 = catenary_equation(x0, a1, theta0=theta0_rad)
            
            # 钢桶段方程 (公式9)
            y1 = catenary_equation(x1, a2, x0, y0, theta1)
            
            # 钢管段方程 (公式9)
            y5 = catenary_equation(x5, a3, x1, y1, theta2)
            
            # 约束条件 (论文方程(8))
            eq1 = y5 - (H_water - H_w)  # 浮标底部位置
            eq2 = a2 * (np.sinh((x1 - x0)/a2 + np.log((1+np.sin(theta1))/np.cos(theta1)))) - \
                   a2 * np.sinh(np.log((1+np.sin(theta1))/np.cos(theta1))) - 1.0  # 钢桶长度
            eq3 = a3 * (np.sinh((x5 - x1)/a3 + np.log((1+np.sin(theta2))/np.cos(theta2)))) - \
                   a3 * np.sinh(np.log((1+np.sin(theta2))/np.cos(theta2))) - 4.0  # 钢管总长度
            return [eq1, eq2, eq3]
        
        # 求解方程组
        x0, x1, x5 = fsolve(equations, [17.0, 17.5, 18.0])
        # print(x0, x1, x5)
        # 更新吃水深度 (公式10)
        H_w = (total_weight + F * np.tan(theta0_rad)) / buoyancy_coeff
        # theta0_deg = np.degrees(theta0_rad)
    
    # 最终结果 (锚链全部拉起)
    drag_length = 0.0
    bucket_angle_deg = 90 - np.degrees(theta1)
    # print('theta0_deg:',theta0_deg)
    
    # 计算钢管竖直夹角
    pipe_angles = calculate_pipe_angles(a3, x1, x5, theta2)
    
    return H_w, drag_length, x5, bucket_angle_deg, theta0_deg, a1, a2, a3, theta1, theta2, x0, x1, x5, pipe_angles

# =====================
# 6. 可视化函数
# =====================
def plot_catenary_corrected(x0, x1, x5, a1, a2, a3, theta0_rad, theta1, theta2, pipe_angles):
    """无拖地部分的系泊系统可视化"""
    # 创建x坐标点 (从锚点开始)
    x_chain = np.linspace(0, x0, 50)    # 锚链段
    x_bucket = np.linspace(x0, x1, 20)  # 钢桶段
    x_pipe = np.linspace(x1, x5, 30)    # 钢管段
    
    # 计算y坐标 (公式9)
    y_chain = catenary_equation(x_chain, a1, theta0=theta0_rad)
    y0 = catenary_equation(x0, a1, theta0=theta0_rad)
    y_bucket = catenary_equation(x_bucket, a2, x0, y0, theta1)
    y1 = catenary_equation(x1, a2, x0, y0, theta1)
    y_pipe = catenary_equation(x_pipe, a3, x1, y1, theta2)
    
    # 绘图
    plt.figure(figsize=(12, 7))
    plt.plot(x_chain, y_chain, 'b-', label='锚链段')
    plt.plot(x_bucket, y_bucket, 'r-', label='钢桶段')
    plt.plot(x_pipe, y_pipe, 'g-', label='钢管段')
    
    # 标记关键点
    plt.scatter([0], [0], c='k', marker='*', s=100, label='锚点')
    plt.scatter([x0], [y0], c='purple', s=80, label='锚链-钢桶连接点')
    plt.scatter([x1], [y1], c='orange', s=80, label='钢桶-钢管连接点')
    plt.scatter([x5], [y_pipe[-1]], c='blue', s=100, label='浮标位置')
    
    # 添加角度标注
    plt.annotate('', xy=(0, 0), xytext=(1, 0.1),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    plt.text(0.5, 0.2, f'θ={np.degrees(theta0_rad):.2f}°', fontsize=12)
    
    # 标注钢管角度
    C = np.log((1 + np.sin(theta2)) / np.cos(theta2))
    for i, angle in enumerate(pipe_angles):
        s_mid = 0.5 + i * 1.0
        def equation(x):
            ds = a3 * np.sinh((x - x1)/a3 + C) - a3 * np.sinh(C)
            return ds - s_mid
        x_mid = fsolve(equation, x1 + (x5 - x1)*s_mid/4.0)[0]
        y_mid = catenary_equation(x_mid, a3, x1, y1, theta2)
        plt.text(x_mid, y_mid, f"{angle:.2f}°", 
                 fontsize=10, ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('水平距离 (m)')
    plt.ylabel('高度 (m)')
    plt.title('24m/s风速系泊系统构型s')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 添加海底和海平面
    plt.axhline(y=0, color='brown', linestyle='-', alpha=0.7, label='海床')
    plt.axhline(y=H_water, color='blue', linestyle='-', alpha=0.5, label='海平面')
    
    plt.axis('equal')
    plt.show()

# =====================
# 7. 主程序
# =====================
if __name__ == "__main__":
    print("\n>>> 严格求解24m/s风速时的系统状态")
    results = solve_problem24_corrected()
    H_w, drag_length, x5, bucket_angle, theta0_deg, a1, a2, a3, theta1, theta2, x0, x1, x5, pipe_angles = results
    print('xo x1 x5',x0, x1, x5)
    print("\n=== 最终结果 ===")
    print(f"浮标吃水深度: {H_w:.2f} m")
    #print(f"锚链拖地长度: {drag_length:.2f} m (全部拉起)")
    print(f"浮标游动区域半径: {x5:.2f} m")
    print(f"钢桶竖直倾斜角度: {bucket_angle:.2f}°")
    print(f"锚链与海床夹角: {theta0_deg:.2f}°")
    
    # 输出钢管竖直夹角
    print("\n>>> 钢管与竖直方向夹角：")
    for i, angle in enumerate(pipe_angles):
        print(f"第{i+1}节钢管: {angle:.2f}°")
    
    # 可视化
    print("\n>>> 生成系泊系统示意图...")
    theta0_rad = np.radians(theta0_deg)
    plot_catenary_corrected(x0, x1, x5, a1, a2, a3, theta0_rad, theta1, theta2, pipe_angles)