import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# =====================
# 1. 基本参数定义 (36m/s)
# =====================
L0 = 22.05        # 锚链长度(m)
m0 = 7.0          # 锚链每米质量(kg/m)
m_ball_start = 1800.0  # 初始重物球质量(kg)
m1 = 100.0        # 钢桶质量(kg)
m2 = 10.0         # 每节钢管质量(kg)
H_water = 18.0    # 海水深度(m)
v = 36.0          # 风速(m/s)
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

# 计算固定部件在水中的有效重量
w0 = calculate_buoyant_weight(m0, volume=m0/rho_steel)  # 锚链每米有效重量(N/m)
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
# 6. 求解函数 (给定重物球质量)
# =====================
def solve_for_given_mass(m_ball, v_wind):
    """求解给定重物球质量和风速时的系统状态"""
    # 计算重物球有效重量
    w_ball = calculate_buoyant_weight(m_ball)
    
    # 初始猜测值
    theta0_deg_guess = 0.0  # 锚链与海床夹角初始猜测值(度)
    H_w_guess = 0.0        # 浮标吃水深度初始猜测值(m)
    
    # 计算浮力系数 
    buoyancy_coeff = 1025 * np.pi * 9.8
    
    # 定义求解方程组
    def equations(vars):
        theta0_deg, H_w = vars
        
        # 转换为弧度
        theta0_rad = np.radians(theta0_deg)
        
        # 计算风力 (论文中风力公式)
        S_float = diameter_float * (height_float - H_w)
        F = 0.625 * S_float * v_wind**2
        
        # 计算悬链线参数
        a1 = F / w0
        a2 = F / (w_bucket / 1.0)  # 钢桶段参数
        a3 = F / (w_pipe / 1.0)    # 钢管段参数
        
        # 计算角度参数 (论文中受力分析)
        W_m0m = L0 * w0 + w_ball # 锚链的重力加上重力球的有效重力
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
    try:
        theta0_deg, H_w = fsolve(equations, [theta0_deg_guess, H_w_guess])
    except:
        # 如果求解失败，返回默认值
        return None, None, None, None, None, None
    
    # 计算最终结果
    theta0_rad = np.radians(theta0_deg)
    
    # 重新计算所有参数
    S_float = diameter_float * (height_float - H_w)
    F = 0.625 * S_float * v_wind**2
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
    
    # 计算钢桶竖直倾斜角度
    bucket_angle_deg = 90 - np.degrees(theta1)
    print(f'最大的活动半径：{x5:.2f}')
    
    # 计算4个钢管的竖直倾斜角度
    
    pip_angles = calculate_pipe_angles(a3, x1, x5, theta2)
    print('钢管的倾斜角:')
    for i in pip_angles:
        print(f'{i:.2f}')
    
    
    return theta0_deg, bucket_angle_deg, H_w, x5, m_ball, []

# =====================
# 7. 问题二求解函数
# =====================
def solve_problem2():
    """求解问题二：寻找使锚链与海床夹角为16°的重物球质量"""
    # 参数设置
    target_angle = 0.2  # 目标锚链与海床夹角(度)
    m_ball_min = 2086.5 # 最小重物球质量(kg)  2086.5
    m_ball_max = 6000.0  # 最大重物球质量(kg)
    step = 0.1           # 搜索步长(kg)
    
    # 存储结果
    results = []
    optimal_mass = None
    
    print(f"开始搜索重物球质量范围: {m_ball_min}kg 到 {m_ball_max}kg")
    print(f"目标锚链与海床夹角: {target_angle}°")
    
    # 迭代搜索
    m_ball = m_ball_min
    while m_ball <= m_ball_max:
        # 求解当前质量下的系统状态
        theta0_deg, bucket_angle, H_w, x5, _, _ = solve_for_given_mass(m_ball, v)
        
        if theta0_deg is None:
            m_ball += step
            continue
        
        # 记录结果
        results.append({
            'mass': m_ball,
            'theta0': theta0_deg,
            'bucket_angle': bucket_angle,
            'H_w': H_w,
            'x5': x5
        })
        
        # 打印当前结果
        print(f"质量: {m_ball:.2f}kg | 锚链夹角: {theta0_deg:.2f}° | 钢桶倾角: {bucket_angle:.2f}°")
        
        # 检查是否达到目标夹角
        if theta0_deg <= target_angle:
            optimal_mass = m_ball
            print(f"\n找到目标质量: {optimal_mass:.1f}kg")
            print(f"此时锚链夹角: {theta0_deg:.2f}°, 钢桶倾角: {bucket_angle:.2f}°")
            break
        # if bucket_angle <= 5.0:
        #     optimal_mass = m_ball
        #     print(f"钢桶竖直夹角等于5°时的重物球质量:{optimal_mass:.2f}")
        #     break
        m_ball += step
    
    if optimal_mass is None:
        print("\n在给定范围内未找到满足条件的重物球质量")
        return None, results
    
    return optimal_mass, results

# =====================
# 8. 可视化函数
# =====================
def plot_results(results, optimal_mass):
    """可视化搜索结果"""
    if not results:
        return
    
    # 提取数据
    masses = [r['mass'] for r in results]
    theta0s = [r['theta0'] for r in results]
    bucket_angles = [r['bucket_angle'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    # 锚链夹角曲线
    plt.subplot(2, 1, 1)
    plt.plot(masses, theta0s, 'b-o', label='锚链与海床夹角')
    plt.axhline(y=16, color='r', linestyle='--', label='目标夹角(16°)')
    if optimal_mass:
        plt.axvline(x=optimal_mass, color='g', linestyle='--', label=f'最优质量({optimal_mass:.2f}kg)')
    plt.xlabel('重物球质量 (kg)')
    plt.ylabel('角度 (°)')
    plt.title('重物球质量对锚链与海床夹角的影响')
    plt.grid(True)
    plt.legend()
    
    # 钢桶倾角曲线
    plt.subplot(2, 1, 2)
    plt.plot(masses, bucket_angles, 'g-o', label='钢桶倾斜角度')
    plt.axhline(y=5, color='r', linestyle='--', label='安全阈值(5°)')
    if optimal_mass:
        plt.axvline(x=optimal_mass, color='g', linestyle='--', label=f'最优质量({optimal_mass:.2f}kg)')
    plt.xlabel('重物球质量 (kg)')
    plt.ylabel('角度 (°)')
    plt.title('重物球质量对钢桶倾斜角度的影响')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('problem2_results.png', dpi=300)
    plt.show()

# =====================
# 9. 主程序
# =====================
if __name__ == "__main__":
    print("="*50)
    print("问题二求解：风速36m/s时确定重物球质量")
    print("="*50)
    
    # 求解问题二
    optimal_mass, results = solve_problem2()
    
    
    
    if optimal_mass:
        # 可视化结果
        plot_results(results, optimal_mass)
        
        # 输出最优解的详细结果
        print("\n=== 最优解详细参数 ===")
        theta0_deg, bucket_angle, H_w, x5, _, _ = solve_for_given_mass(optimal_mass, v)
        print(f"重物球质量: {optimal_mass:.2f} kg")
        print(f"锚链与海床夹角: {theta0_deg:.2f}°")
        print(f"钢桶倾斜角度: {bucket_angle:.2f}°")
        print(f"浮标吃水深度: {H_w:.2f} m")
        print(f"浮标游动区域半径: {x5:.2f} m")
        
        print("\n求解完成! 结果已保存到 problem2_results.png")