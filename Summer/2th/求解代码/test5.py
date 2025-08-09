import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

def fangcheng1(x, Mball, v_wind=36):
    """
    锚链有沉底部分的方程组函数
    对应 MATLAB 的 fangcheng1 函数
    
    参数:
    x: 包含所有未知变量的数组
    Mball: 重物球质量 (kg)
    v_wind: 风速 (m/s)，默认为36m/s
    
    返回:
    F: 方程组残差数组
    """
    # =========================================================
    # 1. 解包变量 - 对应MATLAB代码中的变量赋值
    # =========================================================
    Fwind = x[0]    # 风力 (N)
    unuse = x[1]    # 沉底长度 (m)
    d = x[2]        # 吃水深度 (m)
    F1, F2, F3, F4, F5 = x[3:8]  # 钢管拉力 (N)
    theta1, theta2, theta3, theta4 = x[8:12]  # 钢管倾斜角度 (rad)
    beta = x[12]    # 钢桶倾斜角度 (rad)
    gamma1, gamma2, gamma3, gamma4, gamma5 = x[13:18]  # 钢管下端力角度 (rad)
    x1 = x[18]      # 锚链末端横坐标 (m)
    
    # =========================================================
    # 2. 常数定义 - 固定系统参数
    # =========================================================
    H = 18.0          # 水深 (m)
    p = 1025.0        # 海水密度 (kg/m³)
    sigma = 7.0       # 锚链线密度 (kg/m)
    g = 9.8           # 重力加速度 (m/s²)
    maolian = 22.05   # 锚链总长度 (m)
    maolian_eff = maolian - unuse  # 有效锚链长度 (减去沉底部分)
    
    # 钢桶浮力计算 (半径0.15m)
    bucket_radius = 0.15
    floatage_bucket = p * g * math.pi * (bucket_radius ** 2)
    
    # 钢管浮力计算 (半径0.025m)
    pipe_radius = 0.025
    floatage_pipe = p * g * math.pi * (pipe_radius ** 2)
    
    # =========================================================
    # 3. 锚链方程 - 悬链线模型
    # =========================================================
    alpha1 = 0  # 锚链左端与海床夹角 (沉底情况)
    
    # 悬链线方程 y(t)
    def y_func(t):
        term1 = Fwind / (sigma * g) * math.cosh(sigma * g * t / Fwind + math.asinh(math.tan(alpha1)))
        term2 = Fwind / (sigma * g) * math.cosh(math.asinh(math.tan(alpha1)))
        return term1 - term2
    
    # 导数函数 dy/dx
    def dy_dx(t):
        return math.sinh(sigma * g * t / Fwind + math.asinh(math.tan(alpha1)))
    
    # 弧长微分 ds
    def ds(t):
        return math.sqrt(1 + dy_dx(t) ** 2)
    
    # =========================================================
    # 4. 计算游动区域半径 R
    # =========================================================
    R = (math.sin(beta) + math.sin(theta1) + math.sin(theta2) + 
         math.sin(theta3) + math.sin(theta4) + x1 + unuse)
    
    # =========================================================
    # 5. 方程组定义 (19个方程)
    # =========================================================
    F = np.zeros(19)
    
    # 方程1: 锚链长度约束
    F[0], _ = quad(ds, 0, x1)
    F[0] -= maolian_eff
    
    # 方程2: 锚链末端角度计算
    alph2 = math.atan(math.sinh(sigma * g * x1 / Fwind + math.asinh(math.tan(alpha1))))
    y1 = y_func(x1)
    
    # 方程3: 钢桶力矩平衡
    F[1] = (F1 * math.sin(gamma1 - beta) + 
            (Fwind / math.cos(alph2)) * math.sin(math.pi/2 - alph2 - beta) - 
            Mball * g * math.sin(beta))
    
    # 方程4: 钢桶竖直受力平衡
    F[2] = (F1 * math.cos(gamma1) + floatage_bucket - 
            100 * g - Mball * g - 
            Fwind * math.tan(alph2))
    
    # 方程5: 钢桶水平受力平衡
    F[3] = F1 * math.sin(gamma1) - Fwind
    
    # 方程6-8: 钢管1-4力矩平衡
    F[4] = F1 * math.sin(gamma1 - theta1) - F2 * math.sin(theta1 - gamma2)
    F[5] = F2 * math.sin(gamma2 - theta2) - F3 * math.sin(theta2 - gamma3)
    F[6] = F3 * math.sin(gamma3 - theta3) - F4 * math.sin(theta3 - gamma4)
    F[7] = F4 * math.sin(gamma4 - theta4) - F5 * math.sin(theta4 - gamma5)
    
    # 方程9-12: 钢管1-4水平受力平衡
    F[8] = F2 * math.sin(gamma2) - Fwind
    F[9] = F3 * math.sin(gamma3) - Fwind
    F[10] = F4 * math.sin(gamma4) - Fwind
    F[11] = F5 * math.sin(gamma5) - Fwind
    
    # 方程13-16: 钢管1-4竖直受力平衡
    F[12] = (F1 * math.cos(gamma1) + 10 * g - 
             F2 * math.cos(gamma2) - floatage_pipe)
    F[13] = (F2 * math.cos(gamma2) + 10 * g - 
             F3 * math.cos(gamma3) - floatage_pipe)
    F[14] = (F3 * math.cos(gamma3) + 10 * g - 
             F4 * math.cos(gamma4) - floatage_pipe)
    F[15] = (F4 * math.cos(gamma4) + 10 * g - 
             F5 * math.cos(gamma5) - floatage_pipe)
    
    # 方程17: 浮标竖直受力平衡
    buoy_radius = 1.0  # 浮标半径 (m)
    F[16] = (math.pi * (buoy_radius ** 2) * d * g * p - 
             1000 * g - F5 * math.cos(gamma5))
    
    # 方程18: 水深约束
    F[17] = (y1 + math.cos(beta) + 
             math.cos(theta1) + math.cos(theta2) + 
             math.cos(theta3) + math.cos(theta4) + d - H)
    
    # 方程19: 风力平衡
    F[18] = (F5 * math.sin(gamma5) - 
             0.625 * 2 * (2 - d) * (v_wind ** 2))
    
    return F, R

def fangcheng2(x, Mball, v_wind=36):
    """
    锚链无沉底部分的方程组函数
    对应 MATLAB 的 fangcheng2 函数
    
    参数:
    x: 包含所有未知变量的数组
    Mball: 重物球质量 (kg)
    v_wind: 风速 (m/s)，默认为36m/s
    
    返回:
    F: 方程组残差数组
    """
    # =========================================================
    # 1. 解包变量 - 类似fangcheng1但alpha1不同
    # =========================================================
    Fwind = x[0]    # 风力 (N)
    alpha1 = x[1]   # 锚链左端与海床夹角 (rad) - 无沉底情况
    d = x[2]        # 吃水深度 (m)
    F1, F2, F3, F4, F5 = x[3:8]  # 钢管拉力 (N)
    theta1, theta2, theta3, theta4 = x[8:12]  # 钢管倾斜角度 (rad)
    beta = x[12]    # 钢桶倾斜角度 (rad)
    gamma1, gamma2, gamma3, gamma4, gamma5 = x[13:18]  # 钢管下端力角度 (rad)
    x1 = x[18]      # 锚链末端横坐标 (m)
    
    # =========================================================
    # 2. 常数定义 - 同fangcheng1
    # =========================================================
    H = 18.0          # 水深 (m)
    p = 1025.0        # 海水密度 (kg/m³)
    sigma = 7.0       # 锚链线密度 (kg/m)
    g = 9.8           # 重力加速度 (m/s²)
    
    # 钢桶浮力计算
    bucket_radius = 0.15
    floatage_bucket = p * g * math.pi * (bucket_radius ** 2)
    
    # 钢管浮力计算
    pipe_radius = 0.025
    floatage_pipe = p * g * math.pi * (pipe_radius ** 2)
    
    # =========================================================
    # 3. 锚链方程 - 悬链线模型 (无沉底情况)
    # =========================================================
    # 悬链线方程 y(t)
    def y_func(t):
        term1 = Fwind / (sigma * g) * math.cosh(sigma * g * t / Fwind + math.asinh(math.tan(alpha1)))
        term2 = Fwind / (sigma * g) * math.cosh(math.asinh(math.tan(alpha1)))
        return term1 - term2
    
    # 导数函数 dy/dx
    def dy_dx(t):
        return math.sinh(sigma * g * t / Fwind + math.asinh(math.tan(alpha1)))
    
    # 弧长微分 ds
    def ds(t):
        return math.sqrt(1 + dy_dx(t) ** 2)
    
    # =========================================================
    # 4. 计算游动区域半径 R
    # =========================================================
    R = x1 + math.sin(beta) + math.sin(theta1) + math.sin(theta2) + math.sin(theta3) + math.sin(theta4)
    
    # =========================================================
    # 5. 方程组定义 (19个方程)
    # =========================================================
    F = np.zeros(19)
    
    # 方程1: 锚链长度约束
    F[0], _ = quad(ds, 0, x1)
    F[0] -= 22.05  # 锚链总长度
    
    # 方程2: 锚链末端角度计算
    alph2 = math.atan(math.sinh(sigma * g * x1 / Fwind + math.asinh(math.tan(alpha1))))
    y1 = y_func(x1)
    
    # 方程3: 钢桶力矩平衡
    F[1] = (F1 * math.sin(gamma1 - beta) + 
            (Fwind / math.cos(alph2)) * math.sin(math.pi/2 - alph2 - beta) - 
            Mball * g * math.sin(beta))
    
    # 方程4: 钢桶竖直受力平衡
    F[2] = (F1 * math.cos(gamma1) + floatage_bucket - 
            100 * g - Mball * g - 
            Fwind * math.tan(alph2))
    
    # 方程5: 钢桶水平受力平衡
    F[3] = F1 * math.sin(gamma1) - Fwind
    
    # 方程6-8: 钢管1-4力矩平衡
    F[4] = F1 * math.sin(gamma1 - theta1) - F2 * math.sin(theta1 - gamma2)
    F[5] = F2 * math.sin(gamma2 - theta2) - F3 * math.sin(theta2 - gamma3)
    F[6] = F3 * math.sin(gamma3 - theta3) - F4 * math.sin(theta3 - gamma4)
    F[7] = F4 * math.sin(gamma4 - theta4) - F5 * math.sin(theta4 - gamma5)
    
    # 方程9-12: 钢管1-4水平受力平衡
    F[8] = F2 * math.sin(gamma2) - Fwind
    F[9] = F3 * math.sin(gamma3) - Fwind
    F[10] = F4 * math.sin(gamma4) - Fwind
    F[11] = F5 * math.sin(gamma5) - Fwind
    
    # 方程13-16: 钢管1-4竖直受力平衡
    F[12] = (F1 * math.cos(gamma1) + 10 * g - 
             F2 * math.cos(gamma2) - floatage_pipe)
    F[13] = (F2 * math.cos(gamma2) + 10 * g - 
             F3 * math.cos(gamma3) - floatage_pipe)
    F[14] = (F3 * math.cos(gamma3) + 10 * g - 
             F4 * math.cos(gamma4) - floatage_pipe)
    F[15] = (F4 * math.cos(gamma4) + 10 * g - 
             F5 * math.cos(gamma5) - floatage_pipe)
    
    # 方程17: 浮标竖直受力平衡
    buoy_radius = 1.0  # 浮标半径 (m)
    F[16] = (math.pi * (buoy_radius ** 2) * d * g * p - 
             1000 * g - F5 * math.cos(gamma5))
    
    # 方程18: 水深约束
    F[17] = (y1 + math.cos(beta) + 
             math.cos(theta1) + math.cos(theta2) + 
             math.cos(theta3) + math.cos(theta4) + d - H)
    
    # 方程19: 风力平衡
    F[18] = (F5 * math.sin(gamma5) - 
             0.625 * 2 * (2 - d) * (v_wind ** 2))
    
    return F, R

def fun(Mball, v_wind=36):
    """
    求解系泊系统方程组的函数
    对应 MATLAB 的 fun 函数
    
    参数:
    Mball: 重物球质量 (kg)
    v_wind: 风速 (m/s)，默认为36m/s
    
    返回:
    solution: 求解结果数组
    fval: 函数残差
    exitflag: 求解状态
    R: 游动区域半径
    Unuse: 锚链沉底情况 (0-无沉底, 1-有沉底)
    """
    # 初始值设置 (无沉底情况)
    x0 = np.array([
        1372.4,           # Fwind (N)
        0.18,             # alpha1 (rad)
        0.78,             # d (m)
        14496.80, 14592.35, 14687.92, 14783.49, 14879.07,  # F1-F5 (N)
        math.radians(0.09), math.radians(0.09), math.radians(0.09), math.radians(0.09),  # theta1-4 (rad)
        math.radians(0.09),  # beta (rad)
        math.radians(0.09), math.radians(0.09), math.radians(0.09), math.radians(0.09), math.radians(0.09),  # gamma1-5 (rad)
        17.75             # x1 (m)
    ])
    
    # 求解无沉底情况
    solution, infodict, ier, mesg = fsolve(
        lambda x: fangcheng2(x, Mball, v_wind)[0],  # 方程组函数
        x0,        # 初始值
        full_output=True  # 返回完整信息
    )
    
    Unuse = 0  # 初始假设无沉底
    
    # 检查锚链左端角度是否小于0 (表示有沉底)
    if solution[1] < 0:
        # 更新初始值 (有沉底情况)
        x0 = np.array([
            1372.4,           # Fwind (N)
            6.0,              # unuse (m)
            0.78,             # d (m)
            14496.80, 14592.35, 14687.92, 14783.49, 14879.07,  # F1-F5 (N)
            math.radians(0.09), math.radians(0.09), math.radians(0.09), math.radians(0.09),  # theta1-4 (rad)
            math.radians(0.09),  # beta (rad)
            math.radians(0.09), math.radians(0.09), math.radians(0.09), math.radians(0.09), math.radians(0.09),  # gamma1-5 (rad)
            17.75             # x1 (m)
        ])
        
        # 重新求解有沉底情况
        solution, infodict, ier, mesg = fsolve(
            lambda x: fangcheng1(x, Mball, v_wind)[0],  # 方程组函数
            x0,        # 初始值
            full_output=True  # 返回完整信息
        )
        Unuse = 1  # 标记有沉底
    
    # 计算游动区域半径
    if Unuse == 1:
        _, R = fangcheng1(solution, Mball, v_wind)
    else:
        _, R = fangcheng2(solution, Mball, v_wind)
    
    return solution, infodict['fvec'], ier, R, Unuse

def question2():
    """
    主优化函数 - 对应 MATLAB 的 question2 函数
    优化重物球质量，寻找最佳设计
    """
    # 初始化结果存储
    G = []       # 重物球质量列表
    beta = []    # 钢桶倾角列表 (rad)
    alpha1 = []  # 锚链左端角度列表 (rad)
    d = []       # 吃水深度列表 (m)
    R = []       # 游动区域半径列表 (m)
    
    # 重物球质量范围: 1700kg 到 5000kg，步长10kg
    mass_value = np.arange(1700.0, 5001.1, 0.1)
    for Mball1 in mass_value:
        # 质量单位转换
        Mball = Mball1 * 0.869426751592357
        
        # 求解方程组
        x, fval, exitflag, r, Unuse = fun(Mball)
        
        # 检查是否满足约束条件:
        # 1. 锚链左端角度 < 16° (0.279 rad)
        # 2. 吃水深度 < 1.5m
        # 3. 钢桶倾角 < 5° (0.087 rad)
        if ((Unuse == 0 and x[1] > 0.279) or 
            x[2] > 1.5 or 
            x[12] > 0.087):
            continue  # 跳过不满足约束的解
        
        # 保存结果
        G.append(Mball1)
        beta.append(x[12])  # 钢桶倾角 (rad)
        
        # 根据沉底情况处理锚链左端角度
        if Unuse == 0:
            alpha1.append(x[1])  # 无沉底情况，使用实际角度
        else:
            alpha1.append(0)     # 有沉底情况，角度设为0
        
        d.append(x[2])  # 吃水深度
        R.append(r)     # 游动区域半径
    
    # =========================================================
    # 结果可视化 (5个图表)
    # =========================================================
    
    # 图表1: β随重物球质量变化图
    plt.figure(figsize=(10, 6))
    plt.plot(G, np.degrees(beta))
    plt.title('β随重物球质量变化图')
    plt.xlabel('重物球质量/kg')
    plt.ylabel('β/°')
    plt.grid(True)
    plt.savefig('beta_vs_mass.png')
    plt.close()
    
    # 图表2: 区域半径随重物球质量变化图
    plt.figure(figsize=(10, 6))
    plt.plot(G, R)
    plt.title('区域半径随重物球质量变化图')
    plt.xlabel('重物球质量/kg')
    plt.ylabel('半径/m')
    plt.grid(True)
    plt.savefig('radius_vs_mass.png')
    plt.close()
    
    # 图表3: 吃水深度随重物球质量变化图
    plt.figure(figsize=(10, 6))
    plt.plot(G, d)
    plt.title('吃水深度随重物球质量变化图')
    plt.xlabel('重物球质量/kg')
    plt.ylabel('深度/m')
    plt.grid(True)
    plt.savefig('depth_vs_mass.png')
    plt.close()
    
    # 图表4: α1随重物球质量变化图
    plt.figure(figsize=(10, 6))
    plt.plot(G, np.degrees(alpha1))
    plt.title('α1随重物球质量变化图')
    plt.xlabel('重物球质量/kg')
    plt.ylabel('α1/°')
    plt.ylim(0, 16)  # 设置y轴范围0-16度
    plt.yticks(range(0, 17))  # 设置y轴刻度
    plt.grid(True)
    plt.savefig('alpha1_vs_mass.png')
    plt.close()
    
    # 图表5: 优化目标随重物球质量变化图
    plt.figure(figsize=(10, 6))
    # 计算优化目标 (加权和)
    k = [0.1, 0.8, 0.1]  # 权重: 吃水深度:β:区域半径
    # 归一化处理
    d_norm = [val / max(d) for val in d]
    beta_norm = [val / max(beta) for val in beta]
    R_norm = [val / max(R) for val in R]
    # 计算加权和
    y_vals = [d_norm[i] * k[0] + beta_norm[i] * k[1] + R_norm[i] * k[2] for i in range(len(G))]
    
    plt.plot(G, y_vals)
    plt.title('优化目标随重物球质量变化图')
    plt.xlabel('重物球质量/kg')
    plt.ylabel('目标值')
    plt.grid(True)
    plt.savefig('objective_vs_mass.png')
    plt.close()
    
    # =========================================================
    # 寻找最优解
    # =========================================================
    # 找到最小目标值对应的索引
    min_index = np.argmin(y_vals)
    
    # 输出最优解
    print("=" * 70)
    print("最优重物球质量参数:")
    print("=" * 70)
    print(f"重物球质量: {G[min_index]:.4f} kg")
    print(f"钢桶倾角: {np.degrees(beta[min_index]):.4f}°")
    print(f"锚链左端角度: {np.degrees(alpha1[min_index]):.4f}°")
    print(f"吃水深度: {d[min_index]:.4f} m")
    print(f"游动区域半径: {R[min_index]:.4f} m")
    
    # 返回所有结果
    return {
        'G': G,
        'beta': beta,
        'alpha1': alpha1,
        'd': d,
        'R': R,
        'optimal_index': min_index
    }

# =========================================================
# 主程序入口
# =========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("系泊系统优化设计 - 重物球质量优化")
    print("=" * 70)
    
    # 运行优化
    results = question2()
    
    # 显示优化结果
    print("\n优化完成！结果图表已保存为PNG文件")