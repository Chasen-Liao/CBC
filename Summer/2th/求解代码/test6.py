import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math
import time

def question3_uniform_flow():
    """
    主优化函数 - 各点水速相同、水流与风同向情况
    对应MATLAB的question3_junyunshuili函数
    """
    # =========================================================
    # 1. 初始化和参数设置
    # =========================================================
    start_time = time.time()  # 记录开始时间
    SIGMA = [3.2, 7, 12.5, 19.5, 28.12]  # 不同型号锚链的线密度 (kg/m)
    H = 20.0  # 水深 (m) - 这里固定为20m，但原MATLAB中H是变量
    
    # 初始化结果存储列表
    G_list = []        # 重物球质量列表
    BETA_list = []     # 钢桶倾角列表 (rad)
    ALPH1_list = []    # 锚链左端角度列表 (rad)
    D_list = []        # 吃水深度列表 (m)
    RR_list = []       # 游动区域半径列表 (m)
    A_list = []        # 锚链参数组合 [sigma, maolian, Mball]
    THETA_list = []    # 钢管倾角列表
    
    # =========================================================
    # 2. 三重循环遍历参数空间
    # =========================================================
    # 只使用第5种锚链型号 (28.12 kg/m)
    for sigma in [SIGMA[4]]:
        print(f"正在处理锚链型号: {sigma} kg/m")
        
        # 锚链长度范围: 21.1m 到 22.0m, 步长0.1m
        for maolian in np.arange(20.1, 22.1, 0.1):
            print(f"  锚链长度: {maolian:.1f} m")
            
            # 重物球质量范围: 4000kg 到 4002kg, 步长1kg
            for Mball in range(4000, 5003, 1):
                # 求解方程组
                x, fval, exitflag, r, Unuse = fun_uniform_flow(
                    Mball, sigma, maolian, H
                )
                
                # 检查约束条件:
                # 1. 锚链左端角度 < 16° (0.279 rad)
                # 2. 吃水深度 < 1.8m 且 >0
                # 3. 钢桶倾角 < 5° (0.087 rad)
                # 4. 求解成功 (exitflag > 0)
                # 5. 有沉底时沉底长度>0
                constraint_violation = (
                    (Unuse == 0 and x[1] > 0.279) or
                    (x[2] > 1.8) or
                    (x[2] < 0) or
                    (x[12] > 0.087) or
                    (exitflag <= 0) or
                    (Unuse == 1 and x[1] < 0)
                )
                
                if constraint_violation:
                    continue  # 跳过不满足约束的解
                
                # 保存结果
                G_list.append(Mball)
                BETA_list.append(x[12])
                
                # 根据沉底情况处理锚链左端角度
                if Unuse == 0:
                    ALPH1_list.append(x[1])  # 无沉底情况
                else:
                    ALPH1_list.append(0)      # 有沉底情况
                
                D_list.append(x[2])
                RR_list.append(r)
                A_list.append([sigma, maolian, Mball])
                THETA_list.append(x[8:12])  # 存储钢管倾角
    
    # =========================================================
    # 3. 多目标优化计算
    # =========================================================
    # 权重设置: 吃水深度:β:区域半径 = 0.1:0.8:0.1
    k = [0.1, 0.8, 0.1]
    
    # 归一化处理
    D_norm = [d / max(D_list) for d in D_list]
    BETA_norm = [beta / max(BETA_list) for beta in BETA_list]
    RR_norm = [rr / max(RR_list) for rr in RR_list]
    
    # 计算加权目标函数
    y_vals = [
        D_norm[i] * k[0] + 
        BETA_norm[i] * k[1] + 
        RR_norm[i] * k[2] 
        for i in range(len(G_list))
    ]
    
    # 寻找最小目标值
    min_index = np.argmin(y_vals)
    
    # =========================================================
    # 4. 结果输出
    # =========================================================
    print("\n" + "=" * 70)
    print("系泊系统优化结果 (各点水速相同、水流与风同向)")
    print("=" * 70)
    print(f"计算耗时: {time.time() - start_time:.2f} 秒")
    print(f"可行解数量: {len(G_list)}")
    print("\n最优参数组合:")
    print("-" * 70)
    print(f"锚链线密度: {A_list[min_index][0]:.2f} kg/m")
    print(f"锚链长度: {A_list[min_index][1]:.2f} m")
    print(f"重物球质量: {A_list[min_index][2]} kg")
    print(f"钢桶倾角: {np.degrees(BETA_list[min_index]):.4f}°")
    print(f"吃水深度: {D_list[min_index]:.4f} m")
    print(f"游动区域半径: {RR_list[min_index]:.4f} m")
    
    return {
        'G': G_list,
        'BETA': BETA_list,
        'ALPH1': ALPH1_list,
        'D': D_list,
        'RR': RR_list,
        'A': A_list,
        'THETA': THETA_list,
        'optimal_index': min_index
    }

def fun_uniform_flow(Mball, sigma, maolian, H, v_wind=36):
    """
    求解系泊系统方程组 (考虑水流力)
    对应MATLAB的fun函数
    
    参数:
    Mball: 重物球质量 (kg)
    sigma: 锚链线密度 (kg/m)
    maolian: 锚链长度 (m)
    H: 水深 (m)
    v_wind: 风速 (m/s)，默认为36m/s
    
    返回:
    solution: 求解结果
    fval: 残差
    exitflag: 求解状态
    R: 游动区域半径
    Unuse: 沉底标志 (0-无沉底, 1-有沉底)
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
    
    # 尝试无沉底情况求解
    try:
        # 使用更稳健的求解器设置
        solution, infodict, ier, mesg = fsolve(
            lambda x: fangcheng2_uniform_flow(x, Mball, sigma, maolian, H, v_wind)[0],
            x0,
            full_output=True,
            xtol=1e-6,
            maxfev=10000
        )
        Unuse = 0  # 无沉底
        
        # 检查锚链左端角度是否小于0 (表示有沉底)
        if solution[1] < 0:
            # 切换为有沉底情况
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
            
            solution, infodict, ier, mesg = fsolve(
                lambda x: fangcheng1_uniform_flow(x, Mball, sigma, maolian, H, v_wind)[0],
                x0,
                full_output=True,
                xtol=1e-6,
                maxfev=10000
            )
            Unuse = 1  # 有沉底
    except Exception as e:
        print(f"求解失败: {e}")
        return None, None, -1, None, None
    
    # 计算游动区域半径
    if Unuse == 1:
        _, R = fangcheng1_uniform_flow(solution, Mball, sigma, maolian, H, v_wind)
    else:
        _, R = fangcheng2_uniform_flow(solution, Mball, sigma, maolian, H, v_wind)
    
    return solution, infodict['fvec'], ier, R, Unuse

def fangcheng1_uniform_flow(x, Mball, sigma, maolian, H, v_wind=36):
    """
    有沉底情况的系泊系统方程组 (考虑水流力)
    对应MATLAB的fangcheng1函数
    
    参数:
    x: 变量数组
    Mball: 重物球质量 (kg)
    sigma: 锚链线密度 (kg/m)
    maolian: 锚链长度 (m)
    H: 水深 (m)
    v_wind: 风速 (m/s)
    
    返回:
    F: 方程组残差
    R: 游动区域半径
    """
    # =========================================================
    # 1. 解包变量
    # =========================================================
    Fwind = x[0]    # 风力 (N)
    unuse = x[1]    # 沉底长度 (m)
    d = x[2]        # 吃水深度 (m)
    F1, F2, F3, F4, F5 = x[3:8]  # 钢管拉力 (N)
    theta1, theta2, theta3, theta4 = x[8:12]  # 钢管倾角 (rad)
    beta = x[12]    # 钢桶倾角 (rad)
    gamma1, gamma2, gamma3, gamma4, gamma5 = x[13:18]  # 力角度 (rad)
    x1 = x[18]      # 锚链末端横坐标 (m)
    
    # =========================================================
    # 2. 常数定义
    # =========================================================
    p = 1025.0        # 海水密度 (kg/m³)
    g = 9.8           # 重力加速度 (m/s²)
    maolian_eff = maolian - unuse  # 有效锚链长度
    
    # 浮力计算
    bucket_radius = 0.15  # 钢桶半径 (m)
    floatage_bucket = p * g * math.pi * (bucket_radius ** 2) * 1.0  # 钢桶浮力
    
    pipe_radius = 0.025   # 钢管半径 (m)
    floatage_pipe = p * g * math.pi * (pipe_radius ** 2) * 1.0  # 单根钢管浮力
    
    # 水流力参数 (根据论文)
    pipe_water_force = 42.075  # 单根钢管水流力系数
    bucket_water_force = 252.45  # 钢桶水流力系数
    buoy_water_force = 2 * d * 374 * 1.5 ** 2  # 浮标水流力
    
    # 总水平力 (风力+水流力)
    F_total = (Fwind + 
               4 * pipe_water_force + 
               bucket_water_force + 
               buoy_water_force)
    
    # =========================================================
    # 3. 锚链方程 (悬链线模型)
    # =========================================================
    alpha1 = 0  # 锚链左端与海床夹角 (沉底情况)
    
    def y_func(t):
        term1 = F_total / (sigma * g) * math.cosh(
            sigma * g * t / F_total + math.asinh(math.tan(alpha1))
        )
        term2 = F_total / (sigma * g) * math.cosh(math.asinh(math.tan(alpha1)))
        return term1 - term2
    
    def dy_dx(t):
        return math.sinh(sigma * g * t / F_total + math.asinh(math.tan(alpha1)))
    
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
    alph2 = math.atan(math.sinh(sigma * g * x1 / F_total + math.asinh(math.tan(alpha1))))
    y1 = y_func(x1)
    
    # 方程3: 钢桶力矩平衡
    F[1] = (F1 * math.sin(gamma1 - beta) + 
            (F_total / math.cos(alph2)) * math.sin(math.pi/2 - alph2 - beta) - 
            Mball * g * math.sin(beta))
    
    # 方程4: 钢桶竖直受力平衡
    F[2] = (F1 * math.cos(gamma1) + floatage_bucket - 
            100 * g - Mball * g - 
            F_total * math.tan(alph2))
    
    # 方程5: 钢桶水平受力平衡
    F[3] = F1 * math.sin(gamma1) - F_total
    
    # 方程6-9: 钢管1-4力矩平衡
    F[4] = F1 * math.sin(gamma1 - theta1) - F2 * math.sin(theta1 - gamma2)
    F[5] = F2 * math.sin(gamma2 - theta2) - F3 * math.sin(theta2 - gamma3)
    F[6] = F3 * math.sin(gamma3 - theta3) - F4 * math.sin(theta3 - gamma4)
    F[7] = F4 * math.sin(gamma4 - theta4) - F5 * math.sin(theta4 - gamma5)
    
    # 方程10-13: 钢管1-4水平受力平衡
    F[8] = F2 * math.sin(gamma2) - (Fwind + 3 * pipe_water_force + buoy_water_force)
    F[9] = F3 * math.sin(gamma3) - (Fwind + 2 * pipe_water_force + buoy_water_force)
    F[10] = F4 * math.sin(gamma4) - (Fwind + pipe_water_force + buoy_water_force)
    F[11] = F5 * math.sin(gamma5) - (Fwind + buoy_water_force)
    
    # 方程14-17: 钢管1-4竖直受力平衡
    F[12] = (F1 * math.cos(gamma1) + 10 * g - 
             F2 * math.cos(gamma2) - floatage_pipe)
    F[13] = (F2 * math.cos(gamma2) + 10 * g - 
             F3 * math.cos(gamma3) - floatage_pipe)
    F[14] = (F3 * math.cos(gamma3) + 10 * g - 
             F4 * math.cos(gamma4) - floatage_pipe)
    F[15] = (F4 * math.cos(gamma4) + 10 * g - 
             F5 * math.cos(gamma5) - floatage_pipe)
    
    # 方程18: 浮标竖直受力平衡
    buoy_radius = 1.0  # 浮标半径 (m)
    F[16] = (math.pi * (buoy_radius ** 2) * d * g * p - 
             1000 * g - F5 * math.cos(gamma5))
    
    # 方程19: 水深约束
    F[17] = (y1 + math.cos(beta) + 
             math.cos(theta1) + math.cos(theta2) + 
             math.cos(theta3) + math.cos(theta4) + d - H)
    
    # 方程20: 风力平衡
    F[18] = (0.625 * 2 * (2 - d) * v_wind ** 2 - Fwind)
    
    return F, R

def fangcheng2_uniform_flow(x, Mball, sigma, maolian, H, v_wind=36):
    """
    无沉底情况的系泊系统方程组 (考虑水流力)
    对应MATLAB的fangcheng2函数
    """
    # 解包变量
    Fwind = x[0]    # 风力 (N)
    alpha1 = x[1]   # 锚链左端角度 (rad)
    d = x[2]        # 吃水深度 (m)
    F1, F2, F3, F4, F5 = x[3:8]  # 钢管拉力 (N)
    theta1, theta2, theta3, theta4 = x[8:12]  # 钢管倾角 (rad)
    beta = x[12]    # 钢桶倾角 (rad)
    gamma1, gamma2, gamma3, gamma4, gamma5 = x[13:18]  # 力角度 (rad)
    x1 = x[18]      # 锚链末端横坐标 (m)
    
    # 常数定义
    p = 1025.0        # 海水密度 (kg/m³)
    g = 9.8           # 重力加速度 (m/s²)
    
    # 浮力计算
    bucket_radius = 0.15  # 钢桶半径 (m)
    floatage_bucket = p * g * math.pi * (bucket_radius ** 2) * 1.0
    
    pipe_radius = 0.025   # 钢管半径 (m)
    floatage_pipe = p * g * math.pi * (pipe_radius ** 2) * 1.0
    
    # 水流力参数
    pipe_water_force = 42.075
    bucket_water_force = 252.45
    buoy_water_force = 2 * d * 374 * 1.5 ** 2
    
    # 总水平力
    F_total = (Fwind + 
               4 * pipe_water_force + 
               bucket_water_force + 
               buoy_water_force)
    
    # 锚链方程
    def y_func(t):
        term1 = F_total / (sigma * g) * math.cosh(
            sigma * g * t / F_total + math.asinh(math.tan(alpha1))
        )
        term2 = F_total / (sigma * g) * math.cosh(math.asinh(math.tan(alpha1)))
        return term1 - term2
    
    def dy_dx(t):
        return math.sinh(sigma * g * t / F_total + math.asinh(math.tan(alpha1)))
    
    def ds(t):
        return math.sqrt(1 + dy_dx(t) ** 2)
    
    # 计算游动区域半径
    R = x1 + math.sin(beta) + math.sin(theta1) + math.sin(theta2) + math.sin(theta3) + math.sin(theta4)
    
    # 方程组定义
    F = np.zeros(19)
    
    # 方程1: 锚链长度约束
    F[0], _ = quad(ds, 0, x1)
    F[0] -= maolian
    
    # 方程2: 锚链末端角度计算
    alph2 = math.atan(math.sinh(sigma * g * x1 / F_total + math.asinh(math.tan(alpha1))))
    y1 = y_func(x1)
    
    # 方程3: 钢桶力矩平衡
    F[1] = (F1 * math.sin(gamma1 - beta) + 
            (F_total / math.cos(alph2)) * math.sin(math.pi/2 - alph2 - beta) - 
            Mball * g * math.sin(beta))
    
    # 方程4: 钢桶竖直受力平衡
    F[2] = (F1 * math.cos(gamma1) + floatage_bucket - 
            100 * g - Mball * g - 
            F_total * math.tan(alph2))
    
    # 方程5: 钢桶水平受力平衡
    F[3] = F1 * math.sin(gamma1) - F_total
    
    # 方程6-9: 钢管1-4力矩平衡
    F[4] = F1 * math.sin(gamma1 - theta1) - F2 * math.sin(theta1 - gamma2)
    F[5] = F2 * math.sin(gamma2 - theta2) - F3 * math.sin(theta2 - gamma3)
    F[6] = F3 * math.sin(gamma3 - theta3) - F4 * math.sin(theta3 - gamma4)
    F[7] = F4 * math.sin(gamma4 - theta4) - F5 * math.sin(theta4 - gamma5)
    
    # 方程10-13: 钢管1-4水平受力平衡
    F[8] = F2 * math.sin(gamma2) - (Fwind + 3 * pipe_water_force + buoy_water_force)
    F[9] = F3 * math.sin(gamma3) - (Fwind + 2 * pipe_water_force + buoy_water_force)
    F[10] = F4 * math.sin(gamma4) - (Fwind + pipe_water_force + buoy_water_force)
    F[11] = F5 * math.sin(gamma5) - (Fwind + buoy_water_force)
    
    # 方程14-17: 钢管1-4竖直受力平衡
    F[12] = (F1 * math.cos(gamma1) + 10 * g - 
             F2 * math.cos(gamma2) - floatage_pipe)
    F[13] = (F2 * math.cos(gamma2) + 10 * g - 
             F3 * math.cos(gamma3) - floatage_pipe)
    F[14] = (F3 * math.cos(gamma3) + 10 * g - 
             F4 * math.cos(gamma4) - floatage_pipe)
    F[15] = (F4 * math.cos(gamma4) + 10 * g - 
             F5 * math.cos(gamma5) - floatage_pipe)
    
    # 方程18: 浮标竖直受力平衡
    buoy_radius = 1.0
    F[16] = (math.pi * (buoy_radius ** 2) * d * g * p - 
             1000 * g - F5 * math.cos(gamma5))
    
    # 方程19: 水深约束
    F[17] = (y1 + math.cos(beta) + 
             math.cos(theta1) + math.cos(theta2) + 
             math.cos(theta3) + math.cos(theta4) + d - H)
    
    # 方程20: 风力平衡
    F[18] = (0.625 * 2 * (2 - d) * v_wind ** 2 - Fwind)
    
    return F, R

# =========================================================
# 主程序入口
# =========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("系泊系统优化设计 - 各点水速相同、水流与风同向")
    print("=" * 70)
    
    # 运行优化
    results = question3_uniform_flow()
    
    if results:
        # 输出最优解详细信息
        opt_idx = results['optimal_index']
        print("\n最优解详细信息:")
        print("-" * 70)
        print(f"钢桶倾角: {np.degrees(results['BETA'][opt_idx]):.4f}°")
        print(f"钢管1倾角: {np.degrees(results['THETA'][opt_idx][0]):.4f}°")
        print(f"钢管2倾角: {np.degrees(results['THETA'][opt_idx][1]):.4f}°")
        print(f"钢管3倾角: {np.degrees(results['THETA'][opt_idx][2]):.4f}°")
        print(f"钢管4倾角: {np.degrees(results['THETA'][opt_idx][3]):.4f}°")
        print(f"锚链左端角度: {np.degrees(results['ALPH1'][opt_idx]):.4f}°")
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(results['G'], np.degrees(results['BETA']))
        plt.title('钢桶倾角 vs 重物球质量')
        plt.xlabel('重物球质量 (kg)')
        plt.ylabel('钢桶倾角 (°)')
        plt.grid(True)
        plt.savefig('beta_vs_mass.png')
        plt.close()
        
        print("\n优化完成! 结果图表已保存为 'beta_vs_mass.png'")