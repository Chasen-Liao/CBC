import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题


def mooring_system_solution(v_wind):
    """
    系泊系统求解函数 - 对应MATLAB的question1函数
    计算风速为24m/s时的系泊系统参数
    """
    # ===========================================
    # 1. 初始值设置 - 对应MATLAB的x0
    # ===========================================
    # 初始值数组，包含19个元素 
    # 牛顿迭代法的初始猜测值
    x0 = np.array([
        1372.4,          # Fwind - 风力 (N)
        6.0,             # unuse - 沉底长度 (m)
        0.78,            # d - 吃水深度 (m)
        14496.80,        # F1 - 钢管1左侧拉力 (N)
        14592.35,        # F2 - 钢管2左侧拉力 (N)
        14687.92,        # F3 - 钢管3左侧拉力 (N)
        14783.49,        # F4 - 钢管4左侧拉力 (N)
        14879.07,        # F5 - 浮标连接拉力 (N)
        math.radians(0.09),  # theta1 - 钢管1倾斜角度 (rad)
        math.radians(0.09),  # theta2 - 钢管2倾斜角度 (rad)
        math.radians(0.09),  # theta3 - 钢管3倾斜角度 (rad)
        math.radians(0.09),  # theta4 - 钢管4倾斜角度 (rad)
        math.radians(0.09),  # beta - 钢桶倾斜角度 (rad)
        math.radians(0.09),  # gamma1 - 钢管1下端力角度 (rad)
        math.radians(0.09),  # gamma2 - 钢管2下端力角度 (rad)
        math.radians(0.09),  # gamma3 - 钢管3下端力角度 (rad)
        math.radians(0.09),  # gamma4 - 钢管4下端力角度 (rad)
        math.radians(0.09),  # gamma5 - 浮标连接力角度 (rad)
        17.75            # x1 - 锚链末端横坐标 (m)
    ])
    
    # ===========================================
    # 2. 设置求解选项 - 对应MATLAB的optimset
    # ===========================================
    # 最大函数求值次数和最大迭代次数
    maxfev = 10000  # 最大函数求值次数
    maxiter = 10000  # 最大迭代次数
    
    # ===========================================
    # 3. 求解方程组 - 对应MATLAB的fsolve
    # ===========================================
    # 调用fsolve求解方程组
    solution, infodict, ier, mesg = fsolve(
        fangcheng,  # 方程组函数
        x0,        # 初始值
        args=(v_wind,),  # 额外参数
        full_output=True,  # 返回完整输出
        xtol=1e-6,        # 容差设置
        maxfev=maxfev     # 最大函数求值次数
    )
    
    # 检查求解是否成功
    if ier != 1:
        print(f"求解失败: {mesg}")
        return None
    
    # ===========================================
    # 4. 处理结果 - 对应MATLAB的x(9:18)=x(9:18)/pi*180
    # ===========================================
    # 将角度变量从弧度转换为度
    for i in range(8, 18):  # 索引8到17对应角度变量
        solution[i] = math.degrees(solution[i])
    
    # ===========================================
    # 5. 返回结果
    # ===========================================
    return solution

def fangcheng(x, v_wind):
    """
    系泊系统方程组函数 - 对应MATLAB的fangcheng函数
    x: 变量数组
    v_wind: 风速 (m/s)
    """
    # ===========================================
    # 1. 解包变量 - 对应MATLAB的变量赋值
    # ===========================================
    Fwind = x[0]    # 风力 (N)
    unuse = x[1]     # 沉底长度 (m)
    d = x[2]         # 吃水深度 (m)
    F1 = x[3]        # 钢管1左侧拉力 (N)
    F2 = x[4]        # 钢管2左侧拉力 (N)
    F3 = x[5]        # 钢管3左侧拉力 (N)
    F4 = x[6]        # 钢管4左侧拉力 (N)
    F5 = x[7]        # 浮标连接拉力 (N)
    theta1 = x[8]    # 钢管1倾斜角度 (rad)
    theta2 = x[9]    # 钢管2倾斜角度 (rad)
    theta3 = x[10]   # 钢管3倾斜角度 (rad)
    theta4 = x[11]   # 钢管4倾斜角度 (rad)
    beta = x[12]     # 钢桶倾斜角度 (rad)
    gamma1 = x[13]   # 钢管1下端力角度 (rad)
    gamma2 = x[14]   # 钢管2下端力角度 (rad)
    gamma3 = x[15]   # 钢管3下端力角度 (rad)
    gamma4 = x[16]   # 钢管4下端力角度 (rad)
    gamma5 = x[17]   # 浮标连接力角度 (rad)
    x1 = x[18]       # 锚链末端横坐标 (m)
    
    # ===========================================
    # 2. 常数定义 - 对应MATLAB的%%部分
    # ===========================================
    H = 18.0          # 水深 (m)
    p = 1025.0       # 海水密度 (kg/m³)
    sigma = 7.0      # 锚链线密度 (kg/m)
    g = 9.8          # 重力加速度 (m/s²)
    Mball = 1200 * 0.869426751592357  # 重物球质量 (kg)
    maolian = 22.05  # 锚链长度 (m)
    maolian = maolian - unuse  # 减去沉在海底的长度
    
    # 钢桶浮力计算 (对应MATLAB的floatage_bucket)
    bucket_radius = 0.15  # 钢桶半径 (m)
    floatage_bucket = p * g * math.pi * (bucket_radius ** 2)
    
    # 钢管浮力计算 (对应MATLAB的floatage_pipe)
    pipe_radius = 0.025  # 钢管半径 (m)
    floatage_pipe = p * g * math.pi * (pipe_radius ** 2)
    
    # ===========================================
    # 3. 锚链方程 - 对应MATLAB的y和Dy函数
    # ===========================================
    alpha1 = 0  # 锚链左端与海床夹角 (rad)
    
    # 悬链线方程 y(t)
    def y(t):
        term1 = Fwind / (sigma * g) * math.cosh(sigma * g * t / Fwind + math.asinh(math.tan(alpha1)))
        term2 = Fwind / (sigma * g) * math.cosh(math.asinh(math.tan(alpha1)))
        return term1 - term2
    
    # 导数函数 dy/dx
    def dy_dx(t):
        return math.sinh(sigma * g * t / Fwind + math.asinh(math.tan(alpha1)))
    
    # 弧长微分 ds
    def ds(t):
        return math.sqrt(1 + dy_dx(t) ** 2)
    
    # ===========================================
    # 4. 绘制锚链形状 - 对应MATLAB的绘图部分
    # ===========================================
    # 创建锚链点
    xx = np.linspace(0, x1, 100)
    yy = np.array([y(t) for t in xx])
    
    # 添加沉底部分
    xx_unuse = np.linspace(0, unuse, 100)
    yy_unuse = np.zeros_like(xx_unuse)
    
    # 合并坐标
    full_xx = np.concatenate([xx_unuse, xx + unuse])
    full_yy = np.concatenate([yy_unuse, yy])
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(full_xx, full_yy, linewidth=3)
    plt.title('锚链形状')
    plt.xlabel('锚链投影长度/m')
    plt.ylabel('距离海底高度/m')
    plt.grid(True)
    plt.xticks(np.arange(0, full_xx[-1] + 1, 1))
    plt.yticks(np.arange(0, full_yy[-1] + 1, 1))
    plt.savefig('anchor_chain_shape.png')  # 保存图像
    plt.close()
    
    # ===========================================
    # 5. 计算游动区域半径 - 对应MATLAB的R计算
    # ===========================================
    R = (math.sin(beta) + math.sin(theta1) + math.sin(theta2) + 
         math.sin(theta3) + math.sin(theta4) + full_xx[-1])
    
    # ===========================================
    # 6. 方程组定义 - 对应MATLAB的F=ones(19,1)和后续赋值
    # ===========================================
    F = np.zeros(19)
    
    # 锚链长度约束 (方程1)
    F[0], _ = quad(ds, 0, x1) # quad是求定积分的, ds是被积函数
    F[0] -= maolian
    
    # 锚链末端角度 (方程2)
    alph2 = math.atan(math.sinh(sigma * g * x1 / Fwind + math.asinh(math.tan(alpha1))))
    y1 = y(x1)
    
    # 钢桶力矩平衡 (方程3)
    F[1] = (F1 * math.sin(gamma1 - beta) + 
            (Fwind / math.cos(alph2)) * math.sin(math.pi/2 - alph2 - beta) - 
            Mball * g * math.sin(beta))
    
    # 钢桶竖直受力平衡 (方程4)
    F[2] = (F1 * math.cos(gamma1) + floatage_bucket - 
            100 * g - Mball * g - 
            Fwind * math.tan(alph2))
    
    # 钢桶水平受力平衡 (方程5)
    F[3] = F1 * math.sin(gamma1) - Fwind
    
    # 钢管1力矩平衡 (方程6)
    F[4] = F1 * math.sin(gamma1 - theta1) - F2 * math.sin(theta1 - gamma2)
    
    # 钢管2力矩平衡 (方程7)
    F[5] = F2 * math.sin(gamma2 - theta2) - F3 * math.sin(theta2 - gamma3)
    
    # 钢管3力矩平衡 (方程8)
    F[6] = F3 * math.sin(gamma3 - theta3) - F4 * math.sin(theta3 - gamma4)
    
    # 钢管4力矩平衡 (方程9)
    F[7] = F4 * math.sin(gamma4 - theta4) - F5 * math.sin(theta4 - gamma5)
    
    # 钢管1水平受力平衡 (方程10)
    F[8] = F2 * math.sin(gamma2) - Fwind
    
    # 钢管2水平受力平衡 (方程11)
    F[9] = F3 * math.sin(gamma3) - Fwind
    
    # 钢管3水平受力平衡 (方程12)
    F[10] = F4 * math.sin(gamma4) - Fwind
    
    # 钢管4水平受力平衡 (方程13)
    F[11] = F5 * math.sin(gamma5) - Fwind
    
    # 钢管1竖直受力平衡 (方程14)
    F[12] = (F1 * math.cos(gamma1) + 10 * g - 
             F2 * math.cos(gamma2) - floatage_pipe)
    
    # 钢管2竖直受力平衡 (方程15)
    F[13] = (F2 * math.cos(gamma2) + 10 * g - 
             F3 * math.cos(gamma3) - floatage_pipe)
    
    # 钢管3竖直受力平衡 (方程16)
    F[14] = (F3 * math.cos(gamma3) + 10 * g - 
             F4 * math.cos(gamma4) - floatage_pipe)
    
    # 钢管4竖直受力平衡 (方程17)
    F[15] = (F4 * math.cos(gamma4) + 10 * g - 
             F5 * math.cos(gamma5) - floatage_pipe)
    
    # 浮标竖直受力平衡 (方程18)
    buoy_radius = 1.0  # 浮标半径 (m)
    F[16] = (math.pi * (buoy_radius ** 2) * d * g * p - 
             1000 * g - F5 * math.cos(gamma5))
    
    # 浮标水平受力平衡 (方程19)
    F[17] = (F5 * math.sin(gamma5) - 
             0.625 * 2 * (2 - d) * (v_wind ** 2))
    
    # 水深约束 (方程20)
    F[18] = (y1 + math.cos(beta) + 
             math.cos(theta1) + math.cos(theta2) + 
             math.cos(theta3) + math.cos(theta4) + d - H)
    
    return F

# ===========================================
# 主程序
# ===========================================
if __name__ == "__main__":
    # 求解24m/s风速的系泊系统
    v_wind = 36
    solution = mooring_system_solution(v_wind=v_wind)
    
    if solution is not None:
        # 解析结果
        result_labels = [
            "风力 (N)", "沉底长度 (m)", "吃水深度 (m)",
            "F1 (N)", "F2 (N)", "F3 (N)", "F4 (N)", "F5 (N)",
            "钢管1倾角 (°)", "钢管2倾角 (°)", "钢管3倾角 (°)", "钢管4倾角 (°)",
            "钢桶倾角 (°)",
            "gamma1 (°)", "gamma2 (°)", "gamma3 (°)", "gamma4 (°)", "gamma5 (°)",
            "锚链末端横坐标 (m)"
        ]
        
        # 打印结果
        print("=" * 70)
        print(f"系泊系统参数计算结果 (风速{v_wind}m/s)")
        print("=" * 70)
        
        for i, label in enumerate(result_labels):
            # 对角度值进行特殊处理
            if "倾角" in label or "gamma" in label:
                value = solution[i]
            else:
                value = solution[i]
            print(f"{label:>20}: {value:.4f}")
        
        # 显示锚链形状图像
        print("\n锚链形状图已保存为 anchor_chain_shape.png")
    else:
        print("求解失败")