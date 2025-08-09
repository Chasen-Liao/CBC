import numpy as np
from scipy.optimize import minimize, fsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号

# =====================
# 1. 全局参数定义 (论文P.8数据)
# =====================
# 环境参数
V_WIND = 36.0          # 风速(m/s)
V_WATER = 1.5          # 流速(m/s)
H_WATER_OPTIONS = [16, 18, 20]  # 海水深度(m)
G = 9.8                # 重力加速度
RHO_WATER = 1025       # 海水密度(kg/m³)
RHO_STEEL = 7850       # 钢材密度(kg/m³)
C_D = 1.2              # 阻力系数

# 锚链型号参数 (型号I到V)
CHAIN_TYPES = {
    'I': {'d': 0.078, 'm': 3.2},
    'II': {'d': 0.105, 'm': 7},
    'III': {'d': 0.120, 'm': 12.5},
    'IV': {'d': 0.150, 'm': 19.5},
    'V': {'d': 0.180, 'm': 28.8}  # 论文推荐型号
}
SELECTED_CHAIN = 'V'   # 选择V型锚链

# 组件基本参数
# 浮标
FLOAT_DIAMETER = 2.0   # 直径(m)
FLOAT_HEIGHT = 2.0     # 高度(m)
# 钢管
PIPE_LENGTH = 1.0      # 单节长度(m)
PIPE_DIAMETER = 0.1    # 直径(m)
PIPE_MASS = 10.0       # 质量(kg)
NUM_PIPES = 4          # 数量
# 钢桶
BUCKET_LENGTH = 1.0    # 长度(m)
BUCKET_DIAMETER = 0.3  # 直径(m)
BUCKET_MASS = 100.0    # 质量(kg)
# 重物球
BALL_DENSITY = RHO_STEEL  # 重物球密度(kg/m³)

# =====================
# 2. 修复版悬链线求解器 (论文公式5)
# =====================
def solve_catenary_fixed(F_horizontal, components, H_water, h_submerged, ds=0.01):
    """
    修复版悬链线数值求解器
    正确实现水流力分解和倾角计算
    
    参数:
        F_horizontal: 水平方向合力(N)
        components: 组件列表，每个元素为[类型, 长度, 单位有效重力(N/m), 单位水流力(N/m)]
        H_water: 海水深度(m)
        h_submerged: 浮标吃水深度(m)
        ds: 计算步长(m)
    """
    # 初始化状态数组
    x = [0]  # 水平位置（从浮标开始）
    y = [-H_water + h_submerged]  # 垂直位置（浮标底部）
    theta = [np.pi/2]  # 初始方向（竖直向下）
    T = [F_horizontal]  # 初始张力（水平分量）
    
    # 从浮标向锚点递推（逆向求解）
    for comp_type, L, w_eff, f_flow in reversed(components):
        num_steps = int(L / ds)  # 当前组件分段数
        
        for i in range(num_steps):
            # 当前角度和张力
            current_theta = theta[-1]
            current_T = T[-1]
            
            # 水流力分解（正确实现）
            F_flow_x = f_flow * np.cos(current_theta)
            F_flow_y = f_flow * np.sin(current_theta)
            
            # 力学平衡方程（论文公式5）
            dT_ds = -F_flow_x * np.sin(current_theta) - F_flow_y * np.cos(current_theta) + w_eff * np.sin(current_theta)
            dtheta_ds = (F_flow_x * np.cos(current_theta) - F_flow_y * np.sin(current_theta) + w_eff * np.cos(current_theta)) / current_T
            
            # 更新状态
            T_new = current_T + dT_ds * ds
            theta_new = current_theta + dtheta_ds * ds
            dx = np.cos(theta_new) * ds
            dy = np.sin(theta_new) * ds
            
            # 存储新状态
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
            theta.append(theta_new)
            T.append(T_new)
    
    # 计算钢桶倾角（正确方法）
    # 找到钢桶部分的起点和终点索引
    bucket_start_idx = -1
    for i, (comp_type, _, _, _) in enumerate(components):
        if comp_type == "bucket":
            bucket_start_idx = sum(int(c[1]/ds) for c in components[:i]) + 1
            break
    
    if bucket_start_idx > 0 and bucket_start_idx < len(theta) - 10:
        # 取钢桶中点的切线方向
        mid_idx = bucket_start_idx + int(5/ds)  # 中点附近
        tangent_angle = theta[mid_idx]
        # 转换为与竖直方向夹角（度）
        bucket_angle = 90 - np.degrees(tangent_angle)
    else:
        bucket_angle = 0
    
    return x, y, bucket_angle, x[-1], y[-1]

# =====================
# 3. 修复版目标函数
# =====================
def objective_fixed(m_ball, H_water):
    """修复版目标函数，包含物理约束惩罚"""
    chain = CHAIN_TYPES[SELECTED_CHAIN]
    
    # 1. 计算组件参数
    # 重物球
    ball_volume = m_ball / BALL_DENSITY
    w_ball_eff = (m_ball * G) - (RHO_WATER * G * ball_volume)
    
    # 钢桶
    bucket_volume = np.pi * (BUCKET_DIAMETER/2)**2 * BUCKET_LENGTH
    w_bucket_eff = (BUCKET_MASS * G) - (RHO_WATER * G * bucket_volume)
    w_bucket_unit = w_bucket_eff / BUCKET_LENGTH
    f_bucket = flow_force(BUCKET_DIAMETER, BUCKET_LENGTH, V_WATER) / BUCKET_LENGTH
    
    # 钢管
    pipe_volume = np.pi * (PIPE_DIAMETER/2)**2 * PIPE_LENGTH
    w_pipe_eff = (PIPE_MASS * G) - (RHO_WATER * G * pipe_volume)
    w_pipe_unit = w_pipe_eff / PIPE_LENGTH
    f_pipe = flow_force(PIPE_DIAMETER, PIPE_LENGTH, V_WATER) / PIPE_LENGTH
    
    # 锚链
    chain_volume_per_m = chain['m'] / RHO_STEEL
    w_chain_eff = (chain['m'] * G) - (RHO_WATER * G * chain_volume_per_m)
    w_chain_unit = w_chain_eff
    f_chain = flow_force(chain['d'], 1.0, V_WATER)
    
    # 浮标水流力
    f_float = flow_force(FLOAT_DIAMETER, FLOAT_HEIGHT, V_WATER)
    
    # 2. 计算浮标吃水深度
    def float_equation(h_submerged):
        F_wind = 0.625 * FLOAT_DIAMETER * (FLOAT_HEIGHT - h_submerged) * V_WIND**2
        buoyancy = RHO_WATER * G * np.pi * (FLOAT_DIAMETER/2)**2 * h_submerged
        total_weight = w_ball_eff + w_bucket_eff + NUM_PIPES * w_pipe_eff
        return buoyancy - (1000*G + total_weight)
    
    h_submerged = fsolve(float_equation, 1.0)[0]
    
    # 3. 总水平力
    F_horizontal = 0.625 * FLOAT_DIAMETER * (FLOAT_HEIGHT - h_submerged) * V_WIND**2 + f_float
    
    # 4. 组件定义（从浮标到锚点顺序）
    components = [
        ("pipes", PIPE_LENGTH, w_pipe_unit, f_pipe) for _ in range(NUM_PIPES)
    ] + [
        ("bucket", BUCKET_LENGTH, w_bucket_unit, f_bucket),
        ("chain", 22.05, w_chain_unit, f_chain)  # 初始锚链长度
    ]
    
    # 5. 求解悬链线
    try:
        x, y, bucket_angle, swing_radius, y_anchor = solve_catenary_fixed(
            F_horizontal, components, H_water, h_submerged
        )
    except Exception as e:
        print(f"求解失败: {e}")
        return 1000, 0, 0, 0
    
    # 6. 目标值和约束惩罚
    target_angle = 5.0  # 目标倾角
    
    # 主要目标：倾角接近5°
    angle_error = abs(bucket_angle - target_angle)
    
    # 约束惩罚项
    penalty = 0
    
    # 锚链拖地判断（y_anchor应为0）
    if y_anchor > 0.1:  # 允许0.1m误差
        penalty += 10 * abs(y_anchor)
    
    # 物理约束
    if bucket_angle < 0:  # 负角度严重错误
        penalty += 100 * abs(bucket_angle)
    if h_submerged > FLOAT_HEIGHT:  # 浮标完全淹没
        penalty += 50 * (h_submerged - FLOAT_HEIGHT)
    
    return angle_error + penalty, h_submerged, swing_radius, bucket_angle

# =====================
# 4. 辅助函数
# =====================
def flow_force(diameter, length, velocity):
    """计算圆柱体水流力: F = 0.5 * Cd * ρ * A * v²"""
    area = diameter * length  # 迎流面积
    return 0.5 * C_D * RHO_WATER * area * velocity**2

# =====================
# 5. 主求解函数
# =====================
def solve_problem3():
    """主求解函数：遍历水深，优化重物球质量"""
    results = {}
    chain = CHAIN_TYPES[SELECTED_CHAIN]
    
    for H_water in H_WATER_OPTIONS:
        print(f"\n=== 求解水深 {H_water}m ===")
        
        # 优化重物球质量
        res = minimize(
            fun=lambda m: objective_fixed(m, H_water)[0],
            x0=0,
            bounds=[(3000, 5000)],
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 0.1}
        )
        
        if not res.success:
            print(f"! 优化失败: {res.message}")
            m_ball_opt = 4000
        else:
            m_ball_opt = res.x[0]
        
        # 最终计算
        error, h_submerged, swing_radius, bucket_angle = objective_fixed(m_ball_opt, H_water)
        
        # 存储结果
        results[H_water] = {
            'm_ball': m_ball_opt,
            'H_w': h_submerged,
            'swing_radius': swing_radius,
            'bucket_angle': bucket_angle
        }
        print(f"* 优化结果: m_ball={m_ball_opt:.0f}kg, 吃水深度={h_submerged:.2f}m, "
              f"游动半径={swing_radius:.2f}m, 钢桶倾角={bucket_angle:.2f}°")
    
    return results

# =====================
# 6. 可视化函数
# =====================
def plot_results(results):
    """可视化不同水深下的设计参数"""
    plt.figure(figsize=(14, 8))
    
    # 重物球质量对比
    plt.subplot(2, 2, 1)
    m_balls = [res['m_ball'] for res in results.values()]
    plt.bar([str(h) for h in H_WATER_OPTIONS], m_balls, color='skyblue')
    plt.title('重物球质量优化结果')
    plt.xlabel('水深(m)')
    plt.ylabel('质量(kg)')
    
    # 吃水深度对比
    plt.subplot(2, 2, 2)
    h_ws = [res['H_w'] for res in results.values()]
    plt.plot([str(h) for h in H_WATER_OPTIONS], h_ws, 'o-', markersize=8)
    plt.axhline(y=FLOAT_HEIGHT, color='r', linestyle='--', label='浮标高度')
    plt.title('浮标吃水深度')
    plt.xlabel('水深(m)')
    plt.ylabel('深度(m)')
    plt.legend()
    
    # 钢桶倾角对比
    plt.subplot(2, 2, 3)
    angles = [res['bucket_angle'] for res in results.values()]
    plt.bar([str(h) for h in H_WATER_OPTIONS], angles, color='lightgreen')
    plt.axhline(y=5, color='r', linestyle='--', label='目标倾角')
    plt.title('钢桶倾斜角度')
    plt.xlabel('水深(m)')
    plt.ylabel('角度(°)')
    plt.legend()
    
    # 游动半径对比
    plt.subplot(2, 2, 4)
    radii = [res['swing_radius'] for res in results.values()]
    plt.plot([str(h) for h in H_WATER_OPTIONS], radii, 's-', markersize=8)
    plt.title('浮标游动半径')
    plt.xlabel('水深(m)')
    plt.ylabel('半径(m)')
    
    plt.tight_layout()
    plt.savefig('系泊系统优化结果.png', dpi=300)
    plt.show()

# =====================
# 7. 主程序
# =====================
if __name__ == "__main__":
    print("="*60)
    print(f"近浅海观测网系泊系统优化设计")
    print(f"工况: 风速={V_WIND}m/s, 流速={V_WATER}m/s, 锚链型号={SELECTED_CHAIN}")
    print("="*60)
    
    # 求解多工况
    results = solve_problem3()
    
    # 打印最终结果
    print("\n=== 最终设计结果 ===")
    for h, res in results.items():
        print(f"水深{h}m: 重物球={res['m_ball']:.0f}kg, 吃水={res['H_w']:.2f}m, "
              f"游动半径={res['swing_radius']:.2f}m, 倾角={res['bucket_angle']:.2f}°")
    
    # 论文推荐配置
    print("\n=== 论文推荐配置 ===")
    print(f"锚链型号: {SELECTED_CHAIN}型")
    print(f"重物球质量: {results[18]['m_ball']:.0f}kg (以18m水深为基准)")
    
    # 可视化
    plot_results(results)
    print("可视化结果已保存为 '系泊系统优化结果.png'")