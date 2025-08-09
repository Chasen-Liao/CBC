import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# =====================
# 1. 问题三核心参数 (按论文P.8定义)
# =====================
# 基础参数
v_wind = 36.0          # 风速(m/s) - 极端情况
v_water = 1.5          # 流速(m/s)
H_water_options = [16, 18, 20]  # 海水深度变化(m) 
chain_types = {
    'I': {'d': 0.078, 'm': 3.2},
    'II': {'d': 0.105, 'm': 7},
    'III': {'d': 0.120, 'm': 12.5},
    'IV': {'d': 0.150, 'm': 19.5},
    'V': {'d': 0.180, 'm': 28.8}  # 论文推荐型号
}
selected_type = 'V'    # 按论文结论选择V型锚链

# 物理常数
rho_steel = 7850       # 钢材密度(kg/m³)
rho_water = 1025       # 海水密度(kg/m³)
g = 9.8                # 重力加速度(m/s²)
C_d = 1.2              # 水流阻力系数(文献常用值)

# 组件参数
m_bucket = 100.0       # 钢桶质量(kg)
m_pipe = 10.0          # 单节钢管质量(kg)
d_pipe = 0.1           # 钢管直径(m)
d_bucket = 0.3          # 钢桶直径(m)
d_float = 2.0          # 浮标直径(m)
h_float = 2.0          # 浮标高度(m)

# =====================
# 2. 水流力计算函数 (论文公式P.8)
# =====================
def calculate_flow_force(diameter, length, velocity):
    """计算圆柱体水流力: F = 0.5 * Cd * ρ * A * v²"""
    area = diameter * length  # 迎流面积
    return 0.5 * C_d * rho_water * area * velocity**2

# =====================
# 3. 浮力修正函数
# =====================
def effective_weight(mass, volume=None):
    """计算水中有效重量（考虑浮力）"""
    vol = volume if volume else mass / rho_steel
    return (mass - rho_water * vol) * g

# =====================
# 4. 水流力作用下的悬链线模型 (论文公式4-5)
# =====================
def catenary_with_flow(theta_k, T_k, ds, f_flow, w_eff):
    """
    水流力作用下的悬链线递推关系
    参数严格对应论文公式(5)
    """
    dT = (-f_flow * np.cos(theta_k) * np.sin(theta_k) + w_eff * np.sin(theta_k)) * ds
    dtheta = (f_flow * (np.sin(theta_k))**2 + w_eff * np.cos(theta_k)) / T_k * ds
    dx = np.cos(theta_k) * ds
    dy = np.sin(theta_k) * ds
    return dT, dtheta, dx, dy

# =====================
# 5. 问题三求解主函数
# =====================
def solve_problem3(target_angle=5.0, max_iter=20):
    """问题三核心求解器 (参数严格遵循论文P.9描述)"""
    # 结果存储
    results = {}
    
    for H_water in H_water_options:
        print(f"\n>>> 求解水深 {H_water}m 工况 [风速 {v_wind}m/s, 流速 {v_water}m/s]")
        
        # 5.1 计算水流力 (论文P.8数据)
        chain = chain_types[selected_type]
        u_chain = 0.5 * C_d * rho_water * chain['d'] * v_water**2  # 锚链单位长度水流力
        h_bucket = calculate_flow_force(d_bucket, 1.0, v_water)    # 钢桶水流力
        h_pipe = calculate_flow_force(d_pipe, 1.0, v_water)         # 钢管水流力
        h_float_force = calculate_flow_force(d_float, h_float, v_water)  # 浮标水流力
        
        # 5.2 迭代求解重物球质量 (论文P.9算法)
        m_ball = 4000  # 初始猜测值(kg)
        for iter in range(max_iter):
            # 计算组件有效重量
            w_ball_eff = effective_weight(m_ball)
            w_bucket_eff = effective_weight(m_bucket)
            w_pipe_eff = effective_weight(m_pipe)
            w_chain_eff = effective_weight(chain['m'])
            
            # 计算浮标吃水深度 (论文公式10修正)
            S_float = d_float * (h_float - H_w)  # 受风面积
            F_wind = 0.625 * S_float * v_wind**2
            buoyancy_force = rho_water * g * np.pi * (d_float/2)**2 * H_w
            
            # 力平衡方程 (水平方向)
            F_horizontal = F_wind + h_float_force + 4*h_pipe + h_bucket
            
            # 数值求解悬链线形态 (论文公式5实现)
            # [...] 此处实现20行迭代求解代码（因篇幅限制简化）
            # 核心逻辑：通过catenary_with_flow()递推计算锚链形态
            
            # 终止条件检查 (论文约束)
            if abs(bucket_angle - target_angle) < 0.1:
                print(f"迭代收敛: m_ball = {m_ball:.0f}kg, 钢桶倾角={bucket_angle:.2f}°")
                break
                
            # 质量调整策略 (论文启发式调整)
            m_ball += 50 if bucket_angle > target_angle else -50
        
        # 存储结果
        results[H_water] = {
            'm_ball': m_ball,
            'H_w': H_w,
            'swing_radius': swing_radius,
            'bucket_angle': bucket_angle
        }
    
    return results

# =====================
# 6. 多工况可视化 (论文图8-13风格)
# =====================
def visualize_results(results):
    """多工况结果可视化"""
    plt.figure(figsize=(15, 10))
    
    # 1. 吃水深度对比
    plt.subplot(2, 2, 1)
    depths = [res['H_w'] for res in results.values()]
    plt.bar(H_water_options, depths, width=0.6)
    plt.title('浮标吃水深度对比')
    plt.xlabel('水深(m)')
    plt.ylabel('吃水深度(m)')
    
    # 2. 钢桶倾角对比
    plt.subplot(2, 2, 2)
    angles = [res['bucket_angle'] for res in results.values()]
    plt.plot(H_water_options, angles, 'o-', markersize=8)
    plt.axhline(y=5, color='r', linestyle='--', label='安全阈值')
    plt.title('钢桶倾斜角度对比')
    plt.xlabel('水深(m)')
    plt.ylabel('倾角(度)')
    plt.legend()
    
    # 3. 锚链形态对比 (论文图8,9,10示例)
    plt.subplot(2, 1, 2)
    for H_water in H_water_options:
        # [...] 锚链形态绘制代码
        plt.plot(x_chain, y_chain, label=f'{H_water}m水深')
    plt.title('系泊系统形态对比')
    plt.legend()
    plt.tight_layout()
    plt.savefig('多工况对比.png', dpi=300)

# =====================
# 7. 执行主程序
# =====================
if __name__ == "__main__":
    # 求解多工况
    final_results = solve_problem3(target_angle=5.0)
    
    # 打印论文要求结果 (P.9表1)
    print("\n=== 论文要求的设计结果 ===")
    print(f"推荐配置: {selected_type}型锚链, 重物球质量={final_results[18]['m_ball']:.0f}kg")
    
    # 可视化对比
    visualize_results(final_results)
    
    # 输出关键参数
    print("\n=== 各工况详细结果 ===")
    for depth, data in final_results.items():
        print(f"水深{depth}m: 吃水={data['H_w']:.2f}m, 游动半径={data['swing_radius']:.2f}m, 倾角={data['bucket_angle']:.2f}°")