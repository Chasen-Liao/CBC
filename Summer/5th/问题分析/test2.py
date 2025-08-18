import numpy as np
import pandas as pd
from scipy.optimize import bisect
from scipy import interpolate
import matplotlib.pyplot as plt
from datetime import datetime
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# ======================================================
# 阿基米德螺线模型（向心运动）
# ======================================================
class CentripetalSpiral:
    def __init__(self, pitch):
        self.b = pitch / (2 * np.pi)  # 螺距系数
        # 预计算弧长-角度映射表（0到1000pi）
        self.theta_max = 1000 * np.pi
        self.theta_values = np.linspace(0, self.theta_max, 100000)
        self.s_values = self.arc_length(self.theta_values)
        # 创建插值函数
        self.s_to_theta = interpolate.interp1d(
            self.s_values, self.theta_values, 
            kind='linear', bounds_error=False,
            fill_value=(0, self.theta_max)
        )
    
    def arc_length(self, theta):
        """计算从中心到角度θ的弧长"""
        if isinstance(theta, np.ndarray):
            return 0.5 * self.b * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))
        else:
            if theta == 0:
                return 0.0
            return 0.5 * self.b * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))
    
    def theta_from_arc_length(self, s):
        """通过弧长反解角度(使用插值)"""
        if s <= 0:
            return 0.0
        return float(self.s_to_theta(s))
    
    def position(self, theta):
        """计算笛卡尔坐标位置(向心运动)"""
        r = self.b * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)  # 顺时针旋转
        return x, y
    
    def velocity(self, theta):
        """计算速度矢量(向心运动)"""
        if theta == 0:
            return 0.0, 0.0
        
        denom = np.sqrt(theta**2 + 1)
        # 向心运动时θ减小，速度方向相反
        vx = (np.cos(theta) - theta * np.sin(theta)) / denom
        vy = (np.sin(theta) + theta * np.cos(theta)) / denom
        return vx, vy

# ======================================================
# 舞龙队物理模型（龙尾在A点起始，逆时针生长）
# ======================================================
class GrowthDragonDance:
    def __init__(self, growth_time=300, move_speed=1.0):
        # 龙体参数
        self.head_length = 2.86  # 龙头两个把手间距(m)
        self.body_length = 1.65  # 龙身/龙尾两个把手间距(m)
        
        # 创建螺线模型（螺距0.55米）
        self.spiral = CentripetalSpiral(0.55)
        
        # A点初始位置（最外侧点）: 32圈
        self.theta_A = 32 * np.pi
        self.s_A = self.spiral.arc_length(self.theta_A)  # A点初始弧长
        
        # 总长度 = 龙头长度 + 222节龙身
        self.L_full = self.head_length + 222 * self.body_length
        
        # 运动参数
        self.growth_speed = self.L_full / growth_time  # 生长速度(m/s)
        self.move_speed = move_speed      # 向心运动速度(m/s)
        
        # 创建累积距离数组（从龙尾到龙头）
        self.cum_lengths = np.zeros(224)  # 224个点
        # 龙尾起始点（A点）
        self.cum_lengths[0] = 0
        # 龙尾后把手
        self.cum_lengths[1] = self.body_length
        # 龙身各点
        for i in range(2, 223):
            self.cum_lengths[i] = self.cum_lengths[i-1] + self.body_length
        # 龙头
        self.cum_lengths[223] = self.cum_lengths[222] + self.head_length
    
    def get_state(self, t):
        positions = []
        velocities = []
        
        # 计算生长长度和移动距离
        growth_length = min(t * self.growth_speed, self.L_full)  # 当前生长长度
        s_move = t * self.move_speed  # 向心运动的弧长减少量
        
        # 龙尾位置（A点移动后位置）
        s_tail = max(0, self.s_A - s_move)
        theta_tail = self.spiral.theta_from_arc_length(s_tail)
        
        # 处理所有点（从龙尾到龙头）
        for i in range(224):
            # 如果尚未生长到该点
            if self.cum_lengths[i] > growth_length:
                # 使用当前生长前沿的位置
                s_front = s_tail + growth_length
                theta_front = self.spiral.theta_from_arc_length(s_front)
                x, y = self.spiral.position(theta_front)
                vx, vy = 0.0, 0.0  # 未生长点速度为0
            else:
                # 计算当前点位置
                s_current = max(0, s_tail + self.cum_lengths[i])
                theta = self.spiral.theta_from_arc_length(s_current)
                x, y = self.spiral.position(theta)
                vx, vy = self.spiral.velocity(theta)
            
            positions.append((x, y))
            velocities.append((vx, vy))
        
        return positions, velocities

def save_results(results, filename):
    """保存结果到Excel文件"""
    # 创建结果DataFrame
    columns = []
    data = []
    
    for t, (positions, velocities) in enumerate(results):
        row = []
        # 位置数据
        for (x, y) in positions:
            row.extend([x, y])
        # 速度数据(速度大小)
        for (vx, vy) in velocities:
            speed = np.sqrt(vx**2 + vy**2)
            row.append(speed)
        
        data.append(row)
    
    # 创建列名
    for i in range(224):
        columns.append(f"P{i}_x")
        columns.append(f"P{i}_y")
    
    for i in range(224):
        columns.append(f"V{i}")
    
    df = pd.DataFrame(data, columns=columns)
    
    # 保存到Excel
    df.to_excel(filename, index=False, float_format="%.6f")
    print(f"结果已保存到 {filename}")

def generate_report(dragon_dance, time_points):
    """生成论文所需格式的报告"""
    # 位置报告
    position_data = {f"{t}s": {} for t in time_points}
    
    # 速度报告
    velocity_data = {f"{t}s": {} for t in time_points}
    
    # 指定报告点索引
    report_indices = [0, 1, 51, 101, 151, 201, 223]
    report_names = [
        "龙头", "第1节龙身", "第51节龙身", 
        "第101节龙身", "第151节龙身", "第201节龙身", "龙尾(后)"
    ]
    
    for t in time_points:
        positions, velocities = dragon_dance.get_state(t)
        
        for idx, name in zip(report_indices, report_names):
            x, y = positions[idx]
            vx, vy = velocities[idx]
            speed = np.sqrt(vx**2 + vy**2)
            
            # 添加到位置报告
            position_data[f"{t}s"][f"{name}x"] = x
            position_data[f"{t}s"][f"{name}y"] = y
            
            # 添加到速度报告
            velocity_data[f"{t}s"][name] = speed
    
    # 转换为DataFrame
    pos_df = pd.DataFrame(position_data).T
    vel_df = pd.DataFrame(velocity_data).T
    
    return pos_df, vel_df

def visualize_dragon(positions, t):
    """可视化舞龙队位置（俯视图）"""
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    
    plt.figure(figsize=(10, 8))
    
    # 绘制螺线背景
    theta = np.linspace(0, 32*np.pi, 1000)
    r = 0.0875 * theta
    bg_x = r * np.cos(theta)
    bg_y = r * np.sin(theta)
    plt.plot(bg_x, bg_y, 'gray', alpha=0.2)
    
    # 绘制舞龙队
    plt.plot(x, y, 'b-', linewidth=1, alpha=0.7)
    
    # 标记关键点
    plt.plot(x[0], y[0], 'ro', markersize=8, label='龙头')
    plt.plot(x[223], y[223], 'go', markersize=8, label='龙尾')
    
    # 标记初始A点
    if t == 0:
        plt.plot(x[0], y[0], 'k*', markersize=12, label='A点(初始位置)')
    
    plt.title(f"舞龙队位置 (t = {t}s)", fontsize=14)
    plt.xlabel("X (m)", fontsize=12)
    plt.ylabel("Y (m)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.legend(loc='best')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'dragon_position_{t}s.png', dpi=300)
    plt.close()
    print(f"已生成位置图: dragon_position_{t}s.png")

# ======================================================
# 主执行函数
# ======================================================
def main():
    # 初始化生长模型
    dragon = GrowthDragonDance(growth_time=300, move_speed=1.0)
    
    # 模拟时间范围 (0-300秒)
    total_time = 300
    results = []
    
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    # 执行模拟
    start_time = datetime.now()
    print(f"\n开始模拟 {total_time} 秒运动...")
    for t in range(total_time + 1):
        positions, velocities = dragon.get_state(t)
        results.append((positions, velocities))
        
        # 在关键时间点可视化
        if t in [0, 60, 120, 180, 240, 300]:
            print(f"生成 t={t}s 位置图...")
            visualize_dragon(positions, t)
    
    # 保存完整结果
    save_results(results, "results/growth_model_result.xlsx")
    
    # 生成报告
    print("\n生成论文报告表格...")
    report_times = [0, 60, 120, 180, 240, 300]
    pos_report, vel_report = generate_report(dragon, report_times)
    
    # 打印报告
    print("\n=== 位置报告 (单位：米) ===")
    print(pos_report)
    
    print("\n=== 速度报告 (单位：m/s) ===")
    print(vel_report)
    
    # 保存报告
    pos_report.to_excel("results/growth_position_report.xlsx")
    vel_report.to_excel("results/growth_velocity_report.xlsx")
    
    # 计算执行时间
    duration = datetime.now() - start_time
    print(f"\n模拟完成! 总耗时: {duration.total_seconds():.2f}秒")

if __name__ == "__main__":
    main()