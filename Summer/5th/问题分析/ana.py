import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# 设置阿基米德螺旋线参数
a = 0.1 # 起始半径
b = 0.0875    # 螺距系数（每弧度半径增加量）
theta_max = 32 * np.pi  # 最大角度
num_points = 10000      # 采样点数量

# 生成角度数组
theta = np.linspace(0, theta_max, num_points)

# 计算极径（阿基米德螺旋线公式：r = a + b*θ）
r = a + b * theta

# 将极坐标转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

# 创建画布和子图
plt.figure(figsize=(10, 8))

# 在直角坐标系中绘制螺旋线
plt.subplot(2, 1, 1)
plt.plot(x, y, 'b-', linewidth=2)
plt.title(f'阿基米德螺旋线 (直角坐标系)')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.grid(True)
plt.axis('equal')  # 保持坐标轴比例一致

# 在极坐标系中绘制螺旋线
plt.subplot(2, 1, 2, projection='polar')
plt.plot(theta, r, 'r-', linewidth=2)
plt.title(f'阿基米德螺旋线 (极坐标系)', pad=20)
plt.grid(True)

# 调整布局并显示图形
plt.tight_layout()
plt.show()