import numpy as np
import pandas as pd

# ========== 参数 ==========
n = 224             # 身体节数
b = 0.55
pi = np.pi
t_max = 300
v_h = 1             # 龙头速度
d_h = 2.86          # 龙头到第一节身体的距离
d_b = 1.65          # 身体之间的距离
l_h = 3.41          # 龙头长度
l_b = 2.2           # 身体长度
width = 0.3         # 每个矩形身体的宽度（厚度）

# ========== 基于第一问 ==========
def theta(t):
    """计算龙头角度"""
    return np.sqrt(-4*pi*v_h*t/b + 1024*pi*pi)

# 牛顿法相关函数
def f(theta_now, theta_next, d):
    return (b*b/4/pi/pi)*(theta_now**2 + theta_next**2 - 2*theta_now*theta_next*np.cos(theta_now-theta_next)) - d**2

def f_prime(theta_now, theta_next, d):
    return (b*b/4/pi/pi)*(2*theta_next + 2*theta_now*(np.cos(theta_now-theta_next) - theta_next*np.sin(theta_now-theta_next)))

def newton_method(f, f_prime, d, theta_now, x0, tol=1e-6, max_iter=100):
    """牛顿法迭代求解"""
    for i in range(max_iter):
        x = x0 - f(theta_now, x0, d) / f_prime(theta_now, x0, d)
        if abs(x - x0) < tol:
            break
        x0 = x
    return x

# ========== OBB 碰撞检测 ==========
def get_obb_vertices(p1, p2, length, width):
    """
    根据两个点 p1=(x1,y1), p2=(x2,y2)，生成矩形 OBB 的四个顶点
    length: 长度（沿p1->p2方向）
    width: 宽度（垂直方向）
    """
    p1, p2 = np.array(p1), np.array(p2)
    dir_vec = p2 - p1
    if np.linalg.norm(dir_vec) == 0:   # 防止零向量
        dir_vec = np.array([1.0, 0.0])
    else:
        dir_vec = dir_vec / np.linalg.norm(dir_vec)  # 单位方向向量
    ortho = np.array([-dir_vec[1], dir_vec[0]])      # 垂直向量

    # 矩形中心
    center = (p1 + p2) / 2

    # 半长和半宽
    hl = length / 2
    hw = width / 2

    # 四个顶点
    verts = [
        center + dir_vec * hl + ortho * hw,
        center + dir_vec * hl - ortho * hw,
        center - dir_vec * hl - ortho * hw,
        center - dir_vec * hl + ortho * hw
    ]
    return np.array(verts)

def project_polygon(axis, verts):
    """将多边形顶点投影到某一轴上"""
    axis = axis / np.linalg.norm(axis)
    projections = np.dot(verts, axis)
    return np.min(projections), np.max(projections)

def overlap(interval1, interval2):
    """判断两个区间是否重叠"""
    return not (interval1[1] < interval2[0] or interval2[1] < interval1[0])

def obb_intersect(verts1, verts2):
    """
    2D OBB 碰撞检测（分离轴定理 SAT）
    verts1, verts2: 两个矩形的四个顶点
    """
    # 候选分离轴 = 两个矩形的边方向
    axes = []
    for verts in [verts1, verts2]:
        for i in range(4):
            p1, p2 = verts[i], verts[(i+1)%4]
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])  # 垂直方向
            axes.append(axis)

    # 检查每个轴上的投影是否重叠
    for axis in axes:
        min1, max1 = project_polygon(axis, verts1)
        min2, max2 = project_polygon(axis, verts2)
        if not overlap((min1, max1), (min2, max2)):
            return False
    return True

# ========== 主计算 ==========
theta_head=[]
r_head=[]
head_x=[]
head_y=[]
theta_body_i=np.empty((n,2*t_max))
r_body_i=np.empty((n,2*t_max))
body_i_x=np.empty((n,2*t_max))
body_i_y=np.empty((n,2*t_max))

t=0
flag=False
while flag==False:
    # 龙头位置
    theta_now = theta(t)
    theta_head.append(theta_now)
    r_head.append(b*theta_now/(2*pi))
    x = r_head[t]*np.cos(theta_now)
    y = r_head[t]*np.sin(theta_now)
    head_x.append(round(x,6))
    head_y.append(round(y,6))

    # 第一节身体
    theta_next = newton_method(f,f_prime,d_h,theta_head[t],theta_head[t]+pi/2)
    theta_body_i[0][t] = theta_next
    r_body_i[0][t] = theta_next*b/(2*pi)
    body_i_x[0][t] = round(r_body_i[0][t]*np.cos(theta_next),6)
    body_i_y[0][t] = round(r_body_i[0][t]*np.sin(theta_next),6)

    # 其余身体
    for i in range(1,n):
        theta_next = newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2)
        theta_body_i[i][t] = theta_next
        r_body_i[i][t] = theta_next*b/(2*pi)
        body_i_x[i][t] = round(r_body_i[i][t]*np.cos(theta_next),6)
        body_i_y[i][t] = round(r_body_i[i][t]*np.sin(theta_next),6)

    # ==== 碰撞检测（OBB）====
    head_obb = get_obb_vertices((head_x[t],head_y[t]), (body_i_x[0][t],body_i_y[0][t]), l_h, width)

    for i in range(1,n-1):
        body_obb = get_obb_vertices((body_i_x[i][t],body_i_y[i][t]), (body_i_x[i+1][t],body_i_y[i+1][t]), l_b, width)
        if obb_intersect(head_obb, body_obb):
            print(f"在t={t}时刻，龙头与第{i+1}节身体碰撞(粗检测)")
            flag = True
            break
    if flag: break

    t += 1

# ========== 碰撞时刻精细搜索 ==========
dt = 0.01
t_start = max(0, t-2)
t_end = t
flag_refined = False
t_refined = None
collision_index = None

tt = t_start
while tt <= t_end and not flag_refined:
    theta_now = theta(tt)
    r_hh = b*theta_now/(2*pi)
    x_h = r_hh*np.cos(theta_now)
    y_h = r_hh*np.sin(theta_now)

    # 第一节身体
    theta_next = newton_method(f,f_prime,d_h,theta_now,theta_now+pi/2)
    r1 = theta_next*b/(2*pi)
    x1 = r1*np.cos(theta_next)
    y1 = r1*np.sin(theta_next)

    body_pos = [(x1,y1)]
    theta_prev = theta_next
    for i in range(1,n):
        theta_next = newton_method(f,f_prime,d_b,theta_prev,theta_prev+pi/2)
        r_i = theta_next*b/(2*pi)
        x_i = r_i*np.cos(theta_next)
        y_i = r_i*np.sin(theta_next)
        body_pos.append((x_i,y_i))
        theta_prev = theta_next

    head_obb = get_obb_vertices((x_h,y_h), body_pos[0], l_h, width)

    for i in range(1,n-1):
        body_obb = get_obb_vertices(body_pos[i], body_pos[i+1], l_b, width)
        if obb_intersect(head_obb, body_obb):
            print(f"更精确: 在t={round(tt,1)}s时，龙头与第{i+1}节身体碰撞")
            t_refined = round(tt,1)
            collision_index = i+1
            flag_refined = True
            break
    tt += dt

# ========== 碰撞时刻速度计算 ==========
v_body_i=[]
t = t_refined  # 使用精细碰撞时刻
theta_now = theta(t)
r_hh = b*theta_now/(2*pi)
x_h = r_hh*np.cos(theta_now)
y_h = r_hh*np.sin(theta_now)

theta_next = newton_method(f,f_prime,d_h,theta_now,theta_now+pi/2)
r1 = theta_next*b/(2*pi)
x1 = r1*np.cos(theta_next)
y1 = r1*np.sin(theta_next)

theta_list = [theta_now, theta_next]
x_list = [x_h, x1]
y_list = [y_h, y1]

for i in range(1,n):
    theta_next = newton_method(f,f_prime,d_b,theta_list[-1],theta_list[-1]+pi/2)
    r_i = theta_next*b/(2*pi)
    x_i = r_i*np.cos(theta_next)
    y_i = r_i*np.sin(theta_next)
    theta_list.append(theta_next)
    x_list.append(x_i)
    y_list.append(y_i)

# 速度计算
m_0=(y_list[0]-y_list[1])/(x_list[0]-x_list[1])
m_now=(theta_list[0]*np.cos(theta_list[0])+np.sin(theta_list[0]))/(-theta_list[0]*np.sin(theta_list[0])+np.cos(theta_list[0]))
m_next=(theta_list[1]*np.cos(theta_list[1])+np.sin(theta_list[1]))/(-theta_list[1]*np.sin(theta_list[1])+np.cos(theta_list[1]))
alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
v_next=v_h*np.cos(alpha_1)/np.cos(alpha_2)
v_body_i.append(round(v_next,6))

for i in range(2,n+1):
    m_0=(y_list[i]-y_list[i-1])/(x_list[i]-x_list[i-1])
    m_now=(theta_list[i-1]*np.cos(theta_list[i-1])+np.sin(theta_list[i-1]))/(-theta_list[i-1]*np.sin(theta_list[i-1])+np.cos(theta_list[i-1]))
    m_next=(theta_list[i]*np.cos(theta_list[i])+np.sin(theta_list[i]))/(-theta_list[i]*np.sin(theta_list[i])+np.cos(theta_list[i]))
    alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
    alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
    v_next=v_body_i[i-2]*np.cos(alpha_1)/np.cos(alpha_2)
    v_body_i.append(round(v_next,6))

print(f"t={t}s, 龙头位置: x={x_h}, y={y_h}, v={v_h}")

# 导出结果
data={
    'x': x_list[1:],
    "y": y_list[1:],
    "v": v_body_i
}
df=pd.DataFrame(data)
df.to_excel("solution2.xlsx")

# 打印部分节的信息
num_point=[0,50,100,150,200,222]
print(f"time:{t}s (精细检测)")
print(f"head:x:{x_h},y:{y_h},v:1")
for i in num_point:
    print(f"body{i}:x:{x_list[i+1]},y:{y_list[i+1]},v:{v_body_i[i]}")
