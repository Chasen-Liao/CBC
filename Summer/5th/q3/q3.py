import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数
n = 223
pi = np.pi
t_max = 300
v_h = 1
d_h = 2.86
d_b = 1.65
l_h = 3.41
l_b = 2.2
width = 0.3   # 身体厚度（矩形宽度）

# ========== 龙头角度 ==========
def theta(t, b):
    return np.sqrt(-4*pi*v_h*t/b + (32*pi*0.55/b)**2)

# ========== 牛顿法求解位置 ==========
def f(theta_now, theta_next, d, b):
    return (b*b/4/pi/pi)*(theta_now**2+theta_next**2-2*theta_now*theta_next*np.cos(theta_now-theta_next))-d**2

def f_prime(theta_now, theta_next, d, b):
    return (b*b/4/pi/pi)*(2*theta_next+2*theta_now*(np.cos(theta_now-theta_next)-theta_next*np.sin(theta_now-theta_next)))

def newton_method(f, f_prime, d, theta_now, x0, b, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        x = x0 - f(theta_now, x0, d, b) / f_prime(theta_now, x0, d, b)
        if abs(x - x0) < tol:
            break
        x0 = x
    return x    

# ========== OBB 碰撞检测（分离轴定理 SAT）==========
def get_obb_vertices(p1, p2, length, width):
    """
    根据两个点 p1=(x1,y1), p2=(x2,y2)，生成矩形 OBB 的四个顶点
    length: 矩形沿 p1->p2 方向的长度
    width:  矩形宽度（垂直方向）
    """
    p1, p2 = np.array(p1), np.array(p2)
    dir_vec = p2 - p1
    if np.linalg.norm(dir_vec) == 0:
        dir_vec = np.array([1.0, 0.0])
    else:
        dir_vec = dir_vec / np.linalg.norm(dir_vec)  # 单位方向
    ortho = np.array([-dir_vec[1], dir_vec[0]])      # 垂直方向

    center = (p1 + p2) / 2
    hl = length / 2
    hw = width / 2

    verts = [
        center + dir_vec * hl + ortho * hw,
        center + dir_vec * hl - ortho * hw,
        center - dir_vec * hl - ortho * hw,
        center - dir_vec * hl + ortho * hw
    ]
    return np.array(verts)

def project_polygon(axis, verts):
    axis = axis / np.linalg.norm(axis)
    projections = np.dot(verts, axis)
    return np.min(projections), np.max(projections)

def overlap(interval1, interval2):
    return not (interval1[1] < interval2[0] or interval2[1] < interval1[0])

def obb_intersect(verts1, verts2):
    axes = []
    for verts in [verts1, verts2]:
        for i in range(4):
            p1, p2 = verts[i], verts[(i+1)%4]
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])  # 垂直方向
            axes.append(axis)

    for axis in axes:
        min1, max1 = project_polygon(axis, verts1)
        min2, max2 = project_polygon(axis, verts2)
        if not overlap((min1, max1), (min2, max2)):
            return False
    return True

# ========== 计算碰撞半径 ==========
def dis(b_test):
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
        # 计算龙头位置
        theta_now=theta(t,b_test)
        theta_head.append(theta_now)
        r_head.append(b_test*theta_now/(2*pi))
        x=r_head[t]*np.cos(theta_now)
        y=r_head[t]*np.sin(theta_now)
        head_x.append(x)
        head_y.append(y)

        # 第一节身体
        theta_next=newton_method(f,f_prime,d_h,theta_head[t],theta_head[t]+pi/2,b_test)
        theta_body_i[0][t]=theta_next
        r_body_i[0][t]=theta_next*b_test/(2*pi)
        body_i_x[0][t]=r_body_i[0][t]*np.cos(theta_next)
        body_i_y[0][t]=r_body_i[0][t]*np.sin(theta_next)

        # 其余身体
        for i in range(1,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2,b_test)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b_test/(2*pi)
            body_i_x[i][t]=r_body_i[i][t]*np.cos(theta_next)
            body_i_y[i][t]=r_body_i[i][t]*np.sin(theta_next)

        # ==== 碰撞检测（OBB）====
        head_obb = get_obb_vertices((head_x[t],head_y[t]), (body_i_x[0][t],body_i_y[0][t]), l_h, width)

        # 龙头和身体碰撞
        for i in range(1,n-1):
            body_obb = get_obb_vertices((body_i_x[i][t],body_i_y[i][t]),
                                        (body_i_x[i+1][t],body_i_y[i+1][t]), l_b, width)
            if obb_intersect(head_obb, body_obb):
                flag=True
                break

        t = t+1
    t=t-1
    return np.sqrt(head_x[t]**2+head_y[t]**2)

# ========== 二分求解 ==========
'''
[0.35, 0.55] step=0.1  => b = 0.5
[0.4, 0.5] step=0.01 => b = 0.46
[0.45, 0.46] step=0.001 => b = 0.450374
'''
L=0.44
R=0.46
step=0.001
while L<=R:
    mid=(L+R)/2
    if dis(mid)>4.5:
        L=mid+step
    else:
        R=mid-step
print("最优螺距b ≈", R)
