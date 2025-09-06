import numpy as np
import math

# ? 定义一些全局的参数


def jisuan(P_list, 波数list, 入射角=15.0):
    """
    传入极值级数列表和对应波数的列表
    1. 先把波数转换成波长
    2. 利用公式去计算出各个极值下的厚度
    3. 取平均 / 全部返回
    """
    n1 = 2.55 # 外研层的折射率
    
    theta_ru = math.radians(入射角) # 转换为弧度
    sin_theta_ru = math.sin(theta_ru) 
    
    分母 = math.sqrt(n1**2 - sin_theta_ru**2)
    
    # 波数列表转换为波长列表
    wavelens = [1/i for i in 波数list]
    print(wavelens)
    
    res = [] # 存结果
    for P, wave in zip(P_list, wavelens):
        T_i = (P - 0.5) * (0.001 * wave) / 分母
        res.append(T_i)
    
    return T_i
    

波数 = [821.72, 636.27, 517.74]
P_list = [1, 2, 3]
a = jisuan(P_list, 波数)
print(a)


