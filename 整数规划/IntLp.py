import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize
from pulp import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import seaborn as sns

def School():
    '''
    整数规划
    '''
    # 创建一个lp问题
    problem = LpProblem("School Location Problem") #? 不用填任何就是默认求最小值
    
    # 定义变量
    # cat -> category 变量类型 => Binary Continuous Integer 三种
    x1 = LpVariable('x1', cat='Binary') # 01 变量
    x2 = LpVariable('x2', cat='Binary')
    x3 = LpVariable('x3', cat='Binary')
    x4 = LpVariable('x4', cat='Binary')
    x5 = LpVariable('x5', cat='Binary')
    x6 = LpVariable('x6', cat='Binary')
    
    #TODO 1.先定义目标函数
    problem += x1 + x2 + x3 + x4 + x5 + x6, "Total Schools" 
    
    # 添加限制条件
    # 用 problem += 的形式
    problem += x1 + x2 + x3 >= 1, "Constraint 1"  # A1覆盖
    problem += x4 + x6 >= 1, "Constraint 2"       # A5/A8覆盖
    problem += x3 + x5 >= 1, "Constraint 3"       # A1/A3覆盖
    problem += x2 + x4 >= 1, "Constraint 4"       # A2覆盖
    problem += x5 + x6 >= 1, "Constraint 5"       # A6覆盖
    problem += x1 >= 1, "Constraint 6"            # A1必须覆盖
    problem += x2 + x4 + x6 >= 1, "Constraint 7"  # A8覆盖
    
    problem.solve()
    
    # 输出结果
    print("求解状态:", LpStatus[problem.status])
    print("最优解:")
    print(f"x1 = {x1.value()}")
    print(f"x2 = {x2.value()}")
    print(f"x3 = {x3.value()}")
    print(f"x4 = {x4.value()}")
    print(f"x5 = {x5.value()}")
    print(f"x6 = {x6.value()}")
    print("最少需要校址数:", problem.objective.value())
    
    
    
    return

def allocation():
    
    problem = LpProblem("location problem", LpMaximize)
    
    # 定义变量
    # 这里因为是二维的，为了方便表示，利用变量字典
    
    #TODO 1. 定义参数
    num_equipment = 6 # 6台设备
    num_companies = 4 
    
    #TODO 2. 创建变量字典
    #? (设备i分配给公司j)
    x = LpVariable.dicts("x",
                        [(i, j) for i in range(1, num_equipment + 1)
                                for j in range(1, num_companies + 1)],
                                cat = 'Binary'
                        ) #? 这边只是复杂存下标
    # 也可以用 LpVariable.matrix()
    
    #TODO 3. 利润矩阵
    c = {
        (1,1): 4, (1,2): 2, (1,3): 3, (1,4): 4,
        (2,1): 6, (2,2): 4, (2,3): 5, (2,4): 5,
        (3,1): 7, (3,2): 6, (3,3): 7, (3,4): 6,
        (4,1): 7,  (4,2): 8, (4,3): 8, (4,4): 6,
        (5,1): 7, (5,2): 9, (5,3): 8, (5,4): 6,
        (6,1): 7, (6,2): 10, (6,3): 8, (6,4): 6,
    } # 直接用字典映射，方便
    
    #TODO 4. 目标函数
    #? lpSum 是计算一个表达式的累计结果
    problem += lpSum(c[i, j] * x[(i, j)] 
                     for i in range(1, num_equipment + 1) 
                     for j in range(1, num_companies + 1)), "Total_Profit"
    
    #TODO 5. 约束条件
    for j in range(1, num_companies + 1):
        # 每一个企业至少有一个
        problem += lpSum(x[(i, j)] for i in range(1, num_equipment + 1)) >= 1, f"Min_Equip_Company_{j}"
    
    for i in range(1, num_equipment + 1):
        problem += lpSum(x[(i, j)] for i in range(1, num_companies + 1)) == 1, f"Assign_Equip_{i}"
        #? 每一台设备只能被分配到一个公司
    
    problem.solve()
    
    # 输出结果
    print("求解状态:", LpStatus[problem.status])
    print("最大总利润:", value(problem.objective))
    print("\n分配方案:")
    for i in range(1, num_equipment+1):
        for j in range(1, num_companies+1):
            if x[(i,j)].value() == 1:
                print(f"设备{i} → 企业{j}")
    
    



if __name__ == '__main__':
    # School()
    allocation()