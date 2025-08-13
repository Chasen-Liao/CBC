import numpy as np
from scipy.optimize import differential_evolution

# 定义目标函数（求最大值）
def objective_function(x):
    """
    C4烯烃收率最大化目标函数
    x: 变量数组 [x1, x2, x3, x4, T]
    """
    x1, x2, x3, x4, T = x
    
    # 分解目标函数项
    constant = 11787
    linear_terms = -27.7*x1 -77.5*T
    quadratic_terms = -172.8*(x3**2) + 0.1*(T**2)
    interaction_terms = 3.7*x2*x4 + 0.1*x1*T -3.1*x4*T + 482.4*x3*x4
    
    # 组合所有项
    y = constant + linear_terms + quadratic_terms + interaction_terms
    return -y  # 返回负值用于最小化问题

# 定义变量边界（顺序：x1, x2, x3, x4, T）
bounds = [
    (33, 200),   # x1范围
    (33, 200),   # x2范围
    (0.5, 5),    # x3范围
    (0.3, 2.1),  # x4范围
    (250, 400)   # T范围
]

# 使用差分进化算法求解全局最优解
result = differential_evolution(
    func=objective_function,
    bounds=bounds,
    strategy='best1bin',
    popsize=20,           # 较大的种群规模提高全局搜索能力
    mutation=(0.5, 1.9),  # 自适应变异参数
    recombination=0.7,    # 重组概率
    tol=1e-7,             # 收敛公差
    maxiter=2000,         # 最大迭代次数
    polish=True,          # 使用局部优化进行结果精细化
    seed=42               # 固定随机种子保证可重复性
)

# 提取最优解和最优值
optimal_variables = result.x
max_yield = -result.fun  # 转换为原始目标函数值

# 输出结果
print("=== C4烯烃收率优化结果 ===")
print(f"x1 = {optimal_variables[0]:.4f} (范围: 33-200)")
print(f"x2 = {optimal_variables[1]:.4f} (范围: 33-200)")
print(f"x3 = {optimal_variables[2]:.4f} (范围: 0.5-5)")
print(f"x4 = {optimal_variables[3]:.4f} (范围: 0.3-2.1)")
print(f"T = {optimal_variables[4]:.4f}°C (范围: 250-400)")
print(f"\n最大C4烯烃收率 y = {max_yield:.4f}")