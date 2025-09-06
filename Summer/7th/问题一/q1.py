import pulp
import numpy as np

# ========= 示例数据 =========
# 集合索引
plots = ["P1", "P2"]           # 地块
crops = ["Wheat", "Bean"]      # 作物
years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]       # 年份
seasons = [1]                  # 每年只有 1 季（示例）

# 地块面积
A = {"P1": 10, "P2": 8}  # 亩

# 亩产 (kg/亩)
yield_per_mu = {"Wheat": 400, "Bean": 200}

# 每亩成本 (元/亩)
cost = {"Wheat": 300, "Bean": 250}

# 价格 (元/kg)
price = {"Wheat": 2.0, "Bean": 3.0}

# 预期销售量 = 需求上限 (销售量，kg)
# 假设每一年都相同
D = {
    (2024, 1, "Wheat"): 6000,
    (2024, 1, "Bean"): 3000,
    (2025, 1, "Wheat"): 6000,
    (2025, 1, "Bean"): 3000,
}

# ========= 决策变量 =========
# x[p,t,s,c] = 面积 (亩)
x = pulp.LpVariable.dicts("x", (plots, years, seasons, crops), lowBound=0)

# u[t,s,c] = 正常价售出的数量 (kg)
u = pulp.LpVariable.dicts("u", (years, seasons, crops), lowBound=0)

# ========= 模型 =========
model = pulp.LpProblem("CropPlanning", pulp.LpMaximize)

# 约束 1: 每地块每年每季总面积 ≤ 地块面积
for p in plots:
    for t in years:
        for s in seasons:
            model += pulp.lpSum([x[p][t][s][c] for c in crops]) <= A[p], f"Area_{p}_{t}_{s}"

# 产量 Q[t,s,c]
Q = {}
for t in years:
    for s in seasons:
        for c in crops:
            Q[(t, s, c)] = pulp.lpSum([yield_per_mu[c] * x[p][t][s][c] for p in plots])

# 约束 2: 售出量 ≤ 需求上限
for t in years:
    for s in seasons:
        for c in crops:
            model += u[t][s][c] <= D[(t, s, c)], f"Demand_{t}_{s}_{c}"
            model += u[t][s][c] <= Q[(t, s, c)], f"SupplyLimit_{t}_{s}_{c}"

# ========= 收益定义 =========
# 情形 A：超额滞销（只卖 u 部分）
Revenue_A = pulp.lpSum([price[c] * u[t][s][c] for t in years for s in seasons for c in crops])

# 情形 B：超额按 50% 售出
Revenue_B = pulp.lpSum([0.5 * price[c] * Q[(t, s, c)] + 0.5 * price[c] * u[t][s][c]
                        for t in years for s in seasons for c in crops])

# 成本
Cost = pulp.lpSum([cost[c] * x[p][t][s][c]
                   for p in plots for t in years for s in seasons for c in crops])

# ========= 切换目标 =========
# 这里先用情形 B，你可以切换成 Revenue_A
model += Revenue_B - Cost

# ========= 求解 =========
model.solve(pulp.PULP_CBC_CMD(msg=False))

# ========= 输出 =========
print("最优目标值 (利润):", pulp.value(model.objective))
for p in plots:
    for t in years:
        for s in seasons:
            for c in crops:
                val = x[p][t][s][c].value()
                if val > 1e-6:
                    print(f"{p}, {t}, 季{s}, 作物{c}, 面积={val:.2f}")
