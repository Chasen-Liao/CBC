import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# 1. 数据准备函数
def prepare_data() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """读取供应商最大供货量和转运商平均损耗率"""
    # 读取附件1：计算每个供应商历史最大供货量
    supply_data = pd.read_excel('附件1近5年402家供应商的相关数据.xlsx', sheet_name='评价矩阵')
    supplier_max = supply_data[['最大供货能力']]  # 240周中的最大供货量
    supplier_info = pd.DataFrame({
        'supplier_id': supply_data.iloc[:, 0].values.flatten(),
        'material_type': supply_data.iloc[:, 1].values.flatten(),
        'max_supply': supplier_max.values.flatten()
    })
    
    # 读取附件2：计算转运商平均损耗率（忽略0值）
    trans_data = pd.read_excel('附件2近5年8家转运商的相关数据.xlsx')
    trans_loss = {}
    for i in range(len(trans_data)):
        row = trans_data.iloc[i]
        values = row.iloc[1:][row.iloc[1:] > 0]  # 筛选非零损耗率
        trans_loss[row.iloc[0]] = values.mean() / 100  # 百分比转小数
    
    return supplier_info, trans_loss

# 2. 产能最大化模型
def max_capacity_model(
    suppliers: pd.DataFrame,
    trans_loss: Dict[str, float],
    trans_capacity: int = 6000
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """求解单周最大产能及转运方案"""
    # 初始化模型
    model = pulp.LpProblem("Maximize_Production_Capacity", pulp.LpMaximize)
    
    # 决策变量：供应商i分配给转运商t的供货比例
    x = pulp.LpVariable.dicts("x", 
        [(i, t) for i in suppliers.index for t in trans_loss.keys()],
        lowBound=0, upBound=1
    )
    
    # 目标函数：最大化转换后的产能（考虑材料类型转换率）
    material_coeff = {'A': 1/0.6, 'B': 1/0.66, 'C': 1/0.72}
    model += pulp.lpSum(
        x[(i, t)] * suppliers.loc[i, 'max_supply'] * 
        (1 - trans_loss[t]) * 
        material_coeff[suppliers.loc[i, 'material_type']]
        for i in suppliers.index for t in trans_loss.keys()
    )
    
    # 约束1：供应商供货全部分配
    for i in suppliers.index:
        model += pulp.lpSum(x[(i, t)] for t in trans_loss.keys()) == 1
        
    # 约束2：转运商运力限制
    for t in trans_loss.keys():
        model += pulp.lpSum(
            x[(i, t)] * suppliers.loc[i, 'max_supply'] 
            for i in suppliers.index
        ) <= trans_capacity
        
    # 求解模型
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model.status] != 'Optimal':
        raise ValueError(f"模型求解失败，状态: {pulp.LpStatus[model.status]}")
    
    # 提取结果
    capacity = pulp.value(model.objective)
    trans_plan = {}
    for i in suppliers.index:
        supplier_id = suppliers.loc[i, 'supplier_id']
        trans_plan[supplier_id] = {
            t: x[(i, t)].value() * suppliers.loc[i, 'max_supply']
            for t in trans_loss.keys()
        }
    
    return capacity, trans_plan

# 3. 24周方案生成
def generate_24week_plan():
    """生成未来24周订购与转运方案"""
    suppliers, trans_loss = prepare_data()
    base_capacity = 28200  # 原产能
    
    # 存储结果
    order_plan = []
    trans_plan_all = {t: [] for t in trans_loss.keys()}
    capacity_improvements = []
    
    for week in range(1, 25):
        # 模拟运输损耗（正态分布）
        week_loss = {}
        for t in trans_loss.keys():
            while True:
                loss = np.random.normal(loc=trans_loss[t], scale=0.01)
                if 0 < loss < 0.15:  # 限制合理范围
                    week_loss[t] = loss
                    break
        
        # 求解最大产能
        capacity, weekly_trans = max_capacity_model(suppliers, week_loss)
        capacity_improvements.append(capacity - base_capacity)
        
        # 记录订购方案（所有供应商按最大供货量）
        for i, row in suppliers.iterrows():
            order_plan.append({
                'Week': week,
                'Supplier': row['supplier_id'],
                'Material': row['material_type'],
                'OrderQty': row['max_supply']
            })
        
        # 记录转运方案
        for t in trans_loss.keys():
            trans_plan_all[t].append({
                'Week': week,
                'TransQty': sum(weekly_trans[s][t] for s in weekly_trans.keys())
            })
    
    # 转换为DataFrame
    order_df = pd.DataFrame(order_plan)
    trans_df = pd.concat(
        [pd.DataFrame(trans_plan_all[t]).assign(Transporter=t) 
         for t in trans_loss.keys()]
    )
    
    # 输出产能提升分析
    print(f"平均产能提升: {np.mean(capacity_improvements):.2f} m³")
    print(f"最大单周提升: {max(capacity_improvements):.2f} m³")
    
    return order_df, trans_df

# 执行主程序
if __name__ == "__main__":
    order_scheme, trans_scheme = generate_24week_plan()
    
    # 保存结果（符合题目要求的附件格式）
    # order_scheme.to_excel("附件A_订购方案.xlsx", index=False)
    # trans_scheme.to_excel("附件B_转运方案.xlsx", index=False)
    
    # # 输出关键指标
    # print("订购方案和转运方案已保存至附件A/B")