import pulp
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def solve_supplier_selection(
    n_suppliers: int = 50, 
    capacity_requirement: float = 28200,
    p_C: float = 1.0
) -> Tuple[Dict[int, int], int, float]:
    """
    解决供应商选择问题的0-1整数线性规划模型
    
    参数:
    n_suppliers: 考虑的供应商数量（前N个重要供应商）
    capacity_requirement: 每周产能需求（单位：立方米产品）
    p_C: C类原材料基准单价
    
    返回:
    selected_suppliers: 选择的供应商字典 {供应商ID: 选择状态(0/1)}
    min_count: 最小供应商数量
    min_cost: 最小总成本
    """
    # 读取Excel数据（仅前50家供应商）
    data = pd.read_excel('附件1近5年402家供应商的相关数据.xlsx', sheet_name='评价矩阵')
    data = data.head(n_suppliers)  # 只取前N家供应商
    
    # 供应商ID
    supplier_ids = data['供应商ID'].tolist()
    
    # 原材料类型 (A, B, C)
    material_types = data['材料分类'].tolist()
    
    # 实际数据
    df_suppliers = pd.DataFrame({
        'supplier_id': supplier_ids,
        'material_type': material_types,
        'avg_supply': data['平均供货量'],
        'avg_deviation': data['平均偏差值'],
        'fulfillment': data['履约率'],
        'stability': data['稳定性系数']
    })
    
    # 计算有效供应量 eff_supply_i = 平均供货量 × (1 - 平均偏差率) × 履约率 × 稳定性系数
    df_suppliers['eff_supply'] = ( 
        df_suppliers['avg_supply'] 
        * (1 - df_suppliers['avg_deviation']) 
        * df_suppliers['fulfillment'] 
        * df_suppliers['stability']
    )
    
    # 设置原材料单价
    price_map = {'A': 1.2 * p_C, 'B': 1.1 * p_C, 'C': p_C}
    df_suppliers['unit_price'] = df_suppliers['material_type'].map(price_map)
    
    # 计算成本系数 c_i = 单价 × 有效供应量
    df_suppliers['cost_coeff'] = df_suppliers['unit_price'] * df_suppliers['eff_supply']
    
    # 按材料类型分组
    group_A = df_suppliers[df_suppliers['material_type'] == 'A'].index.tolist()
    group_B = df_suppliers[df_suppliers['material_type'] == 'B'].index.tolist()
    group_C = df_suppliers[df_suppliers['material_type'] == 'C'].index.tolist()
    
    # ====================================================
    # 步骤2: 创建0-1整数线性规划模型
    # ====================================================
    model = pulp.LpProblem("Supplier_Selection_Min_Cost", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", df_suppliers.index, cat='Binary')
    
    # ====================================================
    # 步骤3: 设置目标函数 - 最小化总成本
    # ====================================================
    model += pulp.lpSum(df_suppliers.loc[i, 'cost_coeff'] * x[i] for i in df_suppliers.index)
    
    # ====================================================
    # 步骤4: 添加约束条件
    # ====================================================
    # 产能约束
    model += (
        pulp.lpSum(df_suppliers.loc[i, 'eff_supply'] * x[i] / 0.6 for i in group_A) +
        pulp.lpSum(df_suppliers.loc[i, 'eff_supply'] * x[i] / 0.66 for i in group_B) +
        pulp.lpSum(df_suppliers.loc[i, 'eff_supply'] * x[i] / 0.72 for i in group_C)
    ) >= capacity_requirement
    
    # ====================================================
    # 步骤5: 求解模型
    # ====================================================
    solver = pulp.PULP_CBC_CMD(timeLimit=10, msg=True)
    model.solve(solver)
    
   # ====================================================
    # 步骤6: 提取和分析结果
    # ====================================================
    if pulp.LpStatus[model.status] != 'Optimal':
        raise RuntimeError("求解失败，未能找到最优解")
    
    selected_suppliers = {}
    supply_amounts = {}  # 新增：存储供应商供应量
    
    for i in df_suppliers.index:
        supplier_id = df_suppliers.loc[i, 'supplier_id']
        selected = int(x[i].value())
        selected_suppliers[supplier_id] = selected
        
        # 计算供应量：如果被选择则使用有效供应量，否则为0
        supply_amount = df_suppliers.loc[i, 'eff_supply'] if selected == 1 else 0.0
        supply_amounts[supplier_id] = supply_amount
    
    min_count = sum(selected_suppliers.values())
    min_cost = pulp.value(model.objective)
    
    # ====================================================
    # 步骤7: 结果输出（增加供应量显示）
    # ====================================================
    print("\n" + "="*50)
    print(f"前{n_suppliers}家供应商优化结果摘要")
    print("="*50)
    print(f"最小供应商数量: {min_count}")
    print(f"最小总成本: {min_cost:.2f} 元")
    
    # 输出前50个供应商的选择结果和供应量
    print(f"\n前{n_suppliers}个供应商选择状态及供应量:")
    for supplier_id in list(supply_amounts.keys())[:n_suppliers]:
        status = "选择" if selected_suppliers[supplier_id] == 1 else "未选"
        supply = supply_amounts[supplier_id]
        print(f"供应商 {supplier_id}: {status}, 供应量 = {supply:.2f} 立方米")
    
    return selected_suppliers, supply_amounts, min_count, min_cost  # 增加返回供应量字典

if __name__ == "__main__":
    selected, supply_amounts, min_count, min_cost = solve_supplier_selection(n_suppliers=50)