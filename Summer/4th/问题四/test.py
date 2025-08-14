import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time

def preprocess_supplier_data() -> pd.DataFrame:
    """预处理供应商数据，计算有效供应量"""
    data = pd.read_excel('附件1近5年402家供应商的相关数据.xlsx', sheet_name='评价矩阵')
    
    # 计算有效供应量
    data['eff_supply'] = (
        data['平均供货量'] 
        * (1 + data['超额供货率'])
        * (1 - data['短缺供货率'])
    )
    
    return data[['供应商ID', '材料分类', 'eff_supply']]

def preprocess_carrier_data() -> pd.DataFrame:
    """预处理转运商数据，计算平均损耗率"""
    carrier_data = pd.read_excel('附件2近5年8家转运商的相关数据.xlsx')
    
    # 计算平均损耗率（忽略0值）
    carrier_columns = carrier_data.columns[1:]  # 所有周数据列
    carrier_data['avg_loss_rate'] = carrier_data[carrier_columns].apply(
        lambda row: row[row != 0].mean() if (row != 0).any() else 0, axis=1
    )
    
    return carrier_data[['转运商ID', 'avg_loss_rate']]

def solve_max_production(
    suppliers_df: pd.DataFrame,
    carriers_df: pd.DataFrame,
    week: int
) -> Tuple[Dict[int, float], Dict[int, str], float]:
    """
    求解单周的最大产能问题
    返回: (订购方案, 转运方案, 最大产能)
    """
    n_suppliers = len(suppliers_df)
    n_carriers = len(carriers_df)
    
    # 创建问题实例
    prob = pulp.LpProblem(f"Maximize_Production_Week{week}", pulp.LpMaximize)
    
    # 决策变量
    x = pulp.LpVariable.dicts("select", suppliers_df.index, cat='Binary')  # 供应商选择
    y = pulp.LpVariable.dicts("assign", 
                             [(i, j) for i in suppliers_df.index for j in carriers_df.index], 
                             cat='Binary')  # 供应商-转运商分配
    
    # 目标函数：最大化产能
    objective = 0
    for i in suppliers_df.index:
        material_type = suppliers_df.loc[i, '材料分类']
        eff_supply = suppliers_df.loc[i, 'eff_supply']
        
        # 根据材料类型确定转换系数
        if material_type == 'A':
            conversion_rate = 0.6
        elif material_type == 'B':
            conversion_rate = 0.66
        else:  # 'C'
            conversion_rate = 0.72
            
        for j in carriers_df.index:
            loss_rate = carriers_df.loc[j, 'avg_loss_rate'] / 100  # 转换为小数
            objective += (eff_supply * (1 - loss_rate) / conversion_rate) * y[(i, j)]
    
    prob += objective
    
    # 约束条件
    # 1. 每个供应商最多由一个转运商服务
    for i in suppliers_df.index:
        prob += pulp.lpSum(y[(i, j)] for j in carriers_df.index) == x[i]
    
    # 2. 转运商运输能力限制
    for j in carriers_df.index:
        prob += pulp.lpSum(
            suppliers_df.loc[i, 'eff_supply'] * y[(i, j)] 
            for i in suppliers_df.index
        ) <= 6000
    
    # 3. 供应商选择约束（可选，但模型已隐含）
    
    # 求解
    solver = pulp.PULP_CBC_CMD(timeLimit=10, msg=False)
    prob.solve(solver)
    
    # 检查求解状态
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise RuntimeError(f"求解失败，状态: {pulp.LpStatus[prob.status]}")
    
    # 提取结果
    order_plan = {}
    carrier_plan = {}
    
    for i in suppliers_df.index:
        supplier_id = suppliers_df.loc[i, '供应商ID']
        if x[i].value() == 1:
            order_plan[supplier_id] = suppliers_df.loc[i, 'eff_supply']
            
            # 查找分配的转运商
            for j in carriers_df.index:
                if y[(i, j)].value() == 1:
                    carrier_plan[supplier_id] = carriers_df.loc[j, '转运商ID']
                    break
        else:
            order_plan[supplier_id] = 0
            carrier_plan[supplier_id] = "0"  # 0表示未被选择
    
    max_production = pulp.value(prob.objective)
    
    return order_plan, carrier_plan, max_production

def generate_24_week_plan():
    """生成24周的订购和转运方案"""
    start_time = time.time()
    
    # 预处理数据
    suppliers_df = preprocess_supplier_data()
    carriers_df = preprocess_carrier_data()
    
    # 初始化结果存储
    all_orders = {supplier_id: [0]*24 for supplier_id in suppliers_df['供应商ID']}
    all_carriers = {supplier_id: ["0"]*24 for supplier_id in suppliers_df['供应商ID']}
    weekly_production = []
    
    # 每周求解
    for week in range(24):
        print(f"求解第{week+1}周...")
        try:
            order_plan, carrier_plan, max_prod = solve_max_production(
                suppliers_df, carriers_df, week+1
            )
            
            # 存储结果
            for supplier_id, amount in order_plan.items():
                all_orders[supplier_id][week] = amount
                
            for supplier_id, carrier in carrier_plan.items():
                all_carriers[supplier_id][week] = carrier
                
            weekly_production.append(max_prod)
            
            print(f"第{week+1}周最大产能: {max_prod:.2f}立方米")
        except Exception as e:
            print(f"第{week+1}周求解失败: {str(e)}")
            weekly_production.append(0)
    
    # 创建结果DataFrame
    order_df = pd.DataFrame({
        '供应商ID': suppliers_df['供应商ID'],
        '材料分类': suppliers_df['材料分类']
    })
    
    for week in range(24):
        week_data = [all_orders[sid][week] for sid in order_df['供应商ID']]
        order_df[f'第{week+1}周'] = week_data
    
    carrier_df = pd.DataFrame({
        '供应商ID': suppliers_df['供应商ID']
    })
    
    for week in range(24):
        week_data = [all_carriers[sid][week] for sid in carrier_df['供应商ID']]
        carrier_df[f'第{week+1}周'] = week_data
    
    # 保存结果
    order_df.to_csv('附件A.csv', index=False)
    carrier_df.to_csv('附件B.csv', index=False)
    
    # 输出产能提升分析
    base_capacity = 28200
    avg_increase = np.mean(weekly_production) - base_capacity
    min_week = np.argmin(weekly_production) + 1
    max_week = np.argmax(weekly_production) + 1
    
    print("\n" + "="*50)
    print("产能提升分析")
    print("="*50)
    print(f"原始周产能: {base_capacity}立方米")
    print(f"最大周产能: {max(weekly_production):.2f}立方米 (第{max_week}周)")
    print(f"最小周产能: {min(weekly_production):.2f}立方米 (第{min_week}周)")
    print(f"平均周产能: {np.mean(weekly_production):.2f}立方米")
    print(f"平均产能提升: {avg_increase:.2f}立方米 ({avg_increase/base_capacity*100:.2f}%)")
    print(f"总耗时: {time.time()-start_time:.2f}秒")
    
    return order_df, carrier_df, weekly_production

if __name__ == "__main__":
    order_plan, carrier_plan, weekly_production = generate_24_week_plan()