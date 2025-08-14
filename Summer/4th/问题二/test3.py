import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ====================================================
# 核心优化函数 - 未来24周订购与转运方案
# ====================================================
def solve_24week_plan(
    selected_suppliers: List[int], 
    capacity_requirement: float = 28200,
    p_C: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    制定未来24周的最经济订购方案和最少损耗转运方案
    
    参数:
    selected_suppliers: 选定的供应商ID列表
    capacity_requirement: 每周产能需求（立方米产品）
    p_C: C类原材料基准单价
    
    返回:
    order_plan: 24周订购方案DataFrame
    transport_plan: 24周转运方案DataFrame
    analysis_results: 方案实施效果分析
    """
    # 1. 数据准备
    suppliers_data = pd.read_excel('附件1近5年402家供应商的相关数据.xlsx', sheet_name='评价矩阵')
    trans_data = pd.read_excel('附件2近5年8家转运商的相关数据.xlsx')
    
    # 筛选选定的供应商
    suppliers_data = suppliers_data[suppliers_data['供应商ID'].isin(selected_suppliers)]
    supplier_ids = suppliers_data['供应商ID'].tolist()
    material_types = suppliers_data['材料分类'].tolist()
    
    # 计算供应商平均供货能力
    supplier_capacity = {}
    for idx, row in suppliers_data.iterrows():
        supplier_id = row['供应商ID']
        # 考虑历史波动性（80%置信度）
        supplier_capacity[supplier_id] = row['平均供货量'] * 0.8
        
    # 转运商平均损耗率计算
    trans_names = trans_data['转运商ID'].tolist()
    trans_loss_rates = trans_data.mean(axis=1).tolist()  # 各转运商历史平均损耗率
    
    # 2. 建立24周订购优化模型
    order_model = pulp.LpProblem("24Week_Order_Plan", pulp.LpMinimize)
    
    # 决策变量: 每周对每个供应商的订购量
    order_vars = pulp.LpVariable.dicts("Order",
                                      ((w, s) for w in range(24) for s in supplier_ids),
                                      lowBound=0,
                                      cat='Continuous')
    
    # 目标函数: 最小化总采购成本
    price_map = {'A': 1.2 * p_C, 'B': 1.1 * p_C, 'C': p_C}
    order_model += pulp.lpSum(
        order_vars[w, s] * price_map[suppliers_data[suppliers_data['供应商ID'] == s]['材料分类'].values[0]]
        for w in range(24) for s in supplier_ids
    )
    
    # 3. 约束条件
    # 产能约束 (考虑原材料转换率)
    for w in range(24):
        order_model += pulp.lpSum(
            order_vars[w, s] / {'A': 0.6, 'B': 0.66, 'C': 0.72}[suppliers_data[suppliers_data['供应商ID'] == s]['材料分类'].values[0]]
            for s in supplier_ids
        ) >= capacity_requirement
    
    # 供应商能力约束
    for s in supplier_ids:
        for w in range(24):
            order_model += order_vars[w, s] <= supplier_capacity[s]
    
    # 4. 求解订购模型
    order_model.solve()
    
    # 5. 提取订购方案
    order_results = []
    for w in range(24):
        for s in supplier_ids:
            if pulp.value(order_vars[w, s]) > 0:
                order_results.append({
                    '周次': w+1,
                    '供应商ID': s,
                    '材料类型': suppliers_data[suppliers_data['供应商ID'] == s]['材料分类'].values[0],
                    '订购量': pulp.value(order_vars[w, s])
                })
    order_plan = pd.DataFrame(order_results)
    
    # 6. 建立转运优化模型
    transport_plan = []
    for w in range(24):
        trans_model = pulp.LpProblem(f"Week_{w+1}_Transport", pulp.LpMinimize)
        
        # 决策变量: 供应商-转运商分配
        assign_vars = pulp.LpVariable.dicts("Assign",
                                          ((s, t) for s in supplier_ids for t in trans_names),
                                          cat='Binary')
        
        # 目标函数: 最小化总损耗
        trans_model += pulp.lpSum(
            assign_vars[s, t] * order_plan[(order_plan['周次']==w+1) & 
                                         (order_plan['供应商ID']==s)]['订购量'].values[0] * 
            trans_loss_rates[trans_names.index(t)]
            for s in supplier_ids for t in trans_names
        )
        
        # 约束条件
        # 每个供应商只分配一个转运商
        for s in supplier_ids:
            trans_model += pulp.lpSum(assign_vars[s, t] for t in trans_names) == 1
        
        # 转运商能力约束
        for t in trans_names:
            trans_model += pulp.lpSum(
                assign_vars[s, t] * order_plan[(order_plan['周次']==w+1) & 
                                             (order_plan['供应商ID']==s)]['订购量'].values[0]
                for s in supplier_ids
            ) <= 6000  # 转运商运输能力
        
        # 求解
        trans_model.solve()
        
        # 提取转运方案
        for s in supplier_ids:
            for t in trans_names:
                if pulp.value(assign_vars[s, t]) == 1:
                    transport_plan.append({
                        '周次': w+1,
                        '供应商ID': s,
                        '转运商': t,
                        '运输量': order_plan[(order_plan['周次']==w+1) & 
                                           (order_plan['供应商ID']==s)]['订购量'].values[0]
                    })
    
    transport_plan = pd.DataFrame(transport_plan)
    
    # 7. 方案效果分析
    analysis_results = {
        'total_cost': pulp.value(order_model.objective),
        'avg_transport_loss': transport_plan.groupby('周次')['运输量'].sum().mean(),
        'capacity_utilization': capacity_requirement * 24 / 
                               (order_plan.groupby('周次')['订购量'].sum().sum() / 
                                {'A': 0.6, 'B': 0.66, 'C': 0.72}[suppliers_data['材料分类'].mode()[0]])
    }
    
    return order_plan, transport_plan, analysis_results

# ====================================================
# 主执行流程
# ====================================================
if __name__ == "__main__":
    # 步骤1: 获取50家重要供应商 (从问题1结果)
    # 实际应用中需替换为问题1的供应商选择结果
    important_suppliers = [201, 140 ,348, 151, 229 ,361 ,374, 108 ,139 ,126 ,330 ,395, 308 ,340 ,282 ,275 ,329 ,307
 ,268 ,131 ,356 ,306 ,194 , 37 ,352 ,143 ,247 ,266  ,31 ,284 ,294 ,346 ,365 ,338  ,40 ,364,
 367,  55 ,244  ,80, 123, 218  ,86 ,210  , 3, 114  ,74  , 7, 273, 189]  # 50家供应商ID列表
    
    # 步骤2: 求解24周方案
    order_plan, transport_plan, analysis = solve_24week_plan(
        selected_suppliers=important_suppliers
    )
    
    # 步骤3: 结果输出
    print("="*50)
    print("未来24周订购方案摘要")
    print("="*50)
    print(f"总采购成本: {analysis['total_cost']:.2f}元")
    print(f"平均运输损耗率: {analysis['avg_transport_loss']:.2%}")
    print(f"产能利用率: {analysis['capacity_utilization']:.2%}")
    
    # 保存结果到附件A和附件B
    # order_plan.to_excel("附件A.xlsx", index=False)
    # transport_plan.to_excel("附件B.xlsx", index=False)