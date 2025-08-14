import pulp
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

def solve_supplier_selection_multiobjective(
    n_suppliers: int = 50, 
    capacity_requirement: float = 28200 * (1 + 0.0139),
    p_C: float = 1.0,
    w_cost: float = 0.6,    # 成本权重
    w_A: float = 0.3,       # A类供应商数量权重
    w_C: float = 0.1,       # C类供应商数量权重
    verbose: bool = True
) -> Tuple[Dict[int, int], Dict[int, float], int, float, float, pd.DataFrame, int, int]:
    
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
        'stability': data['稳定性系数'],
        '超额供货率': data['超额供货率'],
        '短缺供货率': data['短缺供货率']
    })
    
    # 计算有效供应量
    df_suppliers['eff_supply'] = (
        df_suppliers['avg_supply'] 
        * (1 + df_suppliers['超额供货率'])
        * (1 - df_suppliers['短缺供货率'])
    )
    
    # 设置原材料单价
    price_map = {'A': 1.2 * p_C, 'B': 1.1 * p_C, 'C': p_C}
    df_suppliers['unit_price'] = df_suppliers['material_type'].map(price_map)
    
    # 计算成本系数
    df_suppliers['cost_coeff'] = df_suppliers['unit_price'] * df_suppliers['eff_supply']
    
    # 按材料类型分组
    group_A = df_suppliers[df_suppliers['material_type'] == 'A'].index.tolist()
    group_B = df_suppliers[df_suppliers['material_type'] == 'B'].index.tolist()
    group_C = df_suppliers[df_suppliers['material_type'] == 'C'].index.tolist()
    
    # 创建多目标整数线性规划模型
    model = pulp.LpProblem("Supplier_Selection_Multiobjective", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", df_suppliers.index, cat='Binary')
    
    # 计算各目标分量
    cost_obj = pulp.lpSum(df_suppliers.loc[i, 'cost_coeff'] * x[i] for i in df_suppliers.index)
    A_count_obj = pulp.lpSum(x[i] for i in group_A)  # 要最大化A类数量
    C_count_obj = pulp.lpSum(x[i] for i in group_C)  # 要最小化C类数量
    
    # 目标归一化系数（防止量纲不一致）
    max_cost = df_suppliers['cost_coeff'].sum()
    max_A = len(group_A) or 1  # 避免除以0
    max_C = len(group_C) or 1
    
    # 构建综合目标函数
    model += (
        w_cost * (cost_obj / max_cost) - 
        w_A * (A_count_obj / max_A) + 
        w_C * (C_count_obj / max_C)
    )
    
    # 添加约束条件
    model += (
        pulp.lpSum(df_suppliers.loc[i, 'eff_supply'] * x[i] / 0.6 for i in group_A) +
        pulp.lpSum(df_suppliers.loc[i, 'eff_supply'] * x[i] / 0.66 for i in group_B) +
        pulp.lpSum(df_suppliers.loc[i, 'eff_supply'] * x[i] / 0.72 for i in group_C)
    ) >= capacity_requirement
    
    # 求解模型
    solver = pulp.PULP_CBC_CMD(timeLimit=10, msg=False)
    model.solve(solver)
    
    # 检查求解状态
    if pulp.LpStatus[model.status] != 'Optimal':
        raise RuntimeError(f"求解失败，状态: {pulp.LpStatus[model.status]}")
    
    selected_suppliers = {}
    supply_amounts = {}
    total_capacity = 0.0
    
    # 准备结果数据框
    results_df = df_suppliers.copy()
    results_df['selected'] = 0
    results_df['supply_amount'] = 0.0
    results_df['capacity_contribution'] = 0.0
    
    # 计算实际目标值
    actual_cost = 0.0
    actual_A_count = 0
    actual_C_count = 0
    
    for i in df_suppliers.index:
        supplier_id = df_suppliers.loc[i, 'supplier_id']
        selected = int(x[i].value())
        selected_suppliers[supplier_id] = selected
        
        # 计算供应量
        supply_amount = df_suppliers.loc[i, 'eff_supply'] if selected == 1 else 0.0
        supply_amounts[supplier_id] = supply_amount
        
        # 计算产能贡献
        capacity_contribution = 0.0
        if selected == 1:
            material_type = df_suppliers.loc[i, 'material_type']
            if material_type == 'A':
                capacity_contribution = supply_amount / 0.6
                actual_A_count += 1
                actual_cost += df_suppliers.loc[i, 'cost_coeff']
            elif material_type == 'B':
                capacity_contribution = supply_amount / 0.66
                actual_cost += df_suppliers.loc[i, 'cost_coeff']
            elif material_type == 'C':
                capacity_contribution = supply_amount / 0.72
                actual_C_count += 1
                actual_cost += df_suppliers.loc[i, 'cost_coeff']
            total_capacity += capacity_contribution
        
        # 填充结果数据框
        results_df.loc[i, 'selected'] = selected
        results_df.loc[i, 'supply_amount'] = supply_amount
        results_df.loc[i, 'capacity_contribution'] = capacity_contribution
    
    min_count = sum(selected_suppliers.values())
    
    # 控制台输出摘要
    if verbose:
        print("\n" + "="*50)
        print(f"前{n_suppliers}家供应商多目标优化结果摘要")
        print("="*50)
        print(f"目标权重配置: 成本={w_cost}, A类={w_A}, C类={w_C}")
        print(f"供应商总数: {min_count}")
        print(f"总成本: {actual_cost:.2f} 元")
        print(f"A类供应商数量: {actual_A_count}/{len(group_A)}")
        print(f"C类供应商数量: {actual_C_count}/{len(group_C)}")
        print(f"总产能: {total_capacity:.2f} 立方米")
        print(f"产能需求: {capacity_requirement} 立方米")
        
        if total_capacity >= capacity_requirement:
            print(f"✅ 产能满足需求 (超出: {total_capacity - capacity_requirement:.2f} 立方米)")
        else:
            print(f"❌❌ 产能不足 (缺少: {capacity_requirement - total_capacity:.2f} 立方米)")
    
    return selected_suppliers, supply_amounts, min_count, actual_cost, total_capacity, results_df, actual_A_count, actual_C_count

def run_24_week_simulation_multiobjective():
    # 初始化结果存储
    summary_data = []
    all_week_details = []
    
    # 初始剩余产能
    remain = 0
    
    for week in range(1, 25):
        # 计算本周实际需求
        demand_this_week = 28200 - remain
        required_capacity = demand_this_week * (1 + 0.0139)
        
        # 输出周信息
        print("\n" + "="*50)
        print(f"第 {week} 周")
        print(f"上周剩余产能: {remain:.2f} 立方米")
        print(f"本周需求产能: {demand_this_week:.2f} 立方米")
        print(f"考虑损耗后的实际要求产能: {required_capacity:.2f} 立方米")
        print("="*50)
        
        # 求解本周供应商选择
        try:
            (selected, supply_amounts, min_count, 
             min_cost, total_capacity, results_df,
             actual_A_count, actual_C_count) = solve_supplier_selection_multiobjective(
                 capacity_requirement=required_capacity,
                 w_cost=0.6,  # 每周可调整权重
                 w_A=0.3,
                 w_C=0.1,
                 verbose=True
             )
        except Exception as e:
            print(f"第 {week} 周求解失败: {str(e)}")
            continue
        
        # 正态随机数 -> 运损率
        while True:
            random_num = np.random.normal(loc=1.00, scale=1.33, size=1)[0]
            if abs(random_num - 1.00) <= 1.33 and 0 <= random_num <= 1.5:
                break
        
        lost_rate = random_num / 100
        
        # 计算接收量和剩余产能
        received = total_capacity * (1 - lost_rate)
        new_remain = received + remain - 28200
        
        # 更新剩余产能用于下周
        remain = new_remain if new_remain > 0 else 0
        
        # 添加摘要信息
        summary_data.append({
            'week': week,
            'last_remain': round(remain, 2),
            'demand_this_week': round(demand_this_week, 2),
            'required_capacity': round(required_capacity, 2),
            'total_capacity': round(total_capacity, 2),
            'received': round(received, 2),
            'new_remain': round(new_remain, 2),
            'min_suppliers': min_count,
            'min_cost': round(min_cost, 2),
            'A_count': actual_A_count,
            'C_count': actual_C_count
        })
        
        # 添加本周供应商详情
        results_df['week'] = week
        all_week_details.append(results_df)
        
        # 打印本周结果摘要
        print(f"\n本周结果:")
        print(f"供应商提供总产能: {total_capacity:.2f} 立方米")
        print(f"运输后接收产能: {received:.2f} 立方米")
        print(f"本周剩余产能: {new_remain:.2f} 立方米")
        print(f"供应商数量: {min_count}, 总成本: {min_cost:.2f} 元")
        print(f"A类供应商: {actual_A_count}, C类供应商: {actual_C_count}")
    
    # 创建结果DataFrame
    summary_df = pd.DataFrame(summary_data)
    details_df = pd.concat(all_week_details, ignore_index=True)
    
    # 保存为CSV文件
    summary_df.to_csv('supplier_selection_multiobjective_summary_24weeks.csv', index=False)
    details_df.to_csv('supplier_selection_multiobjective_details_24weeks.csv', index=False)
    
    print("\n多目标优化模拟完成! 结果已保存为CSV文件")
    return summary_df, details_df

if __name__ == "__main__":
    print("开始24周多目标优化模拟...")
    summary, details = run_24_week_simulation_multiobjective()
    
    # 打印最终摘要
    print("\n24周多目标优化模拟最终摘要:")
    print(summary[['week', 'min_suppliers', 'min_cost', 'A_count', 'C_count', 'total_capacity', 'new_remain']])
    
    # 分析权重效果
    avg_A = summary['A_count'].mean()
    avg_C = summary['C_count'].mean()
    avg_cost = summary['min_cost'].mean()
    print(f"\n平均每周A类供应商: {avg_A:.2f}, C类供应商: {avg_C:.2f}, 平均成本: {avg_cost:.2f}")
    # print(f"A/C比例: {avg_A/avg_C:.2f}:1 (权重配置: 成本0.6, A类0.3, C类0.1)")