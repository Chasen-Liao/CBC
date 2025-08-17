import os
import csv

def calculate_transport_plan(week_file):
    """计算单周的转运方案"""
    # 转运商配置 (按TOPSIS得分排序)
    transporters = [
        {'id': 'T3', 'capacity': 6000, 'remaining': 6000},
        {'id': 'T6', 'capacity': 6000, 'remaining': 6000},
        {'id': 'T8', 'capacity': 6000, 'remaining': 6000},
        {'id': 'T2', 'capacity': 6000, 'remaining': 6000},
        {'id': 'T4', 'capacity': 6000, 'remaining': 6000},
        {'id': 'T5', 'capacity': 6000, 'remaining': 6000},
        {'id': 'T1', 'capacity': 6000, 'remaining': 6000},
        {'id': 'T7', 'capacity': 6000, 'remaining': 6000}
    ]
    
    # 读取该周的订货量数据
    orders = {}
    with open(week_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            supplier_id = row['供应商ID']
            quantity = float(row['订货量'])
            orders[supplier_id] = quantity
    
    # 按订货量降序排列供应商
    sorted_orders = sorted(orders.items(), key=lambda x: x[1], reverse=True)
    
    # 存储分配结果
    assignments = []
    
    # 贪心算法分配
    for supplier, amount in sorted_orders:
        assigned = False
        
        # 按优先级顺序寻找可用转运商
        for t in transporters:
            if t['remaining'] >= amount:
                # 分配订单并更新剩余容量
                assignments.append((supplier, t['id'], amount))
                t['remaining'] -= amount
                assigned = True
                break
        
        # 异常处理（如果容量不足）
        if not assigned:
            print(f"警告: 转运商容量不足，无法分配供应商 {supplier} 的订单 ({amount} 立方米)")
    
    return assignments

def generate_weekly_transport_plans():
    """生成所有24周的转运方案"""
    # 确保输出目录存在
    output_dir = '转运方案'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理24周的订货方案
    for week in range(1, 25):
        # 构造文件名
        week_str = f"第{week:02d}周"
        input_file = f"问题2_{week_str}.csv"
        output_file = os.path.join(output_dir, f"转运方案_{week_str}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 缺少输入文件 {input_file}, 跳过")
            continue
        
        # 计算转运方案
        transport_plan = calculate_transport_plan(input_file)
        
        # 写入CSV文件
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['供应商ID', '转运商ID', '订货量'])
            for assignment in transport_plan:
                writer.writerow(assignment)
        
        print(f"已生成: {output_file}")

if __name__ == "__main__":
    # 生成所有周的转运方案
    generate_weekly_transport_plans()
    print("所有周的转运方案已生成完毕！")