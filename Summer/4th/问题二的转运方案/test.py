import re

def parse_problem_data(text, color_region):
    """
    解析指定颜色区域的订购方案数据
    """
    supplier_data = {}
    # 查找指定颜色区域的文本块
    pattern = f'<{color_region}>(.*?)</{color_region}>'
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        print(f"未找到{color_region}的数据!")
        return supplier_data
        
    data_text = match.group(1).strip()
    lines = data_text.split('\n')
    
    # 定位数据起始行(跳过说明行)
    start_index = None
    for i, line in enumerate(lines):
        if "供应商ID" in line:
            start_index = i + 1
            break
            
    if start_index is None:
        print(f"未找到供应商ID标题行: {color_region}")
        return supplier_data
    
    # 处理数据行
    for line in lines[start_index:]:
        line = line.strip()
        if not line or "合计" in line:
            continue  # 跳过空行和合计行
        
        # 识别供应商ID和订货量
        suppliers = []
        quantities = []
        
        tokens = re.split(r'\s+', line)
        for token in tokens:
            if re.match(r'^S\d{3,4}$', token):
                suppliers.append(token)
            elif token and not any(c.isalpha() for c in token):
                # 数字格式处理: 允许小数和负号(实际无负数)
                quantities.append(float(token) if '.' in token else int(token))
        
        # 没有订货量的供应商跳过
        if not quantities:
            continue
            
        # 为每个供应商创建24周的订货量列表(不足补0)
        q_list = quantities[:24] + [0] * max(0, 24 - len(quantities))
        
        # 多个供应商共享相同订货量模式的情况
        for sid in suppliers:
            supplier_data[sid] = q_list[:24]  # 确保长度24
    
    return supplier_data

def write_weekly_solutions(problem_name, supplier_data):
    """
    输出每周的转运方案到CSV文件
    """
    # 按周重组数据
    weekly_data = {}
    for week in range(24):
        week_data = {}
        for sid, quantities in supplier_data.items():
            if quantities[week] > 0:
                week_data[sid] = quantities[week]
        weekly_data[f"第{week+1:02d}周"] = week_data
    
    # 写入CSV文件
    for week_name, week_data in weekly_data.items():
        filename = f"{problem_name}_{week_name}.csv"
        with open(filename, 'w', encoding='gbk') as f:
            f.write("供应商ID,订货量\n")
            for sid, qty in sorted(week_data.items()):
                f.write(f"{sid},{qty}\n")

# 主程序
input_data = """[您提供的完整文本数据]"""

# 解析不同问题的订购方案
problem2_data = parse_problem_data(input_data, "问题2的订购方案结果")
problem3_data = parse_problem_data(input_data, "问题3的订购方案结果")
problem4_data = parse_problem_data(input_data, "问题4的订购方案结果")

# 生成每周转运方案
write_weekly_solutions("问题2", problem2_data)
write_weekly_solutions("问题3", problem3_data)
write_weekly_solutions("问题4", problem4_data)

print("文件已生成完成！")