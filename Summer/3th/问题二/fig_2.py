import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
file_path = '附件1.xlsx'  # 替换为实际文件路径
df = pd.read_excel(file_path)

# 确保列名与数据匹配（根据实际列名调整）
# 如果列名不同，请修改这里的列名映射
column_mapping = {
    '催化剂组合编号': '催化剂组合编号',
    '温度': '温度',
    '乙醇转化率(%)': '乙醇转化率(%)'
}
df = df.rename(columns=column_mapping)

# 获取前20个唯一的温度值
unique_temps = df['温度'].unique()
if len(unique_temps) > 20:
    unique_temps = sorted(unique_temps)[:20]  # 取前20个温度

# 创建图表
plt.figure(figsize=(15, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 为每个温度绘制一条曲线
for temp in unique_temps:
    # 筛选该温度下的数据
    temp_df = df[df['温度'] == temp]
    # 按催化剂编号排序
    temp_df = temp_df.sort_values('催化剂组合编号')
    
    # 绘制折线图
    plt.plot(temp_df['催化剂组合编号'], 
             temp_df['乙醇转化率(%)'], 
             marker='o', 
             label=f'{temp}℃',
             linewidth=2)

# 添加图表元素
plt.title('不同温度下催化剂对乙醇转化率的影响', fontsize=16)
plt.xlabel('催化剂组合编号', fontsize=14)
plt.ylabel('乙醇转化率(%)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='温度(℃)', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()