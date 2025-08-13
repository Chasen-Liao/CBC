# 预处理数据
# 1. 空值处理
# 2. 


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 读取数据
try:
    # 尝试读取文件，如果找不到则提供明确的错误信息
    import os
    if os.path.exists('附件.xlsx'):
        xls = pd.ExcelFile('附件.xlsx')
    elif os.path.exists('data.xlsx'):
        xls = pd.ExcelFile('data.xlsx')
    else:
        print("错误: 找不到Excel文件。请确保'附件.xlsx'或'data.xlsx'文件存在于当前目录。")
        import sys
        sys.exit(1)
except Exception as e:
    print(f"读取Excel文件时出错: {e}")
    print("提示: 如果缺少openpyxl模块，请运行 'pip install openpyxl' 安装它。")
    import sys
    sys.exit(1)
df1 = xls.parse('表单1')
df2 = xls.parse('表单2')

# 合并表单数据
df2['文物编号'] = df2['文物采样点'].str[:2]  # 提取前两位作为文物编号

# 确保两个数据框中的'文物编号'列具有相同的数据类型（转换为字符串）
df1['文物编号'] = df1['文物编号'].astype(str)
df2['文物编号'] = df2['文物编号'].astype(str)

# 打印数据类型以便调试
print("df1['文物编号']的数据类型:", df1['文物编号'].dtype)
print("df2['文物编号']的数据类型:", df2['文物编号'].dtype)

merged = pd.merge(df1, df2, on='文物编号', how='inner')

# 处理缺失值（题目说明空白表示未检测到）
chemical_cols = [
    '二氧化硅(SiO2)', '氧化钠(Na2O)', '氧化钾(K2O)', '氧化钙(CaO)',
    '氧化镁(MgO)', '氧化铝(Al2O3)', '氧化铁(Fe2O3)', '氧化铜(CuO)',
    '氧化铅(PbO)', '氧化钡(BaO)', '五氧化二磷(P2O5)', '氧化锶(SrO)',
    '氧化锡(SnO2)', '二氧化硫(SO2)'
]
merged[chemical_cols] = merged[chemical_cols].fillna(0)

# 成分总和有效性检查
merged['总和'] = merged[chemical_cols].sum(axis=1)
valid_data = merged[(merged['总和'] >= 85) & (merged['总和'] <= 105)]

# 特征工程
# 1. 分类变量编码
le = LabelEncoder()
valid_data['纹饰编码'] = le.fit_transform(valid_data['纹饰'])
valid_data['类型编码'] = le.fit_transform(valid_data['类型'])
valid_data['颜色编码'] = le.fit_transform(valid_data['颜色'])

# 2. 标准化处理
scaler = StandardScaler()
feature_cols = chemical_cols + ['纹饰编码', '类型编码', '颜色编码']
valid_data_scaled = valid_data.copy()
valid_data_scaled[feature_cols] = scaler.fit_transform(valid_data[feature_cols])

# 保存标准化参数到文件（关键步骤！）
scaler_params = {
    'mean': scaler.mean_,
    'scale': scaler.scale_
}
np.save('scaler_params.npy', scaler_params)

# 划分训练集和测试集
train_data = valid_data_scaled[valid_data_scaled['表面风化'] == '无风化']
test_data = valid_data_scaled[valid_data_scaled['表面风化'] == '风化']

# 输出SVR训练表格（以SiO2为例）
def prepare_svr_data(data, target_col):
    X = data[feature_cols]
    y = data[target_col]
    return X, y

# 训练集（未风化样本）
X_train, y_train = prepare_svr_data(train_data, '二氧化硅(SiO2)')

# 测试集（风化样本）
X_test, y_test = prepare_svr_data(test_data, '二氧化硅(SiO2)')

# 合并为DataFrame方便查看
train_df = X_train.copy()
train_df['目标_SiO2'] = y_train
test_df = X_test.copy()
test_df['目标_SiO2'] = y_test

# 保存结果
train_df.to_csv('svr_train.csv', index=False)
test_df.to_csv('svr_test.csv', index=False)

print("训练集样本数:", len(train_df))
print("测试集样本数:", len(test_df))
print("\n训练集前5行:")
print(train_df.head())
