import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# 自定义拟合函数（保留HAP和Co/SiO2一次项）
def optimized_model(X, a0, a1, a3, a5, a7, a10, a16, a17):
    """
    优化后的五元二次回归模型
    保留项：常数项、Co/SiO2一次项、HAP一次项、温度一次项、Co²项、温度²项、Co*乙醇浓度、Co*温度
    X: [Co/SiO2, Co, HAP, 乙醇浓度, 温度]
    """
    x1, x2, x3, x4, x5 = X  # 解包五个自变量
    
    # 模型公式 (保留显著项和指定的一次项)
    y = (a0 + 
         a1 * x1 +          # Co/SiO2一次项
         a3 * x3 +          # HAP一次项
         a5 * x5 +          # 温度线性项
         a7 * x2**2 +       # Co²项
         a10 * x5**2 +      # 温度²项
         a16 * x2 * x4 +    # Co*乙醇浓度交叉项
         a17 * x2 * x5      # Co*温度交叉项
        )
    return y

# 加载并预处理数据
def load_and_process_data(file_path):
    """
    加载Excel文件并预处理数据
    1. 读取结构化数据部分
    2. 筛选所需列
    3. 计算因变量
    """
    try:
        df = pd.read_excel(file_path, header=0)
    except UnicodeDecodeError:
        try:
            df = pd.read_excel(file_path, encoding='gbk', header=0)
        except:
            df = pd.read_excel(file_path, encoding='latin1', header=0)
    
    # 清理列名：去除多余空格和不可见字符
    df.columns = df.columns.str.strip().str.replace('\n', '')
    
    # 确认列名（根据实际数据调整）
    column_mapping = {
        'Co/SiO2': ['Co/SiO2', 'Co/SiO2质量'],
        'Co': ['Co', 'Co负载量(wt%)'],
        'HAP': ['HAP', 'HAP质量(mg)'],
        '乙醇浓度': ['乙醇浓度', '乙醇浓度(ml/min)'],
        '温度': ['温度'],
        '乙醇转化率': ['乙醇转化率(%)'],
        'C4烯烃选择性': ['C4烯烃选择性(%)']
    }
    
    # 寻找最佳匹配的列名
    selected_cols = []
    for col_type, possible_names in column_mapping.items():
        found = False
        for name in possible_names:
            if name in df.columns:
                selected_cols.append(name)
                found = True
                break
        if not found:
            raise ValueError(f"未找到匹配的列: {col_type} (尝试名称: {possible_names})")
    
    # 筛选所需数据列
    data = df[selected_cols].copy()
    
    # 重命名列以便后续处理
    rename_map = {
        selected_cols[0]: 'Co/SiO2',
        selected_cols[1]: 'Co',
        selected_cols[2]: 'HAP',
        selected_cols[3]: '乙醇浓度',
        selected_cols[4]: '温度',
        selected_cols[5]: '乙醇转化率',
        selected_cols[6]: 'C4选择性'
    }
    
    data.rename(columns=rename_map, inplace=True)
    
    # 计算因变量：乙醇转化率(%) * C4烯烃选择性(%)
    data['因变量'] = data['乙醇转化率'] * data['C4选择性']
    
    # 数据清洗：去除无效值
    data = data.dropna()
    
    return data


# 示例数据准备 (替换为您的实际数据)
data = load_and_process_data('附件1.xlsx')
print(data.head())

# 准备输入数据
X_data = data[['Co/SiO2', 'Co', 'HAP', '乙醇浓度', '温度']].values.T
y_data = data['因变量'].values

# 初始参数猜测
initial_guess = np.ones(8)

# 执行拟合
params_opt, params_cov = curve_fit(
    optimized_model, 
    X_data, 
    y_data, 
    p0=initial_guess,
    maxfev=10000  # 增加最大迭代次数确保收敛
)

# 提取拟合参数
a0, a1, a3, a5, a7, a10, a16, a17 = params_opt
param_names = ['常数项', 'Co/SiO2', 'HAP', '温度', 'Co²', '温度²', 'Co*乙醇浓度', 'Co*温度']

# 输出拟合结果
print("优化后的模型参数:")
for name, value in zip(param_names, params_opt):
    print(f"{name}: {value:.4f}")

# 计算预测值
y_pred = optimized_model(X_data, *params_opt)

# 模型评估
residuals = y_data - y_pred
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
r_squared = 1 - np.var(residuals)/np.var(y_data)

print(f"\n模型评估指标:")
print(f"均方误差(MSE): {mse:.2f}")
print(f"均方根误差(RMSE): {rmse:.2f}")
print(f"决定系数(R²): {r_squared:.4f}")

# 可视化预测结果 (可选)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_data, y_pred, alpha=0.6)
plt.plot([min(y_data), max(y_data)], [min(y_data), max(y_data)], 'r--')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.grid(True)
plt.show()