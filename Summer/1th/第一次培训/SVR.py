import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文编码支持

# ==================== 预处理文件 (preprocessing.py) ====================
# (与之前一致，生成svr_train.csv、svr_test.csv和scaler_params.npy)

# ==================== 模型训练文件 (train.py) ====================
# 1. 加载数据
train_df = pd.read_csv('svr_train.csv')
test_df = pd.read_csv('svr_test.csv')

# 2. 分离特征和目标（以SiO2为例）
feature_cols = [col for col in train_df.columns if col != '目标_SiO2']
X_train = train_df[feature_cols]
y_train = train_df['目标_SiO2']
X_test = test_df[feature_cols]
y_test = test_df['目标_SiO2']

# 3. 初始化SVR模型
svr = SVR()

# 4. 参数网格设置
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'epsilon': [0.01, 0.1, 0.5]
}

# 5. 网格搜索调参
grid_search = GridSearchCV(
    estimator=svr,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# 6. 训练模型
grid_search.fit(X_train, y_train)

# 7. 输出最佳参数
print("最佳参数组合:")
print(grid_search.best_params_)

# 8. 使用最佳参数训练最终模型
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 9. 预测标准化值
y_pred_scaled = best_model.predict(X_test)  # 标准化空间的预测值

# 10. 加载标准化参数
scaler_params = np.load('scaler_params.npy', allow_pickle=True).item()
mean = scaler_params['mean']
scale = scaler_params['scale']

# 11. 计算原始单位的预测值（关键转换）
siO2_index = 0  # 确保特征列顺序一致
y_pred_original = y_pred_scaled * scale[siO2_index] + mean[siO2_index]

# 12. 获取原始单位的真实值
y_test_original = test_df['目标_SiO2']

# 13. 生成结果表格
results_df = test_df[['目标_SiO2']].copy()  # 真实值
results_df['预测_SiO2_原始'] = y_pred_original
results_df.to_csv('svr_predictions.csv', index=False)

# ==================== 可视化（标准化空间） ====================
# 14. 准备标准化空间的绘图数据
y_test_scaled = (y_test_original - mean[siO2_index]) / scale[siO2_index]  # 将真实值转换为标准化

# 15. 绘制标准化空间散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test_scaled, y_pred_scaled, alpha=0.7, label='预测点')
plt.plot([y_test_scaled.min(), y_test_scaled.max()], 
         [y_test_scaled.min(), y_test_scaled.max()], 'r--', lw=2, label='理想拟合线')
plt.xlabel('标准化后的真实SiO2含量')
plt.ylabel('标准化后的预测SiO2含量')
plt.title('SVR标准化空间预测结果')
plt.legend()
plt.grid(True)
plt.show()

# 16. 绘制原始单位散点图（可选）
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.7, label='预测点')
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2, label='理想拟合线')
plt.xlabel('原始单位真实SiO2含量')
plt.ylabel('原始单位预测SiO2含量')
plt.title('SVR原始单位预测结果')
plt.legend()
plt.grid(True)
plt.show()