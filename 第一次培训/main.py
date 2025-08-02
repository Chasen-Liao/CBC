import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==================== 预处理文件 ====================
def preprocess_data():
    # 读取数据
    xls = pd.ExcelFile('data.xlsx')
    df1 = xls.parse('表单1')
    df2 = xls.parse('表单2')
    # 确保数据框中的'文物编号'列具有正确的数据类型（转换为字符串）
    df1['文物编号'] = df1['文物编号'].astype(str)
    
    # 从文物采样点提取文物编号
    df2['文物编号'] = df2['文物采样点'].str[:2]
    merged = pd.merge(df1, df2, on='文物编号', how='inner')
    
    

    # 填充缺失值
    chemical_cols = [
        '二氧化硅(SiO2)', '氧化钠(Na2O)', '氧化钾(K2O)', '氧化钙(CaO)',
        '氧化镁(MgO)', '氧化铝(Al2O3)', '氧化铁(Fe2O3)', '氧化铜(CuO)',
        '氧化铅(PbO)', '氧化钡(BaO)', '五氧化二磷(P2O5)', '氧化锶(SrO)',
        '氧化锡(SnO2)', '二氧化硫(SO2)'
    ]
    merged[chemical_cols] = merged[chemical_cols].fillna(0)

    # 成分有效性检查
    merged['总和'] = merged[chemical_cols].sum(axis=1)
    valid_data = merged[(merged['总和'] >= 85) & (merged['总和'] <= 105)]

    # 特征工程 - 编码分类变量
    le = LabelEncoder()
    valid_data['纹饰编码'] = le.fit_transform(valid_data['纹饰'])
    valid_data['类型编码'] = le.fit_transform(valid_data['类型'])
    valid_data['颜色编码'] = le.fit_transform(valid_data['颜色'])

    # 保存原始目标值（关键修正）
    # 注意：这里需要为每个目标成分单独处理，此处以SiO2为例
    target_col = '氧化钡(BaO)'

    # 准备特征列（排除目标列）
    feature_cols = [col for col in chemical_cols if col != target_col] + ['纹饰编码', '类型编码', '颜色编码']

    # 标准化特征（不包含目标列）
    scaler = StandardScaler()
    X = valid_data[feature_cols]
    X_scaled = scaler.fit_transform(X)

    # 保存标准化参数
    scaler_params = {
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'feature_names': feature_cols
    }
    np.save('scaler_params.npy', scaler_params)

    # 划分训练集和测试集
    train_data = valid_data[valid_data['表面风化'] == '无风化']
    test_data = valid_data[valid_data['表面风化'] == '风化']

    # 保存预处理数据（包含原始目标值）
    train_df = train_data[feature_cols + [target_col]].copy()
    test_df = test_data[feature_cols + [target_col]].copy()
    
    train_df.to_csv('svr_train.csv', index=False)
    test_df.to_csv('svr_test.csv', index=False)
    
    return target_col, feature_cols

# ==================== 模型训练文件 ====================
def train_model():
    target_col, feature_cols = preprocess_data()
    
    # 加载数据
    train_df = pd.read_csv('svr_train.csv')
    test_df = pd.read_csv('svr_test.csv')
    
    # 分离特征和目标
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]  # 原始目标值
    
    # 训练SVR
    param_grid = {
    'C': [0.1, 1, 10, 100, 1000],      # 扩大搜索范围
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # 精细化gamma
    'epsilon': [0.01, 0.05, 0.1, 0.5],         # 调整误差容忍度
    'kernel': ['rbf', 'poly', 'linear']        # 尝试不同核函数
    }
    
    grid = GridSearchCV(SVR(), param_grid, cv=10, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    
    # 输出SVR训练结果
    print(f"最佳参数: {grid.best_params_}")
    print(f"最佳交叉验证分数: {-grid.best_score_:.4f} MSE")
    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")
    
    # 使用最佳模型预测
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)  # 原始单位的预测值
    
    # 计算并输出测试集性能指标
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"测试集MSE: {mse:.4f}")
    print(f"测试集R²: {r2:.4f}")
    
    # 生成结果表格
    results = test_df[[target_col]].copy()
    results['预测值'] = y_pred
    results.to_csv('predictions.csv', index=False)
    
    
    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文编码支持
    # # 可视化（标准化空间）
    # scaler_params = np.load('scaler_params.npy', allow_pickle=True).item()
    # X_test_scaled = (X_test - scaler_params['mean']) / scaler_params['scale']
    
    # plt.figure(figsize=(10,6))
    # plt.scatter(X_test_scaled.mean(axis=1), 
    #             (y_pred - scaler_params['mean'][0])/scaler_params['scale'][0], 
    #             alpha=0.7)
    # plt.xlabel('标准化特征均值')
    # plt.ylabel('标准化预测值')
    # plt.title('标准化空间预测可视化')
    # plt.show()

if __name__ == '__main__':
    train_model()