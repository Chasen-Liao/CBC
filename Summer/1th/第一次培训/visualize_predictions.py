import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_predictions():
    # 读取预测结果
    predictions = pd.read_csv('predictions.csv')
    
    # 提取真实值和预测值
    true_values = predictions.iloc[:, 0].values  # 第一列是真实值
    predicted_values = predictions.iloc[:, 1].values  # 第二列是预测值
    
    # 计算评估指标
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    
    # 1. 散点图：真实值 vs 预测值
    plt.subplot(2, 2, 1)
    plt.scatter(true_values, predicted_values, alpha=0.7)
    
    # 添加对角线（理想情况下，点应该落在这条线上）
    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('真实值 vs 预测值')
    
    # 添加评估指标文本
    plt.annotate(f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}', 
                 xy=(0.05, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
    
    # 2. 残差图：真实值 vs 残差
    plt.subplot(2, 2, 2)
    residuals = true_values - predicted_values
    plt.scatter(true_values, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('真实值')
    plt.ylabel('残差 (真实值 - 预测值)')
    plt.title('残差分布')
    
    # 3. 残差直方图
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('残差值')
    plt.ylabel('频数')
    plt.title('残差直方图')
    
    # 4. 真实值和预测值的条形图比较
    plt.subplot(2, 2, 4)
    
    # 创建索引
    indices = np.arange(len(true_values))
    
    # 设置条形宽度
    bar_width = 0.35
    
    # 创建条形图
    plt.bar(indices - bar_width/2, true_values, bar_width, alpha=0.7, label='真实值')
    plt.bar(indices + bar_width/2, predicted_values, bar_width, alpha=0.7, label='预测值')
    
    plt.xlabel('样本索引')
    plt.ylabel('值')
    plt.title('真实值与预测值比较')
    plt.xticks(indices)
    plt.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('prediction_visualization.png', dpi=300)
    
    # 显示图表
    plt.show()
    
    print(f"可视化结果已保存为 'prediction_visualization.png'")
    print(f"评估指标:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 额外创建一个详细的真实值vs预测值表格
    comparison_df = pd.DataFrame({
        '真实值': true_values,
        '预测值': predicted_values,
        '绝对误差': np.abs(true_values - predicted_values),
        '相对误差(%)': np.abs((true_values - predicted_values) / true_values) * 100
    })
    
    # 保存比较表格
    comparison_df.to_csv('prediction_comparison.csv', index=True, float_format='%.4f')
    print(f"详细比较结果已保存为 'prediction_comparison.csv'")

if __name__ == "__main__":
    visualize_predictions()