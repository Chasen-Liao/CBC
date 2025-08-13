# 五元二次函数拟合


# 自变量: X:{温度, 负载, Co/SiO2, HAP, 乙醇} 
# 因变量: Y:{C4烯烃收率}

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

# 自定义拟合函数（五元二次方程模型）
def quadratic_model(X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, 
                   a11, a12, a13, a14, a15, a16, a17, a18, a19, a20):
    """
    五元二次回归模型
    X: 包含五个自变量的数组 [Co/SiO2, Co, HAP, 乙醇浓度, 温度]
    """
    x1, x2, x3, x4, x5 = X
    
    # 二次多项式模型项：
    # 常数项
    # 线性项 (a1*x1 + a2*x2 + ...)
    # 二次项 (a6*x1², a7*x2², ...)
    # 交叉项 (a11*x1*x2, a12*x1*x3, ...)
    y = (
        a0 + 
        a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5 +
        a6*x1**2 + a7*x2**2 + a8*x3**2 + a9*x4**2 + a10*x5**2 +
        a11*x1*x2 + a12*x1*x3 + a13*x1*x4 + a14*x1*x5 +
        a15*x2*x3 + a16*x2*x4 + a17*x2*x5 +
        a18*x3*x4 + a19*x3*x5 +
        a20*x4*x5
    )
    return y

def optimized_quadratic_model(X, a0, a5, a7, a10, a16, a17):
    """
    优化后的五元二次回归模型（仅保留显著项）
    X: 包含五个自变量的数组 [Co/SiO2, Co, HAP, 乙醇浓度, 温度]
    显著项：常数项、Co²、温度、温度²、Co*乙醇浓度、Co*温度
    """
    _, x2, _, x4, x5 = X  # 仅需提取 Co(x2)、乙醇浓度(x4)、温度(x5)
    
    # 保留显著项：
    # 常数项(a0) + 温度线性项(a5*x5) + Co²项(a7*x2²) 
    # 温度²项(a10*x5²) + Co*乙醇浓度交叉项(a16*x2*x4) + Co*温度交叉项(a17*x2*x5)
    y = (
        a0 + 
        a5 * x5 + 
        a7 * x2**2 + 
        a10 * x5**2 + 
        a16 * x2 * x4 + 
        a17 * x2 * x5
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

# 主函数
def main():
    # 设置文件路径（替换为实际路径）
    file_path = '附件1.xlsx'
    
    try:
        # 1. 加载并预处理数据
        data = load_and_process_data(file_path)
        print(f"成功加载数据，共 {len(data)} 行")
        print("变量统计信息:")
        print(data.describe())
        
        # 2. 准备拟合数据
        X = data[['Co/SiO2', 'Co', 'HAP', '乙醇浓度', '温度']].values.T
        y = data['因变量'].values
        
        # 3. 数据标准化（解决量纲差异）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.T).T  # 对每个特征进行标准化
        
        # 4. 初始参数猜测 (21个参数)
        initial_guess = np.ones(21)
        
        # 5. 执行曲线拟合（使用标准化数据）
        print("\n开始拟合五元二次模型...")
        popt, pcov = curve_fit(quadratic_model, X_scaled, y, p0=initial_guess, maxfev=10000)
        print("拟合完成")
        
        # 6. 计算拟合指标
        y_pred = quadratic_model(X_scaled, *popt)
        r_squared = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = np.mean(np.abs(y - y_pred))
        
        # 7. 获取参数统计显著性
        perr = np.sqrt(np.diag(pcov))
        t_values = popt / perr
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), len(y) - len(popt)))
        
        # 8. 输出结果
        print("\n" + "="*60)
        print("拟合结果报告 - 五元二次方程模型")
        print("="*60)
        
        print("\n模型系数及统计显著性:")
        param_names = [
            '常数项', 
            'Co/SiO2', 'Co', 'HAP', '乙醇浓度', '温度',
            '(Co/SiO2)^2', 'Co^2', 'HAP^2', '乙醇浓度^2', '温度^2',
            'Co/SiO2*Co', 'Co/SiO2*HAP', 'Co/SiO2*乙醇浓度', 'Co/SiO2*温度',
            'Co*HAP', 'Co*乙醇浓度', 'Co*温度',
            'HAP*乙醇浓度', 'HAP*温度',
            '乙醇浓度*温度'
        ]
        
        print("\n{:<25} {:<15} {:<15} {:<15}".format("参数", "估计值", "标准误差", "p值"))
        print("-"*60)
        for i, name in enumerate(param_names):
            significance = "显著" if p_values[i] < 0.05 else "不显著"
            print("{:<25} {:<15.4f} {:<15.4f} {:<15.4f} ({})".format(
                name, popt[i], perr[i], p_values[i], significance))
        
        print("\n模型评价指标:")
        print(f"决定系数(R²): {r_squared:.4f}")
        print(f"均方根误差(RMSE): {rmse:.4f}")
        print(f"平均绝对误差(MAE): {mae:.4f}")
        
        # 9. 可视化拟合效果
        plt.figure(figsize=(12, 8))
        
        # 实际值 vs 预测值散点图
        plt.subplot(2, 2, 1)
        plt.scatter(y, y_pred, alpha=0.6, edgecolors='w')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('实际值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.title('实际值 vs 预测值', fontsize=14)
        plt.grid(alpha=0.3)
        
        # 残差图
        plt.subplot(2, 2, 2)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值', fontsize=12)
        plt.ylabel('残差', fontsize=12)
        plt.title('残差分析', fontsize=14)
        plt.grid(alpha=0.3)
        
        # 误差分布
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=20, alpha=0.7)
        plt.xlabel('残差', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.title('误差分布', fontsize=14)
        plt.grid(alpha=0.3)
        
        # 特征重要性图
        plt.subplot(2, 2, 4)
        abs_coefs = np.abs(popt[1:6])  # 主要线性项的重要性
        feature_names = ['Co/SiO2', 'Co', 'HAP', '乙醇浓度', '温度']
        plt.barh(feature_names, abs_coefs, alpha=0.7)
        plt.xlabel('系数绝对值', fontsize=12)
        plt.title('特征重要性(线性项)', fontsize=14)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quadratic_fit_results.png', dpi=300)
        print("\n拟合分析图已保存为 quadratic_fit_results.png")
        
        # 10. 保存预测结果
        data['预测值'] = y_pred
        data['残差'] = residuals
        data.to_excel('fit_results.xlsx', index=False)
        print("完整预测结果已保存为 fit_results.xlsx")
        
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请检查路径是否正确。")
    except ValueError as ve:
        print(f"数据处理错误: {str(ve)}")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
