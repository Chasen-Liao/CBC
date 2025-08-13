import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# 1. 数据读取与预处理
def load_and_preprocess_data(file_path):
    """
    从Excel文件加载数据并进行预处理
    返回包含特征和目标变量的DataFrame
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 数据清洗：删除不相关的列
    columns_to_keep = [
        'Co负载量(wt%)', 'Co/SiO2质量', 'HAP质量(mg)', 
        '乙醇浓度(ml/min)', '温度', 'C4烯烃收率'
    ]
    df = df[columns_to_keep]
    
    # 创建新特征：装料比 = Co/SiO2质量 / HAP质量
    df['装料比'] = df['Co/SiO2质量'] / df['HAP质量(mg)']
    
    # 删除原始质量列，保留装料比
    df = df.drop(['Co/SiO2质量', 'HAP质量(mg)'], axis=1)
    
    # 处理缺失值（如果有）
    df = df.dropna()
    
    return df

# 2. 特征工程
def create_features(df):
    """
    创建多项式特征和交互项
    返回特征矩阵X和目标向量y
    """
    # 选择特征列
    feature_columns = ['Co负载量(wt%)', '乙醇浓度(ml/min)', '装料比', '温度']
    
    # 创建多项式特征（二次项和交互项）
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(df[feature_columns])
    
    # 获取特征名称
    feature_names = poly.get_feature_names_out(feature_columns)
    
    # 创建特征DataFrame
    X = pd.DataFrame(X_poly, columns=feature_names)
    y = df['C4烯烃收率'].values
    
    return X, y

# 3. 建立回归模型
def build_regression_model(X, y):
    """
    建立并训练多项式回归模型
    返回训练好的模型
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建并训练回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 评估模型
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"模型评估: 训练集R² = {train_score:.3f}, 测试集R² = {test_score:.3f}")
    
    return model

# 4. 优化函数
def optimize_yield(model, feature_names, temperature_constraint=None):
    """
    使用回归模型优化C4烯烃收率
    temperature_constraint: 温度上限约束
    """
    # 定义目标函数（负收率，因为minimize求解最小值）
    def objective(x):
        # 创建特征矩阵
        X_pred = pd.DataFrame([x], columns=feature_names)
        
        # 预测收率（返回负值因为我们要最大化）
        return -model.predict(X_pred)[0]
    
    # 设置变量边界（基于数据范围）
    bounds = [
        (0.5, 5),      # Co负载量(wt%)
        (0.3, 2.1),    # 乙醇浓度(ml/min)
        (0.1, 10),     # 装料比
        (250, 450)     # 温度
    ]
    
    # 设置约束条件
    constraints = []
    if temperature_constraint is not None:
        # 添加温度约束：温度 <= temperature_constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: temperature_constraint - x[3]  # 温度是第四个变量
        })
    
    # 初始猜测（使用数据的平均值）
    initial_guess = [2.5, 1.2, 1.0, 350]
    
    # 执行优化
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',  # 适用于约束优化
        bounds=bounds,
        constraints=constraints
    )
    
    # 解析结果
    if result.success:
        optimal_params = result.x
        max_yield = -result.fun
        print(f"优化成功! 最大收率: {max_yield:.2f}")
        print(f"最优参数: Co负载量={optimal_params[0]:.2f}wt%, 乙醇浓度={optimal_params[1]:.2f}ml/min, "
              f"装料比={optimal_params[2]:.2f}, 温度={optimal_params[3]:.2f}°C")
        return optimal_params, max_yield
    else:
        print("优化失败:", result.message)
        return None, None

# 5. 可视化函数
def visualize_results(df, model, feature_names, optimal_params):
    """
    可视化模型预测结果和优化点
    """
    # 创建3D图展示温度和装料比的影响
    fig = plt.figure(figsize=(12, 8))
    
    # 固定其他参数在最优值
    fixed_co = optimal_params[0]  # Co负载量
    fixed_ethanol = optimal_params[1]  # 乙醇浓度
    
    # 创建网格数据
    temp_range = np.linspace(250, 450, 50)
    ratio_range = np.linspace(0.1, 10, 50)
    T, R = np.meshgrid(temp_range, ratio_range)
    
    # 计算预测值
    Z = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            features = [fixed_co, fixed_ethanol, R[i, j], T[i, j]]
            X_pred = pd.DataFrame([features], columns=feature_names)
            Z[i, j] = model.predict(X_pred)[0]
    
    # 绘制3D表面图
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, R, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    
    # 标记最优解
    ax.scatter(optimal_params[3], optimal_params[2], -optimal_params[0], 
               color='red', s=100, label='最优解')
    
    # 添加标签
    ax.set_xlabel('温度 (°C)')
    ax.set_ylabel('装料比')
    ax.set_zlabel('C4烯烃收率')
    ax.set_title('C4烯烃收率优化')
    plt.legend()
    plt.show()

# 主函数
def main():
    # 指定Excel文件路径
    excel_path = '附件1.xlsx'  # 替换为实际文件路径
    
    try:
        # 1. 加载并预处理数据
        print("步骤1: 加载和预处理数据...")
        df = load_and_preprocess_data(excel_path)
        print(f"数据加载成功! 共 {len(df)} 条记录")
        
        # 2. 特征工程
        print("\n步骤2: 创建多项式特征...")
        X, y = create_features(df)
        print(f"创建的特征: {list(X.columns)}")
        
        # 3. 建立回归模型
        print("\n步骤3: 训练回归模型...")
        model = build_regression_model(X, y)
        
        # 4. 全局优化（无温度约束）
        print("\n步骤4: 全局优化 - 最大化C4烯烃收率")
        optimal_global, max_yield_global = optimize_yield(model, list(X.columns))
        
        # 5. 温度约束优化（温度<350°C）
        print("\n步骤5: 约束优化 - 温度<350°C")
        optimal_constrained, max_yield_constrained = optimize_yield(
            model, list(X.columns), temperature_constraint=350
        )
        
        # 6. 结果可视化
        print("\n步骤6: 可视化结果...")
        visualize_results(df, model, list(X.columns), optimal_global)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()