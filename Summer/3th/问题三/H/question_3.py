#!/usr/bin/env python3
# Problem 3 analysis with LassoCV feature selection and single-objective optimization
import pandas as pd, numpy as np, re
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

# ===== 安全转换函数 =====
def to_float_safe(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except:
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else np.nan

# ===== 主函数 =====
def run_analysis(excel_path, out_csv='optimization_summary.csv'):
    df = pd.read_excel(excel_path)
    # 计算 C4 收率
    df['C4收率(%)'] = df['乙醇转化率(%)'] * df['C4烯烃选择性(%)'] / 100.0

    predictor_cols = ['Co/SiO2质量', 'Co负载量(wt%)', 'HAP质量(mg)', '乙醇浓度(ml/min)', '温度']
    for c in predictor_cols:
        df[c] = df[c].apply(to_float_safe)

    modes = df['装料方式'].unique().tolist()
    summary_rows = []

    for mode in modes:
        dfm = df[df['装料方式'] == mode].dropna(subset=predictor_cols + ['C4收率(%)']).copy()
        if len(dfm) < 5:
            continue

        X = dfm[predictor_cols].astype(float)
        y = dfm['C4收率(%)'].astype(float)

        # 标准化
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # 多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xpoly = poly.fit_transform(Xs)
        feature_names = poly.get_feature_names_out(predictor_cols)

        # ===== LassoCV 特征筛选 =====
        lasso = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50), max_iter=10000).fit(Xpoly, y)
        mask = lasso.coef_ != 0
        selected_features = [name for name, m in zip(feature_names, mask) if m]

        # ===== 在筛选后的特征上重新回归 =====
        Xpoly_sel = Xpoly[:, mask]
        model = LinearRegression().fit(Xpoly_sel, y)
        y_pred = model.predict(Xpoly_sel)

        # 回归方程
        equation = f"C4收率 = {model.intercept_:.4f}"
        for name, coef in zip(selected_features, model.coef_):
            if coef >= 0:
                equation += f" + {coef:.4f}*{name}"
            else:
                equation += f" - {abs(coef):.4f}*{name}"

        # 模型精度
        r2 = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred) ** 0.5

        # ===== 优化函数（在原始变量空间） =====
        def predict_from_raw(x_raw):
            arr = np.array(x_raw).reshape(1, -1)
            arr_s = scaler.transform(arr)
            arr_p = poly.transform(arr_s)[:, mask]  # 只保留筛选特征
            return float(model.predict(arr_p)[0])

        def objective(x_raw):
            return -predict_from_raw(x_raw)

        bounds = [(float(X[c].min()), float(X[c].max())) for c in predictor_cols]
        
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
        res = minimize(objective, x0=x0, bounds=bounds, method='SLSQP', options={'maxiter': 1000})
        if not res.success:
            grid_steps = 25
            grids = [np.linspace(b[0], b[1], grid_steps) for b in bounds]
            mesh = np.meshgrid(*grids)
            mesh_flat = np.stack([m.ravel() for m in mesh], axis=1)
            preds = np.array([predict_from_raw(row) for row in mesh_flat])
            idx = preds.argmax()
            x_opt = mesh_flat[idx]
            y_opt = preds[idx]
        else:
            x_opt = res.x
            y_opt = -res.fun

        # 最优温度
        best_temp = x_opt[-1]

        # 温度 ≤ 350 限制
        bounds_T350 = bounds.copy()
        bounds_T350[-1] = (bounds_T350[-1][0], min(bounds_T350[-1][1], 350.0))
        x0_2 = [(b[0] + b[1]) / 2 for b in bounds_T350]
        res2 = minimize(objective, x0=x0_2, bounds=bounds_T350, method='SLSQP', options={'maxiter': 1000})
        if not res2.success:
            grid_steps = 25
            grids2 = [np.linspace(b[0], b[1], grid_steps) for b in bounds_T350]
            mesh2 = np.meshgrid(*grids2)
            mesh_flat2 = np.stack([m.ravel() for m in mesh2], axis=1)
            preds2 = np.array([predict_from_raw(row) for row in mesh_flat2])
            idx2 = preds2.argmax()
            x_opt_350 = mesh_flat2[idx2]
            y_opt_350 = preds2[idx2]
        else:
            x_opt_350 = res2.x
            y_opt_350 = -res2.fun

        best_temp_350 = x_opt_350[-1]

        # ===== 保存结果 =====
        # 保存变量范围信息
        var_ranges = '; '.join([f'{p}=[{b[0]:.4f}, {b[1]:.4f}]' for p, b in zip(predictor_cols, bounds)])
        
        summary_rows.append({
            '装料方式': mode,
            '样本数': len(y),
            'R2': r2,
            'RMSE': rmse,
            '筛选特征数': len(selected_features),
            '筛选特征': ', '.join(selected_features),
            '模型方程': equation,
            '变量范围': var_ranges,  # 添加变量范围信息
            '最佳预测C4收率(%)': y_opt,
            '最佳温度(°C)': best_temp,
            '最佳变量(原始单位)': '; '.join([f'{p}={v:.6f}' for p, v in zip(predictor_cols, x_opt)]),
            '最佳预测C4收率_T<=350(%)': y_opt_350,
            '最佳温度_T<=350(°C)': best_temp_350,
            '最佳变量_T<=350': '; '.join([f'{p}={v:.6f}' for p, v in zip(predictor_cols, x_opt_350)])
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False)
    return summary_df

def format_output(df):
    """美化输出格式，使结果更加清晰易读"""
    # 设置pandas显示选项
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 120)
    
    # 创建美化后的输出
    result = "" + "="*80 + ""
    result += "特征筛选多项式回归模型与单目标优化结果" + "="*80 + ""
    
    for _, row in df.iterrows():
        result += f"【装料方式 {row['装料方式']}】"
        result += f"{'-'*80}"
        
        # 基本信息部分
        result += f"📊 基本统计信息:\n"
        result += f"  • 样本数: {row['样本数']}\n"
        result += f"  • R² 值: {row['R2']:.6f}\n"
        result += f"  • RMSE: {row['RMSE']:.6f}\n"
        
        # 特征筛选信息
        result += f"🔍 特征筛选结果:\n"
        result += f"  • 筛选特征数: {row['筛选特征数']}\n"
        
        # 将特征列表格式化为多行，每行最多2个特征
        # 处理可能的特殊情况，如特征名称中包含逗号
        features_text = row['筛选特征']
        # 先尝试按照标准模式分割
        features = []
        current_feature = ""
        depth = 0
        
        for char in features_text:
            if char == ',' and depth == 0:
                if current_feature.strip():
                    features.append(current_feature.strip())
                current_feature = ""
            else:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                current_feature += char
        
        if current_feature.strip():
            features.append(current_feature.strip())
        
        # 格式化显示
        result += "  • 筛选特征列表:\n"
        for i in range(0, len(features), 2):
            group = features[i:i+2]
            result += f"      {', '.join(group)}"
        result += ""
        
        # 模型方程 - 格式化为多行，美化显示
        equation = row['模型方程']
        
        # 处理负号，确保正确分割
        equation = equation.replace(' + -', ' - ')
        
        # 分割方程
        if ' - ' in equation:
            parts = []
            for part in equation.split(' + '):
                if ' - ' in part:
                    subparts = part.split(' - ')
                    parts.append(subparts[0])
                    for sp in subparts[1:]:
                        parts.append(f"-{sp}")
                else:
                    parts.append(part)
        else:
            parts = equation.split(' + ')
        
        # 格式化方程
        formatted_eq = parts[0]  # 常数项
        for part in parts[1:]:
            if part.startswith('-'):
                formatted_eq += f"- {part[1:]}"
            else:
                formatted_eq += f"+ {part}"
        
        # 美化方程显示
        formatted_eq = formatted_eq.replace('*', ' × ')
        formatted_eq = formatted_eq.replace('^2', '²')
        # 修复可能的双重负号
        formatted_eq = formatted_eq.replace('- -', '+ ')
        
        result += f"📝 模型方程:{formatted_eq}\n"
        
        # 显示变量范围信息
        result += f"📏 变量范围:\n"
        var_ranges = row['变量范围'].split('; ')
        for var_range in var_ranges:
            name, range_str = var_range.split('=')
            result += f"  • {name}: {range_str}\n"
        result += "\n"
        
        # 最佳预测结果
        result += f"🎯 最佳预测结果:\n"
        result += f"  • C4收率(%): {row['最佳预测C4收率(%)']:.4f}\n"
        result += f"  • 最佳温度(°C): {row['最佳温度(°C)']:.1f}\n"
        
        # 格式化最佳变量，使其更易读
        vars_original = row['最佳变量(原始单位)'].split('; ')
        result += "  • 最佳变量组合:\n"
        for var in vars_original:
            name, value = var.split('=')
            # 根据数值大小决定保留的小数位数
            try:
                float_val = float(value)
                if float_val >= 100:
                    formatted_val = f"{float_val:.2f}"
                elif float_val >= 10:
                    formatted_val = f"{float_val:.3f}"
                else:
                    formatted_val = f"{float_val:.4f}"
                result += f"      {name} = {formatted_val}"
            except:
                result += f"      {name} = {value}"
        result += ""
        
        # 温度限制条件下的最佳预测结果
        result += f"🌡️ 温度限制条件下(T≤350°C)的最佳预测结果:\n"
        result += f"  • C4收率(%): {row['最佳预测C4收率_T<=350(%)']:.4f}\n"
        result += f"  • 最佳温度(°C): {row['最佳温度_T<=350(°C)']:.1f}\n"
        
        # 格式化温度限制下的最佳变量
        vars_t350 = row['最佳变量_T<=350'].split('; ')
        result += "  • 最佳变量组合:"
        for var in vars_t350:
            name, value = var.split('=')
            try:
                float_val = float(value)
                if float_val >= 100:
                    formatted_val = f"{float_val:.2f}"
                elif float_val >= 10:
                    formatted_val = f"{float_val:.3f}"
                else:
                    formatted_val = f"{float_val:.4f}"
                result += f"      {name} = {formatted_val}"
            except:
                result += f"      {name} = {value}"
        
        result += "" + "="*80 + ""
    
    return result

if __name__ == '__main__':
    excel_path = '性能数据表_结构化.xlsx'
    out_csv = 'optimization_summary.csv'
    print(f'Running analysis on {excel_path} ...')
    df_summary = run_analysis(excel_path, out_csv)
    
    # 打印美化后的结果
    print(format_output(df_summary))
