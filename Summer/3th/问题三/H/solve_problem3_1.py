#!/usr/bin/env python3
# Reproduce Problem 3 analysis: polynomial regression and single-objective optimization
# Requires: pandas, numpy, scikit-learn, scipy, matplotlib (for plotting if needed)
import pandas as pd, numpy as np, re
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

# 定义安全转换函数
def to_float_safe(x):
    # 处理NaN值
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except:
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        if m:
            return float(m.group(0))
        else:
            return np.nan

# 主函数
def run_analysis(excel_path, out_csv='optimization_summary.csv'):
    df = pd.read_excel(excel_path)
    df['C4收率(%)'] = df['乙醇转化率(%)'] * df['C4烯烃选择性(%)'] / 100.0
    predictor_cols = ['Co/SiO2质量', 'Co负载量(wt%)', 'HAP质量(mg)', '乙醇浓度(ml/min)', '温度']
    for c in predictor_cols:
        df[c] = df[c].apply(to_float_safe)
    modes = df['装料方式'].unique().tolist()
    summary_rows = []
    for mode in modes:
        dfm = df[df['装料方式']==mode].dropna(subset=predictor_cols+['C4收率(%)']).copy()
        if len(dfm) < 5:
            continue
        X = dfm[predictor_cols].astype(float)
        y = dfm['C4收率(%)'].astype(float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xpoly = poly.fit_transform(Xs)
        # 获取特征名称
        feature_names = poly.get_feature_names_out(predictor_cols)
        model = LinearRegression().fit(Xpoly, y)
        y_pred = model.predict(Xpoly)
        # 生成模型方程字符串
        equation = "C4收率 = {:.4f}".format(model.intercept_)
        for name, coef in zip(feature_names, model.coef_):
            equation += " + {:.4f}*{}".format(coef, name)
        r2 = r2_score(y, y_pred)
        rmse = (mean_squared_error(y, y_pred))**0.5
        bounds = [(float(X[c].min()), float(X[c].max())) for c in predictor_cols]
        def predict_from_raw(x_raw):
            arr = np.array(x_raw).reshape(1,-1)
            arr_s = scaler.transform(arr)
            arr_p = poly.transform(arr_s)
            return float(model.predict(arr_p)[0])
        def objective(x_raw):
            return -predict_from_raw(x_raw)
        x0 = [(b[0]+b[1])/2 for b in bounds]
        res = minimize(objective, x0=x0, bounds=bounds, method='SLSQP', options={'maxiter':1000})
        if not res.success:
            # 网格搜索
            grid_steps=25
            grids=[np.linspace(b[0], b[1], grid_steps) for b in bounds]
            mesh=np.meshgrid(*grids)
            mesh_flat=np.stack([m.ravel() for m in mesh], axis=1)
            preds=np.array([predict_from_raw(row) for row in mesh_flat])
            idx=preds.argmax()
            x_opt=mesh_flat[idx]; y_opt=preds[idx]; opt_msg='grid_fallback'
        else:
            x_opt=res.x; y_opt=-res.fun; opt_msg=res.message
        # temperature constrained to <=350
        bounds_T350 = bounds.copy(); bounds_T350[-1]=(bounds_T350[-1][0], min(bounds_T350[-1][1], 350.0))
        x0_2 = [(b[0]+b[1])/2 for b in bounds_T350]
        res2 = minimize(objective, x0=x0_2, bounds=bounds_T350, method='SLSQP', options={'maxiter':1000})
        if not res2.success:
            grid_steps=25
            grids2=[np.linspace(b[0], b[1], grid_steps) for b in bounds_T350]
            mesh2=np.meshgrid(*grids2)
            mesh_flat2=np.stack([m.ravel() for m in mesh2], axis=1)
            preds2=np.array([predict_from_raw(row) for row in mesh_flat2])
            idx2=preds2.argmax()
            x_opt_350=mesh_flat2[idx2]; y_opt_350=preds2[idx2]; opt_msg2='grid_fallback'
        else:
            x_opt_350=res2.x; y_opt_350=-res2.fun; opt_msg2=res2.message
        summary_rows.append({
            '装料方式': mode,
            '样本数': len(y),
            'R2': r2,
            'RMSE': rmse,
            '模型方程': equation,
            '最佳预测C4收率(%)': y_opt,
            '最佳变量(原始单位)': '; '.join([f'{p}={v:.6f}' for p,v in zip(predictor_cols, x_opt)]),
            '最佳预测C4收率_T<=350(%)': y_opt_350,
            '最佳变量_T<=350': '; '.join([f'{p}={v:.6f}' for p,v in zip(predictor_cols, x_opt_350)])
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False)
    return summary_df

def format_output(df):
    """格式化输出结果，使其更加清晰易读"""
    # 设置pandas显示选项
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 200)
    
    # 创建一个更易读的表格
    formatted_df = df.copy()
    
    # 格式化数值列
    formatted_df['R2'] = formatted_df['R2'].apply(lambda x: f"{x:.6f}")
    formatted_df['RMSE'] = formatted_df['RMSE'].apply(lambda x: f"{x:.6f}")
    formatted_df['最佳预测C4收率(%)'] = formatted_df['最佳预测C4收率(%)'].apply(lambda x: f"{x:.6f}")
    formatted_df['最佳预测C4收率_T<=350(%)'] = formatted_df['最佳预测C4收率_T<=350(%)'].apply(lambda x: f"{x:.6f}")
    
    # 格式化模型方程，使其更加清晰易读
    def format_equation(eq):
        # 将方程分成多行，每行一个项
        eq = eq.replace(' + ', '\n    + ')
        # 替换乘法符号，使其更易读
        eq = eq.replace('*', ' × ')
        # 美化平方符号
        eq = eq.replace('^2', '²')
        return eq
    
    formatted_df['模型方程'] = formatted_df['模型方程'].apply(format_equation)
    
    # 格式化最佳变量，使其更加清晰易读
    def format_variables(var_str):
        # 将变量分成多行
        var_list = var_str.split('; ')
        # 对每个变量进行格式化，保留更少的小数位
        formatted_vars = []
        for var in var_list:
            name, value = var.split('=')
            try:
                # 尝试将值转换为浮点数并格式化
                float_val = float(value)
                # 根据数值大小决定保留的小数位数
                if float_val >= 100:
                    formatted_val = f"{float_val:.2f}"
                elif float_val >= 10:
                    formatted_val = f"{float_val:.3f}"
                else:
                    formatted_val = f"{float_val:.4f}"
                formatted_vars.append(f"{name} = {formatted_val}")
            except:
                formatted_vars.append(f"{name} = {value}")
        return '\n    '.join(formatted_vars)
    
    formatted_df['最佳变量(原始单位)'] = formatted_df['最佳变量(原始单位)'].apply(format_variables)
    formatted_df['最佳变量_T<=350'] = formatted_df['最佳变量_T<=350'].apply(format_variables)
    
    # 打印每个装料方式的结果
    result = "\n" + "="*100 + "\n"
    result += "多项式回归模型与单目标优化结果\n" + "="*100 + "\n\n"
    
    for _, row in formatted_df.iterrows():
        result += f"【装料方式 {row['装料方式']}】\n"
        result += f"{'-'*80}\n\n"
        
        # 基本信息
        result += f"📊 基本统计信息:\n"
        result += f"  • 样本数: {row['样本数']}\n"
        result += f"  • R² 值: {row['R2']}\n"
        result += f"  • RMSE: {row['RMSE']}\n\n"
        
        # 模型方程
        result += f"📝 模型方程:\n"
        result += f"{row['模型方程']}\n\n"
        
        # 最佳预测结果
        result += f"🔍 最佳预测结果:\n"
        result += f"  • C4收率(%): {row['最佳预测C4收率(%)']}\n"
        result += f"  • 最佳变量组合:\n    {row['最佳变量(原始单位)']}\n\n"
        
        # 温度限制条件下的最佳预测结果
        result += f"🌡️ 温度限制条件下(T≤350°C)的最佳预测结果:\n"
        result += f"  • C4收率(%): {row['最佳预测C4收率_T<=350(%)']}\n"
        result += f"  • 最佳变量组合:\n    {row['最佳变量_T<=350']}\n\n"
        
        result += "="*100 + "\n\n"
    
    return result

if __name__=='__main__':
    print('Running analysis on 性能数据表_结构化.xlsx ...')
    df_summary = run_analysis('性能数据表_结构化.xlsx', out_csv='optimization_summary.csv')
    
    # 打印格式化的结果
    print(format_output(df_summary))
