#!/usr/bin/env python3
# Reproduce Problem 3 analysis: polynomial regression and single-objective optimization
# Requires: pandas, numpy, scikit-learn, scipy, matplotlib (for plotting if needed)
import pandas as pd, numpy as np, re
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

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
        if m:
            return float(m.group(0))
        else:
            return np.nan

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
        model = LinearRegression().fit(Xpoly, y)
        y_pred = model.predict(Xpoly)
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
            # coarse grid fallback
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
            '最佳预测C4收率(%)': y_opt,
            '最佳变量(原始单位)': '; '.join([f'{p}={v:.6f}' for p,v in zip(predictor_cols, x_opt)]),
            '最佳预测C4收率_T<=350(%)': y_opt_350,
            '最佳变量_T<=350': '; '.join([f'{p}={v:.6f}' for p,v in zip(predictor_cols, x_opt_350)])
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False)
    return summary_df

if __name__=='__main__':
    print('Running analysis on 性能数据表_结构化.xlsx ...')
    df_summary = run_analysis('性能数据表_结构化.xlsx', out_csv='optimization_summary.csv')
    print(df_summary.to_string(index=False))
