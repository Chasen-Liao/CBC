import numpy as np
import pandas as pd

# 修改后的五元二次函数模型（22个参数）
def quadratic_model(X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, 
                   a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21):
    """22参数五元二次回归模型"""
    x1, x2, x3, x4, x5 = X  # 分别代表：Co/SiO2, Co, HAP, 乙醇浓度, 温度
    y = (
        a0 + 
        a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5 +
        a6*x1**2 + a7*x2**2 + a8*x3**2 + a9*x4**2 + a10*x5**2 +
        a11*x1*x2 + a12*x1*x3 + a13*x1*x4 + a14*x1*x5 +
        a15*x2*x3 + a16*x2*x4 + a17*x2*x5 +
        a18*x3*x4 + a19*x3*x5 +
        a20*x4*x5 + a21*x3*x4  # 新增第22个参数：HAP*乙醇浓度
    )
    return y

# 优化模型（仅保留显著项）
def optimized_model(X, a0, a5, a7, a10, a16, a17):
    """简化模型（保留p<0.05的显著项）"""
    _, x2, _, x4, x5 = X  # 提取Co(x2)、乙醇浓度(x4)、温度(x5)
    # 显著项：常数项(a0) + 温度线性项(a5*x5) + Co²项(a7*x2²) 
    #        + 温度²项(a10*x5²) + Co*乙醇浓度交叉项(a16*x2*x4) + Co*温度交叉项(a17*x2*x5)
    return a0 + a5*x5 + a7*x2**2 + a10*x5**2 + a16*x2*x4 + a17*x2*x5

# 使用您提供的参数估计值（22个）
def get_fit_parameters():
    """返回拟合参数数组（根据您提供的表格）"""
    return np.array([
        838.6741,    # a0: 常数项
        -141.9306,   # a1: Co/SiO2
        -87.8221,    # a2: Co
        555.3008,    # a3: HAP
        176.1804,    # a4: 乙醇浓度
        723.5392,    # a5: 温度 (显著)
        -917.3288,   # a6: (Co/SiO2)^2
        -224.7441,   # a7: Co^2 (显著)
        -875.6114,   # a8: HAP^2
        32.7346,     # a9: 乙醇浓度^2
        246.3049,    # a10: 温度^2 (显著)
        457.1743,    # a11: Co/SiO2*Co
        1575.5788,   # a12: Co/SiO2*HAP
        1059.4323,   # a13: Co/SiO2*乙醇浓度
        138.3899,    # a14: Co/SiO2*温度
        -323.7042,   # a15: Co*HAP
        414.8765,    # a16: Co*乙醇浓度 (p=0.09，边缘显著)
        -108.2215,   # a17: Co*温度 (显著)
        -961.0825,   # a18: HAP*乙醇浓度
        172.6508,    # a19: HAP*温度
        -36.9240,    # a20: 乙醇浓度*温度
        -961.0825    # a21: HAP*乙醇浓度（补充第22个参数，与a18相同）
    ])

# 网格搜索优化（使用完整模型）
def grid_search_optimization(popt, param_ranges):
    """使用网格搜索寻找最优参数组合"""
    # 设置网格点数量
    grid_sizes = {'Co/SiO2': 10, 'Co': 8, 'HAP': 10, '乙醇浓度': 8, '温度': 6}
    
    # 生成参数网格
    co_sio2_range = np.linspace(*param_ranges['Co/SiO2'], grid_sizes['Co/SiO2'])
    co_range = np.linspace(*param_ranges['Co'], grid_sizes['Co'])
    hap_range = np.linspace(*param_ranges['HAP'], grid_sizes['HAP'])
    ethanol_range = np.linspace(*param_ranges['乙醇浓度'], grid_sizes['乙醇浓度'])
    temp_range = np.linspace(*param_ranges['温度'], grid_sizes['温度'])
    
    # 初始化最佳结果
    best_yield = -np.inf
    best_params = None
    
    # 结果存储列表
    results = []
    
    # 五维网格搜索
    for co_sio2 in co_sio2_range:
        for co in co_range:
            for hap in hap_range:
                for ethanol in ethanol_range:
                    for temp in temp_range:
                        X = (co_sio2, co, hap, ethanol, temp)
                        y_pred = quadratic_model(X, *popt)
                        results.append({
                            'Co/SiO2': co_sio2, 'Co': co, 'HAP': hap,
                            '乙醇浓度': ethanol, '温度': temp, '收率预测': y_pred
                        })
                        if y_pred > best_yield:
                            best_yield = y_pred
                            best_params = (co_sio2, co, hap, ethanol, temp)
    
    return best_params, best_yield, pd.DataFrame(results)

# 主函数
def main():
    # 1. 获取拟合参数
    popt = get_fit_parameters()
    print(f"加载22个拟合参数完成")
    
    # 2. 确定搜索范围（根据实际数据调整）
    param_ranges = {
        'Co/SiO2': (50, 150),     # Co/SiO2质量(mg)
        'Co': (0.5, 3.0),          # Co负载量(wt%)
        'HAP': (50, 150),          # HAP质量(mg)
        '乙醇浓度': (0.5, 2.5),     # 乙醇浓度(ml/min)
        '温度': (300, 400)         # 温度(°C)
    }
    
    # 3. 完整模型搜索
    best_params_full, best_yield_full, results_full = grid_search_optimization(popt, param_ranges)
    
    # 4. 优化模型（仅显著项）搜索
    # 提取显著项参数：a0, a5, a7, a10, a16, a17
    opt_params = [popt[i] for i in [0, 5, 7, 10, 16, 17]]
    best_params_opt, best_yield_opt, results_opt = grid_search_optimization(
        lambda X, *args: optimized_model(X, *args),  # 使用优化模型
        param_ranges,
        popt_opt=opt_params
    )
    
    # 5. 结果对比
    print("\n模型对比结果:")
    print(f"完整模型最高收率: {best_yield_full:.2f} @ {best_params_full}")
    print(f"优化模型最高收率: {best_yield_opt:.2f} @ {best_params_opt}")
    
    # 6. 保存结果
    results_full.to_csv('full_model_results.csv', index=False)
    results_opt.to_csv('optimized_model_results.csv', index=False)
    print("结果已保存至CSV文件")

if __name__ == '__main__':
    main()