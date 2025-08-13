import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import seaborn as sns

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

class CatalystAnalyzer:
    def __init__(self):
        """初始化催化剂分析器"""
        self.catalyst_data = {}
        self.fitted_models = {}
        
    def load_data_from_excel(self, filepath):
        """
        从Excel文件加载数据
        """
        # 这里是数据加载的示例结构
        # 实际使用时需要根据Excel文件格式调整
        try:
            # 读取Excel文件
            self.raw_data = pd.read_excel(filepath, engine='openpyxl')
            print(f"成功读取Excel文件: {filepath}")
            print(f"数据形状: {self.raw_data.shape}")
            print(f"列名: {list(self.raw_data.columns)}")
            
            # 显示前几行数据
            print("\n前5行数据预览:")
            print(self.raw_data.head())
            
            return self.raw_data
            
        except Exception as e:
            print(f"读取Excel文件时出错: {e}")
            print("将使用示例数据进行分析...")
            return None
    
    def define_catalyst_groups(self):
        """定义催化剂组合及其数据"""
        # 根据提供的数据定义每个催化剂组合
        catalyst_groups = {
            'A1': {
                'description': '200mg 1wt%Co/SiO2 + 200mg HAP + 1.68ml/min',
                'temperatures': [250, 275, 300, 325, 350],
                'ethanol_conversion': [2.07, 5.85, 14.97, 19.68, 36.80],
                'c4_selectivity': [34.05, 37.43, 46.94, 49.7, 47.21],
                'ethylene_selectivity': [1.17, 1.63, 3.02, 7.97, 12.46],
                'c4_yield': None  # 将通过计算得到
            },
            'A2': {
                'description': '200mg 2wt%Co/SiO2 + 200mg HAP + 1.68ml/min',
                'temperatures': [250, 275, 300, 325, 350],
                'ethanol_conversion': [4.60, 17.20, 38.92, 56.38, 67.88],
                'c4_selectivity': [18.07, 17.28, 19.6, 30.62, 39.1],
                'ethylene_selectivity': [0.61, 0.51, 0.85, 1.43, 2.76]
            },
            'A3': {
                'description': '200mg 1wt%Co/SiO2 + 200mg HAP + 0.9ml/min',
                'temperatures': [250, 275, 300, 325, 350, 400, 450],
                'ethanol_conversion': [9.7, 19.2, 29.3, 37.6, 48.9, 83.7, 86.4],
                'c4_selectivity': [5.5, 8.04, 17.01, 28.72, 36.85, 53.43, 49.9],
                'ethylene_selectivity': [0.13, 0.33, 0.71, 1.83, 2.85, 6.76, 14.84]
            },
            'A4': {
                'description': '200mg 0.5wt%Co/SiO2 + 200mg HAP + 1.68ml/min',
                'temperatures': [250, 275, 300, 325, 350, 400],
                'ethanol_conversion': [4.0, 12.1, 29.5, 43.3, 60.5, 88.4],
                'c4_selectivity': [9.62, 8.62, 10.72, 18.89, 27.25, 41.02],
                'ethylene_selectivity': [0.27, 0.36, 0.54, 1.02, 2.23, 3.67]
            },
            'A5': {
                'description': '200mg 2wt%Co/SiO2 + 200mg HAP + 0.3ml/min',
                'temperatures': [250, 275, 300, 325, 350, 400],
                'ethanol_conversion': [14.8, 12.4, 20.8, 28.3, 36.8, 76.0],
                'c4_selectivity': [1.96, 6.65, 10.12, 13.86, 18.75, 38.23],
                'ethylene_selectivity': [0.14, 0.19, 0.74, 1.14, 3.11, 6.38]
            },
            'A6': {
                'description': '200mg 5wt%Co/SiO2 + 200mg HAP + 1.68ml/min',
                'temperatures': [250, 275, 300, 350, 400],
                'ethanol_conversion': [13.4, 12.8, 25.5, 55.8, 83.3],
                'c4_selectivity': [3.3, 7.1, 7.18, 10.65, 37.33],
                'ethylene_selectivity': [0.2, 0.38, 1.46, 14.49, 6.18]
            },
            'A7': {
                'description': '50mg 1wt%Co/SiO2 + 50mg HAP + 0.3ml/min',
                'temperatures': [250, 275, 300, 350, 400],
                'ethanol_conversion': [19.7, 29.0, 40.0, 58.6, 76.0],
                'c4_selectivity': [5.75, 6.56, 8.84, 18.64, 33.25],
                'ethylene_selectivity': [0.18, 0.31, 0.57, 2.28, 8.31]
            },
            'A8': {
                'description': '50mg 1wt%Co/SiO2 + 50mg HAP + 0.9ml/min',
                'temperatures': [250, 275, 300, 350, 400],
                'ethanol_conversion': [6.3, 8.8, 13.2, 31.7, 56.1],
                'c4_selectivity': [5.63, 8.52, 13.82, 25.89, 41.42],
                'ethylene_selectivity': [0.14, 0.2, 0.52, 1.45, 6.4]
            }
        }
        
        # 计算C4烯烃收率
        for catalyst_id, data in catalyst_groups.items():
            conversion = np.array(data['ethanol_conversion'])
            selectivity = np.array(data['c4_selectivity'])
            data['c4_yield'] = (conversion * selectivity / 100).tolist()
        
        return catalyst_groups
    
    def fit_temperature_relationship(self, temperatures, values, model_type='polynomial'):
        """
        拟合温度与性能指标的关系
        
        Parameters:
        temperatures: 温度数据
        values: 性能指标数据
        model_type: 模型类型 ('polynomial', 'exponential', 'logarithmic', 'power')
        """
        temps = np.array(temperatures)
        vals = np.array(values)
        
        models = {}
        
        # 多项式模型 (1次、2次、3次)
        for degree in [1, 2, 3]:
            try:
                coeffs = np.polyfit(temps, vals, degree)
                poly_func = np.poly1d(coeffs)
                y_pred = poly_func(temps)
                r2 = r2_score(vals, y_pred)
                
                models[f'poly_{degree}'] = {
                    'coefficients': coeffs,
                    'function': poly_func,
                    'r2_score': r2,
                    'equation': self._get_polynomial_equation(coeffs, degree)
                }
            except:
                continue
        
        # 指数模型: y = a * exp(b * x)
        try:
            def exp_func(x, a, b):
                return a * np.exp(b * x)
            
            popt, _ = curve_fit(exp_func, temps, vals, maxfev=1000)
            y_pred = exp_func(temps, *popt)
            r2 = r2_score(vals, y_pred)
            
            models['exponential'] = {
                'parameters': popt,
                'function': lambda x: exp_func(x, *popt),
                'r2_score': r2,
                'equation': f'y = {popt[0]:.4f} * exp({popt[1]:.6f} * T)'
            }
        except:
            pass
        
        # 对数模型: y = a * ln(x) + b
        try:
            def log_func(x, a, b):
                return a * np.log(x) + b
            
            popt, _ = curve_fit(log_func, temps, vals)
            y_pred = log_func(temps, *popt)
            r2 = r2_score(vals, y_pred)
            
            models['logarithmic'] = {
                'parameters': popt,
                'function': lambda x: log_func(x, *popt),
                'r2_score': r2,
                'equation': f'y = {popt[0]:.4f} * ln(T) + {popt[1]:.4f}'
            }
        except:
            pass
        
        # 幂函数模型: y = a * x^b
        try:
            def power_func(x, a, b):
                return a * (x ** b)
            
            popt, _ = curve_fit(power_func, temps, vals, maxfev=1000)
            y_pred = power_func(temps, *popt)
            r2 = r2_score(vals, y_pred)
            
            models['power'] = {
                'parameters': popt,
                'function': lambda x: power_func(x, *popt),
                'r2_score': r2,
                'equation': f'y = {popt[0]:.4f} * T^{popt[1]:.4f}'
            }
        except:
            pass
        
        # 选择最佳模型
        best_model = max(models.items(), key=lambda x: x[1]['r2_score'])
        return models, best_model
    
    def _get_polynomial_equation(self, coeffs, degree):
        """生成多项式方程字符串"""
        terms = []
        for i, coeff in enumerate(coeffs):
            power = degree - i
            if power == 0:
                terms.append(f"{coeff:.4f}")
            elif power == 1:
                terms.append(f"{coeff:.4f}*T")
            else:
                terms.append(f"{coeff:.4f}*T^{power}")
        return "y = " + " + ".join(terms)
    
    def analyze_single_catalyst(self, catalyst_id, data):
        """分析单个催化剂组合的温度效应"""
        temperatures = data['temperatures']
        conversion = data['ethanol_conversion']
        selectivity = data['c4_selectivity']
        c4_yield = data['c4_yield']
        
        results = {}
        
        # 拟合转化率-温度关系
        conv_models, conv_best = self.fit_temperature_relationship(temperatures, conversion)
        results['conversion'] = {
            'models': conv_models,
            'best_model': conv_best
        }
        
        # 拟合选择性-温度关系
        sel_models, sel_best = self.fit_temperature_relationship(temperatures, selectivity)
        results['selectivity'] = {
            'models': sel_models,
            'best_model': sel_best
        }
        
        # 拟合收率-温度关系
        yield_models, yield_best = self.fit_temperature_relationship(temperatures, c4_yield)
        results['yield'] = {
            'models': yield_models,
            'best_model': yield_best
        }
        
        return results
    
    def plot_catalyst_analysis(self, catalyst_groups):
        """绘制所有催化剂组合的分析图"""
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        
        for idx, (catalyst_id, data) in enumerate(catalyst_groups.items()):
            if idx >= 8:
                break
                
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            temperatures = data['temperatures']
            conversion = data['ethanol_conversion']
            selectivity = data['c4_selectivity']
            c4_yield = data['c4_yield']
            
            # 拟合模型
            analysis_results = self.analyze_single_catalyst(catalyst_id, data)
            
            # 绘制原始数据点
            ax.scatter(temperatures, conversion, color='blue', label='乙醇转化率', s=50)
            ax.scatter(temperatures, selectivity, color='red', label='C4烯烃选择性', s=50)
            ax.scatter(temperatures, c4_yield, color='green', label='C4烯烃收率', s=50)
            
            # 绘制拟合曲线
            temp_smooth = np.linspace(min(temperatures), max(temperatures), 100)
            
            # 转化率拟合曲线
            conv_func = analysis_results['conversion']['best_model'][1]['function']
            ax.plot(temp_smooth, conv_func(temp_smooth), '--', color='blue', alpha=0.7)
            
            # 选择性拟合曲线
            sel_func = analysis_results['selectivity']['best_model'][1]['function']
            ax.plot(temp_smooth, sel_func(temp_smooth), '--', color='red', alpha=0.7)
            
            # 收率拟合曲线
            yield_func = analysis_results['yield']['best_model'][1]['function']
            ax.plot(temp_smooth, yield_func(temp_smooth), '--', color='green', alpha=0.7)
            
            ax.set_title(f'{catalyst_id}: {data["description"][:30]}...', fontsize=10)
            ax.set_xlabel('温度 (°C)')
            ax.set_ylabel('百分比 (%)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_analysis_report(self, catalyst_groups):
        """生成详细的分析报告"""
        print("=" * 80)
        print("问题一：催化剂组合温度效应分析报告")
        print("=" * 80)
        
        for catalyst_id, data in catalyst_groups.items():
            print(f"\n【{catalyst_id}组合分析】")
            print(f"催化剂配置: {data['description']}")
            print(f"温度范围: {min(data['temperatures'])}°C - {max(data['temperatures'])}°C")
            
            # 分析结果
            analysis_results = self.analyze_single_catalyst(catalyst_id, data)
            
            # 转化率分析
            conv_best = analysis_results['conversion']['best_model']
            print(f"\n乙醇转化率-温度关系:")
            print(f"  最佳拟合模型: {conv_best[0]} (R² = {conv_best[1]['r2_score']:.4f})")
            print(f"  拟合方程: {conv_best[1]['equation']}")
            
            # 选择性分析
            sel_best = analysis_results['selectivity']['best_model']
            print(f"\nC4烯烃选择性-温度关系:")
            print(f"  最佳拟合模型: {sel_best[0]} (R² = {sel_best[1]['r2_score']:.4f})")
            print(f"  拟合方程: {sel_best[1]['equation']}")
            
            # 收率分析
            yield_best = analysis_results['yield']['best_model']
            print(f"\nC4烯烃收率-温度关系:")
            print(f"  最佳拟合模型: {yield_best[0]} (R² = {yield_best[1]['r2_score']:.4f})")
            print(f"  拟合方程: {yield_best[1]['equation']}")
            
            # 性能特征分析
            temperatures = data['temperatures']
            conversion = data['ethanol_conversion']
            selectivity = data['c4_selectivity']
            c4_yield = data['c4_yield']
            
            print(f"\n性能特征:")
            print(f"  最高转化率: {max(conversion):.2f}% @ {temperatures[conversion.index(max(conversion))]}°C")
            print(f"  最高选择性: {max(selectivity):.2f}% @ {temperatures[selectivity.index(max(selectivity))]}°C")
            print(f"  最高收率: {max(c4_yield):.2f}% @ {temperatures[c4_yield.index(max(c4_yield))]}°C")
            
            print("-" * 60)
    
    def analyze_time_series_data(self, time_data=None):
        """分析附件2中350°C时的时间序列数据"""
        print("\n" + "=" * 80)
        print("附件2：350°C时催化剂时间效应分析")
        print("=" * 80)
        
        # 示例时间序列数据（实际需要从附件2获取）
        if time_data is None:
            # 这里应该是从附件2读取的实际数据
            time_data = {
                'times': [0, 30, 60, 90, 120, 150, 180],  # 分钟
                'ethanol_conversion': [25.0, 35.2, 42.1, 38.7, 36.5, 34.2, 33.8],
                'c4_selectivity': [45.2, 48.3, 46.7, 44.1, 42.8, 41.5, 40.9],
                'catalyst_description': '350°C时给定催化剂组合'
            }
        
        times = time_data['times']
        conversion = time_data['ethanol_conversion']
        selectivity = time_data['c4_selectivity']
        
        # 计算收率
        c4_yield = [(c*s/100) for c, s in zip(conversion, selectivity)]
        
        print(f"催化剂配置: {time_data['catalyst_description']}")
        print(f"反应温度: 350°C")
        print(f"测试时间范围: {min(times)} - {max(times)} 分钟")
        
        # 绘制时间序列图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 转化率随时间变化
        ax1.plot(times, conversion, 'bo-', linewidth=2, markersize=6)
        ax1.set_title('乙醇转化率随时间变化')
        ax1.set_ylabel('转化率 (%)')
        ax1.grid(True, alpha=0.3)
        
        # 选择性随时间变化
        ax2.plot(times, selectivity, 'ro-', linewidth=2, markersize=6)
        ax2.set_title('C4烯烃选择性随时间变化')
        ax2.set_ylabel('选择性 (%)')
        ax2.grid(True, alpha=0.3)
        
        # 收率随时间变化
        ax3.plot(times, c4_yield, 'go-', linewidth=2, markersize=6)
        ax3.set_title('C4烯烃收率随时间变化')
        ax3.set_xlabel('时间 (min)')
        ax3.set_ylabel('收率 (%)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 分析催化剂稳定性
        print(f"\n时间效应分析:")
        print(f"  初始转化率: {conversion[0]:.2f}%")
        print(f"  最高转化率: {max(conversion):.2f}% @ {times[conversion.index(max(conversion))]}min")
        print(f"  最终转化率: {conversion[-1]:.2f}%")
        print(f"  转化率变化范围: {max(conversion) - min(conversion):.2f}%")
        
        print(f"  初始选择性: {selectivity[0]:.2f}%")
        print(f"  最高选择性: {max(selectivity):.2f}% @ {times[selectivity.index(max(selectivity))]}min")
        print(f"  最终选择性: {selectivity[-1]:.2f}%")
        print(f"  选择性变化范围: {max(selectivity) - min(selectivity):.2f}%")
        
        print(f"  最高收率: {max(c4_yield):.2f}% @ {times[c4_yield.index(max(c4_yield))]}min")
        
        # 稳定性评估
        if len(times) > 2:
            # 计算后期稳定性（后一半时间点的标准差）
            mid_point = len(times) // 2
            late_conversion_std = np.std(conversion[mid_point:])
            late_selectivity_std = np.std(selectivity[mid_point:])
            
            print(f"\n稳定性评估（后期标准差）:")
            print(f"  转化率稳定性: ±{late_conversion_std:.2f}%")
            print(f"  选择性稳定性: ±{late_selectivity_std:.2f}%")
        
        return fig

# 主分析函数
def main_analysis():
    """执行问题一的完整分析"""
    analyzer = CatalystAnalyzer()
    
    # 1. 定义催化剂组合数据
    catalyst_groups = analyzer.define_catalyst_groups()
    
    # 2. 绘制所有催化剂的分析图
    print("正在生成催化剂组合分析图...")
    analyzer.plot_catalyst_analysis(catalyst_groups)
    
    # 3. 生成详细分析报告
    analyzer.generate_analysis_report(catalyst_groups)
    
    # 4. 分析时间序列数据（附件2）
    analyzer.analyze_time_series_data()
    
    print("\n问题一分析完成！")

if __name__ == "__main__":
    main_analysis()