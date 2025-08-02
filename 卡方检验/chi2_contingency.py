import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import PermutationMethod


# 构造列联表数据

# 研究"表面风化"和"纹路"之间是否相关
'''
行: 表面风化(是/否)
列: 纹饰(A/B/C)
'''
contingency_table_1 = np.array([
    [11, 11],
    [0, 6],
    [11, 17]
])

# 研究"表面风化"和"颜色"
'''
行: 表面风化(是/否)
列: 颜色(浅绿/浅蓝/深绿/深蓝/紫/绿/蓝绿/黑)
'''
contingency_table_2 = np.array([
    [2, 1],
    [6, 12],
    [3, 4],
    [2, 0],
    [2, 2],
    [1, 0],
    [6, 9],
    [0, 6]
])

# 研究"表面风化"和"类型"
'''
行: 表面风化(是/否)
列: 类型(铅钡/高钾)
'''
contingency_table_3 = np.array([
    [12, 28],
    [10, 6]
])


# 2. 进行卡方独立性检验
def perform_chi2_test(table, table_name, alpha=None):
    chi2, p_value, dof, expected = chi2_contingency(table, correction=False, method=PermutationMethod())
    # 以为样本量比较小, 使用置换检验计算p值更为精准
    print(f"\n===== {table_name}的卡方检验结果 =====")
    print(f"卡方统计量: {chi2:.4f}")
    print(f"p值: {p_value:.4f}")
    # print(f"自由度: {dof}")
    print(f"期望频数:\n{expected}")
    
    # 解释p值的含义
    if alpha is None:
        alpha_input = input(f"请输入显著性水平(默认0.05): ").strip()
        alpha = 0.05 if alpha_input == "" else float(alpha_input)
    if p_value < alpha:
        print(f"\n结论: p值({p_value:.4f}) < 显著性水平({alpha})，拒绝原假设，表明表面风化与{table_name}之间存在显著关联。")
    else:
        print(f"\n结论: p值({p_value:.4f}) >= 显著性水平({alpha})，不能拒绝原假设，表明没有足够证据表明表面风化与{table_name}之间存在显著关联。")

# 用户选择分析表
print("请选择要进行的关联分析:")
print("1 - 表面风化与纹饰的关联分析")
print("2 - 表面风化与颜色的关联分析")
print("3 - 表面风化与类型的关联分析")
print("4 - 分析全部关联关系")

choice = input("请输入选择(1/2/3/4): ")

if choice == '1':
    perform_chi2_test(contingency_table_1, "纹饰")
elif choice == '2':
    perform_chi2_test(contingency_table_2, "颜色")
elif choice == '3':
    perform_chi2_test(contingency_table_3, "类型")
elif choice == '4':
    print("\n========== 开始分析全部关联关系 ==========")
    perform_chi2_test(contingency_table_1, "纹饰")
    perform_chi2_test(contingency_table_2, "颜色")
    perform_chi2_test(contingency_table_3, "类型")
    print("\n========== 全部分析完成 ==========")
else:
    print("无效的选择，请输入1、2、3或4。")