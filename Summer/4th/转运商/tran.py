#!/usr/bin/env python
# coding: utf-8

# # 转运商的评价 -- topsis

# In[14]:


import pandas as pd
import numpy as np


# In[15]:


# 读取数据
# 数据读取
data = pd.read_excel('附件2近5年8家转运商的相关数据.xlsx', sheet_name='运输损耗率（%）')
data = data[['次数', '求和', '均值']] 
data.head(10)


# In[16]:


def normalize_indicators(data_matrix, indicator_types, ideal_values=None):
    """
    根据指标类型对数据进行正向化处理
    
    :param data_matrix: 二维数组，行为供应商，列为指标
    :param indicator_types: 列表，指示每个指标的类型
        'positive' - 极大型指标（越大越好）
        'negative' - 极小型指标（越小越好）
        'moderate' - 中间型指标（越接近某个值越好）
    :param ideal_values: 中间型指标的理想值列表
    :return: 正向化后的数据矩阵
    """
    if isinstance(data_matrix, pd.DataFrame):
        data_matrix = data_matrix.values
    normalized_matrix = np.zeros_like(data_matrix, dtype=float)
    n_indicators = data_matrix.shape[1]
    
    for j in range(n_indicators):
        col = data_matrix[:, j]
        indicator_type = indicator_types[j]
        
        if indicator_type == 'positive':
            # 极大型指标：无需处理
            normalized_matrix[:, j] = col
            
        elif indicator_type == 'negative':
            # 极小型指标：转化为极大型
            # 方法1：取倒数（适用于无零值的情况）
            # normalized_matrix[:, j] = 1 / (col + 1e-10)
            
            # 方法2：用最大值减去当前值（更常用）
            max_val = np.max(col)
            normalized_matrix[:, j] = max_val - col
            
        elif indicator_type == 'moderate':
            # 中间型指标：转化为极大型（越接近理想值越好）
            if ideal_values is None:
                raise ValueError("必须为中间型指标提供理想值")
                
            ideal = ideal_values[j]
            # 计算与理想值的绝对偏差
            deviation = np.abs(col - ideal)
            # 最大偏差（避免除零）
            max_dev = np.max(deviation) if np.max(deviation) > 0 else 1
            # 转化为极大型：偏差越小越好
            normalized_matrix[:, j] = 1 - deviation / max_dev
    
    return normalized_matrix


# In[17]:


indicator_types = ['positive', 'negative', 'negative']
ideal_values = [None, None, None]
normalize_matrix = normalize_indicators(data, indicator_types, ideal_values)

normalize_matrix


# In[18]:


def topsis_method(data_matrix):
    """
    TOPSIS方法完整实现
    :param data_matrix: 二维数组，行为供应商，列为指标
    :return: 
        weights: 指标权重数组
        scores: 供应商TOPSIS得分数组
        rankings: 供应商排名数组
    """
    # 1. 矩阵归一化（向量归一化）
    # 计算每列的平方和
    col_sums = np.sqrt(np.sum(data_matrix**2, axis=0))
    # 避免除零错误
    col_sums[col_sums == 0] = 1e-10
    # 归一化矩阵
    norm_matrix = data_matrix / col_sums
    
    # 2. 利用熵权法确定指标权重
    # 计算指标比重
    p_matrix = norm_matrix / np.sum(norm_matrix, axis=0)
    
    # 计算信息熵
    m = data_matrix.shape[0]  # 供应商数量
    k = 1 / np.log(m)  # 熵计算系数
    e_j = np.zeros(data_matrix.shape[1])
    
    for j in range(data_matrix.shape[1]):
        col = p_matrix[:, j]
        # 避免log(0)错误
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = np.sum(col * np.log(col + 1e-10))
        e_j[j] = -k * entropy
    
    # 计算信息效用值
    d_j = 1 - e_j
    
    # 计算权重
    weights = d_j / np.sum(d_j)
    
    # 3. 构建加权规范化矩阵
    weighted_matrix = norm_matrix * weights
    
    # 4. 确定正负理想解
    positive_ideal = np.max(weighted_matrix, axis=0)
    negative_ideal = np.min(weighted_matrix, axis=0)
    
    # 5. 计算欧氏距离
    d_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal)**2, axis=1))
    d_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal)**2, axis=1))
    
    # 6. 计算相对接近度（TOPSIS得分）
    scores = d_negative / (d_positive + d_negative + 1e-10)
    
    # 7. 供应商排名
    rankings = np.argsort(-scores)  # 从高到低排序
    rankings = rankings + 1
    
    return weights, scores, rankings


# In[19]:


weights, scores, rankings = topsis_method(normalize_matrix)

print(weights)
# print(sum(weights))
print(scores)
print(rankings)


# In[20]:


try:   
    get_ipython().system('jupyter nbconvert --to python q1.ipynb')
    # python即转化为.py，script即转化为.html
except:
    pass

