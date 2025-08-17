# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 22:10:29 2021

@author: 淞
"""

#随机森林实现  
from sklearn.ensemble import RandomForestClassifier

from scipy import io
import numpy as np
#数据集初始化
feature = io.loadmat('supply_oneweek.mat')
feature = feature['unnamed']

label = io.loadmat('label.mat')
label = label['label']
Label = label[:,1]

clf = RandomForestClassifier(criterion = 'gini',max_depth=4,random_state=0)
clf.fit(feature,Label)

print("随机森林精度：",clf.score(feature,Label))

weight = clf.feature_importances_
prob = clf.predict_proba(feature)

io.savemat("./weight.mat", {'weight':weight})
io.savemat("./prob.mat", {'prob':prob})
