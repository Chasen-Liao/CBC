# -*- coding: utf-8 -*-  
from scipy import io
import numpy as np
import matplotlib.pyplot as plt

#读取数据
data = io.loadmat('ABC.mat')
feature = data['class_week']
Y = feature.T
x1 = []
x2 = []
x3 = []
for i in range(24):
    x1.append(i)
    x2.append(i)
    x3.append(i)
X1 = np.array(x1)
X1 = X1.reshape(24,1)
X2 = np.array(x2)
X2 = X2.reshape(24,1)
X3 = np.array(x3)
X3 = X3.reshape(24,1)

y1 = Y[0]
y1 = y1[215:239]
y2 = Y[1]
y2 = y2[215:239]
y3 = Y[2]
y3 = y3[215:239]

plt.scatter(X1,y1, label="train", c="red", marker="x")
from sklearn.ensemble import AdaBoostRegressor
clf1 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X1,y1)
clf2 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X2,y2)
clf3 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X3,y3)
print(clf1.score(X1,y1))
print(clf2.score(X2,y2))
print(clf3.score(X3,y3))

Y_predict = clf1.predict(X1)

##要预测的 横坐标 x 输入 predict函数   用 y_predict接收
y_predict = clf1.predict([[1]])
 
plt.plot(X1,Y_predict)
plt.show()


plt.scatter(X2,y2, label="train", c="red", marker="x")


Y_predict2 = clf2.predict(X2)

##要预测的 横坐标 x 输入 predict函数   用 y_predict接收
y_predict2 = clf2.predict([[1]])
 
plt.plot(X2,Y_predict2)
plt.show()



plt.scatter(X3,y3, label="train", c="red", marker="x")


Y_predict3 = clf3.predict(X3)

##要预测的 横坐标 x 输入 predict函数   用 y_predict接收
y_predict3 = clf3.predict([[1]])
 
plt.plot(X3,Y_predict3)
plt.show()

x1_pre = []
x2_pre = []
x3_pre = []

for j in range(24):
    x1_pre.append(j)
    x2_pre.append(j)
    x3_pre.append(j)
    
X1_pre = np.array(x1_pre)
X1_pre = X1_pre.reshape(24,1)
X2_pre = np.array(x2_pre)
X2_pre = X2_pre.reshape(24,1)
X3_pre = np.array(x3_pre)
X3_pre = X3_pre.reshape(24,1)

predict_y1 = clf1.predict(X1_pre)
predict_y2 = clf2.predict(X2_pre)
predict_y3 = clf3.predict(X3_pre)

io.savemat("./predict_A.mat", {'predict_A':predict_y1})
io.savemat("./predict_B.mat", {'predict_B':predict_y2})
io.savemat("./predict_C.mat", {'predict_C':predict_y3})
