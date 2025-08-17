from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
#数据导入 
data = io.loadmat('waste.mat')
feature = data['waste']
Y = feature
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []

for i in range(24):
    x1.append(i)
    x2.append(i)
    x3.append(i)   
    x4.append(i)
    x5.append(i)
    x6.append(i)
    x7.append(i)
    x8.append(i)
X1 = np.array(x1)
X1 = X1.reshape(24,1)
X2 = np.array(x2)
X2 = X2.reshape(24,1)
X3 = np.array(x3)
X3 = X3.reshape(24,1)
X4 = np.array(x4)
X4 = X4.reshape(24,1)
X5 = np.array(x5)
X5 = X5.reshape(24,1)
X6 = np.array(x6)
X6 = X6.reshape(24,1)
X7 = np.array(x7)
X7 = X7.reshape(24,1)
X8 = np.array(x8)
X8 = X8.reshape(24,1)


y1 = Y[0]
y1 = y1[215:239]
y2 = Y[1]
y2 = y2[215:239]
y3 = Y[2]
y3 = y3[215:239]
y4 = Y[3]
y4 = y4[215:239]
y5 = Y[4]
y5 = y5[215:239]
y6 = Y[5]
y6 = y6[215:239]
y7 = Y[6]
y7 = y7[215:239]
y8 = Y[7]
y8 = y8[215:239]

clf1 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X1,y1)
clf2 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X2,y2)
clf3 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X3,y3)
clf4 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X4,y4)
clf5 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X5,y5)
clf6 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X6,y6)
clf7 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X7,y7)
clf8 = AdaBoostRegressor(random_state=0, n_estimators=100).fit(X8,y8)
print(clf1.score(X1,y1))
print(clf2.score(X2,y2))
print(clf3.score(X3,y3))
print(clf4.score(X4,y4))
print(clf5.score(X5,y5))
print(clf6.score(X6,y6))
print(clf7.score(X7,y7))
print(clf8.score(X8,y8))


x1_pre = []
x2_pre = []
x3_pre = []
x4_pre = []
x5_pre = []
x6_pre = []
x7_pre = []
x8_pre = []

for j in range(24):
    x1_pre.append(j)
    x2_pre.append(j)
    x3_pre.append(j)
    x4_pre.append(j)
    x5_pre.append(j)
    x6_pre.append(j)
    x7_pre.append(j)
    x8_pre.append(j)
    
X1_pre = np.array(x1_pre)
X1_pre = X1_pre.reshape(24,1)
X2_pre = np.array(x2_pre)
X2_pre = X2_pre.reshape(24,1)
X3_pre = np.array(x3_pre)
X3_pre = X3_pre.reshape(24,1)
X4_pre = np.array(x4_pre)
X4_pre = X4_pre.reshape(24,1)
X5_pre = np.array(x5_pre)
X5_pre = X5_pre.reshape(24,1)
X6_pre = np.array(x6_pre)
X6_pre = X6_pre.reshape(24,1)
X7_pre = np.array(x7_pre)
X7_pre = X7_pre.reshape(24,1)
X8_pre = np.array(x8_pre)
X8_pre = X8_pre.reshape(24,1)


predict_y1 = clf1.predict(X1_pre)
predict_y2 = clf2.predict(X2_pre)
predict_y3 = clf3.predict(X3_pre)
predict_y4 = clf4.predict(X4_pre)
predict_y5 = clf5.predict(X5_pre)
predict_y6 = clf6.predict(X6_pre)
predict_y7 = clf7.predict(X7_pre)
predict_y8 = clf8.predict(X8_pre)


io.savemat("./predict_1.mat", {'predict_1':predict_y1})
io.savemat("./predict_2.mat", {'predict_2':predict_y2})
io.savemat("./predict_3.mat", {'predict_3':predict_y3})
io.savemat("./predict_4.mat", {'predict_4':predict_y4})
io.savemat("./predict_5.mat", {'predict_5':predict_y5})
io.savemat("./predict_6.mat", {'predict_6':predict_y6})
io.savemat("./predict_7.mat", {'predict_7':predict_y7})
io.savemat("./predict_8.mat", {'predict_8':predict_y8})

