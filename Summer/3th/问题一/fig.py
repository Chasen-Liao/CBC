import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# 设置温度范围 (200°C 到 600°C)
x = np.linspace(250, 600, 100)

# 创建2x2的子图画布
plt.figure(figsize=(15, 12))

# ======================= 温度-乙醇-线性回归 =======================
plt.subplot(2, 2, 1)
# A3
y_A3 = -95.883 + 0.42 * x
plt.plot(x, y_A3, label='A3: y = -95.883 + 0.42x (R^2=0.964)')

# A4
y_A4 = -144.571 + 0.582 * x
plt.plot(x, y_A4, label='A4: y = -144.571 + 0.582x (R^2=0.995)')

# A6
y_A6 = -119.833 + 0.502 * x
plt.plot(x, y_A6, label='A6: y = -119.833 + 0.502x (R^2=0.967)')

# A7
y_A7 = -74.26 + 0.378 * x
plt.plot(x, y_A7, label='A7: y = -74.26 + 0.378x (R^2=0.999)')

# A8
y_A8 = -83.776 + 0.34 * x
plt.plot(x, y_A8, label='A8: y = -83.776 + 0.34x (R^2=0.955)')

# A11
y_A11 = -18.067 + 0.076 * x
plt.plot(x, y_A11, label='A11: y = -18.067 + 0.076x (R^2=0.938)')

# A12
y_A12 = -74.814 + 0.286 * x
plt.plot(x, y_A12, label='A12: y = -74.814 + 0.286x (R^2=0.927)')

# A14
y_A14 = -86.687 + 0.336 * x
plt.plot(x, y_A14, label='A14: y = -86.687 + 0.336x (R^2=0.929)')

# B1
y_B1 = -73.143 + 0.279 * x
plt.plot(x, y_B1, label='B1: y = -73.143 + 0.279x (R^2=0.926)')

plt.title('温度-乙醇-线性回归')
plt.xlabel('温度 (°C)')
plt.ylabel('乙醇转化率')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=8)

# ======================= 温度-乙醇-非线性回归 =======================
plt.subplot(2, 2, 2)
# A5
y_A5 = 231.943 - 1.669 * x + 0.003 * x**2
plt.plot(x, y_A5, label='A5: y = 231.943 - 1.669x + 0.003x^2 (R^2=0.993)')

# A9
y_A9 = 186.287 - 1.343 * x + 0.002 * x**2
plt.plot(x, y_A9, label='A9: y = 186.287 - 1.343x + 0.002x^2 (R^2=0.990)')

# A10
y_A10 = 135.651 - 0.987 * x + 0.001 * x**2
plt.plot(x, y_A10, label='A10: y = 135.651 - 0.987x + 0.001x^2 (R^2=0.993)')

# A13
y_A13 = 164.302 - 1.210 * x + 0.002 * x**2
plt.plot(x, y_A13, label='A13: y = 164.302 - 1.210x + 0.002x^2 (R^2=0.996)')

# B2
y_B2 = 188.25 - 1.36 * x + 0.002 * x**2
plt.plot(x, y_B2, label='B2: y = 188.25 - 1.36x + 0.002x^2 (R^2=0.991)')

# B3
y_B3 = 106.953 - 0.771 * x + 0.001 * x**2
plt.plot(x, y_B3, label='B3: y = 106.953 - 0.771x + 0.001x^2 (R^2=0.991)')

# B4
y_B4 = 155.274 - 1.129 * x + 0.002 * x**2
plt.plot(x, y_B4, label='B4: y = 155.274 - 1.129x + 0.002x^2 (R^2=0.986)')

# B5
y_B5 = 185.740 - 1.354 * x + 0.002 * x**2
plt.plot(x, y_B5, label='B5: y = 185.740 - 1.354x + 0.002x^2 (R^2=0.991)')

# B6
y_B6 = 190.083 - 1.445 * x + 0.002 * x**2
plt.plot(x, y_B6, label='B6: y = 190.083 - 1.445x + 0.002x^2 (R^2=0.990)')

# B7
y_B7 = 228.710 - 1.711 * x + 0.003 * x**2
plt.plot(x, y_B7, label='B7: y = 228.710 - 1.711x + 0.003x^2 (R^2=0.996)')

plt.title('温度-乙醇-非线性回归')
plt.xlabel('温度 (°C)')
plt.ylabel('乙醇转化率')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=8)

# ======================= 温度-烯烃-线性回归 =======================
plt.subplot(2, 2, 3)
# A3
y_A3_olefin = -59.195 + 0.261 * x
plt.plot(x, y_A3_olefin, label='A3: y = -59.195 + 0.261x (R^2=0.913)')

# A4
y_A4_olefin = -52.411 + 0.227 * x
plt.plot(x, y_A4_olefin, label='A4: y = -52.411 + 0.227x (R^2=0.917)')

# A5
y_A5_olefin = -57.813 + 0.23 * x
plt.plot(x, y_A5_olefin, label='A5: y = -57.813 + 0.23x (R^2=0.94)')

# A7
y_A7_olefin = -44.262 + 0.187 * x
plt.plot(x, y_A7_olefin, label='A7: y = -44.262 + 0.187x (R^2=0.937)')

# A8
y_A8_olefin = -57.257 + 0.242 * x
plt.plot(x, y_A8_olefin, label='A8: y = -57.257 + 0.242x (R^2=0.983)')

# A9
y_A9_olefin = -59.095 + 0.254 * x
plt.plot(x, y_A9_olefin, label='A9: y = -59.095 + 0.254x (R^2=0.995)')

# A11
y_A11_olefin = -13.307 + 0.052 * x
plt.plot(x, y_A11_olefin, label='A11: y = -13.307 + 0.052x (R^2=0.978)')

# A12
y_A12_olefin = -47.727 + 0.205 * x
plt.plot(x, y_A12_olefin, label='A12: y = -47.727 + 0.205x (R^2=0.967)')

# A13
y_A13_olefin = -35.889 + 0.163 * x
plt.plot(x, y_A13_olefin, label='A13: y = -35.889 + 0.163x (R^2=0.977)')

# A14
y_A14_olefin = -35.116 + 0.138 * x
plt.plot(x, y_A14_olefin, label='A14: y = -35.116 + 0.138x (R^2=0.92)')

# B1
y_B1_olefin = -56.728 + 0.24 * x
plt.plot(x, y_B1_olefin, label='B1: y = -56.728 + 0.24x (R^2=0.972)')

# B2
y_B2_olefin = -56.728 + 0.24 * x
plt.plot(x, y_B2_olefin, label='B2: y = -56.728 + 0.24x (R^2=0.972)')

# B3
y_B3_olefin = -28.294 + 0.12 * x
plt.plot(x, y_B3_olefin, label='B3: y = -28.294 + 0.12x (R^2=0.943)')

# B5
y_B5_olefin = -34.599 + 0.146 * x
plt.plot(x, y_B5_olefin, label='B5: y = -34.599 + 0.146x (R^2=0.956)')

# B6
y_B6_olefin = -45.757 + 0.19 * x
plt.plot(x, y_B6_olefin, label='B6: y = -45.757 + 0.19x (R^2=0.965)')

# B7
y_B7_olefin = -56.451 + 0.234 * x
plt.plot(x, y_B7_olefin, label='B7: y = -56.451 + 0.234x (R^2=0.986)')

plt.title('温度-烯烃-线性回归')
plt.xlabel('温度 (°C)')
plt.ylabel('C4烯烃选择性')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=7)

# ======================= 温度-烯烃-非线性回归 =======================
plt.subplot(2, 2, 4)
# A2
y_A2_olefin = 234.745 - 1.646 * x + 0.003 * x**2
plt.plot(x, y_A2_olefin, label='A2: y = 234.745 - 1.646x + 0.003x^2 (R^2=0.98)')

# A6
y_A6_olefin = 176.615 - 1.233 * x + 0.002 * x**2
plt.plot(x, y_A6_olefin, label='A6: y = 176.615 - 1.133x + 0.002x^2 (R^2=0.999)')

# A10
y_A10_olefin = 59.711 - 0.403 * x + 0.0006 * x**2
plt.plot(x, y_A10_olefin, label='A10: y = 59.711 - 0.403x + 0.0006x^2 (R^2=0.977)')

# B4
y_B4_olefin = 81.073 - 0.548 * x + 0.001 * x**2
plt.plot(x, y_B4_olefin, label='B4: y = 81.073 - 0.548x + 0.001x^2 (R^2=0.972)')

plt.title('温度-烯烃-非线性回归')
plt.xlabel('温度 (°C)')
plt.ylabel('C4烯烃选择性')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')

# 调整布局并显示图形
plt.tight_layout()
plt.savefig('temperature_regression_analysis.png', dpi=300)
plt.show()