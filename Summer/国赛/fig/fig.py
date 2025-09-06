import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("附件1.xlsx")


# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(df['波数 (cm-1)'], df['反射率 (%)'], label='反射率 (%)', color='blue')

# 添加图标题和轴标签
plt.title('波数与反射率的关系')
plt.xlabel('波数 (cm-1)')
plt.xticks(rotation=45)
plt.ylabel('反射率 (%)')

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()