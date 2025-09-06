import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数 ====================
filename = "附件3.xlsx"   # 修改为你的文件路径
n = 3.42                 # 薄膜折射率
theta_deg = 10.0         # 入射角（度）
window_length = 21       # Savitzky-Golay 平滑窗口长度
polyorder = 3            # Savitzky-Golay 多项式阶数

# ==================== 读取数据 ====================
df = pd.read_excel(filename, header=None, skiprows=1, names=['wavenumber', 'reflectivity'])
k = df['wavenumber'].values      # 波数 (cm^-1)
R_raw = df['reflectivity'].values

# 平滑
R = savgol_filter(R_raw, window_length, polyorder)

# ==================== FFT 方法 ====================
N = len(k)
dk = np.mean(np.diff(k))    # 波数间隔
R_detrend = R - np.mean(R)  # 去趋势
Y = fft(R_detrend)
freqs = fftfreq(N, d=dk)

# 只取正频部分
mask = freqs > 0
freqs_pos = freqs[mask]
Y_pos = np.abs(Y[mask])

# 找主峰
idx_max = np.argmax(Y_pos)
f_fft = freqs_pos[idx_max]

# 厚度计算
theta_rad = np.deg2rad(theta_deg)
A = np.sqrt(n**2 - np.sin(theta_rad)**2)
d_fft = f_fft / (2 * A)

# 不确定度估计：用峰半高宽近似
half_max = Y_pos[idx_max] / 2
idxs = np.where(Y_pos > half_max)[0]
fwhm = freqs_pos[idxs[-1]] - freqs_pos[idxs[0]] if len(idxs) > 1 else 0.0
sigma_f = fwhm / 2.355 if fwhm > 0 else f_fft * 0.01
sigma_d_fft = sigma_f / (2 * A)

# ==================== 非线性拟合方法 ====================
def cos_model(k, A0, A1, f, phi):
    return A0 + A1 * np.cos(2 * np.pi * f * k + phi)

# 初始猜测
guess = [np.mean(R), (np.max(R) - np.min(R)) / 2, f_fft, 0.0]

popt, pcov = curve_fit(cos_model, k, R, p0=guess)
A0_fit, A1_fit, f_fit, phi_fit = popt
sigma_params = np.sqrt(np.diag(pcov))

# 厚度
d_fit = f_fit / (2 * A)
sigma_d_fit = sigma_params[2] / (2 * A)

# ==================== 输出结果 ====================
print("FFT 方法:")
print(f"  厚度 d = {d_fft*1e4:.2f} μm ± {sigma_d_fft*1e4:.2f} μm")

print("非线性拟合方法:")
print(f"  厚度 d = {d_fit*1e4:.2f} μm ± {sigma_d_fit*1e4:.2f} μm")

# ==================== 作图 ====================
fig, axes = plt.subplots(3, 1, figsize=(10, 14))

# 1. 光谱数据 + 峰点
axes[0].plot(k, R_raw, label="原始光谱", alpha=0.5)
axes[0].plot(k, R, label="平滑光谱", linewidth=2)

# 标出峰点
peaks, _ = find_peaks(R, prominence=0.5)
axes[0].scatter(k[peaks], R[peaks], color="red", marker="^", label="峰点")

axes[0].legend()
axes[0].set_title("光谱数据 (原始+平滑+峰点)")
axes[0].set_xlabel("波数 (cm⁻¹)")
axes[0].set_ylabel("反射率")

# 2. 非线性拟合
axes[1].plot(k, R, label="平滑光谱")
axes[1].plot(k, cos_model(k, *popt), "--", label="非线性拟合", linewidth=2)
axes[1].legend()
axes[1].set_title("非线性拟合")
axes[1].set_xlabel("波数 (cm⁻¹)")
axes[1].set_ylabel("反射率")

# 3. FFT谱
axes[2].plot(freqs_pos, Y_pos, label="FFT幅值")
axes[2].axvline(f_fft, color="r", linestyle="--", label=f"主频 {f_fft:.2f}")
axes[2].legend()
axes[2].set_title("FFT谱")
axes[2].set_xlabel("频率 (cm)")
axes[2].set_ylabel("幅值")

plt.tight_layout()
plt.show()
