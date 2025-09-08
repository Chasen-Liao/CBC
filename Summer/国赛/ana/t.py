import random
import sys

def add_noise_to_peaks(data, peak_noise_scale=5.0, base_noise_scale=1.0):
    """
    在光谱数据的峰值位置添加噪声（仅处理波数和反射率两列数据）
    
    参数：
    data - 包含两列的列表，每行是 [波数, 反射率]
    peak_noise_scale - 峰值位置的噪声强度（倍数）
    base_noise_scale - 非峰值位置的噪声强度
    
    返回：
    new_data - 添加噪声后的数据集（相同格式）
    """
    # 提取反射率列
    reflectances = [row[1] for row in data]
    n = len(reflectances)
    
    # 1. 识别峰值位置（使用梯度法）
    peaks = []
    for i in range(1, n - 1):
        if (reflectances[i] > reflectances[i - 1] and 
            reflectances[i] > reflectances[i + 1]):
            peaks.append(i)
    
    # 2. 生成含噪声的反射率数据
    new_data = []
    for i, row in enumerate(data):
        wave_number = row[0]
        reflectance = row[1]
        
        # 基础噪声：标准差为基值×反射率值的1%
        base_std = max(0.01 * abs(reflectance), 0.001)  # 避免零值问题
        noise = base_noise_scale * random.gauss(0, base_std)
        
        # 如果是峰值，添加额外噪声：标准差为峰值×反射率值的5%
        if i in peaks:
            peak_std = max(0.05 * abs(reflectance), 0.005)  # 避免零值问题
            noise += peak_noise_scale * random.gauss(0, peak_std)
        
        # 应用噪声并确保在[0, 100]范围内
        new_val = reflectance + noise
        new_val = max(0.0, min(100.0, new_val))
        new_data.append([wave_number, new_val])
    
    return new_data

def main(input_file, output_file):
    """读取输入文件，添加噪声，输出结果文件"""
    # 读取数据文件
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            # 跳过空行
            if not line.strip():
                continue
            parts = line.split()
            # 确保有两列数据
            if len(parts) < 2:
                continue
            try:
                wave_number = float(parts[0])
                reflectance = float(parts[1])
                data.append([wave_number, reflectance])
            except ValueError:
                continue  # 跳过无法解析的行
    
    if not data:
        print("错误：未读取到有效数据")
        return
    
    # 添加噪声
    noisy_data = add_noise_to_peaks(data)
    
    # 输出结果到文件
    with open(output_file, 'w') as f:
        for wave_number, reflectance in noisy_data:
            # 波数保留4位小数，反射率保留8位小数
            f.write(f"{wave_number:.4f} {reflectance:.8f}\n")
    
    print(f"处理完成！结果已保存到: {output_file}")
    print(f"共处理 {len(data)} 行数据，检测到 {len(peaks)} 个峰值")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python add_noise.py 输入文件 输出文件")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_file, output_file)