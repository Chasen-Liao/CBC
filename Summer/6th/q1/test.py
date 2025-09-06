from scipy.stats import beta

# -------------------------------
# 参数
# -------------------------------
p0 = 0.10
alpha_reject = 0.05   # 95% 置信拒收
alpha_accept = 0.10   # 90% 置信接收
Nmax = 1000            # 搜索的最大样本数上限

# -------------------------------
# Clopper–Pearson 区间
# -------------------------------
def clopper_pearson(x, n, alpha):
    """返回 (low, high) 的Clopper–Pearson置信区间"""
    if x == 0:
        low = 0.0
    else:
        low = beta.ppf(alpha/2, x, n-x+1)

    if x == n:
        high = 1.0
    else:
        high = beta.ppf(1-alpha/2, x+1, n-x)

    return low, high

# -------------------------------
# 搜索最小n的接收/拒收方案
# -------------------------------
def find_min_sample():
    reject_plan = None
    accept_plan = None

    for n in range(1, Nmax+1):
        for x in range(0, n+1):
            # 95% 拒收条件
            low95, _ = clopper_pearson(x, n, alpha=alpha_reject)
            if low95 > p0 and reject_plan is None:
                reject_plan = (n, x, low95)
            
            # 90% 接收条件
            _, high90 = clopper_pearson(x, n, alpha=alpha_accept)
            if high90 <= p0 and accept_plan is None:
                accept_plan = (n, x, high90)
        
        # 如果两个方案都找到，就提前停止
        if reject_plan and accept_plan:
            break

    return reject_plan, accept_plan

# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    reject_plan, accept_plan = find_min_sample()

    if reject_plan:
        n, x, low95 = reject_plan
        print(f"最小拒收方案: n={n}, 次品={x}, 95%下界={low95:.3f} > {p0}")
    else:
        print("未找到拒收方案（请增加 Nmax）")

    if accept_plan:
        n, x, high90 = accept_plan
        print(f"最小接收方案: n={n}, 次品={x}, 90%上界={high90:.3f} <= {p0}")
    else:
        print("未找到接收方案（请增加 Nmax）")
