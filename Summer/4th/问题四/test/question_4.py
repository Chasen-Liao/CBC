# -*- coding: utf-8 -*-
"""
Top50 供应商 | 未来24周产能最大化订购方案（按流程图）
作者：你
依赖：pandas, numpy, openpyxl
输入：
  - SUPPLY_EXCEL：你的“附件一”（前50家供应商，240周数据）
    要求：前两列分别是 供应商名称(name)、类别(type: A/B/C)；后240列为 W1..W240（周度供货量）
输出：
  - 未来24周_产能与库存计划.xlsx
  - 未来24周_供应商订购计划.xlsx
"""

import os
import numpy as np
import pandas as pd

# =============== 参数区（按需调整） ===============
SUPPLY_EXCEL = r"附件1近5年402家供应商的相关数据.xlsx"  # 你的附件一路径（这里填你上传的文件路径）
OUT_CAP_PATH = "未来24周_产能与库存计划.xlsx"
OUT_ORDER_PATH = "未来24周_供应商订购计划.xlsx"

# 产能与原料参数
BASE_CAP = 28200.0  # 基准产能（m³/周）
COEFF = {"A": 0.60, "B": 0.66, "C": 0.72}  # 生产1 m³产品所需原料体积（A/B/C）
WEEKS_ALL = 240
WEEKS_PER_YEAR = 48
HORIZON = 24

# 阶梯式提升（对应周1~24）
LIFT_STEPS = ([0.3] * 4) + ([0.6] * 8) + ([0.8] * 8) + ([1.0] * 4)

# “稳健可到货量”口径（对同周序的5个历史值取分位数）
QUANTILE = 0.20  # 更激进可改 0.5（中位数）或 0.7 等

# 安全库存要求（按“等效产品库存”>= 未来 k 周目标产能之和）
SAFETY_WEEKS_PRIMARY = 2     # 首选要求：≥2周
SAFETY_WEEKS_FALLBACKS = [1, 0]  # 若不可行，依次放宽到1周、0周

# =============== 工具函数 ===============
def read_top50_supply(excel_path: str) -> pd.DataFrame:
    """
    读取附件一（前50家供应商的历史供货），自动定位包含数据的工作表。
    期望格式：前两列 name,type；后240列为 W1..W240。若列名不一致，程序会重命名。
    """
    sheets = pd.read_excel(excel_path, sheet_name=None, header=None)
    # 优先找含“供货/到货/供给”等关键词的表；找不到就取第一张
    def like(n, keys): return any(k in str(n) for k in keys)
    key = next((n for n in sheets.keys() if like(n, ["供", "到货", "供给"])), None)
    if key is None:
        key = list(sheets.keys())[0]
    raw = sheets[key].copy()

    # 只保留前 2 + 240 列
    need_cols = 2 + WEEKS_ALL
    if raw.shape[1] < need_cols:
        raise ValueError("附件列数不足：应至少包含 name、type 以及 240 周的历史数据。")
    raw = raw.iloc[:, :need_cols].copy()

    # 规范列名
    cols = ["name", "type"] + [f"W{k}" for k in range(1, WEEKS_ALL + 1)]
    raw.columns = cols
    raw["name"] = raw["name"].astype(str).str.strip()
    raw["type"] = raw["type"].astype(str).str.strip().str.upper()
    for c in cols[2:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)

    # 基本检查：只有 A/B/C
    if not set(raw["type"]).issubset({"A", "B", "C"}):
        raise ValueError("type 列必须只包含 A/B/C。")
    return raw


def build_weekly_profile_24(supply_df: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    """
    将 240 周数据按“同周序（48周制）”聚合，对每个供应商每个周序取分位数，得到未来 24 周的稳健可到货量。
    返回列：name, type, w1..w24
    """
    arr = supply_df[[f"W{k}" for k in range(1, WEEKS_ALL + 1)]].to_numpy(dtype=float)  # (n,240)
    n = arr.shape[0]
    # 48 个周序：每个周序对应5个历史周
    def woy_idx(k):  # 1..240 -> 0..47
        return (k - 1) % WEEKS_PER_YEAR
    prof48 = np.zeros((n, WEEKS_PER_YEAR), dtype=float)
    for w in range(WEEKS_PER_YEAR):
        cols = [c for c in range(WEEKS_ALL) if woy_idx(c + 1) == w]  # 5 列
        block = arr[:, cols]  # (n,5)
        prof48[:, w] = np.quantile(block, quantile, axis=1)

    out = pd.DataFrame({"name": supply_df["name"].values, "type": supply_df["type"].values})
    for j in range(1, HORIZON + 1):
        out[f"w{j}"] = prof48[:, j - 1]
    return out


def simulate_and_check(weekly_supply_ABC, base_cap, lift_steps, coeff, safety_weeks):
    """
    给定“每周 A/B/C 可到货量”与安全库存要求，返回可行性检查函数 feasible(L)：
    - 到货后，以 A→B→C 顺序消耗，目标产能 Pj = base_cap + L * lift_steps[j]
    - 库存以“等效产品数”衡量：A/0.6 + B/0.66 + C/0.72
    - 结束时要求 等效库存 >= 未来 safety_weeks 周的目标产能之和
    """
    def feasible(L):
        IA = IB = IC = 0.0
        his = {
            "week": [], "P_target": [],
            "prod_A": [], "prod_B": [], "prod_C": [],
            "inv_A": [], "inv_B": [], "inv_C": [], "shortage": []
        }
        ok = True
        for j in range(1, HORIZON + 1):
            # 本周到货入库
            IA += weekly_supply_ABC[j - 1]["A"]
            IB += weekly_supply_ABC[j - 1]["B"]
            IC += weekly_supply_ABC[j - 1]["C"]

            # 本周目标产能
            Pj = base_cap + L * lift_steps[j - 1]
            need = Pj

            # A -> B -> C
            prod_A = min(IA / coeff["A"], need); IA -= prod_A * coeff["A"]; need -= prod_A
            prod_B = min(IB / coeff["B"], need); IB -= prod_B * coeff["B"]; need -= prod_B
            prod_C = min(IC / coeff["C"], need); IC -= prod_C * coeff["C"]; need -= prod_C

            shortage = max(0.0, need)
            if shortage > 1e-6:
                ok = False  # 本周无法满足目标产能

            # 安全库存（按等效产品量）
            eq_inv = IA / coeff["A"] + IB / coeff["B"] + IC / coeff["C"]
            remain = HORIZON - j
            k = min(safety_weeks, remain)
            future_need = 0.0
            for h in range(1, k + 1):
                future_need += base_cap + L * lift_steps[j - 1 + h]
            if eq_inv + 1e-6 < future_need:
                ok = False

            his["week"].append(j)
            his["P_target"].append(Pj)
            his["prod_A"].append(prod_A)
            his["prod_B"].append(prod_B)
            his["prod_C"].append(prod_C)
            his["inv_A"].append(IA)
            his["inv_B"].append(IB)
            his["inv_C"].append(IC)
            his["shortage"].append(shortage)

            if not ok:
                break
        return ok, pd.DataFrame(his)
    return feasible


def binary_search_L(feasible_fn, L_low=0.0, L_high=80000.0, tol=1.0, max_iter=50):
    """
    在 [L_low, L_high] 上二分搜索最大提升量 L，使24周均可行。
    """
    best_L = L_low
    best_traj = None
    low, high = L_low, L_high
    for _ in range(max_iter):
        mid = (low + high) / 2
        ok, traj = feasible_fn(mid)
        if ok:
            best_L, best_traj = mid, traj
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return best_L, best_traj


def build_order_plan(top50_w24: pd.DataFrame) -> pd.DataFrame:
    """
    订购方案：对每个供应商，每周订货量 = 稳健可到货量（w1..w24）。
    同时在输出里对 A/B/C 内部按24周总量降序排列，便于查看“优先级”。
    """
    plan = top50_w24.copy()
    plan["sum24"] = plan[[f"w{j}" for j in range(1, HORIZON + 1)]].sum(axis=1)
    plan = pd.concat([
        plan[plan["type"] == "A"].sort_values("sum24", ascending=False),
        plan[plan["type"] == "B"].sort_values("sum24", ascending=False),
        plan[plan["type"] == "C"].sort_values("sum24", ascending=False),
    ], ignore_index=True).drop(columns=["sum24"])
    return plan


# =============== 主流程 ===============
def main():
    # 1) 读取前50家供应商数据（附件一即为Top50）
    supply_df = read_top50_supply(SUPPLY_EXCEL)

    # 2) 生成未来24周“稳健可到货量”画像（同周序分位数）
    top50_w24 = build_weekly_profile_24(supply_df, quantile=QUANTILE)

    # 3) 聚合得到每周 A/B/C 的总可到货量
    weekly_supply_ABC = []
    for j in range(1, HORIZON + 1):
        sA = top50_w24.loc[top50_w24["type"] == "A", f"w{j}"].sum()
        sB = top50_w24.loc[top50_w24["type"] == "B", f"w{j}"].sum()
        sC = top50_w24.loc[top50_w24["type"] == "C", f"w{j}"].sum()
        weekly_supply_ABC.append({"A": float(sA), "B": float(sB), "C": float(sC)})

    # 4) 二分搜索“最大提升量 L”，优先尝试 2 周安全库存，不行则降到 1/0 周
    mode = "严格(安全库存=2周)"
    safety_used = SAFETY_WEEKS_PRIMARY
    feasible_fn = simulate_and_check(weekly_supply_ABC, BASE_CAP, LIFT_STEPS, COEFF, safety_used)
    L_star, traj = binary_search_L(feasible_fn)

    if traj is None:
        # 依次尝试放宽
        for sw in SAFETY_WEEKS_FALLBACKS:
            feasible_fn = simulate_and_check(weekly_supply_ABC, BASE_CAP, LIFT_STEPS, COEFF, sw)
            L_star, traj = binary_search_L(feasible_fn)
            if traj is not None:
                mode = f"放宽(安全库存={sw}周)"
                safety_used = sw
                break

    # 若仍然没有可行轨迹（极端紧张），给出“最大可生产轨迹”（不强行满足目标产能）
    if traj is None:
        mode = "资源受限(仅最大可生产)"
        safety_used = 0
        IA = IB = IC = 0.0
        rec = {"week": [], "P_target": [], "prod_A": [], "prod_B": [], "prod_C": [],
               "inv_A": [], "inv_B": [], "inv_C": [], "shortage": []}
        for j in range(1, HORIZON + 1):
            IA += weekly_supply_ABC[j - 1]["A"]
            IB += weekly_supply_ABC[j - 1]["B"]
            IC += weekly_supply_ABC[j - 1]["C"]
            # 不设目标，全部尽量生产
            prod_A = IA / COEFF["A"]; IA = 0.0
            prod_B = IB / COEFF["B"]; IB = 0.0
            prod_C = IC / COEFF["C"]; IC = 0.0
            rec["week"].append(j)
            rec["P_target"].append(BASE_CAP)
            rec["prod_A"].append(prod_A)
            rec["prod_B"].append(prod_B)
            rec["prod_C"].append(prod_C)
            rec["inv_A"].append(IA)
            rec["inv_B"].append(IB)
            rec["inv_C"].append(IC)
            rec["shortage"].append(max(0.0, BASE_CAP - (prod_A + prod_B + prod_C)))
        traj = pd.DataFrame(rec)
        L_star = 0.0

    # 5) 形成订购计划（50 × 24）
    order_plan = build_order_plan(top50_w24)

    # 6) 汇总输出
    traj["prod_total"] = traj["prod_A"] + traj["prod_B"] + traj["prod_C"]
    traj["inv_equiv_product"] = (
        traj["inv_A"] / COEFF["A"] + traj["inv_B"] / COEFF["B"] + traj["inv_C"] / COEFF["C"]
    )

    # 6.1 周度产能与库存计划
    with pd.ExcelWriter(OUT_CAP_PATH, engine="openpyxl") as writer:
        overview = pd.DataFrame({
            "参数": ["模式", "安全库存(周)", "BASE_CAP", "QUANTILE", "L_star(提升量)", "24周总产量"],
            "取值": [mode, safety_used, BASE_CAP, QUANTILE,
                   round(float(L_star), 2), round(float(traj["prod_total"].sum()), 2)]
        })
        overview.to_excel(writer, index=False, sheet_name="概览")
        traj.to_excel(writer, index=False, sheet_name="周度轨迹")

    # 6.2 供应商订购明细（直接将稳健到货量作为下单量）
    order_plan.to_excel(OUT_ORDER_PATH, index=False)

    print("=== 运行完成 ===")
    print(f"模式：{mode} | 安全库存(周)={safety_used}")
    print(f"最大可行提升量 L* = {L_star:.2f} m³")
    print(f"24周总产量 = {traj['prod_total'].sum():.2f} m³")
    print(f"已输出：\n - {os.path.abspath(OUT_CAP_PATH)}\n - {os.path.abspath(OUT_ORDER_PATH)}")


if __name__ == "__main__":
    # 防止中文乱码：确保默认 UTF-8，Excel 写入使用 openpyxl，不额外设置也可正常显示中文
    main()
