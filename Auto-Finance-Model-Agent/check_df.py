print(">>> check_df.py is running <<<")
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 180)

df = pd.read_parquet("data.pq")

print("=== 0) 基本信息 ===")
print("shape:", df.shape)
print(df.head(3))

print("\n=== 1) 列名概览 ===")
print("总列数:", len(df.columns))
print("前20列:", list(df.columns[:20]))
print("后20列:", list(df.columns[-20:]))

print("\n=== 2) dtype 概览（哪些是数值/时间/类别）===")
print(df.dtypes.value_counts())

from pandas.api.types import is_numeric_dtype

print("\n=== 3) 非数值列（可能需要特殊处理）===")
non_num = [c for c in df.columns if not is_numeric_dtype(df[c])]
print("非数值列:", non_num)

print("\n=== 4) datetime 列 ===")
dt_cols = [c for c in df.columns if str(df[c].dtype).startswith("datetime")]
print("datetime列:", dt_cols)

if dt_cols:
    for c in dt_cols:
        print(f"\n[{c}] dtype={df[c].dtype}")
        print("min:", df[c].min(), "max:", df[c].max(), "na:", df[c].isna().mean())

if "trade_date" in df.columns:
    df_sorted = df.sort_values("trade_date")
else:
    df_sorted = df

y_cols = [c for c in df.columns if c.startswith("Y")]
for y in y_cols:
    y_raw = df_sorted[y]
    mask = y_raw.notna()
    if mask.sum() == 0:
        print(f"{y}: 全缺失")
        continue
    y_bin = (y_raw[mask] > 0).astype(int)
    print(f"{y}: 缺失率={1-mask.mean():.4f}  有效样本={mask.sum()}  正类比例(>0)={y_bin.mean():.4f}")

print("\n=== 5) X列检查（数量、是否全数值、是否存在inf）===")
x_cols = [c for c in df.columns if c.startswith("X")]
print("X列数量:", len(x_cols))
if len(x_cols) > 0:
    # 是否全是数值
    non_numeric_x = [c for c in x_cols if not np.issubdtype(df[c].dtype, np.number)]
    print("非数值X列:", non_numeric_x)

    # inf检查（只抽样部分列，避免太慢）
    sample_cols = x_cols[:50] if len(x_cols) > 50 else x_cols
    arr = df[sample_cols].to_numpy()
    has_inf = np.isinf(arr).any()
    print("抽样前50个X列是否存在 inf:", has_inf)

print("\n=== 6) 面板结构检查： (trade_date, underlying) 是否像唯一键？===")
# 这个很关键：决定你是不是“每个标的每天一条记录”的面板数据
needed = ("trade_date" in df.columns) and ("underlying" in df.columns)
print("是否同时存在 trade_date 和 underlying:", needed)

if needed:
    # 唯一组合数量 vs 总行数
    nunique_pair = df[["trade_date","underlying"]].drop_duplicates().shape[0]
    print("总行数:", df.shape[0])
    print("(trade_date, underlying) 唯一组合数:", nunique_pair)
    print("重复行数(总行数-唯一组合):", df.shape[0] - nunique_pair)

    # 每个组合出现次数分布（抽样计算，避免太慢）
    grp = df.groupby(["trade_date","underlying"]).size()
    print("\n每个(trade_date, underlying)出现次数 分布describe:")
    print(grp.describe())

    # underlying数量、每天标的数量
    print("\nunderlying 唯一数量:", df["underlying"].nunique())
    per_day = df.groupby("trade_date")["underlying"].nunique()
    print("每天underlying数量 describe:")
    print(per_day.describe())
    print("最早日期/最晚日期:", df["trade_date"].min(), df["trade_date"].max())

print("\n=== 7) 时间序列检查：start_time/end_time是否和trade_date一致？===")
for c in ["start_time","end_time","trade_date"]:
    if c in df.columns:
        print(c, "dtype:", df[c].dtype, "示例:", df[c].iloc[0])

if ("start_time" in df.columns) and ("end_time" in df.columns):
    # 看看 start_time/end_time 是否都落在同一天范围（抽样）
    tmp = df[["start_time","end_time"]].dropna().head(5)
    print("\nstart_time/end_time 前5行:")
    print(tmp)

print("\n=== 8) 泄漏风险快速扫描（非常粗略）：任何列名含Y/label/target/return? ===")
suspicious = [c for c in df.columns if any(k in c.lower() for k in ["label","target","y", "ret", "return", "future"])]
print("可疑列(名称包含关键词):", suspicious[:80], "..." if len(suspicious) > 80 else "")