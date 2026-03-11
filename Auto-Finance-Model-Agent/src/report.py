# Q2/src/report.py
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def make_data_report(df: pd.DataFrame, label_prefix: str = "Y", x_prefix: str = "X") -> dict:
    """
    对原始 dataframe 做“结构体检”，返回一个 dict 报告（便于后续打印/保存 markdown）。
    """
    report = {}

    # 0) 基本信息
    report["shape"] = tuple(df.shape)
    report["n_cols"] = int(df.shape[1])

    # 1) 列名概览
    cols = list(df.columns)
    report["first_20_cols"] = cols[:20]
    report["last_20_cols"] = cols[-20:]

    # 2) dtype 概览
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    report["dtype_counts"] = dtype_counts

    # 3) X/Y 列
    x_cols = [c for c in cols if c.startswith(x_prefix)]
    y_cols = [c for c in cols if c.startswith(label_prefix)]
    report["x_cols_count"] = len(x_cols)
    report["y_cols_count"] = len(y_cols)
    report["x_cols_head"] = x_cols[:10]
    report["x_cols_tail"] = x_cols[-10:]
    report["y_cols_head"] = y_cols[:12]

    # 4) 非数值列（注意：datetime 也算非数值）
    non_numeric_cols = [c for c in cols if not is_numeric_dtype(df[c])]
    report["non_numeric_cols"] = non_numeric_cols

    # 5) datetime 列详细信息
    dt_cols = [c for c in cols if str(df[c].dtype).startswith("datetime")]
    dt_info = {}
    for c in dt_cols:
        dt_info[c] = {
            "dtype": str(df[c].dtype),
            "min": str(df[c].min()),
            "max": str(df[c].max()),
            "na_rate": float(df[c].isna().mean()),
        }
    report["datetime_cols"] = dt_cols
    report["datetime_info"] = dt_info

    # 6) 标签列分布（缺失率、有效样本、正类比例(>0)）
    y_info = {}
    for y in y_cols:
        y_raw = df[y]
        mask = y_raw.notna()
        n_valid = int(mask.sum())
        if n_valid == 0:
            y_info[y] = {"na_rate": 1.0, "n_valid": 0, "pos_rate_gt0": None}
        else:
            pos_rate = float((y_raw[mask] > 0).mean())
            y_info[y] = {
                "na_rate": float(y_raw.isna().mean()),
                "n_valid": n_valid,
                "pos_rate_gt0": pos_rate,
            }
    report["y_info"] = y_info

    # 7) X 列数值健康检查：缺失率描述 + inf 检查（抽样）
    if len(x_cols) > 0:
        x_df = df[x_cols]

        # 缺失率统计
        miss_rate = x_df.isna().mean()
        report["x_missing_rate_describe"] = miss_rate.describe().to_dict()

        # inf 检查（抽样列，避免太慢）
        sample_cols = x_cols[: min(50, len(x_cols))]
        x_sample = x_df[sample_cols].to_numpy(dtype=np.float64, copy=False)
        report["x_inf_in_first50"] = bool(np.isinf(x_sample).any())
    else:
        report["x_missing_rate_describe"] = None
        report["x_inf_in_first50"] = None

    # 8) (trade_date, underlying) 唯一性检查（如果存在）
    key_check = {}
    if "trade_date" in df.columns and "underlying" in df.columns:
        n_rows = int(df.shape[0])
        n_unique = int(df[["trade_date", "underlying"]].drop_duplicates().shape[0])
        key_check["has_trade_date_and_underlying"] = True
        key_check["n_rows"] = n_rows
        key_check["n_unique_pairs"] = n_unique
        key_check["n_duplicates"] = n_rows - n_unique

        # 每个pair出现次数分布
        vc = df.groupby(["trade_date", "underlying"]).size()
        key_check["pair_count_describe"] = vc.describe().to_dict()

        # 每天 underlying 数量分布
        day_cnt = df.groupby("trade_date")["underlying"].nunique()
        key_check["underlying_unique_count"] = int(df["underlying"].nunique())
        key_check["per_day_underlying_describe"] = day_cnt.describe().to_dict()
    else:
        key_check["has_trade_date_and_underlying"] = False
    report["key_check"] = key_check

    # 9) 泄漏风险快速扫描（列名中包含一些可疑关键词）
    leak_keywords = ["label", "target", "return", "profit", "y", "Y", "future"]
    suspicious = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in ["label", "target", "return", "profit", "future"]):
            suspicious.append(c)
    # 另外把真实标签列 & underlying/trade_date 也列出来给你核对
    report["leak_suspicious_cols_by_name"] = suspicious
    report["note_check_these_cols"] = ["trade_date", "underlying"] + y_cols[:]

    return report


def print_data_report(report: dict) -> None:
    """把 report 用人类可读方式打印出来。"""
    print("===== Data Report =====")
    print("shape:", report["shape"])
    print("total cols:", report["n_cols"])
    print("\n--- columns overview ---")
    print("first_20:", report["first_20_cols"])
    print("last_20 :", report["last_20_cols"])

    print("\n--- dtype counts ---")
    for k, v in report["dtype_counts"].items():
        print(f"{k}: {v}")

    print("\n--- X/Y columns ---")
    print("X count:", report["x_cols_count"], "| head:", report["x_cols_head"], "| tail:", report["x_cols_tail"])
    print("Y count:", report["y_cols_count"], "| head:", report["y_cols_head"])

    print("\n--- non-numeric cols ---")
    print(report["non_numeric_cols"])

    print("\n--- datetime cols ---")
    for c, info in report["datetime_info"].items():
        print(f"[{c}] dtype={info['dtype']} min={info['min']} max={info['max']} na_rate={info['na_rate']}")

    print("\n--- Y distribution (gt0 as positive) ---")
    for y, info in report["y_info"].items():
        print(f"{y}: na_rate={info['na_rate']:.4f} n_valid={info['n_valid']} pos_rate_gt0={info['pos_rate_gt0']}")

    print("\n--- X missing/inf quick check ---")
    print("X missing_rate describe:", report["x_missing_rate_describe"])
    print("X inf in first50:", report["x_inf_in_first50"])

    print("\n--- key check (trade_date, underlying) ---")
    kc = report["key_check"]
    if kc.get("has_trade_date_and_underlying"):
        print("n_rows:", kc["n_rows"])
        print("unique pairs:", kc["n_unique_pairs"])
        print("duplicates:", kc["n_duplicates"])
        print("pair_count_describe:", kc["pair_count_describe"])
        print("underlying_unique_count:", kc["underlying_unique_count"])
        print("per_day_underlying_describe:", kc["per_day_underlying_describe"])
    else:
        print("trade_date/underlying not found, skipped.")

    print("\n--- leakage name scan (very rough) ---")
    print("suspicious cols:", report["leak_suspicious_cols_by_name"])
    print("manual check cols:", report["note_check_these_cols"])