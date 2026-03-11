import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)


def search_best_threshold(y_true, y_prob, t_min=0.01, t_max=0.5, n=100):
    """
    在验证集上搜索最佳阈值（默认以 F1 最大为目标）
    y_true: 0/1
    y_prob: 预测为 1 的概率
    """
    best_f1 = -1.0
    best_threshold = 0.5

    for t in np.linspace(t_min, t_max, n):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, float(best_f1)


def calc_metrics(y_true, y_prob, threshold=0.5):
    """
    给定 y_true + 预测概率 y_prob + 阈值 threshold
    输出: AUC/Precision/Recall/F1 + threshold + confusion(tn/fp/fn/tp)
    """
    import numpy as np
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
    )

    y_pred = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

def topk_uplift(y_true, y_prob, k_list=(0.01, 0.05, 0.1, 0.2)):
    """
    Top-K 交易分析（越靠前越“强信号”）
    返回一个 list[dict]，同时打印结果
    """
    base = float(np.mean(y_true))
    print("\n===== Top-K 交易分析（越靠前越“强信号”）=====")
    print("测试集整体正类比例(base rate):", base)

    order = np.argsort(-y_prob)  # 概率从高到低
    n = len(y_prob)

    rows = []
    for k in k_list:
        top_n = int(n * k)
        top_n = max(top_n, 1)

        idx = order[:top_n]
        hit = float(np.mean(y_true[idx]))
        lift = hit / base if base > 0 else np.nan

        rows.append(
            {
                "top_pct": float(k),
                "top_n": int(top_n),
                "hit_rate": float(hit),
                "lift": float(lift),
            }
        )

        print(f"Top {int(k*100):>2d}% | 样本数={top_n:>5d} | 正类比例={hit:.4f} | Lift={lift:.2f}x")

    return rows

def topk_by_day(df_test, y_true, y_prob, k_list=(0.01, 0.05, 0.1, 0.2)):
    """
    按 trade_date 分组做 TopK 评估（每天挑最靠前的若干个）
    要求：df_test 的行顺序必须与 y_true / y_prob 一一对应（我们当前就是这样）
    """
    import numpy as np
    import pandas as pd

    if "trade_date" not in df_test.columns:
        print("No trade_date column, cannot do daily TopK.")
        return None

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(df_test) != len(y_true) or len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: len(df_test)={len(df_test)}, len(y_true)={len(y_true)}, len(y_prob)={len(y_prob)}"
        )

    temp = df_test.copy().reset_index(drop=True)
    temp["y_true"] = y_true
    temp["y_prob"] = y_prob

    base_rate = float(temp["y_true"].mean())
    print("\n===== Daily TopK Analysis =====")
    print(f"Overall base rate: {base_rate:.4f}")

    results = []

    # 每天算一次 topk 命中率，然后对“天”求平均（避免某天标的多就权重大）
    for k in k_list:
        daily_rates = []
        for _, group in temp.groupby("trade_date"):
            n = len(group)
            top_n = max(1, int(n * k))

            top_group = group.sort_values("y_prob", ascending=False).head(top_n)
            daily_rate = float(top_group["y_true"].mean())
            daily_rates.append(daily_rate)

        avg_rate = float(np.mean(daily_rates)) if daily_rates else 0.0
        lift = (avg_rate / base_rate) if base_rate > 0 else np.nan

        print(f"Top {int(k*100):>2d}% | avg_pos_rate={avg_rate:.4f} | Lift={lift:.2f}x")
        results.append((k, avg_rate, lift))

    return results

def topk_by_day_fixedk_with_filter(
    df_test,
    y_true,
    y_prob,
    top_k_list=(5, 10),
    min_prob=0.57,
    date_col="trade_date",
    prob_col="y_prob",
):
    """
    按天做 Fixed-K TopK + 不交易过滤：
    - 每天先看该日 max(y_prob)，若 max < min_prob => 这天不交易（跳过）
    - 否则对当天样本按 y_prob 降序取 Top-K，计算 Top-K 的正类比例与 Lift
    - 调了好久，run.py那里交易日一直只是293，后面发现是y_prob的问题，调了好久
    """
    import numpy as np
    import pandas as pd

    # -------- 0) 对齐与校验（非常关键） --------
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    if len(df_test) != len(y_true) or len(df_test) != len(y_prob):
        raise ValueError(
            f"Length mismatch: len(df_test)={len(df_test)}, len(y_true)={len(y_true)}, len(y_prob)={len(y_prob)}"
        )

    temp = df_test.copy().reset_index(drop=True)
    temp["_y_true"] = y_true.astype(int)
    temp[prob_col] = y_prob.astype(float)

    if date_col not in temp.columns:
        raise ValueError(f"date_col='{date_col}' not in df_test columns")

    # 全局 base rate
    base_rate = float(temp["_y_true"].mean())
    unique_days = temp[date_col].nunique()

    # -------- 1) 逐日过滤 + 统计 --------
    # traded_days: 通过过滤的交易日数量
    traded_days = 0

    # 每个K累计：sum_pos_rate / 交易日数
    topk_sum_pos = {int(k): 0.0 for k in top_k_list}

    # 逐日
    for d, g in temp.groupby(date_col, sort=True):
        daily_max = float(g[prob_col].max())

        # 过滤：这天不交易
        if daily_max < float(min_prob):
            continue

        traded_days += 1

        g_sorted = g.sort_values(prob_col, ascending=False)

        for k in top_k_list:
            k = int(k)
            kk = min(k, len(g_sorted))
            topk = g_sorted.head(kk)
            pos_rate = float(topk["_y_true"].mean())
            topk_sum_pos[k] += pos_rate

    # -------- 2) 输出 --------
    print("\n===== Daily TopK (Fixed-K + Filter) =====")
    print(f"Overall base rate: {base_rate:.4f} | unique days: {unique_days}")
    print(f"Filter rule: trade only if daily max_prob >= {float(min_prob):.4f}")

    if traded_days == 0:
        print("No traded days under this min_prob. (traded_days=0)")
        return {
            "base_rate": base_rate,
            "unique_days": int(unique_days),
            "min_prob": float(min_prob),
            "traded_days": 0,
            "topk": [],
        }

    rows = []
    for k in top_k_list:
        k = int(k)
        avg_pos = topk_sum_pos[k] / traded_days
        lift = (avg_pos / base_rate) if base_rate > 0 else float("nan")
        rows.append({"k": k, "avg_pos_rate": float(avg_pos), "lift": float(lift)})

        print(f"TopK={k:<3d} | traded_days={traded_days}/{unique_days} | avg_pos_rate={avg_pos:.4f} | Lift={lift:.2f}x")

    return {
        "base_rate": base_rate,
        "unique_days": int(unique_days),
        "min_prob": float(min_prob),
        "traded_days": int(traded_days),
        "topk": rows,
    }