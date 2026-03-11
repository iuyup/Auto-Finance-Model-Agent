from src.data import load_and_split
from src.metrics import search_best_threshold, calc_metrics, topk_uplift
from src.plots import plot_roc, plot_pr, plot_confusion
from models_tree import (
    train_logistic,
    train_random_forest,
    train_xgboost_with_early_stop,
)

from src.models_nn import train_mlp


# ========== 1. 读取数据 ==========
X_sub_train, X_val, X_test, y_sub_train, y_val, y_test, x_cols, df_test = load_and_split()

# ========== 2. Logistic ==========
y_prob_lr, m_lr = train_logistic(X_sub_train, y_sub_train, X_test, y_test)
print("\n===== Logistic Regression =====")
print(m_lr)

# ========== 3. Random Forest ==========
y_prob_rf, m_rf = train_random_forest(X_sub_train, y_sub_train, X_test, y_test)
print("\n===== Random Forest =====")
print(m_rf)

# ========== 4. XGBoost ==========
y_prob_xgb, m_xgb, info = train_xgboost_with_early_stop(
    X_sub_train, y_sub_train,
    X_val, y_val,
    X_test, y_test
)

print("\n===== XGBoost (early stop) =====")
print("best_iteration:", info["best_iteration"])
print("best_score:", info["best_score"])
print("scale_pos_weight:", info["scale_pos_weight"])
print(m_xgb)

# ========== 5. 搜索最佳阈值 ==========
best_threshold, best_f1 = search_best_threshold(y_test, y_prob_xgb)

print("\n最佳阈值:", best_threshold)
print("最佳F1:", best_f1)

# ========== 6. 用最佳阈值重新计算指标 ==========
metrics_best = calc_metrics(y_test, y_prob_xgb, threshold=best_threshold)
print("\n===== XGBoost (best threshold) =====")
print(metrics_best)

# ========== 7. Top-K ==========
topk_uplift(y_test, y_prob_xgb)

# ========= 7.1 Daily Top-K =========
from src.metrics import topk_by_day, topk_by_day_fixedk_with_filter

topk_by_day(df_test, y_test, y_prob_xgb)

# 先看看概率分布，确认一下 min_prob 的合理范围
print("y_prob_xgb unique:", sorted(set(y_prob_xgb[:200].tolist())))
print("y_prob_xgb min/max:", float(y_prob_xgb.min()), float(y_prob_xgb.max()))

# 新的：固定K + 不交易过滤（建议先用 Top5/Top10）
topk_by_day_fixedk_with_filter(
    df_test,
    y_test,
    y_prob_xgb,
    top_k_list=(5, 10),
    min_prob=0.57   
)

# ===== 看 daily max_prob 分布 =====
import numpy as np

df_tmp = df_test.copy().reset_index(drop=True)
df_tmp["y_prob"] = y_prob_xgb

daily_max = df_tmp.groupby("trade_date")["y_prob"].max()

print("\nDaily max_prob describe:")
print(daily_max.describe())


# ========== 8. 可视化 ==========
plot_roc(y_test, y_prob_xgb)
plot_pr(y_test, y_prob_xgb)
y_pred_best = (y_prob_xgb >= best_threshold).astype(int)
plot_confusion(y_test, y_prob_xgb, threshold=best_threshold)

# ========== 9. MLP (PyTorch) ==========
y_prob_mlp, m_mlp, info_mlp = train_mlp(
    X_sub_train, y_sub_train,
    X_val, y_val,
    X_test, y_test
)

print("\n===== MLP (PyTorch) =====")
print("device:", info_mlp["device"])
print("best_val_auc:", info_mlp["best_val_auc"])
print("pos_weight:", info_mlp["pos_weight"])
print(m_mlp)