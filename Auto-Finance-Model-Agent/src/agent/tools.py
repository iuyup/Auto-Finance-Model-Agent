"""
Agent 工具集 —— 整合所有模型与预处理
每个 tool 函数对应 Agent 工作流中的一个"工具调用"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import pandas as pd


@dataclass
class RunResult:
    name: str
    metrics: Dict[str, Any]
    extra: Dict[str, Any]
    y_prob: Optional[np.ndarray] = field(default=None, repr=False)


# ============================================================
# Tool 1: 数据加载（原始 DataFrame，不做预处理）
# ============================================================
def tool_load_raw(label_col: str = "Y1") -> Dict[str, Any]:
    """
    加载原始数据，按时间切分训练/验证/测试集。
    返回原始 DataFrame 片段（未做 impute/scale），供 Agent 诊断。
    """
    df = pd.read_parquet("data.pq")
    df = df[df[label_col].notna()].copy()
    df["y"] = (df[label_col] > 0).astype(int)
    df = df.sort_values("trade_date")

    dates = df["trade_date"].sort_values().unique()
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    train_df = df[df["trade_date"].isin(train_dates)].copy()
    test_df = df[df["trade_date"].isin(test_dates)].copy()

    # 训练集再切验证集
    all_train_dates = train_df["trade_date"].sort_values().unique()
    val_split = int(len(all_train_dates) * 0.8)
    sub_train_dates = all_train_dates[:val_split]
    val_dates = all_train_dates[val_split:]

    sub_train_df = train_df[train_df["trade_date"].isin(sub_train_dates)].copy()
    val_df = train_df[train_df["trade_date"].isin(val_dates)].copy()

    x_cols = [c for c in df.columns if c.startswith("X")]

    return {
        "df": df,
        "sub_train_df": sub_train_df,
        "val_df": val_df,
        "test_df": test_df,
        "x_cols": x_cols,
        "label_col": label_col,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


# ============================================================
# Tool 2: 数据诊断
# ============================================================
def tool_diagnose(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Agent 调用此工具诊断数据质量"""
    from src.preprocessing import diagnose_data
    from src.report import make_data_report, print_data_report

    df = raw["df"]

    # 结构报告
    data_report = make_data_report(df)
    print_data_report(data_report)

    # 诊断
    diagnosis = diagnose_data(df)
    print("\n===== 数据诊断结果 =====")
    for k, v in diagnosis.items():
        print(f"  {k}: {v}")

    return {"data_report": data_report, "diagnosis": diagnosis}


# ============================================================
# Tool 3: 数据泄漏检测
# ============================================================
def tool_leakage_check(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Agent 调用此工具检测数据泄漏"""
    from src.leakage import run_all_leakage_checks

    result = run_all_leakage_checks(
        df=raw["df"],
        train_dates=raw["train_dates"],
        test_dates=raw["test_dates"],
        x_cols=raw["x_cols"],
        label_col=raw["label_col"],
    )
    return result


# ============================================================
# Tool 4: 数据预处理（Agent 根据诊断结果决策）
# ============================================================
def tool_preprocess(
    raw: Dict[str, Any],
    preprocess_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Agent 根据 decide_preprocessing() 的输出调用此工具。
    返回处理好的 numpy 数组。
    """
    from src.preprocessing import PreprocessingPipeline

    sub_train_df = raw["sub_train_df"]
    val_df = raw["val_df"]
    test_df = raw["test_df"]
    x_cols = raw["x_cols"]
    label_col = raw["label_col"]

    # 取出原始 numpy
    X_sub_raw = sub_train_df[x_cols].values.astype(np.float64)
    X_val_raw = val_df[x_cols].values.astype(np.float64)
    X_test_raw = test_df[x_cols].values.astype(np.float64)

    y_sub = (sub_train_df[label_col] > 0).astype(int).to_numpy()
    y_val = (val_df[label_col] > 0).astype(int).to_numpy()
    y_test = (test_df[label_col] > 0).astype(int).to_numpy()

    # 构建 pipeline
    pipeline = PreprocessingPipeline()

    # 提取配置（去掉 reasons）
    cfg = {k: v for k, v in preprocess_config.items() if k != "reasons"}

    X_sub = pipeline.fit_transform(X_sub_raw, **cfg)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    return {
        "X_sub_train": X_sub,
        "X_val": X_val,
        "X_test": X_test,
        "y_sub_train": y_sub,
        "y_val": y_val,
        "y_test": y_test,
        "x_cols": x_cols,
        "df_test": raw["test_df"],
        "pipeline": pipeline,
        "X_sub_raw": X_sub_raw,  # 保留原始数据供可视化
    }


# ============================================================
# Tool 5: 训练 + 评估所有模型
# ============================================================
def tool_train_and_eval(models: List[str], pack: Dict[str, Any]) -> List[RunResult]:
    """
    支持的模型：
    - 线性: lr, svm
    - 树模型: rf, xgb, lgbm, catboost
    - 神经网络: mlp, lstm, tab_transformer
    """
    import sys, os
    # 确保 Q2 目录在 path 中（models_tree.py 在 Q2 根目录）
    q2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if q2_dir not in sys.path:
        sys.path.insert(0, q2_dir)

    X_sub = pack["X_sub_train"]
    X_val = pack["X_val"]
    X_test = pack["X_test"]
    y_sub = pack["y_sub_train"]
    y_val = pack["y_val"]
    y_test = pack["y_test"]

    results: List[RunResult] = []

    def _run_one(name: str, fn):
        print(f"\n--- 训练模型: {name} ---")
        t0 = time.time()
        y_prob, m, extra = fn()
        extra = dict(extra)
        extra["time_sec"] = round(time.time() - t0, 3)
        print(f"  {name}: AUC={m['auc']:.4f} F1={m['f1']:.4f} 耗时={extra['time_sec']:.1f}s")
        results.append(RunResult(name=name, metrics=m, extra=extra, y_prob=y_prob))

    # ---------- 线性模型 ----------
    if "lr" in models:
        from models_tree import train_logistic
        _run_one("lr", lambda: (*train_logistic(X_sub, y_sub, X_test, y_test), {}))

    if "svm" in models:
        from src.models_tree_extended import train_svm
        _run_one("svm", lambda: train_svm(X_sub, y_sub, X_test, y_test))

    # ---------- 树模型 ----------
    if "rf" in models:
        from models_tree import train_random_forest
        _run_one("rf", lambda: (*train_random_forest(X_sub, y_sub, X_test, y_test), {}))

    if "xgb" in models:
        from models_tree import train_xgboost_with_early_stop
        def _xgb():
            y_prob, m, info = train_xgboost_with_early_stop(X_sub, y_sub, X_val, y_val, X_test, y_test)
            # 不要在 info 里保存 bst 对象（不可 JSON 序列化）
            info_clean = {k: v for k, v in info.items() if k != "bst"}
            return y_prob, m, info_clean
        _run_one("xgb", _xgb)

    if "lgbm" in models:
        from src.models_tree_extended import train_lightgbm
        _run_one("lgbm", lambda: train_lightgbm(X_sub, y_sub, X_val, y_val, X_test, y_test))

    if "catboost" in models:
        from src.models_tree_extended import train_catboost
        _run_one("catboost", lambda: train_catboost(X_sub, y_sub, X_val, y_val, X_test, y_test))

    # ---------- 神经网络 ----------
    if "mlp" in models:
        from src.models_nn import train_mlp
        _run_one("mlp", lambda: train_mlp(X_sub, y_sub, X_val, y_val, X_test, y_test))

    if "lstm" in models:
        from src.models_nn_extended import train_lstm
        _run_one("lstm", lambda: train_lstm(X_sub, y_sub, X_val, y_val, X_test, y_test))

    if "tab_transformer" in models:
        from src.models_nn_extended import train_tab_transformer
        _run_one("tab_transformer", lambda: train_tab_transformer(
            X_sub, y_sub, X_val, y_val, X_test, y_test))

    return results


# ============================================================
# Tool 6: 选择最优模型
# ============================================================
def pick_best(results: List[RunResult], key: str = "auc") -> RunResult:
    """Agent 决策：按指定指标选最优模型"""
    return sorted(results, key=lambda r: float(r.metrics.get(key, -1)), reverse=True)[0]


# ============================================================
# Tool 7: 可视化
# ============================================================
def tool_visualize(results: List[RunResult], y_test: np.ndarray, pack: Dict[str, Any] = None):
    """Agent 调用此工具生成所有可视化"""
    from src.plots_extended import (
        plot_roc_overlay,
        plot_pr_overlay,
        plot_metrics_comparison,
        plot_confusion_matrices,
        plot_best_model_detail,
        plot_training_time,
        plot_preprocessing_effect,
    )

    # 准备数据
    viz_data = []
    model_metrics = {}
    model_times = {}

    for r in results:
        viz_data.append({
            "name": r.name,
            "y_true": y_test,
            "y_prob": r.y_prob,
        })
        model_metrics[r.name] = {
            k: v for k, v in r.metrics.items()
            if k in ("auc", "precision", "recall", "f1")
        }
        model_times[r.name] = r.extra.get("time_sec", 0)

    print("\n===== 生成可视化 =====")

    # 1) 预处理效果（如果有原始数据）
    if pack and "X_sub_raw" in pack:
        print("\n[1/6] 数据预处理效果对比")
        plot_preprocessing_effect(pack["X_sub_raw"], pack["X_sub_train"])

    # 2) 多模型 ROC 叠加
    print("\n[2/6] 多模型 ROC 曲线")
    plot_roc_overlay(viz_data)

    # 3) 多模型 PR 叠加
    print("\n[3/6] 多模型 PR 曲线")
    plot_pr_overlay(viz_data)

    # 4) 指标对比柱状图
    print("\n[4/6] 模型指标对比")
    plot_metrics_comparison(model_metrics)

    # 5) 混淆矩阵网格
    print("\n[5/6] 混淆矩阵")
    plot_confusion_matrices(viz_data)

    # 6) 训练时间对比
    print("\n[6/6] 训练时间对比")
    plot_training_time(model_times)

    # 7) 最优模型详情
    best = pick_best(results, key="auc")
    print(f"\n[额外] 最优模型 {best.name} 性能详情")
    plot_best_model_detail(y_test, best.y_prob, best.name)

    return {"status": "ok", "n_plots": 7}
