"""
增强版可视化模块
- 多模型 ROC 曲线叠加
- 多模型 PR 曲线叠加
- 模型指标对比柱状图
- 混淆矩阵网格
- 数据预处理效果对比
- 最优模型性能详情
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc as sklearn_auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from typing import Dict, List, Any, Optional


# ============ 风格设置 ============
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

MODEL_COLORS = {
    "lr": "#1f77b4",
    "svm": "#ff7f0e",
    "rf": "#2ca02c",
    "xgb": "#d62728",
    "lgbm": "#9467bd",
    "catboost": "#8c564b",
    "mlp": "#e377c2",
    "lstm": "#7f7f7f",
    "tab_transformer": "#bcbd22",
}


def get_color(name: str) -> str:
    return MODEL_COLORS.get(name, "#17becf")


# ============ 多模型 ROC 叠加 ============
def plot_roc_overlay(
    results: List[Dict[str, Any]],
    title: str = "ROC Curves - All Models",
):
    """
    results: list of dict, 每个 dict 包含:
      - name: str
      - y_true: array
      - y_prob: array
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        auc_val = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{r["name"]} (AUC={auc_val:.4f})',
                color=get_color(r["name"]), linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


# ============ 多模型 PR 叠加 ============
def plot_pr_overlay(
    results: List[Dict[str, Any]],
    title: str = "Precision-Recall Curves - All Models",
):
    fig, ax = plt.subplots(figsize=(9, 7))

    for r in results:
        precision, recall, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        ap = average_precision_score(r["y_true"], r["y_prob"])
        ax.plot(recall, precision, label=f'{r["name"]} (AP={ap:.4f})',
                color=get_color(r["name"]), linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


# ============ 指标对比柱状图 ============
def plot_metrics_comparison(
    model_metrics: Dict[str, Dict[str, float]],
    metrics_keys: List[str] = None,
    title: str = "Model Performance Comparison",
):
    """
    model_metrics: {model_name: {auc: ..., precision: ..., recall: ..., f1: ...}}
    """
    if metrics_keys is None:
        metrics_keys = ["auc", "precision", "recall", "f1"]

    names = list(model_metrics.keys())
    n_models = len(names)
    n_metrics = len(metrics_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(names):
        values = [model_metrics[name].get(k, 0) for k in metrics_keys]
        bars = ax.bar(x + i * width, values, width, label=name,
                      color=get_color(name), alpha=0.85)
        # 标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([k.upper() for k in metrics_keys])
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


# ============ 混淆矩阵网格 ============
def plot_confusion_matrices(
    results: List[Dict[str, Any]],
    threshold: float = 0.5,
    title: str = "Confusion Matrices",
):
    """
    results: list of dict with name, y_true, y_prob
    """
    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, r in enumerate(results):
        y_pred = (r["y_prob"] >= threshold).astype(int)
        cm = confusion_matrix(r["y_true"], y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i], colorbar=False)
        axes[i].set_title(f'{r["name"]}')

    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


# ============ 数据预处理效果 ============
def plot_preprocessing_effect(
    X_before: np.ndarray,
    X_after: np.ndarray,
    title: str = "Preprocessing Effect",
):
    """
    对比预处理前后的特征分布（取前 6 个特征的直方图）
    """
    n_features = min(6, X_before.shape[1], X_after.shape[1])
    fig, axes = plt.subplots(2, n_features, figsize=(3 * n_features, 6))

    for i in range(n_features):
        # before
        data_before = X_before[:, i]
        data_before = data_before[~np.isnan(data_before)]
        axes[0, i].hist(data_before, bins=50, alpha=0.7, color="steelblue")
        axes[0, i].set_title(f"Feature {i+1} (原始)")
        axes[0, i].tick_params(labelsize=8)

        # after
        data_after = X_after[:, i]
        data_after = data_after[~np.isnan(data_after)]
        axes[1, i].hist(data_after, bins=50, alpha=0.7, color="darkorange")
        axes[1, i].set_title(f"Feature {i+1} (处理后)")
        axes[1, i].tick_params(labelsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


# ============ 最优模型性能详情 ============
def plot_best_model_detail(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
):
    """
    为最优模型画详细的性能图：ROC + PR + 概率分布 + 混淆矩阵
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = sklearn_auc(fpr, tpr)
    axes[0, 0].plot(fpr, tpr, color="darkorange", linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0, 0].set_title(f"ROC Curve (AUC={auc_val:.4f})")
    axes[0, 0].set_xlabel("FPR")
    axes[0, 0].set_ylabel("TPR")
    axes[0, 0].grid(True, alpha=0.3)

    # 2) PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[0, 1].plot(recall, precision, color="green", linewidth=2)
    axes[0, 1].set_title(f"PR Curve (AP={ap:.4f})")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].grid(True, alpha=0.3)

    # 3) 概率分布
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    axes[1, 0].hist(neg_probs, bins=50, alpha=0.6, label="Negative", color="blue", density=True)
    axes[1, 0].hist(pos_probs, bins=50, alpha=0.6, label="Positive", color="red", density=True)
    axes[1, 0].axvline(x=threshold, color="black", linestyle="--", label=f"Threshold={threshold:.2f}")
    axes[1, 0].set_title("Predicted Probability Distribution")
    axes[1, 0].set_xlabel("Probability")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4) 混淆矩阵
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[1, 1], colorbar=False)
    axes[1, 1].set_title(f"Confusion Matrix (threshold={threshold:.2f})")

    fig.suptitle(f"Best Model: {model_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig


# ============ 训练时间对比 ============
def plot_training_time(
    model_times: Dict[str, float],
    title: str = "Training Time Comparison",
):
    names = list(model_times.keys())
    times = [model_times[n] for n in names]
    colors = [get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, times, color=colors, alpha=0.85)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", fontsize=10)

    ax.set_xlabel("Time (seconds)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig
