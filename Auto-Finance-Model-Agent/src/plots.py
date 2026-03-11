import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
)


def plot_roc(y_true, y_prob, title_prefix="ROC Curve"):
    """
    画 ROC 曲线，并返回 AUC（用于标题展示或日志）
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # roc_auc_score 也能算，但这里用曲线配套的 AP 一样的风格：标题里显示数值
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} (AUC={auc:.4f})")
    plt.show()

    return float(auc)


def plot_pr(y_true, y_prob, title_prefix="PR Curve"):
    """
    画 Precision-Recall 曲线，并返回 AP（Average Precision）
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} (AP={ap:.4f})")
    plt.show()

    return float(ap)


def plot_confusion(y_true, y_prob, threshold=0.5, title_prefix="Confusion Matrix"):
    """
    传入 y_true + y_prob + threshold
    函数内部生成 y_pred，再画混淆矩阵
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{title_prefix} (threshold={threshold:.4f})")
    plt.show()
    return cm