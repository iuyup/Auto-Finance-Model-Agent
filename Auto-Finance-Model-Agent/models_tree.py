import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb


def eval_binary(y_true, y_prob, threshold=0.5):
    """给定概率 + 阈值，输出 AUC/Precision/Recall/F1 和 y_pred"""
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1}


def train_logistic(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = eval_binary(y_test, y_prob, threshold=0.5)
    return y_prob, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    metrics = eval_binary(y_test, y_prob, threshold=0.5)
    return y_prob, metrics


def train_xgboost_with_early_stop(
    X_sub_train, y_sub_train,
    X_val, y_val,
    X_test, y_test,
    max_depth=4,
    eta=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    gamma=1.0,
    min_child_weight=10,
    num_boost_round=3000,
    early_stopping_rounds=50,
):
    # 类别不平衡：只用 sub_train 计算
    pos = int((y_sub_train == 1).sum())
    neg = int((y_sub_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_sub_train, label=y_sub_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight,

        "max_depth": max_depth,
        "eta": eta,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,

        "lambda": reg_lambda,
        "gamma": gamma,
        "min_child_weight": min_child_weight,

        "seed": 42,
        "nthread": -1
    }

    evals = [(dtrain, "train"), (dval, "val")]

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )

    # 用最佳迭代预测测试集概率
    y_prob_test = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

    # 先默认阈值 0.5（阈值搜索我们放到 evaluation.py 去做）
    metrics = eval_binary(y_test, y_prob_test, threshold=0.5)

    info = {
        "bst": bst,
        "best_iteration": bst.best_iteration,
        "best_score": bst.best_score,
        "scale_pos_weight": scale_pos_weight,
    }
    return y_prob_test, metrics, info