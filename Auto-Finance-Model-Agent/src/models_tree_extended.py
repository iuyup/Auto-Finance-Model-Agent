"""
扩展树模型：LightGBM + CatBoost
与 models_tree.py 中的 XGBoost/LR/RF 配合使用
"""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
import catboost as cb  #本来想做的，还有LSTM和Transformer，但是时间不够了，就没做了
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from src.metrics import calc_metrics


def train_lightgbm(
    X_sub_train, y_sub_train,
    X_val, y_val,
    X_test, y_test,
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=3000,
    early_stopping_rounds=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_child_samples=20,
):
    pos = int((y_sub_train == 1).sum())
    neg = int((y_sub_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_samples=min_child_samples,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_sub_train, y_sub_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = calc_metrics(y_test, y_prob_test, threshold=0.5)

    info = {
        "best_iteration": model.best_iteration_,
        "best_score": model.best_score_.get("valid_0", {}).get("auc", None),
        "scale_pos_weight": scale_pos_weight,
        "num_leaves": num_leaves,
    }
    return y_prob_test, metrics, info


def train_catboost(
    X_sub_train, y_sub_train,
    X_val, y_val,
    X_test, y_test,
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    early_stopping_rounds=50,
):
    pos = int((y_sub_train == 1).sum())
    neg = int((y_sub_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = cb.CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        scale_pos_weight=scale_pos_weight,
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=early_stopping_rounds,
    )

    model.fit(
        X_sub_train, y_sub_train,
        eval_set=(X_val, y_val),
        verbose=0,
    )

    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = calc_metrics(y_test, y_prob_test, threshold=0.5)

    info = {
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score().get("validation", {}).get("AUC", None),
        "scale_pos_weight": scale_pos_weight,
        "depth": depth,
    }
    return y_prob_test, metrics, info


def train_svm(X_train, y_train, X_test, y_test, C=1.0, max_iter=5000):
    """
    线性 SVM + Platt 标定得到概率输出
    """
    base = LinearSVC(C=C, max_iter=max_iter, random_state=42, dual="auto")
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = calc_metrics(y_test, y_prob, threshold=0.5)

    info = {"C": C, "max_iter": max_iter}
    return y_prob, metrics, info
