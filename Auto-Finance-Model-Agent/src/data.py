import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_and_split(label_col="Y1"):
    df = pd.read_parquet("data.pq")

    # 去标签缺失
    df = df[df[label_col].notna()].copy()
    df["y"] = (df[label_col] > 0).astype(int)

    # 时间排序
    df = df.sort_values("trade_date")

    # 时间切分（用排序后的 unique，避免潜在乱序）
    dates = df["trade_date"].sort_values().unique()
    split_index = int(len(dates) * 0.8)

    train_dates = dates[:split_index]
    test_dates = dates[split_index:]

    train_df = df[df["trade_date"].isin(train_dates)].copy()
    test_df = df[df["trade_date"].isin(test_dates)].copy()

    # ✅ 这就是你要传出去做 daily topk 的 df_test
    df_test = test_df.copy()

    # 再切验证集（仍按时间）
    train_dates_all = train_df["trade_date"].sort_values().unique()
    val_split = int(len(train_dates_all) * 0.8)

    sub_train_dates = train_dates_all[:val_split]
    val_dates = train_dates_all[val_split:]

    sub_train_df = train_df[train_df["trade_date"].isin(sub_train_dates)].copy()
    val_df = train_df[train_df["trade_date"].isin(val_dates)].copy()

    # 特征列
    x_cols = [c for c in df.columns if c.startswith("X")]

    # X / y
    X_sub_train = sub_train_df[x_cols]
    X_val = val_df[x_cols]
    X_test = test_df[x_cols]

    y_sub_train = (sub_train_df[label_col] > 0).astype(int).to_numpy()
    y_val = (val_df[label_col] > 0).astype(int).to_numpy()
    y_test = (test_df[label_col] > 0).astype(int).to_numpy()

    # 缺失值（只在 sub_train fit）
    imputer = SimpleImputer(strategy="median")
    X_sub_train = imputer.fit_transform(X_sub_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # 标准化（只在 sub_train fit）
    scaler = StandardScaler()
    X_sub_train = scaler.fit_transform(X_sub_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_sub_train, X_val, X_test, y_sub_train, y_val, y_test, x_cols, df_test