"""
增强版数据预处理模块
Agent 自动分析数据特点，决策预处理方案：
- 缺失值填充（中位数）
- 标准化 / 归一化
- 方差过滤（去掉近零方差特征）
- 相关性过滤（去掉高度冗余特征）
- PCA 降维（可选）
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from typing import Dict, Any, Tuple, List, Optional


class PreprocessingPipeline:
    """
    可配置的预处理流水线。
    Agent 根据数据诊断结果自动选择各步骤的开关与参数。
    """

    def __init__(self):
        self.imputer: Optional[SimpleImputer] = None
        self.scaler = None
        self.var_selector: Optional[VarianceThreshold] = None
        self.pca: Optional[PCA] = None
        self.selected_cols: Optional[np.ndarray] = None
        self.corr_drop_cols: Optional[List[int]] = None
        self.log: List[Dict[str, Any]] = []

    def _log(self, step: str, detail: Dict[str, Any]):
        entry = {"step": step, **detail}
        self.log.append(entry)
        print(f"  [预处理] {step}: {detail}")

    # ---- 第 1 步：缺失值填充 ----
    def fit_imputer(self, X_train: np.ndarray) -> np.ndarray:
        missing_rate = float(np.isnan(X_train).mean())
        self._log("缺失值诊断", {
            "整体缺失率": f"{missing_rate:.4f}",
            "含缺失列数": int(np.any(np.isnan(X_train), axis=0).sum()),
            "总列数": X_train.shape[1],
        })

        if missing_rate > 0:
            self.imputer = SimpleImputer(strategy="median")
            X_train = self.imputer.fit_transform(X_train)
            self._log("缺失值填充", {"策略": "中位数填充", "状态": "完成"})
        else:
            self._log("缺失值填充", {"策略": "无需填充", "状态": "跳过"})

        return X_train

    def transform_imputer(self, X: np.ndarray) -> np.ndarray:
        if self.imputer is not None:
            return self.imputer.transform(X)
        # 即使 imputer 为 None 也要处理可能的 NaN
        if np.isnan(X).any():
            return np.nan_to_num(X, nan=0.0)
        return X

    # ---- 第 2 步：方差过滤 ----
    def fit_variance_filter(self, X_train: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        n_before = X_train.shape[1]
        self.var_selector = VarianceThreshold(threshold=threshold)
        X_train = self.var_selector.fit_transform(X_train)
        n_after = X_train.shape[1]
        n_dropped = n_before - n_after
        self._log("方差过滤", {
            "阈值": threshold,
            "过滤前列数": n_before,
            "过滤后列数": n_after,
            "丢弃列数": n_dropped,
        })
        return X_train

    def transform_variance_filter(self, X: np.ndarray) -> np.ndarray:
        if self.var_selector is not None:
            return self.var_selector.transform(X)
        return X

    # ---- 第 3 步：相关性过滤 ----
    def fit_correlation_filter(self, X_train: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        n_before = X_train.shape[1]
        corr = np.corrcoef(X_train, rowvar=False)
        # 上三角
        upper = np.triu(np.abs(corr), k=1)
        drop_cols = set()
        for i in range(upper.shape[0]):
            for j in range(i + 1, upper.shape[1]):
                if upper[i, j] > threshold:
                    drop_cols.add(j)

        self.corr_drop_cols = sorted(drop_cols)
        keep_cols = [i for i in range(n_before) if i not in drop_cols]
        X_train = X_train[:, keep_cols]
        n_after = X_train.shape[1]

        self._log("相关性过滤", {
            "阈值": threshold,
            "过滤前列数": n_before,
            "过滤后列数": n_after,
            "丢弃列数": n_before - n_after,
        })
        return X_train

    def transform_correlation_filter(self, X: np.ndarray) -> np.ndarray:
        if self.corr_drop_cols is not None and len(self.corr_drop_cols) > 0:
            keep_cols = [i for i in range(X.shape[1]) if i not in self.corr_drop_cols]
            return X[:, keep_cols]
        return X

    # ---- 第 4 步：标准化 / 归一化 ----
    def fit_scaler(self, X_train: np.ndarray, method: str = "standard") -> np.ndarray:
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self._log("标准化", {"策略": "无", "状态": "跳过"})
            return X_train

        X_train = self.scaler.fit_transform(X_train)
        self._log("标准化", {"策略": method, "状态": "完成"})
        return X_train

    def transform_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    # ---- 第 5 步：PCA 降维 ----
    def fit_pca(self, X_train: np.ndarray, n_components: float = 0.95) -> np.ndarray:
        n_before = X_train.shape[1]
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train = self.pca.fit_transform(X_train)
        n_after = X_train.shape[1]
        explained = float(self.pca.explained_variance_ratio_.sum())
        self._log("PCA降维", {
            "目标方差保留": n_components,
            "降维前列数": n_before,
            "降维后列数": n_after,
            "实际解释方差": f"{explained:.4f}",
        })
        return X_train

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        if self.pca is not None:
            return self.pca.transform(X)
        return X

    # ---- 完整 pipeline ----
    def fit_transform(
        self,
        X_train: np.ndarray,
        use_variance_filter: bool = True,
        variance_threshold: float = 0.01,
        use_corr_filter: bool = True,
        corr_threshold: float = 0.95,
        scale_method: str = "standard",
        use_pca: bool = False,
        pca_components: float = 0.95,
    ) -> np.ndarray:
        """在训练集上 fit + transform 全流程"""
        print("\n===== 数据预处理流水线 (fit_transform) =====")

        X = self.fit_imputer(X_train)

        if use_variance_filter:
            X = self.fit_variance_filter(X, threshold=variance_threshold)

        if use_corr_filter:
            X = self.fit_correlation_filter(X, threshold=corr_threshold)

        X = self.fit_scaler(X, method=scale_method)

        if use_pca:
            X = self.fit_pca(X, n_components=pca_components)

        self._log("最终数据维度", {"shape": X.shape})
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """对验证集/测试集做 transform（不能 fit）"""
        X = self.transform_imputer(X)
        X = self.transform_variance_filter(X)
        X = self.transform_correlation_filter(X)
        X = self.transform_scaler(X)
        X = self.transform_pca(X)
        return X

    def get_log(self) -> List[Dict[str, Any]]:
        return self.log


def diagnose_data(df: pd.DataFrame, x_prefix: str = "X") -> Dict[str, Any]:
    """
    Agent 用此函数做数据诊断，返回诊断结果 dict，
    Agent 据此自主决策预处理方案。
    """
    x_cols = [c for c in df.columns if c.startswith(x_prefix)]
    x_df = df[x_cols]

    # 缺失率
    missing_rate_per_col = x_df.isna().mean()
    overall_missing = float(missing_rate_per_col.mean())
    high_missing_cols = int((missing_rate_per_col > 0.3).sum())

    # 方差
    variances = x_df.var(skipna=True)
    near_zero_var = int((variances < 0.01).sum())

    # 值域范围
    ranges = x_df.max() - x_df.min()
    max_range = float(ranges.max()) if len(ranges) > 0 else 0.0
    min_range = float(ranges.min()) if len(ranges) > 0 else 0.0

    # inf 检查
    n_inf = 0
    for c in x_cols:
        vals = pd.to_numeric(x_df[c], errors="coerce")
        n_inf += int(np.isinf(vals.dropna().values).sum())

    # 相关性（抽样快速检查）
    sample_cols = x_cols[:min(50, len(x_cols))]
    sample_data = x_df[sample_cols].dropna()
    if len(sample_data) > 100:
        corr = sample_data.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_pairs = int((upper > 0.95).sum().sum())
    else:
        high_corr_pairs = 0

    diagnosis = {
        "n_features": len(x_cols),
        "n_samples": len(df),
        "overall_missing_rate": overall_missing,
        "high_missing_cols_gt30pct": high_missing_cols,
        "near_zero_variance_cols": near_zero_var,
        "max_feature_range": max_range,
        "min_feature_range": min_range,
        "n_inf_values": n_inf,
        "high_corr_pairs_in_sample50": high_corr_pairs,
    }
    return diagnosis


def decide_preprocessing(diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 的核心决策函数：根据诊断结果自动确定预处理方案。
    返回 PreprocessingPipeline 的参数字典。
    """
    decisions = {}
    reasons = []

    # 1) 方差过滤
    if diagnosis["near_zero_variance_cols"] > 0:
        decisions["use_variance_filter"] = True
        decisions["variance_threshold"] = 0.01
        reasons.append(
            f"发现 {diagnosis['near_zero_variance_cols']} 个近零方差特征，启用方差过滤(阈值=0.01)"
        )
    else:
        decisions["use_variance_filter"] = False
        reasons.append("无近零方差特征，跳过方差过滤")

    # 2) 相关性过滤
    if diagnosis["high_corr_pairs_in_sample50"] > 5:
        decisions["use_corr_filter"] = True
        decisions["corr_threshold"] = 0.95
        reasons.append(
            f"抽样50列中发现 {diagnosis['high_corr_pairs_in_sample50']} 对高相关(>0.95)，启用相关性过滤"
        )
    else:
        decisions["use_corr_filter"] = False
        reasons.append("高相关特征对较少，跳过相关性过滤")

    # 3) 标准化方式
    if diagnosis["max_feature_range"] > 100:
        decisions["scale_method"] = "standard"
        reasons.append(
            f"特征值域差异大(最大范围={diagnosis['max_feature_range']:.1f})，使用 StandardScaler"
        )
    else:
        decisions["scale_method"] = "standard"
        reasons.append("特征值域适中，使用 StandardScaler")

    # 4) PCA
    if diagnosis["n_features"] > 200:
        decisions["use_pca"] = True
        decisions["pca_components"] = 0.95
        reasons.append(
            f"特征数量较多({diagnosis['n_features']})，启用 PCA(保留95%方差)"
        )
    else:
        decisions["use_pca"] = False
        reasons.append("特征数量适中，跳过 PCA")

    decisions["reasons"] = reasons
    return decisions
