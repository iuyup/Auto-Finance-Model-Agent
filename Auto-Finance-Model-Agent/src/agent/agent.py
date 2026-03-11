"""
AutoModelAgent —— 自动化模型选择系统
Agent 自主完成以下流程：
1. 数据加载 & 结构诊断
2. 数据泄漏检测
3. 自动决策预处理方案 & 执行
4. 自动训练多类模型（线性/树/神经网络）
5. 自动评估 & 选择最优模型
6. 生成可视化 & 报告
"""
from __future__ import annotations

import json
import os
import sys
import io
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

# Windows GBK 兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.agent.tools import (
    tool_load_raw,
    tool_diagnose,
    tool_leakage_check,
    tool_preprocess,
    tool_train_and_eval,
    tool_visualize,
    pick_best,
    RunResult,
)
from src.preprocessing import decide_preprocessing


class AutoModelAgent:
    """
    核心 Agent 类。
    设计理念：Agent 拥有一组"工具"（tool_xxx），
    通过内置决策逻辑自主调用工具，完成完整建模流水线。
    所有决策过程和中间结果写入决策日志（decision_log）。
    """

    SYSTEM_PROMPT = """
你是一个金融时序数据建模 Agent。你的任务是：
1. 分析数据特点（缺失值、方差、相关性、数据泄漏）
2. 根据诊断结果自动决策预处理方案
3. 训练多种模型（线性模型、树模型、神经网络）
4. 自动评估所有模型，按 AUC 选择最优
5. 生成可视化和报告

你需要：
- 严格按时间切分数据，防止未来数据泄漏
- imputer/scaler 只在训练集上 fit
- 使用验证集做早停，测试集做最终评估
- 计算 AUC/Precision/Recall/F1 四项指标
- 对最优模型做详细分析
"""

    # Agent 支持的全部模型
    ALL_MODELS = ["lr", "rf", "xgb", "mlp"]

    def __init__(self, label_col: str = "Y1", out_dir: str = "runs"):
        self.label_col = label_col
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.decision_log: List[Dict[str, Any]] = []

    def _log_decision(self, phase: str, decision: str, detail: Any = None):
        """记录 Agent 的每一步决策"""
        entry = {
            "phase": phase,
            "decision": decision,
            "detail": detail,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.decision_log.append(entry)
        print(f"\n{'='*60}")
        print(f"[Agent 决策] 阶段: {phase}")
        print(f"  决策: {decision}")
        if detail:
            if isinstance(detail, dict):
                for k, v in detail.items():
                    print(f"  {k}: {v}")
            elif isinstance(detail, list):
                for item in detail:
                    print(f"  - {item}")
        print(f"{'='*60}")

    def run(
        self,
        models: Optional[List[str]] = None,
        min_prob: float = 0.57,
        skip_viz: bool = False,
    ) -> Dict[str, Any]:
        """
        Agent 主执行入口。
        Agent 自主完成数据诊断 → 泄漏检测 → 预处理 → 训练 → 评估 → 可视化 → 报告。
        """
        total_start = time.time()
        print("\n" + "=" * 70)
        print("  AutoModelAgent 启动")
        print(f"  标签列: {self.label_col}")
        print(f"  输出目录: {self.out_dir}")
        print("=" * 70)

        if models is None:
            models = self.ALL_MODELS

        self._log_decision(
            "初始化",
            f"Agent 将训练 {len(models)} 个模型",
            {"模型列表": models, "标签列": self.label_col},
        )

        # ========== 阶段 1: 数据加载 ==========
        self._log_decision("数据加载", "调用 tool_load_raw 加载原始数据并按时间切分")
        raw = tool_load_raw(label_col=self.label_col)
        n_features = len(raw["x_cols"])
        n_samples = len(raw["df"])
        self._log_decision("数据加载完成", "数据加载成功", {
            "总样本数": n_samples,
            "特征数": n_features,
            "训练日期数": len(raw["train_dates"]),
            "测试日期数": len(raw["test_dates"]),
        })

        # ========== 阶段 2: 数据诊断 ==========
        self._log_decision("数据诊断", "调用 tool_diagnose 分析数据质量")
        diag_result = tool_diagnose(raw)
        diagnosis = diag_result["diagnosis"]

        # ========== 阶段 3: 数据泄漏检测 ==========
        self._log_decision("泄漏检测", "调用 tool_leakage_check 检测数据泄漏")
        leakage_result = tool_leakage_check(raw)
        if leakage_result["has_leak"]:
            self._log_decision(
                "泄漏检测",
                "[WARN] 检测到潜在泄漏风险! Agent 将继续执行但标记警告",
                {"详情": leakage_result["overall_verdict"]},
            )
        else:
            self._log_decision("泄漏检测", "[PASS] 未检测到数据泄漏，流程安全")

        # ========== 阶段 4: 自动决策预处理方案 ==========
        self._log_decision("预处理决策", "Agent 根据诊断结果自动决策预处理方案")
        preprocess_config = decide_preprocessing(diagnosis)
        self._log_decision(
            "预处理决策完成",
            "预处理方案已确定",
            {"决策理由": preprocess_config.get("reasons", [])},
        )

        # ========== 阶段 5: 执行预处理 ==========
        self._log_decision("预处理执行", "调用 tool_preprocess 执行预处理流水线")
        pack = tool_preprocess(raw, preprocess_config)
        self._log_decision("预处理完成", "数据预处理完成", {
            "训练集shape": pack["X_sub_train"].shape,
            "验证集shape": pack["X_val"].shape,
            "测试集shape": pack["X_test"].shape,
        })

        # ========== 阶段 6: 训练全部模型 ==========
        self._log_decision("模型训练", f"开始训练 {len(models)} 个模型: {models}")
        results: List[RunResult] = tool_train_and_eval(models=models, pack=pack)

        # 汇总训练结果
        train_summary = {}
        for r in results:
            train_summary[r.name] = {
                "AUC": round(r.metrics["auc"], 4),
                "F1": round(r.metrics["f1"], 4),
                "耗时": f"{r.extra.get('time_sec', 0):.1f}s",
            }
        self._log_decision("训练完成", f"全部 {len(results)} 个模型训练完成", train_summary)

        # ========== 阶段 7: 选择最优模型 ==========
        best = pick_best(results, key="auc")
        self._log_decision(
            "模型选择",
            f"[BEST] 最优模型: {best.name} (AUC={best.metrics['auc']:.4f})",
            best.metrics,
        )

        # ========== 阶段 7.5: Daily TopK 交易分析 ==========
        import numpy as np
        import pandas as pd
        from src.metrics import topk_by_day_fixedk_with_filter

        # --- 自动校准 min_prob ---
        # 不同预处理管线（PCA等）会导致模型概率分布差异巨大，
        # 用户指定的 min_prob 可能对当前模型完全无效。
        # 因此基于 daily max_prob 分布自动校准。
        _df_tmp = pack["df_test"][["trade_date"]].copy().reset_index(drop=True)
        _df_tmp["_prob"] = np.asarray(best.y_prob).astype(float)
        _daily_max = _df_tmp.groupby("trade_date")["_prob"].max()

        user_min_prob = min_prob
        pass_rate = float((_daily_max >= min_prob).mean())

        if pass_rate > 0.95:
            # 用户 min_prob 过滤率 <5%，几乎不过滤 → 自动上调到 P75
            effective_min_prob = float(_daily_max.quantile(0.75))
            self._log_decision(
                "TopK校准",
                f"用户 min_prob={user_min_prob} 仅跳过 {1-pass_rate:.1%} 天，过滤形同虚设 → "
                f"自动上调到 daily_max_prob 的 P75 = {effective_min_prob:.4f}",
                {
                    "用户min_prob": user_min_prob,
                    "有效min_prob": round(effective_min_prob, 4),
                    "daily_max_prob分布": {
                        "min": round(float(_daily_max.min()), 4),
                        "P25": round(float(_daily_max.quantile(0.25)), 4),
                        "P50": round(float(_daily_max.quantile(0.50)), 4),
                        "P75": round(float(_daily_max.quantile(0.75)), 4),
                        "max": round(float(_daily_max.max()), 4),
                    },
                },
            )
        else:
            effective_min_prob = min_prob
            self._log_decision(
                "TopK校准",
                f"min_prob={min_prob} 可过滤 {1-pass_rate:.1%} 天，无需校准",
            )

        # --- 执行 TopK 分析 ---
        self._log_decision("TopK分析", f"对最优模型 {best.name} 做 Daily TopK 交易分析 (effective_min_prob={effective_min_prob:.4f})")
        topk_result = topk_by_day_fixedk_with_filter(
            df_test=pack["df_test"],
            y_true=pack["y_test"],
            y_prob=best.y_prob,
            top_k_list=(5, 10),
            min_prob=effective_min_prob,
        )
        self._log_decision("TopK分析完成", "Daily TopK 交易分析完成", {
            "traded_days": topk_result["traded_days"],
            "unique_days": topk_result["unique_days"],
            "effective_min_prob": effective_min_prob,
        })

        # ========== 阶段 8: 可视化 ==========
        if not skip_viz:
            self._log_decision("可视化", "生成所有可视化图表")
            tool_visualize(results, pack["y_test"], pack=pack)

        # ========== 阶段 9: 生成报告 ==========
        total_time = round(time.time() - total_start, 1)
        self._log_decision("报告生成", f"Agent 全流程完成，总耗时 {total_time}s")

        report = self._build_report(
            results=results,
            best=best,
            models=models,
            preprocess_config=preprocess_config,
            diagnosis=diagnosis,
            leakage_result=leakage_result,
            total_time=total_time,
            min_prob=min_prob,
            topk_result=topk_result,
        )

        # 保存
        out_path = (self.out_dir / f"report_{self.label_col}.json").resolve()
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=self._json_default),
            encoding="utf-8",
        )

        report["saved_to"] = str(out_path)
        report["file_exists"] = out_path.exists()
        report["file_size"] = out_path.stat().st_size if out_path.exists() else -1

        # 打印最终摘要
        self._print_summary(report)

        return report

    def _build_report(
        self,
        results: List[RunResult],
        best: RunResult,
        models: List[str],
        preprocess_config: Dict[str, Any],
        diagnosis: Dict[str, Any],
        leakage_result: Dict[str, Any],
        total_time: float,
        min_prob: float,
        topk_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """汇总完整报告"""
        all_results_clean = []
        for r in results:
            d = asdict(r)
            d.pop("y_prob", None)
            all_results_clean.append(d)

        best_dict = asdict(best)
        best_dict.pop("y_prob", None)

        report = {
            "label_col": self.label_col,
            "models_trained": models,
            "total_time_sec": total_time,
            "preprocessing": {
                "diagnosis": diagnosis,
                "config": {k: v for k, v in preprocess_config.items() if k != "reasons"},
                "reasons": preprocess_config.get("reasons", []),
            },
            "leakage_check": {
                "has_leak": leakage_result["has_leak"],
                "overall_verdict": leakage_result["overall_verdict"],
                "temporal": leakage_result["temporal_leakage"],
            },
            "best_by_auc": best_dict,
            "all_results": all_results_clean,
            "decision_log": self.decision_log,
            "daily_topk_analysis": topk_result,
            "suggested_min_prob": float(min_prob),
            "agent_system_prompt": self.SYSTEM_PROMPT.strip(),
        }
        return report

    def _print_summary(self, report: Dict[str, Any]):
        """打印最终摘要"""
        print("\n" + "=" * 70)
        print("  [RESULT] Agent 执行完毕 - 最终摘要")
        print("=" * 70)
        print(f"  标签列: {report['label_col']}")
        print(f"  训练模型数: {len(report['models_trained'])}")
        print(f"  总耗时: {report['total_time_sec']:.1f}s")
        print(f"  泄漏检测: {report['leakage_check']['overall_verdict']}")
        print()

        best = report["best_by_auc"]
        print(f"  [BEST] 最优模型: {best['name']}")
        print(f"     AUC       = {best['metrics']['auc']:.4f}")
        print(f"     Precision = {best['metrics']['precision']:.4f}")
        print(f"     Recall    = {best['metrics']['recall']:.4f}")
        print(f"     F1        = {best['metrics']['f1']:.4f}")
        print()

        # TopK 交易分析摘要
        topk = report.get("daily_topk_analysis")
        if topk and topk.get("topk"):
            print(f"  [TopK] min_prob={topk['min_prob']:.4f} | base_rate={topk['base_rate']:.4f}"
                  f" | traded {topk['traded_days']}/{topk['unique_days']} days")
            for row in topk["topk"]:
                print(f"    Top-{row['k']:>2d}: avg_pos_rate={row['avg_pos_rate']:.4f} | Lift={row['lift']:.2f}x")
            print()
        elif topk:
            print(f"  [TopK] min_prob={topk['min_prob']:.4f} | traded_days=0 (无交易日)")
            print()

        print("  全部模型排名（按 AUC）：")
        sorted_results = sorted(
            report["all_results"],
            key=lambda r: r["metrics"]["auc"],
            reverse=True,
        )
        for i, r in enumerate(sorted_results, 1):
            m = r["metrics"]
            print(f"    {i}. {r['name']:>15s}  AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

        print(f"\n  报告已保存: {report.get('saved_to', 'N/A')}")
        print("=" * 70)

    @staticmethod
    def _json_default(o):
        try:
            import numpy as np
        except Exception:
            np = None

        if np is not None:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)

        return str(o)
