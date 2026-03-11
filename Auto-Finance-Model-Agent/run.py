"""
Agent 运行入口
启动 AutoModelAgent，自动完成：
数据诊断 → 泄漏检测 → 预处理 → 模型训练(9种) → 评估 → 可视化 → 报告
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.agent import AutoModelAgent

if __name__ == "__main__":
    agent = AutoModelAgent(label_col="Y1", out_dir="runs")

    # Agent 自动训练 4 个模型（线性 + 树 + 神经网络）
    rep = agent.run(
        models=["lr", "rf", "xgb", "mlp"],
        min_prob=0.57,
    )

    if rep is None:
        raise RuntimeError("agent.run() returned None")

    print("\n[DONE] Agent 执行完毕!")
    print(f"报告路径: {rep.get('saved_to')}")
