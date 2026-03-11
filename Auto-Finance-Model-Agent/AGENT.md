# Agent 系统设计文档

## 1. 系统架构

### 1.1 架构图

```
+================================================================+
|                    AutoModelAgent (主控)                         |
|  - 决策引擎: 根据各阶段输出自主决策下一步操作                      |
|  - 决策日志: 记录每一步的决策理由和执行结果                        |
+================================================================+
        |           |           |           |           |
        v           v           v           v           v
  +-----------+ +----------+ +----------+ +----------+ +----------+
  | Tool 1    | | Tool 2   | | Tool 3   | | Tool 4   | | Tool 5   |
  | 数据加载  | | 数据诊断 | | 泄漏检测 | | 预处理   | | 训练评估 |
  +-----------+ +----------+ +----------+ +----------+ +----------+
        |           |           |           |           |
        v           v           v           v           v
  +-----------+ +----------+ +----------+ +----------+ +----------+
  | data.pq   | | report   | | leakage  | | preproc  | | 4 Models |
  | (Parquet) | | module   | | module   | | pipeline | | (3 类)   |
  +-----------+ +----------+ +----------+ +----------+ +----------+
                                                            |
                                                            v
                                                   +----------------+
                                                   | Tool 6: 可视化 |
                                                   | Tool 7: 报告   |
                                                   +----------------+
```

### 1.2 模块依赖关系

```
src/
├── agent/
│   ├── agent.py          # AutoModelAgent 主类 (决策引擎)
│   └── tools.py          # 7 个工具函数 (Agent 的"手和脚")
├── data.py               # 数据加载与时间切分
├── preprocessing.py      # 预处理流水线 + 诊断 + 决策
├── leakage.py            # 数据泄漏检测 (4 类)
├── metrics.py            # 评估指标计算
├── models_nn.py          # MLP 模型
├── models_nn_extended.py # LSTM + TabTransformer 模型
├── models_tree_extended.py # LightGBM + CatBoost + SVM
├── plots.py              # 基础可视化
├── plots_extended.py     # 增强可视化 (叠加图/对比图/详情图)
└── report.py             # 数据结构报告
```

---

## 2. 完整 Prompt

### 2.1 Agent System Prompt

```
你是一个金融时序数据建模 Agent。你的任务是：
1. 分析数据特点（缺失值、方差、相关性、数据泄漏）
2. 根据诊断结果自动决策预处理方案
3. 训练多种模型（线性模型、树模型、神经网络），包括自定义深度学习架构
4. 自动评估所有模型，按 AUC 选择最优
5. 生成可视化和报告

你需要：
- 严格按时间切分数据，防止未来数据泄漏
- imputer/scaler 只在训练集上 fit
- 使用验证集做早停，测试集做最终评估
- 计算 AUC/Precision/Recall/F1 四项指标
- 对最优模型做详细分析
```

### 2.2 决策 Prompt（预处理）

Agent 的预处理决策逻辑（`decide_preprocessing`）：

| 诊断指标 | 条件 | 决策 |
|---|---|---|
| `near_zero_variance_cols > 0` | 存在近零方差特征 | 启用方差过滤 (threshold=0.01) |
| `high_corr_pairs_in_sample50 > 5` | 抽样50列中高相关对数>5 | 启用相关性过滤 (threshold=0.95) |
| `max_feature_range > 100` | 特征值域差异大 | 使用 StandardScaler |
| `n_features > 200` | 特征数>200 | 启用 PCA (保留95%方差) |

### 2.3 决策 Prompt（模型选择）

Agent 按 AUC 降序排列所有模型，选择 AUC 最高者作为最优模型。

---

## 3. 工作流设计

### 3.1 完整工作流

```
[START]
  │
  ├─ 阶段 1: 数据加载
  │   └─ tool_load_raw() → 原始 DataFrame + 时间切分
  │
  ├─ 阶段 2: 数据诊断
  │   └─ tool_diagnose() → 数据质量报告 + 诊断指标
  │
  ├─ 阶段 3: 泄漏检测
  │   └─ tool_leakage_check() → 4 类泄漏检查结果
  │   └─ [若有泄漏] → 标记警告，继续执行
  │
  ├─ 阶段 4: 预处理决策
  │   └─ decide_preprocessing(diagnosis) → 预处理配置
  │   └─ Agent 记录决策理由
  │
  ├─ 阶段 5: 执行预处理
  │   └─ tool_preprocess(raw, config) → 处理后的 numpy 数组
  │   └─ pipeline: 缺失值 → 方差过滤 → 相关性过滤 → 标准化 → PCA
  │
  ├─ 阶段 6: 模型训练
  │   └─ tool_train_and_eval(models, pack)
  │   └─ 4 个模型: lr, rf, xgb, mlp
  │
  ├─ 阶段 7: 模型选择
  │   └─ pick_best(results, key='auc') → 最优模型
  │
  ├─ 阶段 8: 可视化
  │   └─ tool_visualize() → 7 类图表
  │
  └─ 阶段 9: 报告生成
      └─ JSON 报告 + 决策日志 → runs/report_Y1.json
[END]
```

### 3.2 数据流

```
data.pq
  → [时间排序] → [80/20 时间切分]
    → train (80%)
      → sub_train (64%) ← imputer.fit / scaler.fit
      → val (16%) ← imputer.transform / scaler.transform
    → test (20%) ← imputer.transform / scaler.transform

预处理 pipeline (仅 sub_train 上 fit):
  原始 300 维 → 缺失值中位数填充
    → 方差过滤 (300 → 128)
    → 相关性过滤 (128 → 107)
    → StandardScaler
    → PCA 95% (107 → 66)
  最终: 66 维
```

### 3.3 模型清单

| 类别 | 模型 | 关键参数 | 说明 |
|---|---|---|---|
| 线性 | Logistic Regression | max_iter=2000 | 基线模型 |
| 树 | Random Forest | n_estimators=200 | 集成模型 |
| 树 | XGBoost | early_stop=50, depth=4 | 带验证集早停 |
| 神经网络 | MLP | 256→128→1, dropout=0.2 | 全连接网络 |

---

## 4. 异常处理机制

### 4.1 数据层异常处理

| 异常场景 | 处理策略 |
|---|---|
| 标签列缺失值 | `df[label_col].notna()` 过滤 |
| 特征列 NaN | SimpleImputer(strategy="median") 填充 |
| 特征列 Inf | `np.nan_to_num` 处理 |
| 类别不平衡 | `scale_pos_weight = neg/pos` 自动计算 |

### 4.2 训练层异常处理

| 异常场景 | 处理策略 |
|---|---|
| 模型不收敛 | 早停机制 (patience=5, 监控 val AUC) |
| GPU 不可用 | 自动回退到 CPU |
| 内存不足 | batch 预测 (4096 样本/批) |
| XGBoost Booster 不可序列化 | 过滤 bst 对象后再写入 JSON |

### 4.3 泄漏检测异常处理

| 检测项 | 异常时行为 |
|---|---|
| 时间泄漏 | 标记 `[WARN]`，继续执行但在报告中警告 |
| 标签泄漏 | 标记可疑特征，不自动删除（由用户决策） |
| 列名泄漏 | 列出匹配关键词，提醒人工检查 |
| 预处理泄漏 | 代码架构保证 fit 仅在 sub_train |

### 4.4 系统层异常处理

| 异常场景 | 处理策略 |
|---|---|
| Windows GBK 编码 | `io.TextIOWrapper(stdout, encoding='utf-8')` |
| JSON 序列化 numpy | 自定义 `_json_default` 转换器 |
| 文件写入失败 | 写入后校验 `file_exists` + `file_size` |

---

## 5. 评估指标

所有模型统一计算以下指标：

| 指标 | 说明 |
|---|---|
| AUC | ROC 曲线下面积，主要排序指标 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1 | Precision 和 Recall 的调和平均 |
| Confusion Matrix | TN/FP/FN/TP |

---

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `run.py` | 命令行入口 |
| `agent_notebook.ipynb` | Jupyter Notebook（已运行，含完整输出） |
| `DESIGN.md` | 本设计文档 |
| `src/agent/agent.py` | Agent 主类 |
| `src/agent/tools.py` | Agent 工具集 |
| `src/data.py` | 数据加载与时间切分 |
| `src/preprocessing.py` | 预处理流水线 |
| `src/leakage.py` | 数据泄漏检测 |
| `src/models_nn.py` | MLP 模型 |
| `models_tree.py` | Logistic Regression + Random Forest + XGBoost |
| `src/plots_extended.py` | 增强可视化 |
| `src/metrics.py` | 评估指标计算 |
| `src/report.py` | 数据结构报告 |
| `runs/report_Y1.json` | 运行结果报告 |
