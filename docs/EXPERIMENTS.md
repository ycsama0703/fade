# 实验执行手册

本文档逐阶段说明实验流程，对应 `experiments/` 下的脚本。

## 阶段 1：数据生成（`01_data_generation.py`）

### 1.1 因子池构建

```python
# 三个来源
sources = {
    "qlib_alpha158": load_qlib_alpha158(),       # 158 个
    "synthetic":     generate_random_trees(      # ~5000 candidates
                        n=5000, max_depth=4),
    "academic":      load_academic_anomalies(),  # ~50 个（可选）
}
```

合成因子的随机树生成规则：
- 叶节点从 `{open, high, low, close, volume, vwap}` 采样
- 内部节点从算子库采样（带类型约束：时序算子的子节点必须返回时序）
- 控制深度 ≤ 4

### 1.2 因子筛选

去除以下：
- 表达式恒为常数
- 含未定义运算（如 log(负数)）
- 前 12 个月平均 |IC| < 0.01（弱信号）

预期保留：800–1,500 个因子。

### 1.3 IC 时序回测

对每个因子 f：
```
for t in [discovery_month, ..., min(discovery_month + 60, 2024-12)]:
    rank_score = f.evaluate(market_data[t])
    forward_return = compute_forward_return(t, horizon=20)
    ic[t] = spearman_corr(rank_score, forward_return)
```

输出：`data/ic_series.parquet`，schema:
```
factor_id | expression | discovery_date | t (month) | ic
```

### 1.4 标签生成

调用 `src/data/label_generation.py` 生成三个标签：

```python
labels = {
    "L1_half_life":   compute_half_life(ic_series),
    "L2_survives_24m": compute_binary(ic_series, n_months=24),
    "L3_decay_slope": compute_linear_slope(ic_series),
}
```

输出：`data/labels.parquet`

---

## 阶段 2：特征工程（`02_feature_extraction.py`）

### 2.1 组 A 特征（结构）

需要先解析每个因子表达式为 AST：

```python
ast = parse_expression(factor.expression)
features_A = extract_structural_features(ast)
```

输出列：`tree_depth, node_count, ts_op_ratio, cs_op_ratio, max_lookback, op_diversity, factor_type, raw_data_types`

### 2.2 组 B 特征（早期 IC）

```python
for K in [3, 6, 12]:
    early_ic = ic_series[t0 : t0 + K]
    features_B_K = extract_early_ic_features(early_ic)
```

K 不同会产生不同的特征矩阵，分别训练对比。

### 2.3 组 C 特征（市场环境）

```python
features_C = extract_market_features(
    discovery_date=factor.discovery_date,
    factor_library=existing_factors_at_discovery_date,
    new_factor=factor,
)
```

### 2.4 特征矩阵拼接

```python
X = pd.concat([features_A, features_B, features_C], axis=1)
y = labels
```

输出：`data/feature_matrix.parquet`

---

## 阶段 3：模型训练（`03_train_models.py`）

### 3.1 数据切分

```python
train: discovery_date ∈ [2010-01, 2018-12]
val:   discovery_date ∈ [2019-01, 2020-12]
test:  discovery_date ∈ [2021-01, 2024-12]
```

注意：切分按因子的发现日期，避免未来信息泄露。

### 3.2 模型清单

```python
models = {
    "B1_global_mean":        GlobalMeanBaseline(),
    "B2_type_mean":          TypeMeanBaseline(),
    "B3_hyperbolic_fit":     HyperbolicDecayFit(),
    "B4_xgboost":            XGBRegressor(...),
    "B5_lstm":               LSTMRegressor(...),
    "M1_cox":                CoxPHFitter(),         # 主模型
    "M2_deepsurv":           DeepSurv(...),
}
```

每个模型训练后保存到 `models/checkpoints/`。

### 3.3 超参搜索

仅对 M1/M2/B4/B5 做，使用 Optuna：
- M1: 正则化强度
- B4: max_depth, learning_rate, n_estimators, subsample
- M2/B5: hidden_dim, dropout, learning_rate

---

## 阶段 4：评估（`04_evaluation.py`）

### 4.1 主结果表

为每个模型 × 每个特征组合 × 每个 K 值，输出：

| Metric | M1 Cox | B1 | B4 XGB | B5 LSTM |
|--------|--------|----|----|--------|
| MAE (months) | ... | ... | ... | ... |
| Spearman ρ | ... | ... | ... | ... |
| C-index | ... | -- | -- | -- |

### 4.2 Ablation 分析

```
特征组：    A / B / C / A+B / A+C / B+C / A+B+C
K 值：       3 / 6 / 12
```

热力图可视化各组合的性能。

### 4.3 案例研究

```python
case_factors = [
    "momentum_20d",
    "reversal_5d",
    "value_book_to_price",
    "volatility_60d",
    "turnover_anomaly",
]

for f in case_factors:
    plot_actual_vs_predicted_decay(f)
```

### 4.4 下游应用：衰减感知的因子替换

```python
# 策略：每月用模型预测所有当前因子的"剩余半衰期"
# 当预测剩余半衰期 < 阈值时，触发因子替换

baseline_strategy = RandomReplacement(prob=0.1)        # 每月 10% 概率替换
proactive_strategy = DecayAwareReplacement(model=M1)   # 基于预测

# 对比：组合层面的 IC、ICIR、最大回撤
compare_strategies(baseline_strategy, proactive_strategy)
```

---

## 算力与时间预估

| 任务 | 资源 | 时间 |
|------|------|------|
| 因子合成与回测 | CPU 多进程 | ~2 天 |
| Qlib IC 计算 | CPU | ~12 小时 |
| 特征提取 | CPU | ~1 小时 |
| Cox / XGBoost 训练 | CPU | <30 分钟 |
| LSTM / DeepSurv 训练 | 单 GPU | ~2 小时 |
| Ablation 全量实验 | 同上 | ~1 天 |

**主要瓶颈是回测**，建议提前并行化。

---

## 复现性

所有实验通过单一 YAML 配置驱动（`configs/default.yaml`），seed 固定，输出 manifest 文件记录：
- 因子集 hash
- 数据切分 hash
- 模型超参
- 评估结果

便于审稿人复现。
