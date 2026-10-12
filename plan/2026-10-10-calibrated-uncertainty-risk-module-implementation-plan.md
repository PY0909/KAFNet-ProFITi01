# 第四章任务条件化结构化概率预测模块实现方案

## 1. 章节定位

**建议题目：**面向异步多传感器时序的任务条件化结构化概率预测方法。

第四章只研究如何生成概率预测，第五章研究概率预测的校准、风险和预警：

```text
第四章：历史观测 -> 概率预测产物
第五章：概率预测产物 -> 校准、风险分数、预警事件
```

本方案不以多步预测为创新依据，不构造 FD004 多步 RUL 标签，不要求完整联合采样。

## 2. 方法目标与贡献

在 Unified V3C 历史编码器基础上，增加任务条件化结构化概率解码器：

```text
异步历史观测
 -> Unified V3C 历史编码
 -> 任务/目标查询
 -> location 分支 + uncertainty 分支
 -> 可选低秩共享不确定性因子
 -> 任务特定概率输出
 -> 第四章冻结概率产物
 -> 第五章校准与风险评估
```

三个可验证的模块贡献：

1. 统一任务条件化概率解码接口，适配状态、RUL 和分类概率三种输出语义。
2. 分离 location 与 uncertainty 分支，支持非交叉分位数和异方差参数化。
3. 在 MetroPT 上提供可选低秩共享不确定性因子，表达多传感器共同残差变化。

这些属于结构化概率解码和跨任务适配，不能声称首次提出 Gaussian、分位数、copula 或 conformal。

## 3. 第四章与第五章边界

| 内容 | 第四章 | 第五章 |
|---|---|---|
| 概率头 | 设计、训练、比较 | 读取冻结输出 |
| pinball、NLL、CRPS | 原始概率质量 | 校准前后补充比较 |
| PICP、MPIW、interval score | 原始区间质量 | conformal 前后对照 |
| conformal | 不作为主方法 | validation-only 校准 |
| TEP 概率 | 原始检测/诊断概率 | Brier、ECE、可靠性 |
| 风险分数 | 不形成正式预警结论 | 由冻结概率生成 |
| threshold/persistence | 不使用 | validation-only 拟合 |
| 误报、漏报、检测延迟 | 不作为主结果 | 事件级风险评估 |

## 4. 模型设计

### 4.1 任务查询

```text
q = Query(task_id, target_schema, target_id, context)
```

允许使用任务类型、输出 schema、MetroPT 目标传感器身份、历史可得工况和历史编码表示。禁止使用未来目标、故障标签、fault number、测试统计量和未来工况。

### 4.2 双分支概率解码

连续任务：

```text
mu = f_mu(z, q)
raw_scale = f_sigma(z, q)
sigma = softplus(raw_scale) + epsilon
```

分位数任务：

```text
q_0.5 = base(z, q)
q_k = q_previous + softplus(delta_k(z, q))
```

位置和尺度使用独立输出层，并记录尺度下限、裁剪范围和有限梯度检查。

### 4.3 可选低秩因子

仅在 MetroPT 做扩展：

```text
y = mu + D * epsilon + V * f
Sigma = D^2 + V V^T
```

rank=0 必须等价于 diagonal Gaussian。第一版不要求联合采样。FD004 使用标量 RUL 分布，TEP 使用分类概率。

## 5. 统一概率输出

扩展 `PredictionOutput` 或新增兼容容器：

```python
ProbabilisticOutput(
    point=None,
    quantiles=None,
    probabilities=None,
    location=None,
    scale=None,
    covariance_factor=None,
    metadata={},
)
```

MetroPT 保存 point/quantiles 或 location/scale；FD004 保存 RUL point/quantiles 或 location/scale，并保留 raw/cap metadata；TEP 保存 detection/diagnosis probabilities。第四章只保存原始概率输出。

metadata 至少记录 dataset、task、target schema、head type、分位数或分布名、training cap、split identity、normalization 来源、seed 和 encoder version。

## 6. 第四章评价指标

### 连续预测

实现 pinball、quantile crossing rate、PICP、MPIW、interval score、weighted interval score 和 quantile-wise coverage，并支持 overall、sensor-wise、horizon-wise、engine-wise、run-wise 聚合。

```text
coverage = mean(q_low <= y <= q_high)
width = mean(q_high - q_low)
interval_score = width
                 + 2/alpha * (q_low-y) * I[y<q_low]
                 + 2/alpha * (y-q_high) * I[y>q_high]
```

第一版 CRPS 使用分位数加权近似，并记录分位数网格与权重。Gaussian/Student-t 接入后再增加解析 CRPS。

### 参数化分布

```text
NLL = 0.5 * ((y-mu)/sigma)^2 + log(sigma) + const
```

Student-t 仅在 Gaussian 显示厚尾不足时实施。flow、diffusion 和复杂 copula不属于第一版。

### TEP

第四章报告 detection log loss、diagnosis log loss、raw Brier、macro-F1 和 balanced accuracy。ECE、可靠性图和校准后概率放入第五章。

## 7. 三数据集适配

### MetroPT-3

输出逐传感器 point/quantile，可选 Gaussian location/scale 和低秩因子。报告 MAE、RMSE、pinball、CRPS、NLL、PICP、MPIW 和 interval score。

### C-MAPSS FD004

输出单值 RUL point/quantile 或 location/scale，保留 raw RUL 和 training cap。报告 raw MAE、raw RMSE、NASA score、RUL CRPS/NLL、coverage、width 和 per-engine 误差。不得报告多步 RUL或传感器联合区间。

### TEP

输出 detection probability、diagnosis probability、原始 log loss/Brier。第五章读取这些概率执行校准、阈值、persistence和事件评估。故障标签不进入第四章推理路径。

## 8. 分阶段实现

### Stage 0：协议与概率产物合同

先写测试，固定 point/quantile checkpoint 身份、target schema、raw/cap 语义、概率 artifact 字段以及第四章—第五章读写边界。

门禁：旧 checkpoint 可恢复，既有点预测回归通过。

### Stage 1：概率指标与序列化

新增：

```text
code/kaf_profiti/experiments/probabilistic_metrics.py
code/tests/test_probabilistic_metrics.py
code/tests/test_probability_artifacts.py
```

实现 pinball、CRPS approximation、NLL、PICP、MPIW、interval score、TEP log loss/Brier、分组聚合和原始概率产物保存。

### Stage 2：任务条件化概率解码

新增：

```text
code/kaf_profiti/models/unified/query.py
code/kaf_profiti/models/unified/heads/gaussian.py
code/tests/test_task_conditioned_probability.py
code/tests/test_distribution_heads.py
```

顺序：统一 query shape；实现 Gaussian location/scale；接入 MetroPT/FD004；TEP保留分类概率头；保持旧 TaskHead 接口兼容。

门禁：sigma>0、梯度有限、预测路径不读 targets、三个任务 shape通过。

### Stage 3：条件化尺度消融

比较 point/quantile、unconditional Gaussian、target/context-conditioned Gaussian和参数量匹配确定性头，区分“有尺度头”和“条件化结构”的收益。若只提高 NLL却明显损害点精度，停止扩展。

### Stage 4：MetroPT低秩因子（可选）

比较 diagonal Gaussian、diagonal+low-rank及rank敏感性，检查Cholesky、正定性和有限梯度。只有CRPS/NLL或残差相关性有明确收益时纳入主模型。

### Stage 5：第四章冻结与第五章交接

第四章冻结 checkpoint、raw probability predictions、distribution parameters、quantile levels、target schema、split identity 和 raw probability metrics。第五章独立新增：

```text
code/kaf_profiti/calibration/
    conformal.py
    classification.py
    threshold.py
    persistence.py
code/kaf_profiti/experiments/risk.py
```

这些模块只消费冻结产物，不反向改变第四章模型。

## 9. 第四章实验矩阵

| 编号 | 模型 | 是否重训 | 目的 |
|---|---|---:|---|
| P0 | Unified V3C point | 否/已有 | 点预测基线 |
| P1 | 真正以 quantile loss 训练的 head | 是/需确认 | 非参数概率基线 |
| P2 | Gaussian location/scale | 是 | 参数化概率基线 |
| P3 | Student-t location/scale | 可选 | 厚尾对照 |
| P4 | task/context-conditioned Gaussian | 是 | 条件化尺度消融 |
| P5 | MetroPT diagonal + low-rank Gaussian | 是 | 共享不确定性因子 |
| P6 | 参数量匹配确定性头 | 是 | 排除容量收益 |
| P7 | 条件置换负对照 | 是/重评估 | 检查 query 作用 |

正式结论至少三个 seed；单 seed只筛选高成本分支。

## 10. 第五章接口

第五章读取第四章冻结的概率产物，执行：

1. validation-only conformal；
2. TEP temperature/Platt/isotonic；
3. exceedance calibration；
4. validation-only threshold；
5. persistence/alarm merging；
6. TEP run-level误报、漏报、延迟；
7. MetroPT 健康期误报与故障前风险轨迹；
8. FD004 engine-level RUL 风险分层。

第五章必须区分 marginal coverage、simultaneous coverage、joint coverage 和 event recall。CRC控制期望风险，不能自动保证每个事件被检测。

## 11. 结果图表

第四章：

1. V3C—query—location/scale—probability output结构图；
2. 点预测与概率头对照；
3. MetroPT sensor-wise CRPS/NLL/coverage；
4. FD004 RUL分布质量与 per-engine误差；
5. TEP raw detection/diagnosis概率质量；
6. 低秩或任务条件消融；
7. 参数量、训练、推理和显存。

第五章：

1. 校准前后 coverage/width；
2. TEP reliability diagram、Brier、ECE；
3. 风险阈值与误报/召回权衡；
4. run-level alarm timeline；
5. persistence敏感性；
6. 缺失率鲁棒性和多 seed统计。

## 12. 停止准则

停止新增第四章概率模块的条件：

- Gaussian/Student-t相对分位数没有明确分布评分收益；
- 条件化尺度只增加参数而没有可重复收益；
- 低秩结构没有改善 NLL、CRPS或残差相关性；
- 点精度明显下降；
- 仅一个数据集改善且无任务语义解释；
- 需要测试标签选择概率头或超参数；
- 出现非有限NLL、尺度爆炸或 checkpoint不可恢复。

第五章风险模块停止条件单独记录，不反向修改第四章概率模型。

## 13. 完成标准

- 三个数据集产生正式概率输出；
- 第四章只评价原始概率质量；
- 第五章只消费冻结概率产物；
- FD004 raw/cap语义清晰；
- TEP detection/diagnosis probability为正式字段；
- calibration、threshold和persistence参数全部来自validation；
- 第四章和第五章不混淆风险指标；
- 新增测试通过；
- 至少三个 seed 支撑正式结论；
- 未完成的通用数据集实验不写成通用性结论。

