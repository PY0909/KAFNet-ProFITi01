# 当前基线与后续任务交接

更新时间：2026-10-09

## 1. 当前结论

当前论文实验主基线确立为 **Unified V3C**。后续模型改进、对比实验和结果表均应以 Unified V3C 的协议、数据划分和 evaluator 为参照，不再使用早期 Unified V2、V4 或 persistence-TCN 结果替代主基线。

Unified V3C 的正式身份为：

- 编码器：`unified_temporal_channelwise_v3c`
- 多尺度掩码内容打分时间池化（不是 Q/K 自注意力）
- 保留尺度级摘要 token（不是有序 patch 序列）
- learned sensor pooling
- masked-input invariant
- 参数按数据集独立训练

## 2. 已确认的实验与代码状态

### Unified V3C

V3C 已完成编码器实现、契约测试和 FD004 方向性消融。MetroPT、FD004 和 TEP 的训练入口均使用统一 runner；当前正式比较仍需以各数据集实际保存的 V3C artifact 为准，不把单 seed 结果解释为稳定性结论。

MetroPT 主线采用官方首月协议：第一个月训练，后续时间测试；训练内部保留 validation split；归一化统计量只由 train split 拟合；测试指标恢复到原始传感器单位。

### TCN

TCN 曾出现验证损失正常但测试预测达到百万量级的问题。已完成两类修复验证：

1. persistence + residual 版本可以稳定输出，但改变了标准 TCN 预测形式；
2. 当前标准 TCN 版本保留直接多步线性预测头，并加入残差块 LayerNorm、小增益 Xavier 初始化和统一梯度裁剪。

标准 TCN 已按 MetroPT 官方首月协议重训，但测试仍出现明显外推发散，结果不作为有效 baseline。该问题属于 TCN 实现/结构稳定性问题，不影响 Unified V3C 基线确立。

### 测试状态

当前 Python `torch23` 环境下完整测试已通过。模型比较测试覆盖 TCN、GRU、PatchTST、TimesNet 和 iTransformer 的形状、有限性、任务接口及诊断概率输出。

## 3. 结果使用规则

正式结果必须同时满足：

- 与 Unified V3C 使用相同的数据协议、split identity、归一化来源、seed 和 evaluator；
- checkpoint 只能由 validation loss 选择；
- test 只评估一次，不参与调参；
- MetroPT 报告 raw-unit overall/per-sensor MAE 与 RMSE；
- FD004 明确区分 raw RUL、training cap 和 prediction output；
- TEP 报告分类概率、Macro-F1、Balanced Accuracy、逐类指标及 run-level 结果；
- 单 seed 结果只能作为诊断，不声称多 seed 稳定性。

旧结果目录只读，不手动修改 `metrics.json`、`test_metrics.json` 或预测产物。

## 4. 后续任务顺序

### P0：冻结并核验 Unified V3C

1. 为 MetroPT、FD004、TEP 各选择明确的 V3C checkpoint 和 run identity。
2. 使用 protocol checker 检查 manifest 的 dataset、task、seed、split、target schema、normalizer、history/prediction window 和 evaluator。
3. 将通过检查的 V3C 结果登记为主基线；未通过者不得进入论文表格。

### P1：完成公平对比

1. GRU、PatchTST、TimesNet、iTransformer 按同一正式协议重训或重评估。
2. 对比模型使用与 V3C 相同的 test evaluator 和输出单位。
3. 若 TCN 继续保留，单独开展稳定性修复；在未解决测试发散前，不纳入正式性能排名。

### P2：FD004 针对性优化

1. 保持 V3C 架构不变，按 FD004 特点单独调节 RUL head、training cap、condition normalization、validation sampler 和 history length。
2. 每次只改变一个因素，先单 seed 方向筛选，再对通过的配置运行多 seed。
3. 同时报告 raw MAE、raw RMSE 和 NASA score，避免只优化单一指标。

### P3：概率与第五章产物

1. 补齐 MetroPT 分位数 test 指标、PICP、MPIW 和校准统计。
2. 补齐 TEP 概率指标和 validation-only calibration。
3. 第五章只消费第四章冻结的预测分布或分类概率，不重新使用 test label 搜索阈值。

## 5. 不应重复的工作

- 不再把 Unified V2、V4 或未通过 protocol checker 的结果作为当前主基线。
- 不把 persistence-TCN 结果与标准 TCN baseline 混列。
- 不使用测试集指标选择 epoch、阈值或归一化参数。
- 不把 smoke、单 seed 或存在数值发散的结果写成论文性能结论。
- 不修改已有 `result/` 中的历史 artifact。

## 6. 推荐交接入口

- 模型实现：`code/kaf_profiti/models/unified/`
- 统一训练入口：`code/run_unified_experiment.py`
- 对比模型入口：`compare_code.run_experiment`
- 训练与评估主逻辑：`code/kaf_profiti/experiments/entrypoint.py`
- 协议检查：`code/kaf_profiti/experiments/protocol_checker.py`
- V3 设计依据：`plan/2026-10-07-v3-encoder-ablation-plan.md`
- 当前结果产物：`final_result/ch3/metropt3/`

交接原则：任何新实验先记录协议和目标，再训练；任何新指标先核验 artifact 来源，再进入论文表格。
