V3 编码器结构对照实验方案
========================

**文档状态**：实验设计，尚未产生 V3 实验结果

**实验目标**：验证当前统一编码器中三个可能限制点预测性能的结构因素：

- 多尺度窗口内部使用均值聚合；
- 编码器不保留尺度级 patch 表示；
- 全局表示对传感器进行等权平均。

本方案只改变 Unified 编码器，不改变数据划分、目标定义、任务头接口、训练预算或测试评价口径。V3 结果必须与当前 `unified_temporal_channelwise_v2` 的受控结果分开保存和解释。

**一、研究问题与假设**

| 编号 | 研究问题 | 可检验假设 |
|---|---|---|
| H1 | 均值池化是否抹平局部趋势和突变 | 用可学习 attention pooling 替代均值后，验证集和测试集点预测误差下降 |
| H2 | patch 级时序表示是否包含额外有效信息 | 显式保留 `patch_tokens` 后，MetroPT 多步状态预测和 FD004 RUL 预测改善 |
| H3 | 传感器等权全局平均是否稀释关键变量 | learned sensor pooling 后，FD004 RUL 和 TEP 类别诊断更稳定 |

实验不预设哪一个假设一定成立。若某一变体没有改善，结果应保留并报告为负向消融证据。

**二、模型变体定义**

**V2 基线：当前 Unified 编码器**

```text
通道级 temporal MLP
  -> 全局均值、12/24/48 窗口均值
  -> 多尺度摘要拼接
  -> 跨传感器 MultiheadAttention
  -> z_history
  -> z_global = mean_sensor(z_history)
```

输出接口：

```text
z_history: [B, N, H]
z_global:  [B, H]
patch_tokens: None
```

**V3-A：attention pooling**

只替换时间窗口内部的均值聚合，其他结构保持 V2 一致。

```text
通道级 temporal block
  -> 全局尺度和 12/24/48 尺度的 attention pooling
  -> 多尺度摘要拼接
  -> 跨传感器注意力
  -> z_history
  -> z_global = mean_sensor(z_history)
```

要求：

- 每个尺度独立生成时间权重；
- 权重只依赖历史输入、时间特征和观测掩码；
- 权重在时间维归一化；
- 不读取未来目标或 query 值；
- 全缺失通道仍输出有限表示；
- `patch_tokens` 暂保持 `None`，以确保 V3-A 只检验池化方式。

**V3-B：attention pooling + patch tokens**

在 V3-A 基础上保留尺度级 token，不改变最终任务头的输入接口。

```text
通道级 temporal block
  -> 多尺度 attention pooling
  -> patch_tokens [B, N, S, H]
  -> 尺度融合
  -> 跨传感器注意力
  -> z_history [B, N, H]
  -> z_global = mean_sensor(z_history)
```

其中 `S = 1 + len(patch_lens)`，默认 `S=4`，分别对应全局、12、24、48 四个尺度。

要求：

- `patch_tokens[:, :, s, :]` 对应一个明确的时间尺度；
- token 顺序固定为全局、12、24、48；
- 任务头仍接收 `z_history` 与 `z_global`，不强制修改任务头；
- 额外的 token 融合层必须是轻量模块，避免把 V3-B 变成完整 Transformer；
- 输出 metadata 记录 `patch_token_count` 和 `patch_lens`。

**V3-C：V3-B + learned sensor pooling**

在 V3-B 基础上，将传感器等权平均替换为可学习的传感器注意力池化。

```text
V3-B patch 表示
  -> 跨传感器注意力
  -> z_history [B, N, H]
  -> learned sensor pooling
  -> z_global [B, H]
```

建议的池化形式：

```text
score_n = v^T tanh(W z_history_n + b)
weight = softmax(score, dim=sensor)
z_global = sum_n(weight_n * z_history_n)
```

要求：

- 传感器权重只由历史表示计算；
- 权重在传感器维归一化；
- 传感器数量变化时仍保持接口合法；
- 保存每次前向的权重仅用于诊断，不作为正式预测输出；
- 不根据测试误差、测试标签或人工指定的关键传感器初始化权重。

**三、统一数据与训练协议**

V3-A、V3-B、V3-C 和 V2 必须使用完全相同的：

- 原始数据文件和字段顺序；
- 数据集划分；
- 历史窗口、预测窗口和窗口步长；
- train-only 特征归一化；
- 随机种子；
- batch size、优化器、学习率和 scheduler；
- validation-only checkpoint 选择；
- test 单次评估规则。

数据集配置如下：

| 数据集 | 任务 | history | pred | stride | formal epochs | batch |
|---|---|---:|---:|---:|---:|---:|
| MetroPT-3 | 多传感器状态点预测 | 168 | 24 | 60 | 50 | 128 |
| C-MAPSS FD004 | RUL 点预测 | 50 | 1 | 1 | 80 | 128 |

说明：FD004 的正式任务是 RUL 点预测，`pred_len=1` 是模型任务头的输出长度；数据适配器保留协议中的 `pred-len=10` 参数登记，但不将未来十个周期误解释为十个独立 RUL 标签。

训练目标：

- MetroPT：在 train-only target normalization 后使用 point loss，测试指标恢复到 raw sensor units；
- FD004：训练和验证目标使用 `min(raw_rul, 125)`，测试目标使用官方 raw RUL，预测结果使用 raw cycles；
- 所有模型均以 validation loss 最低 epoch 保存 best checkpoint；
- 不使用 test 指标选择 epoch；
- 正式实验不启用 early stopping，完整执行预注册 epoch 预算。

随机种子：

- 主诊断：`2026`；
- 正式稳定性：`2026、2027、2028`；
- 每个变体和每个数据集独立训练参数；
- V3 结果不得与旧 V2 checkpoint 混合评估。

**四、实现边界**

V3 只允许修改以下编码器相关内容：

- 时间尺度内部的 pooling；
- patch token 的保存和尺度融合；
- 传感器级全局聚合；
- encoder version 和模型配置字段。

本轮不改变：

- 数据集 adapter 和 split；
- `HistoryBatch` 保持既有必需字段不变，并向后兼容地增加可选 `M_obs`；未提供时视为全观测；
- `PredictionOutput` 字段名称；
- MetroPT StateHead、FD004 RULHead、TEPHead 的损失定义；
- 既有缺失掩码语义；V3 对 `M_obs=0` 的输入值先屏蔽，且不得影响预测；
- 结果目录中的旧实验产物。

建议版本标识：

```text
unified_temporal_channelwise_v3a
unified_temporal_channelwise_v3b
unified_temporal_channelwise_v3c
```

配置中应记录：

- `variant`：`v3a`、`v3b` 或 `v3c`；
- `pooling`：`mean` 或 `attention`；
- `patch_lens`；
- `patch_token_count`；
- `sensor_pooling`：`mean` 或 `attention`；
- `hidden_dim`、参数量和训练时间。

**五、测试先行要求**

实现前新增失败测试，至少覆盖：

1. attention pooling 输出时间权重并在时间维归一化；
2. V3-A 输出有限的 `z_history` 和 `z_global`；
3. V3-B 输出形状为 `[B,N,S,H]` 的 `patch_tokens`；
4. patch token 的时间顺序变化会改变输出；
5. V3-C 的传感器权重在传感器维归一化；
6. V3-C 改变单个传感器时会改变 `z_global`；
7. 修改 `M_obs=0` 位置的输入值不会改变 V3-A/B/C 预测；
8. 全缺失通道和不规则时间戳仍能完成前向；
9. state、RUL 和 TEP 三种任务头都能消费 V3 输出；
10. state_dict 保存和恢复后输出一致。

每个失败测试必须先单独运行并确认失败，再实现最小修正，最后运行完整测试集。

**六、实验执行顺序**

第一阶段：结构 smoke

- 使用真实 MetroPT 和 FD004 数据；
- 每个变体各运行 1 epoch；
- 检查前向、反向、显存、输出形状和 artifact；
- smoke 结果不进入论文表格。

第二阶段：单 seed 诊断

- V2、V3-A、V3-B、V3-C；
- MetroPT 和 FD004；
- seed `2026`；
- 按正式 epoch 和 batch 执行；
- 使用同一 evaluator 生成 test prediction artifact。

第三阶段：正式多 seed

- 只对通过单 seed 诊断的变体运行 seed `2026、2027、2028`；
- 每个 dataset、variant、seed 生成独立结果目录；
- 先分别计算每个 seed 的完整 test 指标，再汇总 mean、sample std 和自然单位置信区间。

第四阶段：效率与表示诊断

- 统计参数量、完整训练时间、batch 推理延迟和峰值显存；
- 对 V3-B/V3-C 记录 patch token 数和传感器权重熵；
- 只将权重作为解释性诊断，不将注意力权重直接当作因果贡献。

**七、指标与结果表模板**

MetroPT-3：

| variant | seed | validation MAE | test MAE | test RMSE | valid target count | params | train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 | 2026 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| V3-A | 2026 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| V3-B | 2026 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| V3-C | 2026 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

FD004：

| variant | seed | validation MAE | test MAE cycles | test RMSE cycles | NASA score | engine count | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 | 2026 | 待运行 | 待运行 | 待运行 | 待运行 | 248 | 待运行 |
| V3-A | 2026 | 待运行 | 待运行 | 待运行 | 待运行 | 248 | 待运行 |
| V3-B | 2026 | 待运行 | 待运行 | 待运行 | 待运行 | 248 | 待运行 |
| V3-C | 2026 | 待运行 | 待运行 | 待运行 | 待运行 | 248 | 待运行 |

三 seed 汇总格式：

```text
mean +/- sample_std
```

不得将单 seed 结果写成最终性能结论，不得跨 MetroPT 和 FD004 直接平均原始误差。

**八、判定标准**

V3-A：若两个数据集至少一个主指标改善，且另一数据集没有明显退化，则认为 attention pooling 值得保留。

V3-B：除误差改善外，还必须检查训练时间和显存是否可接受。若 patch token 只增加开销而不改善指标，则不进入后续主模型。

V3-C：重点观察 FD004 RUL 与 TEP 分类；若总体误差改善但传感器权重高度塌缩，需增加正则化或降级为诊断分支。

最终主模型选择必须基于 validation 选择规则、三 seed 结果、效率和稳定性共同判断，不能只按 test MAE 选择。

**九、风险与回退策略**

- attention pooling 可能过度集中到少数时间点：加入最小温度或权重熵监控；
- sensor pooling 可能只依赖单个通道：记录权重熵并保留 mean pooling 回退；
- patch token 可能增加显存：固定 `S=4`，不增加 patch 数量；
- v3 checkpoint 不兼容 v2：禁止自动加载旧 encoder 权重；
- 任何版本若破坏 masked-input invariance，立即停止训练并修复；
- 若单个变体失败，不能修改其他变体的协议或训练预算来补偿。

**十、预期交付物**

- V3-A/B/C 编码器实现和版本化配置；
- 编码器行为测试和任务头兼容性测试；
- MetroPT 与 FD004 的 smoke、单 seed 和多 seed artifact；
- 统一指标汇总表；
- 训练时间、参数量和推理效率表；
- 结构消融结论及未解决问题记录。

在上述交付物完成前，只能称为 V3 架构实验，不能声称统一模型性能已经提升。
