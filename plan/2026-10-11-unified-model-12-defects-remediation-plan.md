# Unified/CUDTR 十二项设计问题分阶段解决方案

## 1. 目标与边界

本文针对当前 Unified V4/V3C 与 CUDTR 实现中已经发现的 12 个问题，给出代码修复、协议修正、结构消融和三数据集验证方案。

MetroPT-3 继续采用官方基准协议：第一个自然月用于训练与 validation，后续月份只用于最终 test。该 train/test 协议本身不修改。需要单独处理的是 quantile calibration 的 validation 内部划分，因为 validation 前后滑动窗口可能共享原始时间点。

三个数据集的任务语义保持不同：

| 数据集 | 任务 | CUDTR 目标语义 |
|---|---|---|
| MetroPT-3 | 多传感器状态预测 | 逐目标传感器、逐源传感器路由 |
| C-MAPSS FD004 | 单值 RUL 预测 | 面向 RUL 的任务级源选择 |
| TEP | 故障检测与诊断 | 检测目标和诊断目标分别评价，不能使用故障标签参与推理路由 |

所有实验必须满足：训练集归一化、validation-only checkpoint/阈值/校准参数、test 只做最终评估、固定 seed 与数据划分、保存完整 manifest。单 seed 结果只能作为诊断，不能直接作为论文性能结论。

## 2. 十二项问题清单

| 编号 | 问题 | 类型 | 主要影响 |
|---|---|---|---|
| P1 | 全缺失源通道仍可能进入路由 | 明确代码缺陷 | 无效源污染、路由解释失真 |
| P2 | CUDTR source 与 target base 位于不同融合层 | 结构不一致 | teacher/student utility 不一致 |
| P3 | validation 校准子集仍存在底层窗口重叠 | 校准协议问题 | 校准指标可能不稳定或偏乐观 |
| P4 | fallback 初始值为 1，路由可能长期退化为 baseline | 训练退化风险 | CUDTR 实际贡献无法确认 |
| P5 | hard top-k 选择不可微且可能过早锁定 | 优化问题 | 错误源选择、梯度稀疏 |
| P6 | 全量 cross-channel attention 与 CUDTR 功能重复 | 结构冗余 | 负迁移、增参但收益有限 |
| P7 | CUDTR 后 MetroPT 又做全局 mean，但 StateHead 不消费 z_global | 接口语义问题 | 配置名与实际作用不一致 |
| P8 | 多尺度历史最终过早压缩为单一传感器向量 | 表达能力问题 | 趋势、突变、尺度冲突丢失 |
| P9 | StateHead 没有 horizon-specific 解码 | 预测头限制 | 近端与远端预测共享同一摘要 |
| P10 | FD004/TEP utility 没有区分任务目标 | 跨数据集适配问题 | RUL、检测、诊断的路由信号混合 |
| P11 | TEP 普通 BCE/CE 未处理类别不平衡 | 训练与评价问题 | 正常类主导、故障召回不足 |
| P12 | TEP 事件阈值固定为 0.5 | 校准问题 | 检测延迟和误报依赖任意阈值 |

## 3. 阶段总览

| 阶段 | 内容 | 解决问题 | 是否需要重训 | 阶段产物 |
|---|---|---|---|---|
| Stage 0 | 建立缺陷回归测试与指标基线 | 全部问题的可观测性 | 否 | 缺陷清单、回归测试、诊断脚本 |
| Stage 1 | 修复输入有效性、路由表示和 fallback 诊断 | P1、P2、P4、P7 | 是 | 修复后的 Unified/CUDTR 前向与路由统计 |
| Stage 2 | 重构 cross-sensor 融合和 top-k 训练 | P5、P6 | 是 | soft-to-hard/CUDTR-only 消融 |
| Stage 3 | 修正 validation 校准隔离 | P3、P12 的协议基础 | 需要受控重训/重评估 | 无底层窗口重叠的 selection/calibration |
| Stage 4 | 保留多尺度与 horizon 信息 | P8、P9 | 是 | patch-aware/horizon-aware 预测头 |
| Stage 5 | 数据集任务语义与 TEP 训练校准 | P10、P11、P12 | TEP 必须重训 | detection/diagnosis 独立结果 |
| Stage 6 | 统一消融、敏感性和三数据集正式验证 | 全部 | 是 | 论文表格、图和结论依据 |

每个涉及代码的阶段都必须遵循：先写测试，再实现；测试通过后单独提交；训练结果写入新的 result 子目录，不覆盖已有结果。

## 4. Stage 0：建立可观测性与回归门禁

### 4.1 实现内容

新增或完善以下诊断字段：

- 每个目标的 `route_weights`、`route_entropy`、`fallback_gate`；
- 每个源的实际选中次数、选中比例和平均权重；
- 全缺失源被选中次数，必须为 0；
- CUDTR 融合表示与 baseline 表示的距离；
- 每个传感器和每个测试时间块的 point/quantile 指标；
- TEP detection/diagnosis 分别记录 AUROC、AUPRC、macro-F1、事件召回、误报和检测延迟。

### 4.2 必测回归

- 修改 `M_obs == 0` 位置的输入值不改变输出；
- 全缺失通道不能进入 route mask；
- target sensor 不得选择自身为 source；
- 无可用 source 时输出必须等于 baseline；
- route weights 在有效 source 上归一化；
- point、quantile、TEP probability 输出形状和 finite 检查；
- test evaluator 不读取 test label 进行 checkpoint、阈值或校准选择。

### 4.3 验收标准

现有回归测试全部通过；新增测试能够在没有真实数据时捕获 P1、P2、P4、P5 和 P7；诊断字段写入预测产物或独立诊断 JSON。

## 5. Stage 1：修复有效源、表示层和 fallback

### 5.1 P1：全缺失源屏蔽

将源可用条件从 `rate >= 0` 改为明确的正观测率条件：

```text
available = source_observation_rate > max(min_observation_rate, 0)
```

当 `M_obs` 不存在时，视为全部观测；当某通道全缺失时，无论 utility prediction 多大，都不能进入 top-k。

### 5.2 P2：统一 source/base 表示层

采用单一融合边界。推荐实现为：

```text
pre-cross sensor tokens
    -> CUDTR target-conditioned sparse routing
    -> selected-source message + target self history
    -> one shared normalization/context block
    -> task head
```

teacher utility 也在同一边界删除 source message。禁止使用 pre-cross source 与 post-cross target base 直接相加。teacher、student、fallback 使用相同的表示层和相同的归一化路径。

### 5.3 P4：fallback 诊断和约束

保留 baseline fallback 作为稳定路径，但增加：

- clean input 下 fallback 均值、分位数和逐目标分布；
- source corruption 下 fallback 变化；
- route message 与 baseline 的相对范数；
- fallback 饱和率，例如 `fallback > 0.95` 的比例；
- clean route 与 corrupted route 的一致性损失。

训练时使用有界 fallback，并把 fallback 统计写入 manifest 和 test prediction diagnostics。若 fallback 长期接近 1，标记为“路由未生效”，不能宣称 CUDTR 有效。

### 5.4 P7：修正传感器汇总语义

MetroPT StateHead 直接消费 `[B, N, H]` 的 `z_history`，因此 `z_global` 的 mean pooling 不参与当前状态预测。应在配置和 manifest 中明确：

- MetroPT：逐目标 sensor representation 直接进入 StateHead；
- FD004/TEP：任务头消费 global representation；
- `sensor_pooling` 只在确实消费 `z_global` 的任务中生效。

不要把 MetroPT 的 mean pooling 作为 CUDTR 贡献的一部分进行解释。

### 5.5 阶段实验

固定 V4 encoder、quantile/point head、seed、batch、学习率和官方协议，比较：

1. 原始 CUDTR；
2. P1 修复；
3. P1+P2；
4. P1+P2+fallback 诊断与约束。

报告 point MAE/RMSE、Pinball、WIS、coverage、fallback 饱和率和 source usage。

## 6. Stage 2：重构稀疏路由与 top-k 优化

### 6.1 P5：soft-to-hard routing

`top-k=2` 是当前启发式值，不是理论规定。它的含义是每个目标最多使用两个外部传感器，目标自身通过 fallback 保留。对于 MetroPT 的 15 个传感器，该设置可能过稀疏；对 FD004 和 TEP 也不应直接照搬。

采用两阶段训练：

1. 前若干 epoch 使用 soft routing，让所有可用 source 获得梯度；
2. 后续逐渐降低温度或使用 straight-through top-k；
3. 最终推理阶段固定稀疏规则，并记录实际 source usage。

### 6.2 P6：取消重复的全量跨传感器融合

设计三条结构臂：

| 结构 | 全量 cross-channel attention | CUDTR sparse routing | 用途 |
|---|---:|---:|---|
| Attention-only | 有 | 无 | V4 结构基线 |
| CUDTR-only | 无 | 有 | 判断 CUDTR 独立贡献 |
| Attention+CUDTR | 有 | 有 | 判断叠加是否有价值 |

正式主模型优先采用 CUDTR-only 或明确的顺序结构，避免全量 attention 已经完成跨传感器混合后，CUDTR 只做重复修正。

### 6.3 top-k 敏感性

只使用 validation 选择，不读取 test 标签。MetroPT 建议比较：

```text
k = 1, 2, 4, 8, all
```

FD004 和 TEP 使用相同候选集合，但报告传感器数量归一化后的稀疏率，不直接把 `k=2` 当作跨数据集固定设定。

### 6.4 阶段验收

至少完成 Attention-only、CUDTR-only、Attention+CUDTR 和 soft-to-hard 四个结构结果。若 CUDTR-only 不优于 Attention-only，应把结论写成“路由提供条件化解释/鲁棒性能力”，不能声称点预测性能提升。

## 7. Stage 3：严格处理 validation 校准隔离

### 7.1 协议边界

MetroPT 官方协议保持：

```text
首月前段/训练部分 -> train
首月后段          -> validation
其余月份          -> test
```

点预测使用完整 validation 做 checkpoint 选择是允许的。只有当 validation 同时承担 checkpoint 选择和 quantile calibration 时，才需要进一步拆分。

### 7.2 无底层窗口重叠的拆分

不能只按 validation sample 列表前后切分。应在生成窗口前按原始时间块确定：

- selection block：只用于 Pinball/MAE checkpoint 选择；
- calibration block：只用于估计校准参数；
- 两块之间设置至少 `history_len + pred_len` 的 gap，或使用不重叠的窗口起点集合。

两个子集都必须位于首月 validation 时间段内，不能改变官方 test 月份边界。

### 7.3 校准方法

比较 validation-only 的：

- median-residual calibration；
- CQR nonconformity；
- 必要时按传感器分组校准。

测试集只报告 raw 与 calibrated coverage、width、WIS，不参与校准方法选择。

### 7.4 阶段验收

manifest 记录 selection/calibration 的时间范围、窗口数量和 gap；自动测试验证底层时间索引不交集；重新训练 CUDTR 与匹配 baseline。

## 8. Stage 4：恢复多尺度和预测步信息

### 8.1 P8：延迟多尺度压缩

当前各尺度最终汇总为一个 `[B, N, H]` 表示，导致 patch 级时序信息丢失。改为保留：

```text
[B, N, scale, H]
```

由预测头或轻量 decoder 学习尺度权重。至少保留 global、recent、trend 三类 token，并输出尺度权重用于解释。

### 8.2 P9：horizon-specific decoder

对 MetroPT 状态预测增加预测步 query：

```text
sensor representation + horizon embedding
    -> horizon decoder
    -> [B, pred_len, N, output]
```

FD004 当前仍是单值 RUL，不伪造多步 RUL 标签；TEP 仍使用任务级分类输出。该阶段只在有真实多步标签的 MetroPT 上启用，其他数据集使用兼容的 global head。

### 8.3 消融

- 单一汇总向量；
- 保留尺度 token；
- 尺度 token + horizon query；
- recent/trend only；
- global/recent/trend 全部输入。

报告近端与远端预测步的 MAE/RMSE，以及整体指标变化。

## 9. Stage 5：FD004 与 TEP 任务适配

### 9.1 P10：任务级 utility 分解

FD004 只有一个 RUL 目标，使用任务级 source utility 是合理的，但应在 manifest 中明确 `target=rul`。

TEP 至少构造两套训练期 utility：

- detection utility：删除 source 后 detection loss 的变化；
- diagnosis utility：删除 source 后 diagnosis loss 的变化。

推理时只使用历史输入、mask 和 context，不读取故障标签。路由器可使用共享 utility encoder，再为 detection/diagnosis 使用独立 query 或独立 utility head。

### 9.2 P11：TEP 类别不平衡

类别权重或 focal 参数只能由 train/validation 统计确定。建议依次比较：

1. 普通 BCE/CE；
2. train-frequency balanced BCE/CE；
3. focal/balanced variant。

测试集不用于选择损失权重。报告 detection AUPRC、AUROC、positive recall、macro-F1、diagnosis macro-F1 和 per-class confusion matrix。

### 9.3 P12：TEP validation-only 阈值和事件规则

在 validation 上确定 detection threshold，并固定应用到 test。阈值目标应预先规定，例如：

- 固定 validation false alarm rate；
- validation F1；
- validation event recall 与 false alarm 的加权目标。

连续报警 persistence 长度也必须在 validation 选择并记录。test 只计算检测延迟、事件召回、漏报率和误报率。

## 10. Stage 6：正式实验矩阵

### 10.1 主比较

三个数据集都执行：

- V4/Unified baseline；
- CUDTR-only；
- CUDTR + fallback；
- 完整改进模型。

所有模型固定数据划分、normalizer、seed、batch size、epoch 上限和 checkpoint 规则。

### 10.2 必做消融

| 消融 | 目的 |
|---|---|
| 去除 CUDTR | 判断新增模块的独立贡献 |
| 去除 fallback | 判断稳定回退的作用 |
| soft vs hard routing | 判断不可微选择的影响 |
| k=1/2/4/8/all | 判断稀疏率敏感性 |
| attention-only/CUDTR-only/组合 | 判断结构冗余 |
| 无 source mask/有 source mask | 验证全缺失通道修复 |
| 单尺度/多尺度 | 验证 P8 |
| 无 horizon query/有 horizon query | 验证 P9，仅 MetroPT |
| 普通/平衡 TEP loss | 验证 P11 |
| 固定 0.5/validation threshold | 验证 P12 |

### 10.3 结果报告

MetroPT：overall 和逐传感器 MAE/RMSE、逐预测步误差、Pinball、coverage、width、WIS、route usage。

FD004：raw RUL MAE/RMSE、NASA score、按 engine 统计、source usage 和 fallback 分布。

TEP：detection AUROC/AUPRC/F1、diagnosis macro-F1、per-class recall、事件召回、误报、漏报、检测延迟。

## 11. 阶段门禁与论文结论边界

每阶段结束必须报告：修改文件、完成内容、测试命令和结果、未解决问题、下一阶段计划。正式论文只使用有完整 run_id、seed、dataset、task、split identity、target schema 和 checkpoint provenance 的结果。

以下情况不得宣称 CUDTR 提升点预测性能：

- 只与不同 head 或不同 pooling 的模型比较；
- 只报告单一 seed；
- fallback 长期接近 1；
- CUDTR-only 未与 Attention-only 比较；
- validation threshold/calibration 使用了 test 标签；
- P8/P9 只在 MetroPT 具备真实多步标签时验证，却推广到 FD004/TEP。

建议的实施顺序是：先完成 Stage 0 和 Stage 1，再进行 Stage 2 的结构消融；Stage 3 独立修正校准协议；最后实施 Stage 4 和 Stage 5，并用 Stage 6 的统一矩阵形成第四章和第五章的可复现证据链。
