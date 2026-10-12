# CUDTR 模块实现与第四章实验方案

状态：方案设计，尚未实现、尚未训练，不构成性能结论。

## 1. 研究定位

### 1.1 模块名称

**CUDTR（Counterfactual Utility-Distilled Target Routing，反事实效用蒸馏的目标条件化依赖路由）**。

名称为工作名称。联网查新已发现 TFT 的样本级变量选择与门控、CAMul 的上下文相关多视图不确定性融合，以及 MoNIG 的不确定性感知模态融合。因此，CUDTR 不能把“门控、变量选择或可靠性加权”本身宣称为原创。其待验证的差异点是：用训练期源变量的反事实预测效用监督推理期路由器，并将目标条件化稀疏路由、稳定回退和概率输出组成一个可审计闭环。

### 1.2 研究问题

固定跨传感器融合默认所有源变量对所有目标、所有样本均有益。当源变量缺失、冲突或只在特定工况下有效时，固定融合可能产生负迁移。CUDTR 要回答：

> 能否用训练期“删除某一源变量后目标损失如何变化”的反事实效用，蒸馏出一个推理期可用的目标条件化路由器，在利用有效跨传感器信息的同时，在信息失效时回退到稳定基线？

该问题适用于 MetroPT-3、C-MAPSS FD004 和 TEP，但三者的目标定义不同，不能强行使用相同的传感器级目标语义。

## 2. 方法总览

```text
X_obs, T_obs, M_obs, context, target_schema
                    |
          Unified V3C / 独立数据集编码器
                    |
       目标表示 z_base 与源表示 {z_j}
                    |
      训练期：反事实效用教师 u_ij（仅 train）
                    |
      推理期：目标条件化效用预测器 u_hat_ij
                    |
       稀疏源路由 + 有界稳定回退融合
                    |
        分位数/Gaussian/分类概率预测头
                    |
       第四章冻结概率产物 -> 风险评估接口
```

`Y_q`、`y_flat`、`M_q`、`mq_flat`、`rul`、fault label 和测试统计量不得进入正式预测路径。反事实效用教师只在训练 batch 中使用训练标签生成蒸馏监督；validation/test 推理不构造教师，也不计算反事实损失。

## 3. 模块定义

### 3.1 编码接口

在现有 `UnifiedTemporalEncoder` 中增加可选的 `pre_cross_tokens`，表示跨传感器混合之前的通道级 token。不要把当前已经经过 `cross_channel` 的 `z_history` 误当作独立源表示。

建议新增：

```text
EncoderOutput.pre_cross_tokens: [B, C, H]
EncoderOutput.z_base: [B, H]
```

对 MetroPT，`C` 为目标传感器数；对 FD004，`z_base` 是 RUL 基线表示，源 token 为选定传感器表示；对 TEP，`z_base` 是检测/诊断基线表示，源 token 为过程变量表示。

### 3.2 目标与源表示

对 MetroPT 目标传感器 `i`：

```text
z_i_base = target token i
z_j_src  = source token j, j != i
```

对 FD004：

```text
z_base = global RUL representation
z_j_src = sensor j representation
```

对 TEP：

```text
z_base = pooled process representation
z_j_src = process-variable representation
```

因此 FD004/TEP 不伪造“自身目标传感器回退”，而使用原 Unified V3C 的任务基线分支作为稳定回退。

### 3.3 反事实效用教师

对训练样本、目标任务 `i` 和源变量 `j`，先用完整输入计算任务损失：

```text
loss_full = L(head(encoder(X, M)), y)
```

再按预先固定的 source-drop 规则删除源变量 `j` 的历史观测，计算：

```text
loss_drop_j = L(head(encoder(X with source j masked)), y)
u_ij = stop_gradient(loss_drop_j - loss_full)
```

`u_ij > 0` 表示删除源变量导致损失增加；`u_ij < 0` 表示该源变量可能造成负迁移。训练中可对固定遮挡比例取少量平均，避免单次遮挡噪声。遮挡方式、比例和随机种子写入配置。

效用是当前模型和当前数据协议下的预测效用，不是因果效应、物理重要性或真实故障贡献。计算教师时必须使用与正式训练相同的归一化、窗口和目标语义。

### 3.4 推理期效用预测器

路由器不能读取 `u_ij`，只使用推理时可得的特征：

```text
h_ij = concat(z_base_i, z_j_src, target_query_i,
              observation_rate_j, recent_variability_j,
              context_i)
u_hat_ij = UtilityMLP(h_ij)
```

缺失统计量由 `M_obs` 计算；波动特征只能来自历史观测。目标查询包括任务类型、目标身份/输出 schema 和可用上下文，不包含未来标签。

路由权重使用可控的稀疏映射，例如 entmax 或预先固定的 top-k softmax：

```text
score_ij = u_hat_ij / temperature
g_ij = sparse_route(score_i)
```

第一版优先使用 top-k softmax，避免引入额外稀疏算子依赖；`k` 在 validation-only 配置中确定。仅对归一化权重施加 L1 不会产生稀疏性，不能作为稀疏约束的替代。

### 3.5 稳定回退融合

源消息：

```text
m_i = sum_j g_ij * ValueProjection(z_j_src)
```

采用有界凸融合：

```text
lambda_i = sigmoid(FallbackGate(z_base_i, m_i, route_entropy_i))
z_fused_i = (1 - lambda_i) * z_base_i + lambda_i * LayerNorm(m_i)
```

为了真正接近基线，`FallbackGate` 的输出偏置初始化为较小值，使初始 `lambda_i` 接近 0；不能把“零初始化 sigmoid”错误地描述为精确的零门控。实现中保留 `lambda_min` 和 `lambda_max`，并测试有限值、范围和梯度。

当源变量全部缺失、路由效用冲突或路由熵过高时，模型应降低 `lambda_i`。回退机制是稳定性设计，不应被解释为因果机制。

### 3.6 概率输出接口

融合表示可接现有任务特定概率头：

- MetroPT：非交叉分位数或 location/scale 头，输出逐传感器状态概率；
- FD004：单值 raw RUL 的分位数或 location/scale，保留 cap150 点预测支路语义；
- TEP：detection probability 与 diagnosis probability，不使用连续回归分位数。

可选保存：

```text
route_weights, utility_prediction, route_entropy,
fallback_gate, evidence_disagreement
```

这些是模型行为诊断字段，不自动等同于传感器重要性或风险概率。

## 4. 训练目标

```text
L = L_task
  + lambda_distill * L_distill
  + lambda_rank * L_rank
  + lambda_sparse * L_sparse
  + lambda_fallback * L_fallback
  + lambda_consistency * L_consistency
```

### 4.1 任务损失

- MetroPT：点损失与真实配置启用的分位数/NLL 损失；
- FD004：raw RUL 语义的点损失和概率损失，训练 cap 与 raw 必须分开记录；
- TEP：detection BCE 与 diagnosis cross-entropy。

### 4.2 效用蒸馏损失

效用回归：

```text
L_distill = mean(Huber(u_hat_ij - clip(u_ij)))
```

效用排序：对同一目标的正效用和低效用源变量构造配对：

```text
L_rank = max(0, margin - (u_hat_ia - u_hat_ib))
```

建议第一版同时记录回归和排序指标，但只将一种作为主训练项；避免无法归因。

### 4.3 稀疏与回退损失

稀疏项优先采用 top-k 结构本身和熵惩罚：

```text
L_sparse = mean(entropy(g_i))
```

熵权重必须较小，防止所有目标只选择一个源变量。

回退项使用源变量破坏版本：

```text
L_fallback = || z_fused(corrupted) - z_base ||_2
```

只在破坏强度达到预先定义的范围时使用，避免模型学会永远关闭跨传感器融合。

### 4.4 训练阶段

1. **Stage A：教师效用生成**。冻结 V3C 和任务头，计算 train-only 的 `u_ij`；不生成 validation/test 教师文件。
2. **Stage B：路由器训练**。冻结或低学习率更新编码器，训练 `UtilityMLP`、top-k 路由和回退门。
3. **Stage C：概率联合微调**。以较小学习率联合训练概率头和路由器。
4. **Stage D：推理闭环检查**。关闭教师分支，确认正式 `predict(history)` 只接收 `HistoryBatch`，输出与训练路径一致。

## 5. 工程实现边界

建议新增：

```text
code/kaf_profiti/models/unified/routing.py
code/kaf_profiti/models/unified/utility_teacher.py
code/tests/test_cudtr_routing.py
code/tests/test_cudtr_no_target_leakage.py
code/tests/test_cudtr_fallback.py
code/tests/test_cudtr_utility_teacher.py
```

建议修改：

```text
code/kaf_profiti/models/unified/encoder.py
code/kaf_profiti/models/unified/model.py
code/kaf_profiti/models/unified/config.py
code/kaf_profiti/experiments/entrypoint.py
code/kaf_profiti/experiments/artifacts.py
```

配置至少记录：

```text
cudtr_enabled
utility_target
counterfactual_drop_mode
counterfactual_drop_rate
route_temperature
route_top_k
lambda_distill
lambda_rank
lambda_sparse
lambda_fallback
lambda_consistency
fallback_gate_init
```

正式 manifest 还需记录 dataset、task、seed、split identity、target schema、encoder version、概率头类型和 raw/cap 语义。用户已要求不绑定 hash，因此不得新增内容 hash 或 manifest hash 字段。

## 6. 必须先写的测试

### 6.1 预测路径无标签

- `predict(HistoryBatch)` 不接受或读取 targets；
- 删除 batch 中的目标字段不会改变预测 API；
- 训练教师存在与否不改变正式推理输出接口；
- 路由模块源码/运行图不读取 `Y_q`、`y_flat`、`rul` 或 fault label。

### 6.2 反事实教师

- 完整输入和源变量遮挡输入产生不同损失；
- `u_ij` 的梯度被 `stop_gradient` 截断；
- 源变量全部无效时效用标签有限；
- FD004 标量 RUL 和 TEP 分类目标均能生成对应效用。

### 6.3 路由与回退

- top-k 权重非负、和为 1；
- 路由温度和 k 的边界有效；
- `lambda_i` 在预定范围内；
- 全部源缺失时输出接近 `z_base`；
- 修改未选源变量不影响输出，或只产生规定的数值容差变化；
- 所有输出和梯度有限。

### 6.4 概率输出

- 分位数不交叉；
- location/scale 的 scale 正值；
- TEP 概率和为 1；
- calibration 只接收 validation split；
- 第四章冻结产物可被第五章读取而不重新训练模型。

## 7. 第四章实验方案

### 7.1 章节题目

建议题目：**基于反事实效用蒸馏目标路由的多传感器概率状态预测方法**。

章节主张必须限定为待验证命题：CUDTR 是否能够在减少无效跨传感器依赖的同时，维持或改善点预测和概率预测质量，并在源变量损坏时保持较稳定的退化。

### 7.2 统一协议

所有模型使用相同数据划分、train-only 归一化、目标语义、缺失掩码、batch size、训练预算、validation-best 规则和 test evaluator。MetroPT 使用官方首月训练、余下测试协议；FD004 使用 engine 隔离和标量 raw RUL 评估；TEP 使用 run 隔离和检测/诊断概率评估。

正式测试只运行一次。校准参数、阈值和 top-k 仅由 validation split 确定。测试集不得用于效用教师、路由超参数搜索或阈值搜索。

### 7.3 实验矩阵

| 编号 | 实验 | 设置 | 目的 |
|---|---|---|---|
| E0 | 工程正确性 | 合成数据、无标签推理、回退测试 | 排除泄露和数值错误 |
| E1 | 融合主对比 | V3C、普通 attention、拼接 MLP、自身基线、CUDTR | 判断收益是否来自路由机制 |
| E2 | 参数量控制 | CUDTR 与加宽普通融合头 | 排除容量收益 |
| E3 | 反事实教师消融 | 完整 CUDTR、无教师、打乱教师、仅正效用 | 检查新监督信号 |
| E4 | 路由结构消融 | 无 target query、无 top-k、无 fallback、无负效用 | 定位机制贡献 |
| E5 | 源破坏鲁棒性 | 随机缺失、块缺失、指定传感器移除、噪声注入 | 检查负迁移抑制和回退 |
| E6 | 概率质量 | point、普通 quantile/Gaussian、CUDTR 概率头 | 检查概率输出收益 |
| E7 | 路由敏感性 | temperature、top-k、效用权重、回退权重、遮挡比例 | 评估稳定区间和计算代价 |
| E8 | 跨任务验证 | MetroPT、FD004、TEP 独立训练 | 检查方法是否超出单一数据集 |

### 7.4 主对比模型

最小可行矩阵：

1. Unified V3C 原始模型；
2. V3C + 普通跨传感器 attention；
3. V3C + 参数量匹配 MLP；
4. V3C + 仅自身/任务基线分支；
5. V3C + CUDTR。

TFT、CAMul 或 MoNIG 只有在数据协议、输出语义和训练预算能够公平复现时才加入正式主表，否则作为文献方法对照和实现性讨论，不能用不完整复现结果排名。

### 7.5 指标

连续任务：MAE、RMSE、Pinball、WIS/区间分数、PICP、MPIW、NLL 或 CRPS（仅在输出定义支持时）。FD004 额外报告 raw NASA score 和 engine-level 指标。

TEP：detection/diagnosis log loss、Brier、ECE、Macro-F1、Balanced Accuracy；事件级延迟和误报属于冻结概率后的风险评估，不与概率主指标混写。

路由机制：效用排序 Spearman/Kendall、top-k 命中率、路由熵、回退门均值、源变量移除后的损失变化、跨扰动路由一致性。

效率：参数量、训练时间、单 batch 推理时间和显存。不得把训练速度收益当作精度机制证据。

### 7.6 统计与报告

正式主结果至少使用三个 seed；单 seed 只用于筛选。MetroPT 以时间块或故障事件为独立单位，FD004 以 engine 为独立单位，TEP 以 run 为独立单位进行 bootstrap 或配对比较。窗口数量不能直接作为独立样本数。

结果表必须同时报告均值、标准差、样本/独立单位数和配置身份。没有真实结果前使用“待验证”，不得填写模拟数字。

## 8. 图表设计

| 图表 | 回答的问题 | 内容 |
|---|---|---|
| 图4-1 | 模块如何工作 | V3C、效用教师、推理路由、回退、概率头和冻结接口 |
| 图4-2 | 是否改善概率质量 | 覆盖率-宽度、WIS/NLL、三数据集分面 |
| 图4-3 | 路由是否学习到效用 | 反事实效用与预测效用散点/排序曲线 |
| 图4-4 | 回退是否有效 | 源变量破坏强度-性能下降-回退门变化 |
| 图4-5 | 哪个组件贡献收益 | E3/E4 消融森林图或配对差值图 |
| 图4-6 | 超参数是否稳定 | temperature、top-k、损失权重敏感性曲线 |
| 表4-1 | 协议是否公平 | 数据划分、目标语义、训练预算和指标 |
| 表4-2 | 主模型是否有效 | 各融合方法的点和概率指标 |
| 表4-3 | 是否具有跨任务适配性 | MetroPT、FD004、TEP 的任务特定结果 |

路由权重图只能说明模型行为，不能用标题或正文写成“物理传感器因果关系”。

## 9. 第四章与第五章接口

第四章冻结：checkpoint、原始概率预测、校准前指标、route weights、utility predictions、fallback gate、split identity 和 target schema。

第五章只消费冻结的概率产物，执行 validation-only calibration、风险分数、阈值、持续报警和事件评估。第五章可以分析路由与风险的关系，但不能用 test 标签重新选择路由温度、top-k 或报警阈值。

## 10. 阶段门禁与否证条件

### Gate 0：查新和接口

完成针对 counterfactual variable utility、target-conditioned routing、negative-transfer suppression 的原文查新；将 TFT、CAMul、MoNIG 作为最近邻方法写入差异表。未完成前不使用“首次提出”。

### Gate 1：工程正确性

所有 CUDTR 测试通过，正式预测路径无标签输入，旧 V3C checkpoint 仍可评估。

### Gate 2：单 seed 筛选

在 MetroPT 和 FD004 先验证：点预测不显著退化、源破坏下性能下降更小、效用排序为正向、回退门按预期变化。

### Gate 3：扩展 TEP

只有 Gate 2 通过后才进行 TEP；TEP 以分类概率和 run-level 指标验证，不强行报告连续区间。

### Gate 4：正式结论

至少两个数据集在相同方向上支持机制证据，并完成三个 seed；否则将贡献限定为“在特定任务上的候选机制”，不写成统一方法已被证明有效。

## 11. 当前未解决问题

1. CUDTR 与已有“反事实变量选择/不确定性感知路由”工作的精确新颖性边界仍需全文查新。
2. 反事实效用计算会增加训练成本，需通过 source batch 化或每步抽样控制开销。
3. 负效用的估计受基线模型偏差影响，不能直接等同于真实变量因果效应。
4. MetroPT 故障事件稀疏，事件级预警结论需要谨慎；主实验仍应报告健康期误报与概率质量。
5. CUDTR 尚未实现、测试、训练或产生任何正式指标。
