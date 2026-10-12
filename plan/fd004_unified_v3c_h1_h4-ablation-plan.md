# FD004 Unified V3C H1-H4 消融方案

## 目标

在 C-MAPSS FD004 上定位 Unified V3C 与 GRU/TimesNet 的性能差异来源。所有实验均使用单 seed（2026）进行方向筛选；不得将单 seed 结果写成稳定性结论。

## 固定协议

- 数据集：C-MAPSS FD004
- 训练/验证/测试：现有 engine-level split，测试使用官方 raw RUL
- 输入传感器：14 个筛选传感器（删除 setting1-3、sensor1、5、6、10、16、18、19）
- 特征归一化：condition-aware，统计量仅由 train split 拟合
- RUL target：train/validation 使用 cap150，test 使用 official raw RUL
- validation：`all_windows`
- 历史长度：50
- batch size：128
- encoder：Unified V3C
- patch lengths：12、24、48
- seed：2026
- device：CUDA（torch23 环境）
- test evaluator：`unified_evaluator_v4`
- 测试指标：MAE、RMSE、NASA score

## 实验矩阵

| 实验 | 唯一变量 | 配置 | 目的 |
|---|---|---|---|
| Baseline | 无 | 当前 V3C，默认学习率 | 复现参考结果 |
| H1 | 学习率 | `1e-3`、`5e-4`、`2e-4` | 判断优化步长是否导致 V3C 落后 |
| H2 | RUL 过估惩罚 | `1.0`、`1.25`、`1.5` | 观察 MAE/RMSE 与 NASA score 的取舍 |
| H3 | RUL head 输入 | `z_global`；`z_global + trend` | 检验显式退化趋势是否提升 RUL |
| H4 | 传感器聚合 | attention；mean；mean+attention residual | 检验跨传感器汇总是否造成信息损失 |

## H1 执行方案

H1 只改变 optimizer learning rate，建议顺序为：

1. `1e-3`：当前 baseline，已有参考结果可复用，但建议用新 run-id 复核。
2. `5e-4`：中等学习率。
3. `2e-4`：较小学习率。

每个配置使用 120 epochs、ReduceLROnPlateau、patience 12，并保存 validation-best checkpoint。不得使用 test 指标选择学习率。

建议 run-id：

```text
cmapss_fd004_unified_v3c_h1_lr1e3_seed2026
cmapss_fd004_unified_v3c_h1_lr5e4_seed2026
cmapss_fd004_unified_v3c_h1_lr2e4_seed2026
```

## 报告规则

每个 run 必须记录：run_id、seed、dataset、model、encoder version、split identity、target schema、normalizer identity、learning rate、best epoch、validation loss、test MAE/RMSE/NASA score、预测均值和标准差。

先选择 validation loss 和预测是否塌缩均正常的配置，再比较 test 指标。若不同学习率的结论冲突，以 validation 规则为主，test 只做一次最终确认。

## 后续顺序

完成 H1 后，选择最多一个学习率进入 H2；H2 完成后再执行 H3；H4 最后执行。任何结构改动都必须与 baseline 使用相同数据协议和 evaluator。

## H1 结果与 H2 执行配置

H1 单 seed 结果已完成。按 validation-best 选择，`lr=1e-3` 最优（best validation loss 17.846）；`5e-4` 和 `2e-4` 未带来改善。因此 H2 固定 `lr=1e-3`。

H2 使用 penalty `1.0`、`1.25`、`1.5`。已有 all-windows Unified V3C baseline 使用默认 `rul_over_penalty=1.0`，作为 H2 基线复用，不重复训练。只新跑：

```text
cmapss_fd004_unified_v3c_h2_penalty1p25_seed2026
cmapss_fd004_unified_v3c_h2_penalty1p5_seed2026
```

除 penalty 外，H2 固定协议与上述 FD004 配置一致；两组都使用 `lr=1e-3`。惩罚系数须记录于 manifest。
