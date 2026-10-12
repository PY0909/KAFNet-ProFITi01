# 第三章实验指标交接说明

## 1. 当前 artifact

目标目录：`result/metropt3_state_seed2026/`

`manifest.json` 表明：

| 字段 | 当前值 | 解释 |
|---|---|---|
| `run_id` | `metropt3_state_seed2026` | 单次 MetroPT-3 状态预测运行 |
| `dataset` / `task` | `metropt3` / `state` | MetroPT-3 状态点预测任务 |
| `seed` | `2026` | 单一随机种子，不能形成均值/标准差 |
| `dataset_schema` | 15 个传感器列 | 已记录输入列身份，需与 loader schema 一致 |
| `split_identity` | `stable_unit_split_v1-units1516948` | 数据划分身份 |
| `normalizer_identity` | `train-zscore-features15` | 训练集 z-score 归一化身份 |
| `target_schema` | `metropt3_state_v1` | 状态目标协议 |
| `model_version` | `unified_temporal_channelwise_v2` | 当前统一编码器版本 |
| `evaluator_version` | `unified_evaluator_v2` | 评估器版本 |
| `best_metric` | `4.265846765041351` | validation loss，不是 MAE/RMSE |
| `checkpoint` | `checkpoint.pt` | 已记录 checkpoint 名称 |

同目录的 `metrics.json` 还记录了 50 个 epoch 的训练损失和 validation loss，最终 validation loss 为 `4.265846765041351`；`test_evaluation_count` 为 `0`。当前目录未发现独立的 `test_metrics.json`、按通道指标、效率 profile 或多 seed 汇总文件。

## 2. 不能直接写入论文的内容

- `best_metric` 只能作为工程训练记录，不能写成第三章的 MAE、RMSE 或主结果。
- 当前没有 test 评估，因此不能声称 MetroPT-3 测试集性能。
- 当前只有 seed 2026，不能报告均值和标准差。
- 当前没有 TCN、GRU、PatchTST、TimesNet、iTransformer 对照结果，不能完成第三章模型比较。
- 当前没有消融、窗口敏感性、参数量、训练时间、推理时间或显存指标。
- 当前 checkpoint/artifact 是单次运行证据，不能推断论文模型优于其他方法。

## 3. 第三章必须补齐的指标

### 3.1 MetroPT-3 主结果

对每个模型、每个 seed、固定 test split 输出：

- overall `MAE`；
- overall `RMSE`；
- 15 个传感器分别的 `MAE` 和 `RMSE`；
- 参数量、训练总时间、单 batch 推理时间和峰值显存；
- `run_id`、seed、checkpoint、dataset schema、split identity、normalizer identity。

### 3.2 第三章矩阵

至少需要以下模型族在同一 MetroPT-3 协议下运行：

`TCN`、`GRU`、`PatchTST`、`TimesNet`、`iTransformer`、`Unified Temporal Encoder`。

每个模型至少运行 seed `2026`；正式结果建议运行 `2026/2027/2028`，并报告均值与标准差。所有模型必须使用相同历史窗口、预测窗口、数据划分、归一化规则、训练预算和 test evaluator。

### 3.3 消融与敏感性

逐项记录：

- 去除多尺度时间块；
- 去除时间编码；
- 去除通道身份编码；
- 去除工况条件（若该数据协议启用）；
- 去除跨传感器交互；
- 历史窗口长度和表示维度敏感性。

每项只改变一个因素，指标仍为 test MAE/RMSE；验证集只用于选择 checkpoint 和超参数，不能用 test 调参。

## 4. 交给下一会话的执行 prompt

```text
请在 /home/new_work/kaf_profiti 继续完成论文第三章点预测实验，不要把当前
result/metropt3_state_seed2026/metrics.json 中的 validation loss 直接当成论文指标。

先读取：
1. AGENTS.md
2. plan/2026-10-06-unified-model-target-and-correction-spec.md
3. plan/2026-10-06-ch3-metrics-handoff.md
4. /home/new_work/new_work/3.2_结构框架.md

目标：补齐 MetroPT-3 第三章 test MAE/RMSE 和效率指标，并建立统一模型族对比。
模型至少包括 TCN、GRU、PatchTST、TimesNet、iTransformer、Unified Temporal Encoder。

硬性要求：
- 不引入观测掩码、缺失率、缺失模式或缺失专用分支；
- 不修改已有 result/metropt3_state_seed2026/ 下的文件；
- test 评估必须显式开启，且只在 validation-best checkpoint 上执行一次；
- 归一化统计只能来自 train split；
- 输出 overall 及 15 个传感器的 MAE/RMSE；
- 每个结果绑定 run_id、seed、checkpoint、split_identity、dataset_schema、normalizer_identity；
- 至少先完成 seed 2026 的可复现实验，再扩展到 2027/2028；
- 没有 test 指标前不得声称模型性能优越。

完成后把结果写入新的 result/<run_id>/，生成 manifest.json、test_metrics.json、
efficiency.json，并在 plan/progress.md 记录命令、commit 和验证结果。
```

## 5. 当前结论

该 artifact 可以证明：MetroPT-3 状态任务的训练流程曾运行到 50 epoch，并保存了 validation-best 指标记录和 checkpoint 身份。它还不能提供第三章正式实验结果。下一会话应先实现或调用任务特定 test evaluator，再运行统一模型对比矩阵。
