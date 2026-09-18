# CH3-S04 中心条件 go/no-go 报告

> 生成：2026-09-18。数据来源：`result/pilot/metropt3/runs/`（7 个完整 run，同机 AutoDL RTX 3090，commit `c535097`，dry-run 验签 verified_complete=7 / expected_new=0，跨机 preflight `identity_sections_match`）。条件：`point_mixed_030`（mixed 机制、目标缺失率 0.30、realized 0.297/0.301/0.304、split_sha256=`eb7b957c…`、mask_seed=2026、seed=2026 单种子）。本报告只给 `go/fix/stop` 诊断，不报告 mean±std、置信区间、显著性或任何"证明优于"表述（单种子 pilot 不支持此类推断）。

## 结论：go

中心条件协议可学习、七模型比较口径一致、无病理性信号。允许进入 CH3-S05（缺失强度与机制单种子扩展）。

## 口径声明（含一项更正）

- 全部 run 指标（train/validation/test 的 MAE/RMSE）与 data_gate 的 `std_micro` floor 同处 standardized 空间（train-split z-score；判据：predictions payload 的 target 值以 0 为中心，raw 物理量如 TP2 压力不可能为负）。naive floor 的正确基准：**valid persistence `std_micro` MAE=0.4024（RMSE=0.8978）**；**test `std_micro` MAE=0.4786（RMSE=0.9991）**。
- **更正（2026-09-18）：** CH34-S03-T02 sanity 的 `beat_naive` 代码比较使用了 `persistence_mae_raw=0.9159`（raw 物理空间），与指标的 standardized 空间不匹配；当日记录的"改善 56.5%"为错误口径，作废。改用正确基准后，5-epoch sanity 的真实改善幅度为：ode_rnn 0.3985 vs 0.4024（1.0%）、li_tcn 0.4013 vs 0.4024（0.3%）——5 epoch 仅触及 floor。门禁结论不变（三标志全 true 且确实低于 floor），但"大幅超越 naive"的证据来自本轮 50-epoch 完整 run，而非 sanity。`pilot_runner.py` 中 beat_naive 的比较基准已列为待修正项（见"遗留注意点"）。
- sanity 的 valid MAE 聚合 masked query 位置，floor 聚合全部 query 位置（mixed@0.30 下预期接近）；两者同为单次训练轨迹，不构成多 seed 统计。

## 主表（validation/test，std 空间）

| 模型 | 参数量 | best_val MAE | vs floor 0.4024 | best epoch | test MAE | test RMSE | vs floor 0.4786 |
|---|---:|---:|---:|---:|---:|---:|---:|
| kst_light\|mlp | 149,597 | **0.2777** | **+31.0%** | 32 | 0.2923 | 0.5827 | +38.9% |
| ff_gru\|linear | 27,816 | 0.2819 | +29.9% | 47 | **0.2389** | **0.5240** | **+50.1%** |
| kst_light\|linear | 145,437 | 0.2956 | +26.5% | 47 | 0.2988 | 0.5618 | +37.6% |
| li_tcn\|linear | 111,208 | 0.3157 | +21.5% | 31 | 0.2605 | 0.5395 | +45.6% |
| masked_tcn\|linear | 111,208 | 0.3464 | +13.9% | 33 | 0.2751 | 0.5569 | +42.5% |
| ode_rnn\|linear | 41,576 | 0.3552 | +11.7% | 13 | 0.2989 | 0.6130 | +37.5% |
| gru_d\|linear | 27,631 | 0.4048 | **−0.6%** | 49 | 0.3501 | 0.6299 | +26.8% |

- validation：6/7 模型低于 naive persistence floor；**kst_light|mlp validation 全场最优**，超过最强 baseline ff_gru（0.2777 vs 0.2819）。
- test：7/7 低于 test floor。test 排序（ff_gru 最优）与 validation 排序存在排名扰动（kst_light|mlp test 第 4）。单种子下 val/test 排名扰动属预期噪声，本文不据此得出任何模型优劣结论；T02 注记已声明留待多 seed formal。
- checkpoint 一律由 best_valid 选择后冻结再评估 test（test_evaluation_count=1×7）。

## 逐传感器 test MAE（std 空间）

| 模型 | TP2 | TP3 | H1 | DV_pressure | Reservoirs | Oil_temp | Motor_current |
|---|---:|---:|---:|---:|---:|---:|---:|
| ff_gru | 0.314 | 0.229 | 0.306 | 0.099 | 0.228 | 0.229 | 0.266 |
| li_tcn | 0.331 | 0.242 | 0.322 | 0.091 | 0.242 | 0.265 | 0.331 |
| masked_tcn | 0.337 | 0.253 | 0.327 | 0.100 | 0.253 | 0.302 | 0.354 |
| kst_light\|mlp | 0.386 | 0.300 | 0.369 | **0.048** | 0.291 | 0.306 | 0.346 |
| kst_light\|linear | 0.385 | 0.322 | 0.407 | **0.072** | 0.306 | 0.270 | 0.330 |
| ode_rnn | 0.380 | 0.278 | 0.365 | 0.138 | 0.278 | 0.323 | 0.330 |
| gru_d | 0.457 | 0.344 | 0.437 | 0.142 | 0.344 | 0.285 | 0.443 |

- 各模型误差分布形状一致：DV_pressure 最易（绝对误差小）、TP2/H1/Motor_current 最难。kst_light 两个 head 在 DV_pressure 上显著优于所有 baseline（0.048/0.072 vs 0.091-0.142）。

## 效率（同机同 batch，FP32）

| 模型 | train (s/50ep) | test 评估 (s) | 推理 (s, 3 次中位) |
|---|---:|---:|---:|
| ff_gru | 564.1 | 5.16 | 4.88 |
| gru_d | 553.4 | 4.94 | 4.81 |
| li_tcn | 572.2 | 5.13 | 5.08 |
| masked_tcn | 549.2 | 4.92 | 4.75 |
| ode_rnn | 701.4 | 5.13 | 5.01 |
| kst_light\|linear | 673.0 | 5.05 | 4.89 |
| kst_light\|mlp | 676.3 | 5.15 | 5.08 |

训练时长同数量级（549-701s），评估与推理几乎无差异（~5s）；ode_rnn 与 kst_light 训练慢 ~20% 属模型结构属性，timing 可比。

## 病理检查（T03 规定四项）

| 检查项 | 结果 | 判定 |
|---|---|---|
| 全模型接近零预测 | 无：7 模型预测 std / target std（masked 位置）= 0.73-0.84，MAE 互异且轴检查断言非全零 | 通过 |
| epoch 1 后持续恶化 | 无：6/7 模型 best epoch 分布 31-49，曲线正常收敛。例外：ode_rnn 自 epoch 13 后 valid 持续变差（0.355→0.525）——best_valid checkpoint 选择已正确取 epoch 13，test 结果未受影响 | 通过（含注记） |
| 单通道支配总误差 | 无：最大通道误差占比 0.18-0.19（7 通道均匀基线 0.14），TP2×5/H1×1/Motor_current×1，无支配 | 通过 |
| timing 不可比 | 无：同机同 batch 同精度，训练时长同数量级，评估/推理一致 | 通过 |

## 诊断与遗留注意点

1. **go 的依据**：协议可学习（6/7 模型 validation 低于 naive floor，最高 +31.0%）、七模型共享完全一致的 protocol_sha/shared_artifacts/code_fingerprint（fairness 无 mismatch）、无病理信号。满足 CH3-S05 扩展的前置条件。
2. **gru_d 未过 valid floor**（0.4048 vs 0.4024，−0.6%）：作为 baseline 的真实属性（GRU-D 在该协议下 50 epoch 收敛极慢，best 在末位 epoch 49），不构成协议问题，不影响 go；多 seed 阶段观察其稳定性。
3. **ode_rnn 后期退化**：epoch 13 后 valid 持续变差，现有 best_valid checkpoint 机制是充分兜底；S05/S06 沿用，无需改训练配方。
4. **beat_naive 比较基准修正待办**：`pilot_runner.py` 的 `_naive_floor_reference`/`beat_naive` 应改用 `std_micro`（当前用 raw）；该修正属代码变更，会变更 code_fingerprint——**不得在 CH3-S05 run 进行中插入**，排入下一轮代码窗口（与后续任务合并 commit），修正后无需重跑本轮 GPU 结果（本报告数字直接读自 manifest/metrics/predictions，不经过 beat_naive 字段）。
5. 边界重申：以上全部为单种子（seed=2026）中心条件结果；排名扰动不支撑任何模型间优劣声明；进入多 seed formal 前不出"证明优于"类结论。
