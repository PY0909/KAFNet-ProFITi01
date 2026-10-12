# V5 有序 Patch 编码器实施记录

V3C 保持冻结主基线。V5 只改变编码器内部表示，任务头、数据协议、归一化和 evaluator 保持不变。

## 阶段

- V5-A：按连续时间块生成有序 patch tokens `[B,C,P,H]`，并用 token 均值汇总到既有任务头接口。
- V5-B：在 V5-A token 序列上加入轻量 depthwise temporal mixer。
- V5-C：在 V5-B 上使用最近 patch 作为 query，对有序 token 做 query-conditioned pooling。

当前实现状态：V5-A、V5-B、V5-C 均已独立提交，并在 MetroPT-3 与 FD004 上完成 seed 2026 单次训练。结果是方向性诊断，不是多 seed 论文结论。

## 单 seed 结果

| 变体 | MetroPT MAE | MetroPT RMSE | FD004 MAE | FD004 RMSE | FD004 NASA score |
|---|---:|---:|---:|---:|---:|
| V5-A | 0.8015 | 2.0825 | 30.2936 | 38.9520 | 52707.7 |
| V5-B | 0.7768 | 2.1097 | 32.6990 | 46.0095 | 2553558.5 |
| V5-C | 0.6838 | 2.0633 | 26.3076 | 34.6743 | 26624.4 |

结果目录分别为 `result/metropt3_unified_v5{a,b,c}_seed2026_20261009/` 和 `result/cmapss_fd004_unified_v5{a,b,c}_cap150_e120_seed2026_20261009/`。V5-C 在本次单 seed 方向性运行中优于 V5-A/V5-B，但仍需与冻结 V3C 及多 seed 结果按 protocol checker 核验后才能进入正式论文结论。

V3C 的 `Linear(H,1)` masked content-score pooling 和尺度摘要 token 不再描述为 Q/K attention 或有序 patch 序列。
