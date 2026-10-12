# CUDTR 概率预测优化任务

## Goal

完成四项 MetroPT-3 验证：无 CUDTR 的匹配分位数基线、区间校准分数对照、checkpoint 选择与校准数据隔离、逐传感器与连续时间块区间诊断。测试集仅用于最终评估，不参与选模或校准器挑选。每个代码阶段先测试、完成验证后单独提交。

## Phases

| Phase | 内容 | 状态 | 提交门禁 |
|---|---|---|---|
| 1 | WIS 与每轮验证诊断；训练 V4 quantile、无 CUDTR 匹配基线 | complete | 指标测试、入口测试、全量测试、独立 run 产物检查 |
| 2 | 实现并比较 median-residual 与标准 CQR 校准；仅 validation 校准 | complete | 校准数学测试、只读 checkpoint 评估、记录验证校准指标 |
| 3 | 按时间顺序拆分 validation 的选模段与校准段，并对 CUDTR/基线使用相同拆分 | complete | 切分无重叠测试、训练与选模/校准来源检查、重新训练受控对照 |
| 4 | 增加逐传感器与连续时间块的区间指标，完成两模型评估 | complete | 时间块顺序与覆盖测试、全量测试、结果完整性审计 |

## Acceptance

- checkpoint 唯一主选择指标仍为 validation Pinball。
- validation WIS、coverage、width、P50 MAE/RMSE 用训练集归一化统计量还原至原始单位后记录。
- 两个模型固定相同 seed、官方首月协议、split、归一化、history/prediction 长度、batch、epoch 上限、优化器与学习率；唯一结构差异为 CUDTR 和其专属辅助目标。
- CUDTR 与无 CUDTR 对照均仅拟合于校准 validation 子集，且测试标签不参与阈值/分数选择。
- 测试报告同时含整体、逐传感器和连续时间块的 raw/calibrated 区间指标；保留原有结果目录。

## Errors

| 问题 | 处理 |
|---|---|
| `conda` 未在 shell PATH | 使用已验证的 torch23 解释器 `/root/anaconda3/envs/torch23/bin/python` |
| 旧训练日志不含命令行回显 | 从 checkpoint config、manifest、原日志及数据协议源码交叉确认训练配置 |
