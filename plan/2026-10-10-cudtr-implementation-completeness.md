# CUDTR 实现完整性清单

更新时间：2026-10-10

## 已完成

| 机制 | 当前状态 | 代码位置 |
|---|---|---|
| 目标查询与 source token 路由 | 已实现 | `code/kaf_profiti/models/unified/model.py`、`routing.py` |
| top-k 路由与全缺失 baseline 回退 | 已实现 | `routing.py` |
| V3C warm-start 的 baseline-preserving 初始化 | 已实现 | `config.py`、`routing.py` |
| 反事实 utility teacher | 已实现 | `utility_teacher.py` |
| utility distillation、ranking、entropy、fallback loss | 已实现 | `cudtr_loss.py` |
| observation rate、masked variability、context source features | 已实现 | `model.py` |
| 最低 observation rate 与 absolute utility gate | 已实现 | `config.py`、`routing.py` |
| route、utility、entropy、fallback diagnostics | 已实现 | `model.py`、`entrypoint.py` |
| CUDTR 参数量统计 | 已修复 | `model.py` |

## 部分实现或仍需验证

| 机制 | 状态 | 说明 |
|---|---|---|
| 训练 dense、推理 hard top-k | 已实现但需实验验证 | 需要比较收敛稳定性和稀疏性 |
| fallback gate 正确性 | 部分实现 | corrupted 样本朝 fallback=1 约束，完整样本尚无显式 fallback=0 约束 |
| MetroPT 逐目标 source-only corruption | 接口已支持 | `build_source_corruption` 可保护目标通道；训练入口尚未按目标逐路由生成 corruption |
| FD004/TEP 任务级路由 | 已实现 | 不等价于 MetroPT 的逐目标传感器路由 |
| utility 的物理/因果解释 | 不支持 | utility 仅表示当前模型和遮挡协议下的预测效用 |
| 三数据集真实训练闭环 | 未验证 | 需要真实数据 smoke、显存、耗时和性能实验 |

## 不应声称

- 尚不能声称 CUDTR 已证明优于 Unified V3C。
- 尚不能将 route weight 或 utility prediction 解释为因果传感器贡献。
- 尚不能声称三个数据集采用完全相同粒度的目标条件化路由。
