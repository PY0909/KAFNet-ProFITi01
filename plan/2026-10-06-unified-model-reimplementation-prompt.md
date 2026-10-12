# Unified Model Reimplementation Prompt

你正在 `/home/new_work/new_work` 中继续实现论文主模型。请先读取并遵守：

1. `AGENTS.md`
2. `README.md`
3. `plan/2026-10-05-shared-encoder-task-head-code-refactor-plan.md`
4. `plan/2026-10-05-shared-encoder-task-head-future-direction.md`
5. `3.2_结构框架.md`
6. `plan/experiment-protocol.md`
7. `plan/2026-10-05-dataset-specific-ch3-ch5-experiment-plan.md`
8. `plan/progress.md`
9. `plan/implementation-plan.md`
10. `plan/stage-gates.md`

## 当前目标

在不破坏 legacy 模型、已有 checkpoint、registry 和现有数据协议的前提下，为论文主线建立独立的统一模型实现目录：

```text
code/kaf_profiti/models/unified/
├── __init__.py
├── config.py
├── encoder.py
├── missingness.py
├── model.py
├── validation.py
└── heads/
    ├── __init__.py
    ├── base.py
    ├── state.py
    ├── rul.py
    └── tep.py
```

这里的 shared encoder 指统一的编码器结构和输入输出接口。MetroPT-3、C-MAPSS FD004、TEP 仍然独立训练参数，不进行跨数据集联合训练。

## 任务边界

- MetroPT-3：未来多传感器状态预测和故障窗口风险。
- C-MAPSS FD004：RUL 点预测和 RUL 分布预测。
- TEP：故障检测、故障诊断概率，以及由事件预测推导的检测延迟。
- 第五章只消费第四章冻结的预测分布或分类概率，校准参数只能使用 validation split。
- 人工缺失是可选压力测试，不得把三个原始数据集都描述成天然异步缺失数据。

## 必须保留的兼容边界

暂时保留以下模块及其公开接口：

- `code/kaf_profiti/models/kst_probflow.py`
- `code/kaf_profiti/models/kst_light.py`
- `code/kaf_profiti/models/kst_flow.py`
- `code/kaf_profiti/models/cross_variable.py`
- `code/kaf_profiti/models/kafnet_encoder.py`
- `code/kaf_profiti/industrial/missing.py`
- `code/kaf_profiti/experiments/masks.py`
- `code/kaf_profiti/experiments/registry.py` 中的 legacy model identity

不得直接删除旧模块、修改旧 checkpoint 的加载语义，或把新模型结果混入旧 artifact。

## 已知问题，必须在新实现中解决

1. 当前 `HierarchicalMissingnessKAFEncoder.encode_with_patches()` 的局部 KAF 路径把原始 `X` 传入 `_batched_local_kaf()`，masked position 的值可能通过卷积影响输出。新 encoder 必须满足 masked-input invariance。
2. `UnifiedTemporalEncoder.VERSION` 当前为 `hierarchical_kaf_v1`，但实际封装了缺失感知 encoder。新包应使用明确且不可歧义的版本字符串。
3. `kafnet_encoder.py` 中 `return` 后仍保留不可达的旧实现。新包不得复制这段死代码。
4. C-MAPSS 的 `target_convention` 必须明确区分 raw target、训练标签 cap 和模型输出约束。
5. TEP 不能只把诊断概率放进 metadata；应提供正式的 detection probability、diagnosis probability 和事件级 latency 评估接口。

## 实现要求

- 先补测试，再写实现；测试应覆盖形状、有限值、history-only、masked-input invariance、保存/恢复和 registry 路由。
- 新模型不能读取 `Y_q`、`y_flat`、`M_q`、`mq_flat` 或 `rul` 产生预测。
- 所有归一化统计量只能来自训练数据。
- 训练、validation、test 的 split identity、seed、target schema 和 checkpoint 必须进入 manifest。
- 不修改 `result/` 中已有 metrics、predictions 或 checkpoint。
- 不使用绝对机器路径写入配置或代码。
- 每完成一个阶段运行相应测试，并单独创建一个 Git commit。

## 推荐阶段

### Stage 1：目录和接口

- 建立 `models/unified/` 包。
- 定义 encoder、head、model、output、version contract。
- 保留 registry 旧入口，新增兼容路由。
- 添加最小单元测试。

### Stage 2：mask-safe encoder

- 实现历史值、时间、观测 mask、context 的编码。
- 统一全局和局部路径的 mask 处理。
- 添加 masked-input invariance、全缺失通道和非规则时间测试。

### Stage 3：任务头

- MetroPT state head：point 和 monotonic quantile。
- C-MAPSS RUL head：point、quantile、raw/capped 语义。
- TEP fault head：detection、diagnosis 和事件延迟评估接口。

### Stage 4：训练入口和 registry

- 切换 `unified_encoder_task` 到新包。
- 保持旧模型 registry identity 不变。
- 验证 config、manifest、checkpoint 和 test evaluator。

### Stage 5：回归和文档

```bash
PYTHONPATH=code python -m pytest code/tests/test_unified_encoder.py -q
PYTHONPATH=code python -m pytest code/tests/test_task_heads.py code/tests/test_unified_task_model.py -q
PYTHONPATH=code python -m pytest code/tests/ -q
git diff --check
```

只有全量测试和真实数据 smoke 通过后，才更新 `plan/progress.md` 并进入正式实验。

## 输出格式

每个阶段结束时报告：

- 修改的文件；
- 阶段目标是否完成；
- 定向测试和全量测试结果；
- 未解决风险；
- commit hash；
- 下一阶段入口。

不要把 smoke 数值写成论文性能结论，也不要在没有真实 TEP 数据验证时声称 TEP 训练已完成。
