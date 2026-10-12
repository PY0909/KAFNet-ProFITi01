# Unified Temporal Encoder and Dataset-Specific Task Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. Each step is independently testable and must be checked off only after its command passes.

**Goal:** 将当前以连续未来值和统一风险头为中心的实验代码，改造成“统一时序编码器结构 + 数据集特定任务头”的可验证代码框架，覆盖 MetroPT-3 状态预测、C-MAPSS FD004 RUL 预测和 TEP 故障检测/诊断。

**Architecture:** 统一编码器结构只负责把历史多变量窗口编码为 `z_history`，不读取未来目标字段；三个数据集分别独立实例化和训练，参数不跨数据集共享。任务头根据数据集和任务类型解释该表示：MetroPT-3 输出未来状态点值/分位数，C-MAPSS 输出 RUL 点值/分位数，TEP 输出二元检测概率及可选多类诊断概率。训练器通过显式 task adapter 选择目标、损失和指标；第五章风险评估只读取第四章冻结的预测结果，不把风险学习重新塞回编码器。

**Tech Stack:** Python 3.12, PyTorch 2.5.1, pandas, NumPy, pytest, 现有 `code/kaf_profiti` registry/dataset/runner 体系。

---

## 1. 当前代码基线与改造边界

### 已确认的现状

- `code/kaf_profiti/models/kafnet_encoder.py` 已有 `MultiScaleKAFEncoder` 和 `HierarchicalMissingnessKAFEncoder`；后者的 `encode_with_patches()` 可返回表示、patch token 和 patch mask，但现有主模型没有把它作为独立统一编码器契约暴露。
- `code/kaf_profiti/models/kst_probflow.py` 中 `KSTProbFlow` 同时持有编码器、动态变量图、flow、分位数头和风险头；其 `loss()` 将 `batch.rul` 统一解释为风险标签，这不适合 TEP 多类诊断，也不适合作为三类任务的长期接口。
- `code/kaf_profiti/industrial/batch.py` 的 `IndustrialBatch` 固定包含 `Y_q/M_q/y_flat/mq_flat/rul`，`IndustrialCollator` 只会构造连续未来目标的扁平张量。
- `code/kaf_profiti/experiments/datasets.py` 已实现 MetroPT-3、C-MAPSS 和 TEP 的协议入口，但三者目前都被包装成连续未来窗口；`TEPWindowDataset` 返回的 `rul` 实际是二元故障标记。
- `code/run_experiment.py` 的训练和评估路径只认识统一的 `loss/predict_mean/predict_quantiles/predict_risk` 契约，不能正确记录分类概率、诊断类别和事件级检测延迟。
- `code/kaf_profiti/experiments/registry.py` 已有模型注册与 `create_model()`，但尚无统一编码器、任务头和任务适配器的独立注册项。

### 本阶段不做的事情

- 不删除或重写已有 `KSTProbFlow`、旧结果目录和历史 metrics JSON；旧模型作为 legacy 回归对象保留。
- 不把 TEP 离散故障标签伪装成连续九分位数回归目标。
- 不把三个数据集的 MAE、RMSE、F1 或风险指标合并成一个跨任务总分。
- 不在本阶段实现第五章完整风险策略；本阶段只提供冻结预测结果和风险计算所需的标准输出。
- 不改变数据集原始协议；主实验使用现有数据集协议函数，并补齐任务语义和标签字段。

## 2. 目标接口合同

本计划中的“统一”只表示结构、输入输出契约和实验接口一致，不表示跨数据集联合训练或参数共享。每个数据集拥有独立的 encoder checkpoint；第三章到第四章在同一数据集内复用冻结的编码器结构/初始化约定，并按章节选择确定性或概率任务头。

### 2.1 统一编码器接口

新增 `code/kaf_profiti/models/unified_encoder.py`，定义：

```python
class SharedTemporalEncoder(nn.Module):
    def encode(self, batch: IndustrialBatch) -> EncoderOutput:
        """Return history-only representation; never read future target fields."""

    def forward(self, X_obs, T_obs, M_obs, context=None) -> Tensor:
        """Return z_history with shape [B, N_or_1, H]."""
```

`EncoderOutput` 至少包含：

```python
@dataclass
class EncoderOutput:
    z_history: Tensor       # [B, N, H] for sensor-state tasks or [B, 1, H] pooled
    z_global: Tensor        # [B, H]
    patch_tokens: Tensor | None
    patch_mask: Tensor | None
```

实现方式：复用 `HierarchicalMissingnessKAFEncoder.encode_with_patches()` 的历史编码路径；用 `z_global = z_history.mean(dim=1)` 作为通用全局表示；保留传感器级表示供 MetroPT-3 状态头使用。`encode()` 内部只访问 `X_obs/T_obs/M_obs/context`，并增加行为测试，扰动 `Y_q/M_q/y_flat/mq_flat/rul` 后输出必须不变。

### 2.2 任务头接口

新增 `code/kaf_profiti/models/task_heads.py`：

```python
class TaskHead(nn.Module):
    def forward(self, encoded: EncoderOutput, batch: IndustrialBatch) -> TaskOutput:
        ...
    def loss(self, output: TaskOutput, batch: IndustrialBatch) -> Tensor:
        ...
    def metrics_payload(self, output: TaskOutput, batch: IndustrialBatch) -> dict:
        ...
```

`TaskOutput` 是统一的结构化结果，不再用字段位置猜测任务：

```python
@dataclass
class TaskOutput:
    task: str
    point: Tensor | None
    quantiles: Tensor | None
    probabilities: Tensor | None
    target: Tensor | None
    metadata: dict
```

具体头：

1. `MetroStateHead`: 从 `[B,N,H]` 或 `[B,H]` 生成未来状态点值；可选九分位数 `0.1..0.9`，输出 `[B,P,N]`，使用 masked pinball loss。
2. `RULHead`: 从 `z_global` 生成每个 engine 的 RUL 点值；概率模式输出 `[B,9]` 分位数，使用 Huber/MSE 点损失和可配置 pinball 损失，明确记录 raw/capped target convention。
3. `TEPFaultHead`: 从 `z_global` 生成二元检测概率 `[B]` 和可选多类诊断概率 `[B,K]`；使用 BCE/Focal 与 cross-entropy，记录 `faultNumber`、`simulationRun` 和检测时间信息。

所有头必须满足：输出有限值；点模式和概率模式共享同一输入表示；概率头不把类别标签扩展为连续分位数；输出中包含 `task`, `dataset`, `target_schema` 和 shape metadata。

### 2.3 模型封装接口

新增 `code/kaf_profiti/models/unified_task_model.py`：

```python
class SharedEncoderTaskModel(nn.Module):
    def encode(self, batch) -> EncoderOutput: ...
    def predict(self, batch, mode: str = "point") -> TaskOutput: ...
    def loss(self, batch) -> Tensor: ...
    def parameter_count(self) -> int: ...
```

`predict()` 只调用头的前向过程；`loss()` 才读取任务目标。模型构造参数必须显式包含 `dataset`, `task`, `head_mode`, `quantiles`, `hidden_dim`, `patch_lens` 和统一编码器版本，写入 checkpoint config。每个数据集单独保存 checkpoint，不允许把一个数据集的权重当作另一个数据集的预训练结果。

## 3. 数据与 batch 改造

### Task 1: 扩展样本语义而不破坏旧 batch

**Files:**

- Modify: `code/kaf_profiti/industrial/batch.py`
- Modify: `code/kaf_profiti/industrial/metropt.py`
- Modify: `code/kaf_profiti/industrial/cmapss.py`
- Modify: `code/kaf_profiti/industrial/tep.py`
- Test: `code/tests/test_task_batch_contract.py`

- [x] 在 `IndustrialBatch` 增加可选字段 `task`, `dataset`, `target`, `target_mask`, `class_label`, `fault_number`, `simulation_run`, `sample_index`，保留旧字段以便 legacy 模型回归。
- [x] 在 `IndustrialCollator` 中根据样本字段构造任务目标：MetroPT 返回 `[B,P,N]` 状态目标；C-MAPSS 返回 `[B]` RUL；TEP 返回 `[B]` fault label 和 `[B]` fault number。
- [x] 为每个 dataset sample 增加 `target_schema` 和任务 metadata，不在 collator 内推断数据集名称。
- [x] 测试三种样本分别得到正确 dtype、shape、标签范围；空 batch、全正常 TEP batch 和单 engine C-MAPSS batch 都不能产生 NaN。

Run:

```bash
/root/anaconda3/envs/torch23/bin/python -m pytest code/tests/test_task_batch_contract.py -q
```

Expected: 新增合同测试 PASS，现有 `IndustrialBatch` 构造调用不回归。

### Task 2: 为 TEP 补齐检测/诊断标签

**Files:**

- Modify: `code/kaf_profiti/industrial/tep.py`
- Modify: `code/kaf_profiti/experiments/datasets.py`
- Test: `code/tests/test_tep_task_labels.py`

- [x] `TEPWindowDataset.__getitem__()` 返回 `fault_label`, `fault_number`, `simulation_run`, `sample_index`；窗口标签使用 `faultNumber > 0 and sample >= fault_start_sample`，并保留完整 run identity。
- [x] `_create_tep()` 的 split metadata 写明 train/valid/test 的 composite run keys、类别数和 label rule。
- [x] 增加 run-level aggregation 所需的 `unit_id` 与窗口起止 sample，禁止按重叠窗口数量直接当作独立样本数。
- [x] 测试故障前窗口为 0、故障后窗口为 1、不同 faultNumber 不被混淆，train/valid/test 的 run key 不交叉。

## 4. 统一编码器和任务头实现

### Task 3: 提取统一编码器

**Files:**

- Create: `code/kaf_profiti/models/unified_encoder.py`
- Modify: `code/kaf_profiti/models/kafnet_encoder.py`
- Test: `code/tests/test_unified_encoder.py`

- [x] 将 `HierarchicalMissingnessKAFEncoder` 包装为 `UnifiedTemporalEncoder`，统一 `encode()`、`forward()` 和 `parameter_count()`；不同数据集分别实例化，禁止隐式共享参数。
- [x] `encode()` 返回 `EncoderOutput`，对 `[B,L,N]` 输入保证 `z_history=[B,N,H]`、`z_global=[B,H]`；patch token 缺省时返回 `None`。
- [x] 对全零可用性输入、单步历史和不同 `patch_lens` 做有限值保护，避免 softmax 全为无效值时产生 NaN。
- [x] 用 `check_history_only` 行为测试验证未来字段不会进入编码器；用保存/加载 state dict 测试形状和输出稳定性。

### Task 4: 实现三类任务头

**Files:**

- Create: `code/kaf_profiti/models/task_heads.py`
- Test: `code/tests/test_task_heads.py`

- [x] 先实现确定性点模式，再实现统一九分位数模式；分位数顺序固定为 `(0.1, 0.2, ..., 0.9)`，用正增量参数化保证单调性。
- [x] `MetroStateHead` 使用 sensor-level representation 和 query projection 生成未来 `[B,P,N]`；不把未来 `Y_q` 传入前向。
- [x] `RULHead` 使用 global representation 生成 `[B]` 或 `[B,9]`；显式记录 raw/capped RUL。
- [x] `TEPFaultHead` 输出二元 probability 和可选 `num_fault_classes` 类别 probability；实现稳定的 BCE/CE loss 和类别缺省策略。
- [x] 测试三个头的 shape、finite、梯度、概率范围、分位数单调性和全正常 TEP batch。

### Task 5: 组装 `SharedEncoderTaskModel`

**Files:**

- Create: `code/kaf_profiti/models/unified_task_model.py`
- Modify: `code/kaf_profiti/models/__init__.py`
- Test: `code/tests/test_unified_task_model.py`

- [x] 构造函数根据 `dataset/task` 选择头，拒绝未知组合；单个模型实例内只创建一个编码器，三个数据集之间不复用参数。
- [x] `predict()` 返回 `TaskOutput`；`loss()` 调用当前头的 loss，并把 task/head/version 写入输出 metadata。
- [x] `parameter_count()` 分别报告 encoder、head 和 total，便于第三章效率表和第四章概率头参数表使用。
- [x] 测试 MetroPT、C-MAPSS、TEP 三种组合均可构造，未知 task、错误 sensor 数和错误 quantile 列表明确报错。

## 5. Registry、训练入口和结果协议

### Task 6: 注册新模型并保留 legacy identity

**Files:**

- Modify: `code/kaf_profiti/experiments/registry.py`
- Modify: `code/kaf_profiti/experiments/model_api.py`
- Test: `code/tests/test_unified_task_registry.py`

- [x] 新增模型 ID `unified_encoder_task`，显示名为 `Unified Temporal Encoder + Task Head`，状态先设为 `pilot_ready`，不覆盖 `kst_probflow`。
- [x] `create_model()` 接收 `dataset`, `task`, `head_mode`, `num_fault_classes`, `quantiles` 并传入模型 config；config 序列化后必须包含完整 task identity 和 `parameter_scope: dataset_independent`。
- [x] 更新模型 API 检查器：连续任务检查 point/quantile；TEP 检查 probability/log-loss/CE；所有 prediction path 都执行 history-only 检查。
- [x] 测试 registry identity 不与 A1/F1-F3、旧 `kst_probflow` 或已有 result variant 冲突。

### Task 7: 增加任务感知训练与评估路径

**Files:**

- Modify: `code/run_experiment.py`
- Create: `code/kaf_profiti/experiments/task_runner.py`
- Modify: `code/kaf_profiti/experiments/metrics.py`
- Test: `code/tests/test_task_runner.py`

- [x] 新 runner 接受 `--task`、`--head-mode`、`--num-fault-classes`、`--quantiles`，并按 task 选择 loss、validation metric 和 checkpoint criterion；每个 dataset/task 单独创建 optimizer、checkpoint 和 run_id。
- [x] MetroPT validation criterion 使用状态 MAE/RMSE 与事件标签的辅助统计；C-MAPSS 使用 engine-level MAE/RMSE/PHM score；TEP 使用 run-level PR-AUC/F1/detection delay。
- [x] 第四章概率模式记录 CRPS/NLL/PICP/MPIW 或 LogLoss/Brier/ECE；第五章所需 prediction artifact 保存 task metadata、unit/run identity 和 checkpoint path。
- [x] 每个 run 的 manifest 必须包含 `run_id`, `dataset_release`, `task`, `model`, `seed`, `split_identity`, `target_schema`, `head_mode`, `checkpoint`, `best_metric`。
- [x] 测试 checkpoint 选择只读取 validation，test evaluator 只在显式 `--evaluate-test` 时执行一次。

### Task 8: 为原始数据协议增加配置项

**Files:**

- Modify: `code/kaf_profiti/experiments/ch3/config.py`
- Modify: `code/kaf_profiti/experiments/datasets.py`
- Create: `configs/ch3/unified_encoder_metropt.yaml`
- Create: `configs/ch4/unified_encoder_cmapss_fd004.yaml`
- Create: `configs/ch5/unified_encoder_tep.yaml`
- Test: `code/tests/test_task_configs.py`

- [x] MetroPT 配置固定其原始时间切分和事件标签策略；状态预测窗口、warning horizon、LPS feature policy 分开记录。
- [x] C-MAPSS 配置固定 FD004 engine split、RUL target convention、raw/capped 选项和官方 test evaluator。
- [x] TEP 配置固定 release、faulty train/test source、composite run split、fault start sample、binary/multiclass mode。
- [x] 配置验证禁止把三个数据集写进同一任务头或同一 label rule；每个配置生成稳定的 recipe identity。

## 6. 测试与验收顺序

### 定向测试

```bash
/root/anaconda3/envs/torch23/bin/python -m pytest \
  code/tests/test_task_batch_contract.py \
  code/tests/test_tep_task_labels.py \
  code/tests/test_unified_encoder.py \
  code/tests/test_task_heads.py \
  code/tests/test_unified_task_model.py \
  code/tests/test_unified_task_registry.py \
  code/tests/test_task_runner.py \
  code/tests/test_task_configs.py -q
```

Expected: all new contracts PASS; legacy model API tests remain PASS.

### Smoke experiments

1. MetroPT-3: 2 epochs, one batch per split, deterministic point head。
2. C-MAPSS FD004: 2 epochs, one batch per split, RUL point head。
3. TEP: 2 epochs, one batch per split, binary fault head and one multiclass forward pass。
4. Repeat the three runs with nine-quantile mode only after point mode passes。

Each smoke run must create an isolated `run_id`, manifest and checkpoint; smoke results are engineering checks and cannot be reported as thesis results.

Execution record (2026-10-05): MetroPT-3 point/quantile and C-MAPSS FD004 point/quantile each completed 2 CPU epochs with one train/validation batch; run artifacts are stored under `result/smoke/shared_encoder_task/`. The TEP task runner completed a two-epoch synthetic run-level smoke in `code/tests/test_task_runner.py`; local TEP RData files are unavailable, so no real-data TEP run was performed. Classification correctly uses probabilities rather than quantiles.

### Full regression

```bash
/root/anaconda3/envs/torch23/bin/python -m pytest code/tests/ -q
git diff --check
```

Full regression passed on 2026-10-05: `426 passed, 6 skipped`; skipped dataset tests require TEP RData files not present in this checkout. `git diff --check` passed. No historical result file was modified. Real-data TEP evaluation remains a precondition before any long experiment.

## 7. Implementation order and commits

1. Batch/task metadata and TEP labels; commit `refactor: add dataset task batch contract`。
2. Unified encoder wrapper and history-only tests; commit `refactor: expose unified temporal encoder`。
3. Three task heads and unit tests; commit `feat: add dataset specific task heads`。
4. Model assembly, registry and API checks; commit `feat: register unified encoder task model`。
5. Task runner, metrics, manifests and configs; commit `feat: add task aware training entrypoint`。
6. Smoke runs, full tests and plan/progress update; commit `test: validate shared encoder task pipeline`。

每个 commit 都必须只包含对应步骤的代码、测试和必要配置；不得提交 `result/` 中生成的 metrics、checkpoint 或本地数据文件。

## 8. Acceptance criteria

- `UnifiedTemporalEncoder.encode()` 对三种数据集输入都输出稳定且有限的历史表示，未来字段扰动不改变预测路径；三套训练参数彼此独立。
- MetroPT-3、C-MAPSS FD004、TEP 分别使用状态头、RUL 头、故障检测/诊断头；头的输出、损失和指标语义与原始任务一致。
- 对每个数据集，第三章点模式和第四章概率模式使用相同的 encoder 结构与 task identity；三个数据集之间不共享训练权重，只有结构和接口一致。
- 第五章可以从第四章 prediction artifact 读取分布或类别概率，并计算风险策略所需输入，但不重新训练或读取测试标签选择阈值。
- 所有 run artifact 有完整 manifest、checkpoint 和 split identity；旧模型和旧结果身份不变。
- 定向测试、全量 pytest、`git diff --check` 全部通过；未完成的性能结论和跨数据集结论明确标为未验证。

## 9. Handoff after plan approval

计划验收后，执行者应从 Task 1 开始按顺序实现，每完成一个任务运行对应定向测试并更新本文件 checkbox。只有在 Task 7/8 完成且全量测试通过后，才允许启动三数据集正式训练；正式训练命令和结果表写入 `plan/progress.md`，不直接修改论文正文或历史 metrics JSON。
