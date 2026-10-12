# 统一时序模型目标方向与整改规格

> 状态：执行基线（2026-10-06）
>
> 适用范围：`/home/new_work/kaf_profiti` 的论文主线与 `code/` 新实现。

## 1. 决策摘要

论文和代码统一采用“规整多传感器历史序列 -> 共享时序编码器 -> 数据集特定任务头 -> 概率输出 -> 风险决策”的单向链路。MetroPT-3、C-MAPSS FD004 和 TEP 使用相同的编码器接口与实验矩阵，但分别训练、分别保存参数，不进行跨数据集联合训练或迁移结论。

本版本明确删除观测缺失建模：主模型不接收观测掩码、缺失率、缺失模式、观测间隔特征或缺失专用分支；论文不以异步/不完整观测为研究问题；实验不生成或读取人工缺失 mask。数据文件中的空值、无穷值、重复时间、错误列和非法标签属于输入协议错误，必须在加载阶段直接报错，不能通过填零、插值、mask 或特殊 token 掩盖。

“共享”只表示结构、张量协议和任务头调用方式共享。每个数据集拥有独立的编码器、任务头、归一化统计量、checkpoint 和 artifact 身份。

## 2. 论文目标方向

论文主题固定为：**面向工业设备多传感器时序状态预测与故障风险评估的概率建模方法研究**。

### 2.1 研究问题

1. 如何从规整的多传感器历史窗口学习具有时间尺度和变量交互能力的统一状态表示？
2. 如何针对连续状态、RUL 和离散故障任务输出可评价的条件概率分布？
3. 如何只使用冻结的预测分布和 validation-only 校准器构造可追溯的风险决策？

### 2.2 模型主线

```text
规整历史窗口 X[T, N] + 时间戳 T[T] + 可选已知工况 C
    -> UnifiedTemporalEncoder
       多尺度时间块 -> 变量级表示 -> 跨变量交互 -> 历史聚合
    -> dataset-specific head
       MetroPT: 多传感器状态头
       C-MAPSS: RUL 头
       TEP: 故障检测/诊断头
    -> point / probability output
    -> validation-only calibration -> risk decision
```

编码器必须保留变量级差异，不能把一个全局向量复制到所有传感器。点输出和概率输出必须来自同一任务头族：点预测是概率头的中心输出（中位数或显式中心参数），不能维护一套彼此独立且语义不一致的 point 分支。

### 2.3 章节职责

| 章节 | 研究对象 | 允许报告的核心结果 |
|---|---|---|
| 第三章 | 表示与点预测 | MAE、RMSE、Macro-F1、参数量、训练/推理开销 |
| 第四章 | 概率预测质量 | NLL、CRPS、Energy Score、PICP、MPIW、Log Loss、Brier、ECE |
| 第五章 | 冻结分布到风险决策 | AUROC、AUPRC、F1、校准、Lead Time、Detection Delay、误报/漏报 |

第三章不调风险阈值，第四章不重复报警策略，第五章不重新训练编码器或重复表示/概率头扫描。三个数据集的指标只在各自任务内解释，不跨任务平均或排序。

## 3. 完全移除的设计与文件范围

### 3.1 新主线接口中必须删除

- `HistorySample.M_obs`、`HistoryBatch.M_obs`、`mask`/`target_mask` 作为模型输入或损失控制项；
- `missingness.py`、`missingness_features.py`、`industrial/missing.py`、`experiments/masks.py` 对主线的依赖；
- `missing-rate`、`missing-mode`、人工 mask 文件和 mask seed；
- `HierarchicalMissingnessKAFEncoder` 及其 mask-safe 命名；
- `M_obs`、`M_q`、`mq_flat`、`missing`、`mask` 出现在 unified 模型、配置、manifest、日志和论文主线中的字段；
- 以“异步观测、不规则缺失、缺失鲁棒性”为创新点、消融项或结论。

执行删除前必须完成全仓库 import/reference 审计。若旧实验或 checkpoint 仍需读取，先导出只读兼容脚本到 `legacy/`，不得让新模型包继续依赖旧缺失接口；不修改已有 `result/` 内容。

### 3.2 仍然保留但不属于缺失建模的检查

- 输入列 schema、dtype、时间单调性、重复时间和有限值检查；
- 训练集拟合的均值/方差或其他归一化统计量；
- 历史窗口与目标窗口的时间因果隔离；
- 已知工况 `context`，仅当它在预测起点前可获得且被数据协议明确声明时使用。

这些检查失败时应抛出带数据集、列名和样本位置的错误，不允许自动修复。

## 4. 必须修正的 P0-P2 缺陷

### P0：模型语义与数据泄漏

1. **变量表示退化**：修正 encoder，使每个变量拥有独立表示并显式进行跨变量交互；添加通道区分测试，任意两个不同输入通道不能恒等输出。
2. **TEP 划分错误**：按 `faultNumber + simulationRun` 先划分实体，再生成窗口；train/validation/test 均必须非空且实体不交叉。正常类为 0，故障类别数为 `max_label + 1`，非法标签直接报错。
3. **C-MAPSS 标签语义错误**：训练/验证 cap 只作用于训练目标并写入 manifest；官方 test RUL 保持原值，不截断；测试窗口按 engine 协议构造并避免未来信息泄漏。
4. **最佳 checkpoint 未恢复**：按任务特定 validation metric 选择并恢复 best checkpoint；test 评估必须显式开启且只执行一次。
5. **归一化泄漏与 NaN 静默处理**：统计量只能由 train split 拟合，保存统计方案与特征 schema；任何非有限输入直接失败。

### P1：任务头与评价

1. State/RUL 的 point 输出必须等于对应概率输出的中心量；统一 target shape、量纲和反归一化路径。
2. TEP 正式返回 detection probability 和 diagnosis probability；事件评价必须报告检测、漏报、误报和延迟分布，不能只对已报警 run 求平均。
3. MetroPT 固定传感器 schema，排除 `Unnamed: 0` 等索引列；目标窗口与历史 context 分离，禁止使用预测起点后的 context。
4. manifest 必须绑定 dataset schema、split identity、normalizer identity、target convention、evaluator version 和 model version；不记录任何 hash。

### P2：工程一致性

1. 删除不可达代码和过时版本名，统一 `UnifiedTemporalEncoder.VERSION` 与实际结构。
2. registry 只注册可运行模型；未实现基线标记为 `not_implemented`，不得进入正式结果表。
3. 训练入口使用显式 config，不从绝对路径、环境变量或隐式全局状态推导数据协议。
4. runner 对空 split、空类别、非有限 loss 和 shape 不一致立即失败，并输出可定位错误。

## 5. 分阶段实施与提交门禁

每一阶段必须单独提交，提交前运行对应测试并更新 `plan/progress.md`。

### Stage 0：边界冻结

- 更新 `3.2_结构框架.md` 的“时间戳、观测掩码”标题为“时间戳与通道身份编码”；
- 更新 `README.md`、`plan/experiment-protocol.md`、CLI 示例，删除缺失参数；
- 建立 `rg` 审计清单，确认 unified 主线不再引用缺失字段。

门禁：`rg -n -i 'missing|缺失|M_obs|mask' code/kaf_profiti/models/unified code/run_unified_experiment.py` 无结果（普通 `isfinite` 检查不得使用这些命名）。

### Stage 1：数据协议与 loader

- 实现固定 schema、有限值检查、train-only normalizer；
- 修正 MetroPT、C-MAPSS、TEP 的划分、标签和窗口语义；
- 为每个数据集增加实体不交叉、无泄漏和标签映射测试。

门禁：合成数据测试全通过；真实文件 smoke 只验证形状、有限值和 split，不产生论文指标。

### Stage 2：无缺失统一 encoder 与任务头

- 删除 unified 输入中的 mask 字段和 missingness 包；
- 实现变量级多尺度时间编码和跨变量交互；
- 统一 point/probability 中心输出，修正 TEP 类别接口和 RUL target convention。

门禁：形状、通道区分、有限值、保存恢复和任务头语义测试通过。

### Stage 3：runner、artifact 与 evaluator

- 恢复 validation-best checkpoint；
- 增加任务特定 validation/test 指标、显式 test 开关和事件级 TEP 评价；
- 完整绑定 manifest 与 checkpoint/config 身份，不记录任何 hash。

门禁：临时输出根可重复构建 artifact；test 未显式开启时不生成 test 指标。

### Stage 4：论文协议与正式实验

- 第三章运行统一模型族点预测矩阵；第四章运行概率质量矩阵；第五章消费冻结结果完成校准和风险评价；
- 每次实验记录 `run_id`、seed、dataset、model、protocol、checkpoint 和指标；
- 正式结果不得混入 smoke、tuning 或 legacy artifact。

门禁：全量测试、数据协议审计、三 seed（如资源允许）和结果 provenance 检查通过后，才可写论文结果性表述。

## 6. 验收标准

- 主模型代码、配置、测试和论文正文无缺失建模设计、字段或参数；
- 三个数据集独立训练，split、normalizer、target 和 evaluator 身份可追溯；
- 所有输入有限、schema 完整，禁止静默填充；
- encoder 的变量表示可区分，点/概率输出语义一致；
- TEP、C-MAPSS、MetroPT 的上述 P0-P2 缺陷均有回归测试；
- `PYTHONPATH=code python -m pytest code/tests/ -q`、`git diff --check` 和真实数据 smoke 通过；
- 没有正式训练结果前，文档只使用“目标、设计、待验证”，不宣称性能提升。

## 7. 当前不做事项

- 不做跨数据集联合训练、参数迁移或统一指标排名；
- 不把原始数据集描述为天然缺失/异步观测数据；
- 不保留旧缺失模型作为新主线 fallback；
- 不在本规格中修改 `result/`、已有 checkpoint 或伪造实验结果。
