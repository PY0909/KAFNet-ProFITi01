# 第三至第五章：分数据集实验、指标与结果产物矩阵

> 日期：2026-10-07  
> 状态：实验规划，所有指标和图表均待真实 formal runs 产生。  
> 适用范围：MetroPT-3、C-MAPSS FD004、Tennessee Eastman Process (TEP)。

## 1. 研究边界与使用规则

论文三章按以下递进关系组织：

```text
第三章：表示与点预测质量
    -> 第四章：预测分布/类别概率质量
        -> 第五章：冻结预测输出上的风险决策质量
```

- 三个数据集使用各自原始数据协议，分别训练和保存参数；不进行跨数据集联合训练、迁移结论或跨任务指标排名。
- 输入为规整、有限值的历史数据。主线不构造或使用观测掩码、人工缺失率、缺失模式或缺失专用模块；非法空值/无穷值按数据协议报错，不静默填充。
- 同一数据集内对比模型共享 split、窗口、训练预算、归一化来源、checkpoint 选择规则和 evaluator。归一化统计只由 train split 拟合。
- 正式结果建议使用 seeds `2026/2027/2028`；每个 seed 先独立计算完整 test 指标，再报告 `mean ± sample std (ddof=1)`。单 seed 结果只能标为诊断/开发结果。
- 所有 test 指标仅在冻结的 validation-best checkpoint 上评估一次。超参数、校准器和报警阈值只由 train/validation 选择。
- 图表只从带 `run_id`、seed、split、schema、normalizer、model/checkpoint/evaluator 身份的正式 artifact 确定性生成。不得手工填入或虚构数值。

### 1.1 模型比较范围与当前实现边界

第三、四章的目标模型族为 TCN、GRU、PatchTST、TimesNet、iTransformer、Unified Temporal Encoder；当前五个比较 backbone 已有连续状态/RUL 点预测与 TEP 类别分类适配。统一 runner 可为 MetroPT、FD004 和 TEP 写入第三章 test 指标与预测产物；这代表代码接口可运行，不代表六模型均已完成真实数据 smoke 或正式多 seed 实验。

| 数据集/任务 | 当前代码能力 | 可纳入正式矩阵的条件 |
|---|---|---|
| MetroPT-3 状态点预测 | Unified 和五个比较 backbone 使用共同 runner；test artifact 含 raw-unit overall/per-sensor MAE、RMSE 与预测/目标张量 | 完成各模型真实数据 smoke、正式训练、多 seed 汇总和 artifact validator 后形成正式主对比 |
| C-MAPSS FD004 RUL 点预测 | native adapter 保留官方 test engines 最后窗口；test artifact 含 raw-cycle MAE、RMSE、NASA score、逐 engine 误差及预测/目标 | 完成各模型真实协议 smoke、正式训练、多 seed 汇总和 artifact validator；当前代码测试不替代真实数据训练 |
| TEP 故障分类/检测 | Unified 输出诊断概率；五个比较 backbone 有分类适配；test evaluator 输出 Macro-F1、Balanced Accuracy、Accuracy、逐类指标、混淆矩阵和 composite-run 汇总，并保存逐窗口概率与标签 | 完成真实数据协议 smoke、正式训练和多 seed/composite-run bootstrap 汇总后，才能形成正式模型对比；第四章概率校准与第五章事件评估仍待单独实现 |

当前代码可生成开发或真实 run 的第三章 test 指标 artifact：MetroPT raw-unit MAE/RMSE，FD004 raw-cycle MAE/RMSE/NASA score 与逐 engine 误差，TEP 分类指标与逐 composite-run 汇总。它们只有在对应真实数据、冻结 validation-best checkpoint 和正式协议运行后才可作为论文结果；单元测试或 smoke 不能替代正式实验。

第五章不重复完整模型扫描，只从第四章中按 validation 规则选取有代表性的冻结模型/概率输出，对照风险映射与校准策略。

## 2. 第三章：表示与点预测

| 数据集 | 主任务与实验类型 | 指标类型 | 结果表 | 结果图 |
|---|---|---|---|---|
| MetroPT-3 | 统一模型族点预测主对比；多尺度/时间/通道身份/工况（如可用）/跨传感器交互消融；历史窗口和表示维度敏感性；冻结模型效率测试 | 主指标：raw-unit overall 与逐传感器 MAE、RMSE；补充 valid target count。效率：参数量、训练时间、batch/sample 推理延迟、峰值显存 | CH3-T-M1 模型主对比；CH3-T-M2 模块消融；CH3-T-M3 窗口/维度敏感性；CH3-T-M4 效率 | CH3-F-M1 各模型 MAE/RMSE 点图（seed 汇总及 CI）；CH3-F-M2 传感器×模型误差热图；CH3-F-M3 代表性预测轨迹；CH3-F-M4 效率-误差权衡图 |
| C-MAPSS FD004 | 统一模型族 RUL 点预测；编码器模块消融；history length/表示维度敏感性；按 engine 的泛化误差分析；效率测试 | 主指标：官方 test engines 上 raw-cycle MAE、RMSE；建议补充 NASA score（非对称惩罚，作为辅助传统指标）；按 engine 报告误差分布。效率同 MetroPT | CH3-T-C1 模型主对比；CH3-T-C2 模块消融；CH3-T-C3 敏感性；CH3-T-C4 效率 | CH3-F-C1 每 engine 真实 RUL vs 预测 RUL 散点及理想线；CH3-F-C2 engine 级绝对误差分布；CH3-F-C3 模型指标/效率对比 |
| TEP | **任务目标：故障类别点分类**（正常类与故障类别）。Unified 和五个比较模型均有分类训练/评估适配；按 `faultNumber:simulationRun` 隔离划分，再切窗，窗口不得跨 run。正式对比和消融仍未运行 | 已实现 Macro-F1、Balanced Accuracy、Accuracy、逐类 Precision/Recall/F1/support、混淆矩阵及 composite-run 指标汇总。Macro-F1 在真实标签或预测中出现的类别上取平均；逐类仍保留零 support 类。正式结果建议进一步按 run bootstrap 汇总 CI | 目标产物：CH3-T-T1 分类模型对比；CH3-T-T2 每类诊断指标；CH3-T-T3 模块消融；CH3-T-T4 敏感性/效率。需真实协议运行后生成论文表 | 目标产物：CH3-F-T1 归一化混淆矩阵；CH3-F-T2 每类 F1/Recall 图；CH3-F-T3 按故障类别的 run 级表现分布。test artifact 已含绘图所需的概率、标签和 run identity |

**第三章指标边界：** 不报告 NLL、Brier、ECE、CRPS 等概率质量指标；不报告 AUROC/AUPRC 风险判别、检测延迟、预警提前量或误报控制。TEP `Log Loss` 属于概率评价，应放第四章，而非点分类主结果。

**TEP 任务锁定说明：** 现有章节资料同时出现“过程状态预测”和“故障分类”。本矩阵将第三章 TEP 主任务收敛为正常/故障类别点分类，以便与第四章类别概率、第五章故障风险构成清楚递进。Unified 通过诊断概率的 argmax 形成类别点预测；比较模型使用共享时序 backbone 和任务专用分类 projection。若论文还需要预测连续过程变量，应另行定义目标变量、预测 horizon、loss 和 MAE/RMSE 表图，作为独立任务，不与故障分类混为一行。

## 3. 第四章：概率预测质量

| 数据集 | 主任务与实验类型 | 指标类型 | 结果表 | 结果图 |
|---|---|---|---|---|
| MetroPT-3 | 目标：状态分位数预测、边际区间覆盖及分布校准；当前 Unified state head 可输出分位数，训练/validation 预测 artifact 可包含 quantiles，但 test evaluator 未汇总概率指标；没有当前证据支持 Gaussian/NLL 或联合密度比较 | 当前可产出：中心点 MAE/RMSE。待实现：test Pinball/分位数覆盖、90%/95% PICP、MPIW、校准统计。NLL、CRPS、Energy Score 仅在密度/评分估计器数学定义和 evaluator 实现验证后纳入 | 目标产物：CH4-T-M1 分位数主对比；CH4-T-M2 分布头/依赖消融；CH4-T-M3 horizon 敏感性；CH4-T-M4 区间与校准统计。正式 test 指标当前不能生成 | 目标产物：CH4-F-M1 状态预测区间；CH4-F-M2 coverage-width；CH4-F-M3 PIT/可靠性或分布诊断；CH4-F-M4 相关结构。test 概率统计未实现前不生成 |
| C-MAPSS FD004 | 目标：RUL 分位数预测与区间质量；当前 RUL head 可输出分位数且 artifacts 可序列化 quantiles，但 test runner 只聚合 point metrics，概率指标尚未实现 | 当前可产出：MAE、RMSE、NASA score。待实现：test Pinball、90%/95% PICP、MPIW、校准统计。NLL 只在实现有效归一化密度后允许报告；CRPS 需定义适用于分位数的估计器 | 目标产物：CH4-T-C1 概率主对比；CH4-T-C2 概率头消融；CH4-T-C3 区间/校准。概率 evaluator 完成前不能生成正式 test 汇总 | 目标产物：CH4-F-C1 RUL 区间；CH4-F-C2 engine 覆盖率/区间宽度；CH4-F-C3 概率可靠性诊断。概率 evaluator 未实现前不生成 |
| TEP | 目标：故障类别条件概率预测；Unified TEP head 输出 detection/diagnosis probability，但 runner 不保存概率或分类预测，baseline 分类头不存在 | 当前仅 runner loss。待实现：Multiclass Log Loss、multiclass Brier、Macro-F1、ECE（预注册 binning）、Balanced Accuracy、逐类 recall。AUROC/AUPRC 明确 one-vs-rest 聚合后才可作为补充 | 目标产物：CH4-T-T1 类别概率对比；CH4-T-T2 校准指标；CH4-T-T3 head/不平衡处理消融。全部需分类/概率 evaluator | 目标产物：CH4-F-T1 reliability diagram；CH4-F-T2 混淆矩阵；CH4-F-T3 run 内类别概率轨迹。当前概率未导出，不能生成 |

第四章现有分位数头并非显式归一化密度。NLL 不可直接套用；CRPS 也必须指定并验证分位数离散近似。当前 runner 的 `head_mode` 不是完整概率评估管线：需要同时实现预测 artifact、按 task 聚合的指标和可复现图表数据后，相关实验才可进入正式矩阵。分类任务不使用连续变量 CRPS/PICP/MPIW。

## 4. 第五章：冻结预测到风险决策

| 数据集 | 主任务与实验类型 | 指标类型 | 结果表 | 结果图 |
|---|---|---|---|---|
| MetroPT-3 | 目标：从第四章冻结状态概率/区间构造故障窗口风险。当前没有概率 artifact、风险标签 evaluator、校准或报警阈值流程 | 目标指标：样本级 AUROC/AUPRC/Brier/ECE/F1；事件级召回、漏检、false alarms/day、time-in-alert、lead time。风险生成、校准和阈值均需 validation-only | 目标产物 CH5-T-M1..M4；风险分数定义、validation/test 分区和事件级汇总器完成前不生成 | 目标产物 CH5-F-M1..M4；当前无风险分数和校准 artifact |
| C-MAPSS FD004 | 目标：从冻结 RUL 概率/区间构造低寿命风险 `P(RUL <= threshold)`。当前没有概率 artifact 或 engine 级风险 evaluator | 目标指标：AUROC/AUPRC/Brier/ECE/F1/Recall、engine-level missed/false alarm、提前量/延迟；RUL 风险阈值须预注册 | 目标产物 CH5-T-C1..C3；需先完成 RUL 概率预测、阈值来源记录和风险 evaluator | 目标产物 CH5-F-C1..C3；当前不能由 point metrics 重建概率校准图 |
| TEP | 目标：由冻结类别概率取 `1 - P(normal)` 作为检测风险，并用故障类别概率诊断。当前 test artifact 已导出类别概率；validation 校准/阈值冻结和完整事件 evaluator 仍待实现 | 目标指标：AUROC/AUPRC/Brier/ECE/F1；检测率、漏检 run、误报 run/小时、Detection Delay 分布（未检测单列）；诊断 Macro-F1/混淆矩阵 | 目标产物 CH5-T-T1..T3；必须补齐 validation 校准/阈值冻结及事件 evaluator | 目标产物 CH5-F-T1..T4；test 概率 artifact 是必要输入但本身不构成风险评估 |

第五章的风险输出只消费第四章冻结 checkpoint 的预测分布/类别概率，不重新训练模型。校准器和报警规则在 validation 冻结，test 只评估一次。MetroPT 的故障报告窗口、C-MAPSS 的 RUL 风险定义、TEP 的注入事件分别解释，不能跨数据集合并原始分数。

## 5. 实施顺序与进入正式实验的门槛

矩阵按“先能稳定生成可审计 artifact，再扩大实验量”执行。建议顺序如下：

1. **第三章 MetroPT/C-MAPSS 点预测**：代码已写入共同 runner 和完整 test 预测/指标 artifact；接下来完成各模型 native data smoke、相同 split/目标/归一化检查和正式多 seed 训练。
2. **第三章 TEP 分类**：已锁定 `faultNumber:simulationRun` 划分、窗口标签语义及 run 统计单位，并补齐统一分类 evaluator、五个 baseline 分类适配和端到端测试；接下来执行真实数据 smoke、正式训练及 run bootstrap 汇总。
3. **第四章概率质量**：将 point 与 quantile/probability head 的训练目标、预测导出和指标 evaluator 分开实现。现有 quantile heads 可作为起点，但不能把 quantile 输出直接叫作 density；实现并验证指标数学定义后再解锁表图。
4. **第五章风险决策**：在第四章 artifact 冻结后实现数据集专属风险标签、validation-only calibrator/threshold、sample/event evaluator、报警策略和 traceability；测试集一次性评估。

每个阶段以代码端到端测试和 artifact schema 检查作为准入条件。缺少预测 artifact、任务指标或独立验证集阈值来源时，该表图状态为“待实现”，不得用 loss、辅助函数输出或 smoke 结果替代。

## 6. 共用表图和 artifact 数据合同

### 6.1 每个 run 必需字段

每个表格数值可追溯至 `run_id`、dataset、task、model/version、seed、split identity、feature/target schema、normalizer identity、checkpoint、evaluator version、配置和有效样本数。每个 test 聚合指标附 test evaluation count；正式 run 必须为 1。

TEP 另外保存 class mapping、每类 support、`faultNumber:simulationRun` 成员清单及 run 级预测聚合规则。连续任务另外保存单位、反归一化规则、horizon/channel 轴定义。第五章另外保存风险标签定义、校准器和阈值来源 split。

### 6.2 表格统计口径

- 连续误差从全局绝对误差和平方误差充分统计量计算；不能平均各 batch 的 MAE/RMSE。
- 分类指标同时保存逐样本预测和逐 run 结果；Macro-F1 对类别逐类计算后等权平均，需明确 undefined class 的处理规则。
- 多 seed 的主表先按 seed 计算 test estimate，再报告 mean ± sample std；置信区间以预注册的自然单位 bootstrap 为准，MetroPT 时间块、FD004 engine、TEP composite run。
- 参数量、训练时间和推理时间必须标明硬件、精度、batch size、warmup/repeat 设置；跨设备不直接比较原始耗时。

### 6.3 推荐的表图目录映射

表图 ID（如 `CH3-T-M1`）是规划 ID，不是最终论文编号。正式排版时再映射到章节表号/图号。建议每张图同时保留生成脚本、源 CSV/JSON 和 PNG/SVG；表格保留可重建的 CSV。

## 7. 需要在正式实验前冻结的事项

1. TEP 第三章主任务固定为故障类别点分类；如还做连续过程状态预测，单列任务、指标和图表。
2. 统一六模型在三个数据集上的 formal matrix。若受数据/实现限制缩减，必须在实验前冻结缩减理由和共同模型集合，不能跑完后挑选有利模型。
3. TEP 是按窗口预测还是按 run/timepoint 汇总；主表应选择一个主要统计单位，另一个可作补充，避免窗口重叠造成伪样本量。
4. MetroPT 与 TEP 风险标签、事件合并规则，以及 C-MAPSS RUL 风险阈值；均需预注册，不能从 test 选择。
5. NLL/CRPS/Energy Score 的具体估计器、概率校准方法、ECE binning 方案、bootstrap 置信区间算法和缺类处理规则。

在上述事项未冻结、正式 artifact 未通过验证前，本文件只作为实验设计，不构成实验结果或性能主张。
