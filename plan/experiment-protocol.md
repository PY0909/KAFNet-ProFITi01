# 三数据集实验协议

> 协议版本：CH3-D0-v1  
> 当前状态：范围已锁定；D0 已由 CH3-P00-T04 关闭，正式实现与实验结果仍未开始  
> 适用章节：第三章至第五章的三数据集原始协议实验

## 1. 协议目的与适用边界

本协议锁定三个数据集的数据角色、实验矩阵、评价指标、公平性约束和正式运行资格。第三章评价共享时序表示与点预测任务适配；第四章评价数据集特定概率预测；第五章评价概率到风险的决策机制。三章使用各自原始数据协议。

当旧版 `README.md`、`统一对比实验方案.md` 与本协议冲突时，以 `plan/implementation-plan.md` 第 11 节及本协议为准。旧版方案中的 validation CRPS、概率指标和风险指标不适用于第三章 point-only 实验。

### 1.1 与第四、第五章的边界

- 第三章只产生 point-only 表示证据（MAE/RMSE、分类指标及效率字段）。
- 第四章三个数据集均使用各自原始观测协议进行概率实验，评价连续任务分布质量和 TEP 故障类别概率质量。
- 第五章三个数据集均使用各自原始观测协议，只消费第四章冻结 checkpoint、validation 校准器和风险标签；风险指标不得回填第三章表格。

## 2. 数据集角色、划分与结论权限

| 数据集 | 数据集 ID | 第三章角色 | 固定划分 | 隔离单位 | 正式模型范围 | 允许结论 |
|---|---|---|---|---|---|---|
| MetroPT-3 | `metropt3_official_first_month` | 第三章状态点预测表示；第四章状态概率预测；第五章故障风险与事件级评估 | 数据起始所在首个日历月为训练期；该月按时间尾段拆出 validation，其余时间只作 test，之后各 split 内生成窗口 | 时间轴与预注册时间块 | 三章均使用对应任务的完整模型族 | 只在 MetroPT-3 原始连续传感器流和维护报告语义内解释 |
| C-MAPSS FD004 | `cmapss_fd004` | 第三章 RUL 点预测表示；第四章 RUL 概率分布；第五章 RUL 风险分层与校准 | 官方 train engines 按固定 `split_seed=2026` 划分 80% train/20% valid；官方 test engines 只作 test | 完整 engine | 三章均使用对应任务的完整模型族 | 只在官方 cycle、run-to-failure 和 RUL 语义内解释 |
| Tennessee Eastman Process | `tep` | 第三章过程状态/故障点预测表示；第四章过程状态或故障概率预测；第五章在线故障检测、诊断与延迟 | faulty training runs 按固定 `split_seed=2026` 划分 80% train/20% valid；official faulty testing runs 只作 test | `faultNumber:simulationRun` 复合 run | 三章均使用对应任务的完整模型族 | 只在原始 process sample、故障场景和运行级评估语义内解释 |

### 2.1 数据集使用硬约束

- MetroPT-3 UCI 原始数据按固定采样记录且字段标注无观测不完整；若构造人工观测不完整，必须明确写为“受控输入扰动/人工多源压力测试”，不能写成原始观测不完整。
- C-MAPSS 的 cycle 轨迹和 TEP 的过程采样通常是规则记录；二者的变长 engine/run、工况变化和故障阶段不等于多源观测不完整。
- FD004 的任何滑动窗口不得跨 engine，归一化不得读取 valid 或 official test engines。
- TEP 必须使用 faulty training/testing 数据；仅有 fault-free 数据时不得完成 TEP 外部验证，也不得降级后仍标记为正式结果。
- TEP 的窗口不得跨 `faultNumber:simulationRun`；availability_map 生成器不得读取 fault label 或故障注入时刻。
- 三个数据集的原始 MAE/RMSE 量纲不同，不跨数据集直接平均、排序或声称绝对误差更优。

### 2.2 确定性 split 算法

所有 unit 列表、排序规则、算法版本、raw source SHA 和最终成员列表写入 canonical JSON；键排序后计算 split SHA。窗口只能在 split 成员确定后生成。

- **MetroPT-3：** 稳定按 timestamp 排序并将相同 timestamp 作为一个整体。默认官方基准取起始时间所在的首个日历月作为训练期；为满足 validation-only checkpoint 选择，将该月唯一 timestamp 组的末尾 `max(1, round(validation_fraction * n_month))` 组留作 validation，其余组为 train。首月之后的所有时间戳均为 test，绝不进入训练、调度或 checkpoint 选择。同一 timestamp 的所有 source rows 不得拆分；train、validation 或 test 任一窗口不足时协议失败。旧 chronological 50/20/30 仅通过显式 `--metropt-protocol chrono_502030` 选择。
- **FD004：** 对官方 train 中升序排列的 engine IDs 使用 `numpy.random.Generator(PCG64(2026)).permutation`。令 `n_valid=max(1, floor(0.20*N+0.5))`，排列后的前 `N-n_valid` 个 engines 为 train，末尾 `n_valid` 个为 valid；两者都必须非空。官方 test engine IDs 原序登记为 test，不参加随机排列。
- **TEP：** unit key 固定为 `faultNumber:simulationRun`。在每个 `faultNumber` 内按数值升序列出 training composite runs，再以 `SeedSequence([2026, faultNumber])` 初始化 PCG64 独立排列；每层 `n_valid=max(1, floor(0.20*N_f+0.5))`，前 `N_f-n_valid` 个为 train、末尾为 valid。每个 fault 层必须同时有 train/valid，否则协议失败；official faulty testing composite runs 只作 test。
- 重复 `(timestamp, source_row_id)`、重复 engine ID、重复 composite-run key、unit 跨 split、成员列表与 source SHA 不符均为 hard failure。重复 timestamp 本身合法，但必须按上述唯一 timestamp 组整体分配；不得通过重新抽 seed 规避。

## 3. 固定窗口与训练预算

| 数据集 | `history_len` | `pred_len` | `stride` | `split_seed` | formal epochs | batch size |
|---|---:|---:|---:|---:|---:|---:|
| MetroPT-3 | 168 | 24 | 60 | 2026 | 50 | 128 |
| C-MAPSS FD004 | 50 | 10 | 1 | 2026 | 80 | 128 |
| TEP | 96 | 24 | 12 | 2026 | 50 | 128 |

- formal 训练完整执行表中 epoch 预算，不启用 formal early stopping；每个 epoch 只访问 train/valid，保存最低 validation MAE 对应的 best checkpoint。
- 同一数据集内所有模型使用相同 batch size、最大 epoch、损失函数和 checkpoint 规则。若显存不足，必须先记录范围变更并对同一比较族全部重跑，不能只为某个模型缩小 batch 或截断数据。
- 第三章统一使用 availability_maped MAE 训练目标、`hidden_dim=64` 和同规格 MLP point head；head 敏感性实验按预注册 linear/MLP 取值单独标识。
- 六模型的学习率只在 train/valid 上从 `1e-3/3e-4/1e-4` 中选择，tuning 使用 seed 2026、最多 15 epochs、patience 5；以最低 validation MAE 决胜，并在 formal 前冻结到 tracked 配置。P08-P12 不得重新选择学习率。
- MetroPT-3 的 formal canonical window catalog 对 E0/E1/E4/E5/E6 统一按 `max_history_len=336`、`pred_len=24`、`stride=60` 判定可用 forecast origins，window ID 由 split、unit、`forecast_start` 和 query timestamps 确定，不含 `history_len`。E5-S history 扫描的 96/168/336 分别只取同一 336 历史父窗口的最后 96/最后 168/全部 336 个位置；train/valid/test 的 forecast-origin、target 和 window ID 集不变，因此中心 168 配置可直接复用 E1 checkpoint。

## 4. 模型、实验族与运行数量

### 4.1 固定模型集合

- MetroPT-3 E1 六模型：`li_tcn`、`ff_gru`、`availability_maped_tcn`、`gru_d`、`ode_rnn`、`kst_light`。
- E4 与外部验证四模型：`li_tcn`、`gru_d`、`ode_rnn`、`kst_light`。
- 显示名固定为 LI+TCN、FF+GRU、Availabilityed TCN、GRU-D、ODE-RNN、KST-Light；同一代码 ID 不得映射到不同结构。
- GRU-D 与 ODE-RNN 必须通过来源、公式、真实时间输入和行为测试的 fidelity gate；简化代理实现不能以正式模型名进入论文。

### 4.2 固定实验矩阵

| 实验族 | 数据集 | 固定条件 | 重复与 coverage |
|---|---|---|---|
| E0 数据协议 | MetroPT-3 | KST-Light × native sparse、linear interpolation、forward fill、availability_maped input | 4 协议 × 3 seeds = 12 formal matrix rows |
| E1 主对比 | MetroPT-3 | 六模型，原始规则采样观测协议 | 6 模型 × 3 seeds = 18 rows |
| E4 MetroPT-3 受控可用率 | MetroPT-3 | 四模型 × random 0%/10%/30%/50%/70% | 可选附加矩阵；不代表原始观测不完整 |
| E4 MetroPT-3 机制压力测试 | MetroPT-3 | 四模型 × random/block_offline/mixed，均 matched 30% | 可选附加矩阵；用于点预测机制鲁棒性，不延伸为概率/风险结论 |
| E5-A 消融 | MetroPT-3 | full、wo_time、wo_delta、wo_availability_map、wo_channel、wo_context、wo_multiscale、wo_cross_sensor | 8 变体 × 3 seeds = 24 rows |
| E5-S 敏感性 | MetroPT-3 | history、hidden、patch、time dim、head 的 OFAT 扫描 | 14 factor-views 映射到 12 formal runs |
| E1-X 外部验证 | FD004、TEP | 各四模型，原始 cycle/sample 观测协议 | 2 数据集 × 4 模型 × 3 seeds = 24 rows；压力测试另行注册 |
| E6 效率 | MetroPT-3 | E1 六模型冻结 checkpoint | 6 模型 × 3 seeds = 18 条独立 profile，不增加训练 run |

核心 formal matrix 由 E0、E1、E5-A、E5-S、E1-X 和 E6 组成；E4 的人工可用性矩阵作为单独注册的附加压力测试，不得改变 E1 主结果的原始观测口径。最终 rows 和唯一训练 run 数由条件 registry 生成，不能沿用旧版 198/165 计数。

### 4.3 E0 四种 history-only adapter 合同

四种 adapter 只接收历史批次 `(X_h, M_h, T_h, time_gap)` 和 train-only fill，不得接收 `Y_q`、`M_q` 或 forecast origin 之后的值。`train_fill` 是 train normalization 后的通道均值，因此在标准化空间固定为 0。每个 adapter 输出相同 shape、window IDs 和 target schema，但输入语义如下：

| protocol | `X_h` 观测不完整位置 | 输出 availability_map | 输出时间/`time_gap` | 边界与全空通道 |
|---|---|---|---|---|
| `native_sparse` | 仅为张量安全写入 `train_fill`，不得标成观测 | 保留原始 `M_h` | 保留真实 `T_h`；由原始 availability_map 和真实时间计算 `time_gap` | 使用 `train_fill`，availability_map 仍为 0 |
| `linear_interpolation` | 只在同一 history window、同一通道的两个历史观测之间按真实时间线性插值 | adapter value availability_map 置 1；原始 availability_map 另存 provenance | 使用规则历史索引；已填充值的 `time_gap=0` | 首段、尾段和全空通道使用 `train_fill` |
| `forward_fill` | 每个通道只沿过去方向复制最近观测 | adapter value availability_map 置 1；原始 availability_map 另存 provenance | 使用规则历史索引；已填充值的 `time_gap=0` | 首个观测前和全空通道使用 `train_fill` |
| `availability_maped_input` | 原始观测不完整位 zero-fill，即标准化空间的 `train_fill` | 保留原始 `M_h` | 使用规则历史索引；按原始 availability_map 计算规则间隔 `time_gap` | 使用 `train_fill`，availability_map 仍为 0 |

- “history window 内”固定为 `t <= forecast_origin` 且属于同一 split；linear 可以使用窗口内较晚的历史观测，但不能读取预测起点及其后的 query/target。
- adapter 不得跨窗口、split、engine 或 composite run 搜索填充值，也不得使用 valid/test normalization。
- 四协议必须记录 raw availability_map SHA、adapter config SHA、输出 availability_map SHA、时间语义和 train-fill SHA；`native_sparse` 与 `availability_maped_input` 的时间语义不同，validator 必须拒绝二者 scientific signature 相同。
- 手工数组测试必须覆盖首段、尾段、内部观测不完整、全空通道、不规则时间、target 扰动和 forecast-origin 后值扰动。

## 5. 随机种子与观测不完整协议

### 5.1 固定种子

- 核心训练 seeds 固定为 `2026`、`2027`、`2028`。
- `split_seed=2026` 固定数据划分；`availability_map_seed` 等于该次训练 seed；模型初始化、Python、NumPy、PyTorch、CUDA 和 DataLoader RNG 均由同一 run seed 派生并写入 manifest。
- 同一数据集、实验条件和 seed 下，所有模型复用完全相同的 split、target、normalization 和 availability_map SHA。
- E5-S 非中心取值只运行 seed 2026，中心配置复用三 seed；单 seed 结果只作描述性比较，不报告伪造的标准差或显著性。

### 5.2 固定输入设置

- E1 主设置为数据集原始观测协议：MetroPT-3 使用原始规则采样；C-MAPSS 使用 engine cycle；TEP 使用 process sample。第四章和第五章同样对三个数据集使用各自原始协议，不得为了统一而把三者默认转换成 mixed 观测不完整。
- E4 附加压力测试使用 MetroPT-3 的代表性 `random@0.30`；如研究问题需要机制对照，再增加 `block_offline@0.30`。不执行完整 `0.00/0.10/0.30/0.50/0.70` 扫描，C-MAPSS/TEP 的压力条件需另行注册并说明物理含义。
- 所有人工可用性映射 必须记录为 `controlled_observation_corruption`，目标实际可用率容许误差 1 个百分点；不得写入 `raw_availability=true`。

第四章和第五章不得将上述 E4 rows 当作概率或风险实验结果；三个数据集的概率与风险主结果必须使用各自原始协议，并重新生成与对应 checkpoint、校准器和风险标签绑定的 artifacts。输入扰动只可作为单独注册的附加实验。
- availability_map 必须先在完整 split 时间轴上生成并固化，再切片到窗口；生成器不得读取 label、未来目标、故障注入时刻或 test metric。
- 相同 `dataset + split + condition + seed` 的 availability_map 只允许一个内容哈希；模型间不得各自重新采样。

## 6. Checkpoint、测试与 formal 判定

### 6.1 Checkpoint 与测试访问

- 训练和超参数选择只访问 train/valid。checkpoint selector 固定为最低 validation MAE；并列时使用预注册配置顺序，不读取 test 排名、效率或参数量决胜。
- best checkpoint 冻结并通过哈希校验后，test loader 只允许完整遍历一次。任何重评估必须引用同一 checkpoint 和既有 prediction artifact，不能借重评估重新选模。
- test 指标只用于最终报告，不用于改变窗口、availability_map、模型、学习率、训练轮数、消融项或统计方法。

### 6.2 formal 资格

一次运行只有同时满足以下条件，才能标记为 `run_level=formal` 并在 validator 通过后取得 `evidence_status=formal_validated`：

1. 使用本协议规定的数据集 ID、窗口、seed、观测不完整条件、模型集合、训练预算和 point-only 目标。
2. 使用 tracked YAML 与冻结超参数；CLI 只覆盖 data/result root、device、run ID、seed 和 resume，不覆盖科学参数。
3. `split_seed=2026`，使用 timeline-first availability_map schema、完整 train/valid/test loader，且 train normalization 无 valid/test 泄漏。
4. 不设置 train/eval batch 上限，不截断 engine、fault、run、时间段或 official test，不以子集替代正式数据。
5. checkpoint 只由 validation MAE 选择，best checkpoint 冻结后 test evaluation count 精确为 1。
6. 保存 clean Git SHA、dirty flag、环境与硬件、resolved config、协议/数据/split/availability_map/code SHA、history、best/final checkpoint、完整 prediction 和 metrics。
7. artifact validator 确认文件、shape、有限指标、有效目标数、相对路径和内容哈希全部有效。

smoke、tuning 和 profile 使用独立状态与目录。限 batch、少 epoch、调试、fixture 或失败恢复中的中间结果均不具备 formal 资格。

### 6.3 profile 与一次性 test 的边界

- `test_evaluation_count` 只计持有 target 的完整 evaluator 遍历；每个 formal run 必须精确为 1。
- E6 profile 只能在 E1 formal checkpoint、prediction 和科学配置全部冻结后创建，固定读取与该 run 对应的预加载 test **feature-only** batch sequence；profile API 不接收 target、target availability_map、loss 或预测指标 evaluator。
- profile 固定 FP32、batch 128、warm-up 10 batches、measure 50 batches、5 repeats；仅保存参数量、batch/sample latency、峰值显存、device fingerprint、batch-sequence SHA 和原始 repeats。
- feature-only benchmark 的重复 forward pass 记录为 `profile_feature_passes`，不计入 `test_evaluation_count`，也不能修改 checkpoint、prediction、metrics 或任何科学配置。
- batch sequence 按预注册 window ID 顺序生成，不按误差、标签或速度挑选；缺 target 隔离证明、source checkpoint SHA 或 batch-sequence SHA 时 profile validator 失败。

## 7. 固定指标与统计口径

- 预测指标只包括 MAE、RMSE 和 valid target count；MAE/RMSE 必须由全局 absolute-error sum、squared-error sum 和有效点数计算，禁止 batch 或窗口等权平均。
- 鲁棒性相对退化统一为 `100 * (metric_condition - metric_reference) / metric_reference`，正值表示误差变差；reference 必须在相同 model、seed 和数据集内配对。
- 效率字段包括参数总数/可训练参数数、完整训练时间、batch/sample inference latency、峰值显存、硬件、精度模式和原始 repeats。
- 三 seed 项先分别计算每 seed 全量有效目标点估计，再报告算术 mean、sample std（`ddof=1`）与 95% CI。
- 置信度固定 0.95，bootstrap resamples 固定 10000，bootstrap seed 固定 2026，CI 固定为 percentile `[2.5%, 97.5%]`。
- MetroPT-3 的 block 秒数固定为 `24 * stride * train_median_interarrival_seconds`；从 test 起点锚定半开、非重叠区间 `[t0+k*block, t0+(k+1)*block)`，末尾不足一块仍保留。每个预测窗口以 `forecast_start`（最早 query timestamp）归入唯一 block，该窗口的全部 target 充分统计量随窗口整体归入；即使 prediction horizon 跨 block 边界也不拆分。block 归属和选择不得读取 test error 或 label。
- FD004 使用完整 engine，TEP 使用 `faultNumber:simulationRun`。每个 seed 内对自然单位有放回抽样，paired comparison 共用同一抽样索引；每次重采样按充分统计量重算全局 MAE/RMSE，再对三个 seed 的估计取算术均值形成 bootstrap replicate。
- 每个直接对比族（含 E5-S）的 bootstrap unit、forecast-origin、target 和 canonical window ID 集必须一致；E5-S history 扫描只允许输入历史长度按上述父窗口尾部截取规则不同。全体条件同为零有效位的 unit 可记录后排除，只有部分条件为零或成员不一致时立即失败。不能仅以三个 seed 做 Wilcoxon 显著性结论。
- 不跨数据集合并原始 MAE/RMSE；外部验证的相对指标只在各数据集内相对同 seed LI+TCN 计算。

## 8. 公平性与防泄漏合同

| 公平维度 | 固定规则 | 机器核验依据 |
|---|---|---|
| split | 同一数据集与 seed 的所有模型使用完全相同的 train/valid/test 单元和窗口边界 | split ID、split SHA、window ID 集合 |
| availability_map | 同一数据集、条件和 seed 共享同一个 timeline-first availability_map，不允许模型内重新采样 | availability_map ID、availability_map SHA、requested/realized rate |
| target | 所有模型预测相同未来通道、时间位置和有效目标；input transform 不得改变 target availability_map | target schema SHA、target availability_map SHA、test window IDs |
| normalization | 均值、标准差或其他缩放统计只由 train split 计算，valid/test 只应用冻结统计量 | normalization SHA、source split=`train` |
| 训练预算 | 同数据集模型使用相同 batch、formal epochs、availability_maped MAE、学习率候选数和 validation-only 选择规则 | resolved config、tuning report、history |
| point head | 表示比较统一使用同规格 MLP point head；只在 E5-S 的预注册 head 敏感性中切换 linear | model manifest、head config SHA |
| checkpoint | 全部方法以最低 validation MAE 选择 best，test 只评估冻结 best 一次 | selector、best SHA、test count |
| 评价 | 使用同一个全局 MAE/RMSE accumulator 和相同有效位；相对指标在同 seed 内配对 | evaluator SHA、valid count、provenance |

### 8.1 方法特异输入的公平边界

- LI+TCN、FF+GRU 的插值或填充只能由当前及历史观测确定，不得利用未来真实值、valid/test 统计量或跨 split 信息。
- GRU-D 必须接收由同一观测 availability_map 计算的真实 `time_gap`；ODE-RNN 必须接收同一事件的真实时间戳。方法所需时间输入不属于额外标签，但其来源和单位必须写入 manifest。
- Availabilityed TCN 与 KST-Light 使用同一原始观测、availability_map 和目标；KST-Light 不能通过 context、观测不完整模式或缓存旁路读取 target。
- 模型结构可以因方法定义不同，但超参数搜索次数、数据访问权限、训练预算、checkpoint 规则和评价脚本必须一致。
- 公平性任一 SHA 不一致时，该比较单元整体失效；不得只删除表现不利的模型或 seed。

## 9. 第三章禁止项

以下内容一律不得用于生成、选择或支撑第三章结论：

1. 禁止实例化或调用 `ProFITiFlowHead`、`LowRankCopulaFlowHead`、`QuantileHead`、`RiskHead`；第三章模型输出只能是 point prediction。
2. 禁止用 NLL、CRPS、Energy Score、PICP、MPIW、AUROC、AUPRC、F1、ECE、风险概率、风险阈值或校准结果证明表示质量。
3. 禁止用 test split 调参、选模型、选 checkpoint、选窗口、选输入机制、选统计方法或追加实验；`test_best` 及任何 test-label 搜索字段只能作为被 validator 拒绝的非法字段。
4. 禁止手工修改 metrics、prediction、availability_map、checkpoint、profile repeats、聚合 CSV 或表图数值；修复必须通过代码重建 artifact。
5. 禁止把 smoke、tuning、fixture、限 batch、少 epoch、截断 engine/run/time range 或调试结果放入正式表图。
6. 禁止以 fault-free TEP 替代 faulty 外部验证，禁止按行随机划分 FD004/TEP，禁止让窗口跨 split、engine 或 composite run。
7. 禁止用不同 split、availability_map、target、normalization、训练预算或评价脚本比较模型，也禁止因结果方向不理想删除 seed、模型或条件。
8. 禁止在 tracked 文档、配置、代码或 artifact manifest 中写死机器绝对路径、账号、主机、IP、端口或远程实例链接。

## 10. Artifact 与可追溯要求

每个正式 run 必须记录 `run_id`、dataset、model、variant、seed、split seed、availability_map seed、窗口、输入设置、实际可用率、训练预算、checkpoint selector、test evaluation count、MAE、RMSE、有效点数、参数量和效率字段。manifest 中所有文件引用使用 result root 相对路径；代码和配置不得依赖已解析的旧机器地址。

每个报告数字必须可追溯到 `run_id -> prediction/metrics -> checkpoint -> resolved config -> split/availability_map/protocol/code hash`。原始 metrics、prediction、availability_map、checkpoint 和 profile repeats 只允许由程序生成，不得手工改写。

## 11. 参考依据

- `3.2_结构框架.md` 第三章结构与 3.6 实验职责。
- `docs/毕业论文研究边界与总体方案.md` 的数据集角色、统一证据矩阵和防泄露规则。
- `plan/implementation-plan.md` 的 CH3-P00 至 CH3-P14 锁定计划。
