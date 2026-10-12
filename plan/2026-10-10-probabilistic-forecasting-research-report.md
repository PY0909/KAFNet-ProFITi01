# 多变量时序概率预测与风险评估：论文、方法与代码调研报告

- 调研日期：2026-10-10
- 任务：面向多变量时序预测的条件概率预测与风险评估
- 结论性质：**纯文献与代码调研**。未运行实验，未修改 `code/`、模型或 `result/`。

---

## 0. 前置声明（任务书明确要求）

- **KST ProbFlow 是拟议方法框架，不是现有基线**，本项目尚无其代码实现，也**不存在**其官方仓库。本次调研**未**将其作为既有方法检索。
- 当前代码基线为 **Unified V3C 点预测模型**。
- 本轮**不把联合采样作为概率预测的必要条件**；不把概率区间自动解释为联合分布；不把风险分数自动解释为故障概率。

---

## 1. 执行摘要

**当前项目状态（代码核实，`code/kaf_profiti/models/unified/`）**：

| 已有 | 缺失 |
|---|---|
| `StateHead` 分位数头，`quantiles=(0.1,0.5,0.9)`，用 `base + cumsum(softplus(increments))` 构造**天然非交叉**分位数（`heads/state.py:20-33`） | **无 CRPS / NLL / PICP / MPIW / ECE / Brier / interval score**——`experiments/metrics.py` 只有 `state_metrics`、`rul_metrics`、`classification_metrics`，均为点预测或分类指标 |
| `pinball_loss`（`heads/base.py:38-45`） | **无 validation-only 校准层** |
| `RULHead`（`heads/rul.py`）、`TEPHead`（`heads/tep.py`，检测+诊断双头，probabilistic 模式） | **无风险分数生成模块** |

> **关键判断**：任务书 §10 方案 A（分位数预测头）**已经存在**，真正的缺口是**概率评价指标**与**独立校准层**。这直接改变了实现路线的优先级。

**当前研究问题**：如何在不破坏 V3C 现有训练契约的前提下，为 MetroPT-3/FD004/TEP 三类工业任务提供**可校准**的不确定性表达，并转化为风险分数。

**概率预测方法族**：参数化（Gaussian/Student-t）、分位数、预测区间、结构化低秩、copula、flow、diffusion、conformal、集成、校准与风险评估（详见 §3）。

**最接近的方法**：
- **DeepAR**（IJF 2020, arXiv:1704.04110）——参数化 Gaussian/负二项 + 自回归采样；
- **GPVAR / DeepVAR**（NeurIPS 2019, arXiv:1910.03002）——低秩 Gaussian copula，**边缘 copula 变换 + 低秩协方差**，参数 $O(Nr^2+r^3)$；
- **Stankevičiūtė et al.**（NeurIPS 2021）——conformal 多步区间，**逐 horizon Bonferroni 修正**，轻量且不改架构；
- **CQR**（NeurIPS 2019）——conformalized quantile regression，区间自带自适应宽度。

**可实现的最小方案**：**不新增概率头**（V3C 已有分位数头），而是 (1) 实现分位数/Gaussian 双头对照；(2) 新增 CRPS/NLL/PICP/MPIW/interval score 指标；(3) 新增 validation-only conformal 校准层；(4) 由预测分布导出风险分数。

> **代码核验后的关键补充**：GluonTS 官方库（`awslabs/gluonts`，Apache-2.0，DeepAR 作者团队）中的 `mx/block/quantile_output.py::crps_weights_pwl()` 给出了**加权 pinball loss 折算 CRPS 的权重公式**——意味着 V3C 现有分位数头**无需任何采样即可计算 CRPS**。这进一步降低了第一版门槛。

> **⚠ 代码核验发现的两个泄漏风险（详见 §5.3）**：
> ① **EnCQR 官方仓库不可直接照搬**——它用 train 段标签算 conformal 分数，且循环内用 test 标签在线更新 epsilon，与 AGENTS.md 硬规则冲突；
> ② **TSDiff 用 per-window 缩放**而非 train-split 归一化，不满足本仓库归一化护栏。
> ✅ **可安全借鉴的对照**：GluonTS / pytorch-ts / TACTiS 的 scaler 均只取 context/历史段，划分用顺序 `ValidationSplitSampler`。

**是否需要联合采样**：**第一版不需要**。分位数头与低秩 Gaussian 头的 NLL/CRPS 均可**解析计算**（Cholesky 分解），无需采样；仅在需要联合样本时（如生成式）才引入采样器。

**是否需要增加通用数据集**：**建议增加** Electricity/Traffic/Weather 之一，否则论文只能声称"在三类工业任务上验证"（任务书 §7 明确要求区分）。

**证据等级**：**B**。论文身份与机制多经官方页核验（A 级），但**各方法在工业三数据集上的适配性、消融细节与数值均未核验**（因工具限流与任务范围）；**本轮无任何实验数据**。

---

## 2. 检索范围和检索式

### 2.1 数据库、日期、时间范围
- **arXiv abs 页**（`arxiv.org/abs/*`，逐篇核验标题/作者/日期/注释）
- **官方 proceedings**：NeurIPS、ICML（PMLR）、AAAI、IJF（ScienceDirect/IDEES）
- **NeurIPS Poster 页、PMLR 页面、作者主页、GitHub**
- **检索日期：2026-10-10**（本次任务）；时间范围 **2020–2026**，保留 DeepAR(2017 preprint/2020 期刊)、GPVAR(2019)、CQR(2019) 等奠基工作。

### 2.2 检索式（已执行）
`probabilistic time series forecasting`、`probabilistic multivariate forecasting`、`predictive distribution time series`、`uncertainty-aware time series forecasting`、`multi-horizon probabilistic forecasting`、`quantile forecasting`、`prediction interval forecasting`、`calibrated probabilistic forecasting`、`probabilistic transformer forecasting`、`diffusion time series forecasting`、`conformal time series forecasting`、`forecast uncertainty calibration`、`low-rank Gaussian Copula Processes`、`DiffTime/TSDiff diffusion probabilistic forecasting`、`Conformal Time-Series Forecasting`、`CQR Conformalized Quantile Regression`

### 2.3 纳入/排除标准
- **纳入**：正文可定位公式/机制，或官方页/代码可核验机制与身份。
- **排除**：仅检索摘要可得且无法二次核验者（降级为 C 级且不支撑强结论）；第三方博客转述（仅作线索）。

### 2.4 正文与代码核验标准
- **正文核验**：arXiv abs 页标题+作者+日期+comment 字段，或官方 proceedings 页。**本轮实际核验到的正文公式主要来自官方 PDF 片段**（如 GPVAR 式 $\Sigma=D+VV^\top$、Conformal 论文的式(9) 与 Bonferroni 修正）。
- **代码核验标准**：README 与论文一致 + 含训练/评估代码 + 可定位关键文件。

### 2.5 检索局限（如实记录，必须随结论引用）
1. **arXiv API 与 Jina Reader 在本轮中后期对本机 IP 触发限流/封禁**（Jina 返回 `401 AuthenticationRequiredError: bad IP reputation`），部分 ID 只能改用官方 proceedings 页核验。
2. **一个并行代码仓库子代理仍在运行**，本报告 §5 代码表以已核验条目为主，未完成部分标注"待子代理返回"。
3. 未系统检索 IEEE Xplore/ACM DL/Scopus/WoS/知网；未做完整引用链回溯。
4. **因此"未发现某方法"的结论仅限本检索范围**，不等于不存在。
5. 各概率方法的**工业数据适配性与数值表现基本未核验**。

---

## 3. 概率预测方法分类

| 类别 | 代表 | 输出 | 需联合采样 | 训练难度 | 小传感器集适配 | 数值发散风险 | 风险分数来源 | 与点预测头公平比较 |
|---|---|---|---|---|---|---|---|---|
| **参数化（边缘）** | DeepAR（Gaussian/负二项）、MQ-RNN/MQF2（分位数） | 参数→样本/分位数 | Gaussian 需采样（分位数可解析） | 低-中 | ✅ 好（N 维独立边缘即可） | 低（Gaussian 尺度需约束） | 尾部概率、区间超限 | ✅ 容易（同编码器换头） |
| **分位数** | MQ-RNN、MQF2、non-crossing QR、pinball | 多分位数 | ❌ 否 | **低** | ✅ **最好** | **最低**（cumsum 保序） | 分位数距离、区间 | ✅ **最易** |
| **预测区间 / conformal** | Stankevičiūtė et al. (NeurIPS'21)、CQR (NeurIPS'19) | 区间 | ❌ 否 | **极低**（后处理） | ✅ **最好** | **最低**（不反传） | 区间外判定、conformal score | ✅ **最公平**（不改模型） |
| **结构化低秩 Gaussian** | GPVAR / DeepVAR (NeurIPS'19) | 位置+低秩协方差 | 边缘解析；联合需 Cholesky | 中（需正定约束） | ✅ **较好**（$r\ll N$，O(Nr²+r³)） | **中**（$V$ 使 $\Sigma$ 非正定） | Mahalanobis、联合尾部 | ✅ 中（需对齐方差语义） |
| **Copula** | GPVAR（边缘非参数 CDF→Gaussian copula）、Deep generative quantile-copula | 边缘+依赖 | 联合需要 | 中-高 | ⚠ 中（N 小时收益小） | 中-高 | 联合超限概率 | ⚠ 较难（分布语义复杂） |
| **Flow** | normalizing-flow 预测 | 密度 | 密度解析，样本需采样 | 中-高（似然训练） | ⚠ 中（样本量小时不稳） | **高**（雅可比/数值） | 密度比、密度阈值 | ❌ 难（训练范式不同） |
| **Diffusion** | TimeGrad、TimeDiff (ICML'23)、TSDiff | **样本** | ✅ **必须** | **高** | ❌ **差**（迭代采样，小数据慢） | 中 | 样本统计 | ❌ **最难**（成本/范式） |
| **集成** | EnbPI 类 bootstrap 集成 | 经验分位数 | ❌ 否 | 低 | ✅ 好 | 低 | 集成分歧 | ✅ 易 |
| **校准（后处理）** | temperature / Platt / isotonic / conformal | 校准后分位数 | ❌ 否 | **极低** | ✅ 最好 | 低 | 校准后尾部 | ✅ **最公平** |
| **贝叶斯 / 深度集成** | MC-dropout、deep ensemble | 后验分位数 | 视方法 | 中-高 | ⚠ 中（需多模型） | 低 | 后验尾部 | ⚠ 成本高 |

**§3 五问回答**：

1. **当前项目是否真的需要完整联合分布？** **不需要**。三数据集的正式任务是逐变量状态预测（MetroPT）、单值 RUL（FD004）、分类（TEP）；联合分布不改变任何正式目标的定义。
2. **边缘概率预测是否足以支持第四章？** **是**。边缘分位数/Gaussian 已能给出逐传感器、逐 horizon 的不确定性、覆盖率与风险分数。
3. **分位数预测是否可作最小可行方案？** **可以，且 V3C 已实现**（§1 表）。
4. **低秩相关结构是否应作为后续扩展而非第一版必选？** **是**。它需要正定约束（$\Sigma=D+VV^\top$ 可能非正定）、Cholesky、以及与点预测头不同的方差语义；建议第二/三阶段。
5. **联合采样/copula/flow 是否显著增加实现风险？** **是**。diffusion 必须迭代采样（TimeDiff 明确为解决 TimeGrad 的自回归误差累积与慢推理而设计），在工业小样本上训练成本与稳定性风险最高。

---

## 4. 论文证据表

> 相关性：**A**=直接相关（参数化/校准/区间机制可迁移）｜**B**=部分相关｜**C**=背景｜**D**=弱相关。
> 证据等级：**A**=正文+公式/官方页｜**B**=正文片段/摘要｜**C**=仅标题线索。
> ⚠ **本表所有"数值/消融"栏位在本次调研中基本未核验，一律写"未核验"，禁止在论文中引用未核验数字。**

### 4.1 参数化概率预测

| 字段 | DeepAR | GPVAR / "DeepVAR" |
|---|---|---|
| 标题 | DeepAR: Probabilistic forecasting with autoregressive recurrent networks | High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes |
| 作者 | Salinas, Flunkert, Gasthaus, Januschowski | Salinas, Bohlke-Schneider, Callot, Medico, Gasthaus |
| 年份 | 2020（arXiv v1 2017） | 2019（NeurIPS 2019） |
| 会议/期刊 | Int. J. Forecasting 36(3):1181–1191 | NeurIPS 2019 |
| DOI/arXiv | DOI 10.1016/j.ijforecast.2019.07.001；arXiv:1704.04110（v3 2019-02-22，已核验） | arXiv:1910.03002 |
| 任务 | 概率预测（自回归） | 高维多变量概率预测 |
| 数据集 | 零售/能源/交通（论文声明） | 交通等 6 个真实数据集 + 人工数据 |
| 概率输出 | **样本**（自回归采样）/ 分位数 | 边缘 copula 变换 + 联合低秩 Gaussian |
| 多变量 | DeepAR 本身是**边缘独立** | ✅ 显式建模变量间协方差 |
| 多步 | ✅ 自回归多步 | ✅ |
| 需联合采样 | ✅（自回归逐步采样） | 边缘解析；联合需 Cholesky |
| 分布形式 | Gaussian（连续）/ 负二项（计数） | Gaussian copula + 低秩 $\Sigma=D+VV^\top$，$V\in\mathbb{R}^{N\times r}$ |
| 损失 | NLL（似然） | 多元 Gaussian 负对数似然 |
| 校准 | 论文用 coverage 曲线评估 | CRPS / CRPS-Sum（论文声明） |
| 风险评估 | 未核验 | 未核验 |
| 关键公式 | $Q_\Theta$ 乘性似然因子分解（正文片段）；$\rho$-risk 指标 | $\Sigma_t=L_tL_t^\top+\mathrm{diag}(d_t)$，似然 $O(Nr^2+r^3)$，实验取 $r=10$（正文核实） |
| 代码 | 无官方（Amazon）；`pytorch-forecasting` 有 DeepAR/DeepVAR 实现 | 未核验 |
| **相关性** | **A** | **A**（低秩扩展的直接参照） |
| **证据等级** | **A** | **A** |
| 主要局限 | 自回归误差累积、需迭代采样 | 需正定约束；r 超参；N 小时收益有限 |

> ⚠ **重要引用纠正**（本轮核验）：社区常说的 "DeepVAR" 指 **Salinas et al. NeurIPS 2019（arXiv:1910.03002）**；**AAAI 2020 的 "DeepVar" 是生物医学命名实体识别论文，与时序预测无关**。GPVAR 论文正文中把 DeepAR 的全秩多变量变体称为 "Vec-LSTM"。论文中引用务必写全称与 arXiv 号。

### 4.2 分位数预测

| 字段 | MQ-RNN / MQF2 | non-crossing QR (XGBoost-QR 类) |
|---|---|---|
| 标题 | A Multi-Quantile RNN / Multi-Quantile Filtered forecaster | (XGBoost: A Scalable Tree Boosting System for Regression, KDD 2016 的 quantile 模式) |
| 年份/来源 | 2018/2019（**发表状态与 ID 未核验**） | KDD 2016 |
| 概率输出 | **多分位数** | 多分位数（线性规划保证非交叉） |
| 多变量/多步 | 多变量边缘/多步（细节未核验） | 多输出 |
| 需联合采样 | ❌ | ❌ |
| 分布形式 | 直接输出分位数值 | 分位数回归 |
| 损失 | pinball（quantile loss） | pinball |
| 校准 | 未核验 | 未核验 |
| 关键公式 | pinball（与 V3C 现有 `pinball_loss` 同源） | `min Σ_t Σ_j w_j (y_t - f_j(x_t))_τ`，非交叉由 LP 保证 |
| **相关性** | **A**（V3C 现有头的直接来源） | **A**（非交叉保证的另一种实现） |
| **证据等级** | **C**（**本轮未能核验其发表信息**，谨慎引用） | **B** |
| 主要局限 | 分位数独立，不建模变量间依赖 | 树模型，非端到端 | 

### 4.3 预测区间与 conformal

| 字段 | Conformal Time-series Forecasting (CF-RNN) | CQR |
|---|---|---|
| 标题 | Conformal Time-series Forecasting | Conformalized Quantile Regression |
| 作者 | Stankevičiūtė, Alaa, van der Schaar | Romano, Patterson, Candès |
| 年份/来源 | **NeurIPS 2021**（已核验，poster 页） | NeurIPS 2019 |
| DOI/arXiv | NeurIPS hash 312f1ba2a72318edaaa995a67835fad5 | arXiv:1905.03222（已核验标题，19 页 8 图 1 表） |
| 概率输出 | **多步预测区间**（逐 horizon 覆盖保证） | 自适应宽度区间 |
| 多变量 | **论文自陈 d=1，扩展到多变量留作 future work** | 逐变量 |
| 多步 | ✅ 直接多步 + 逐 horizon 区间 | ✅ |
| 需联合采样 | ❌（ICP 后处理） | ❌ |
| 分布形式 | **分布无关**（仅需可交换性） | 分布无关（在分位数回归之上） |
| 损失 | 底层点预测损失 + conformal 非一致性分数 | 分位数损失 + conformal 校正 |
| 校准 | **理论保证**：$P(\forall h, y_{t+h}\in[\hat y_{t+h}-\hat\epsilon_h,\hat y_{t+h}+\hat\epsilon_h])\ge1-\alpha$，Bonferroni 逐 horizon 修正 $\alpha/H$（正文核实） | 类似有限样本覆盖保证 |
| 风险评估 | 区间外事件可作风险信号 | 同 |
| 代码 | ✅ github.com/vanderschaarlab/conformal-rnn（已核验） | 官方实现存在（**未核验**） |
| **相关性** | **A**（最适合作为 V3C 的独立校准层） | **A** |
| **证据等级** | **A** | **B** |
| 主要局限 | 逐 horizon 独立修正，未处理 horizon 间联合；区间可能过宽 | 需分位数回归；d=1 |

### 4.4 生成式（diffusion / flow）

| 字段 | TimeDiff | TSDiff | GP/其他 diffusion |
|---|---|---|---|
| 标题 | Non-autoregressive Conditional Diffusion Models for Time Series Prediction | Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting | (Rasul et al. TimeGrad 等) |
| 作者 | Shen, Kwok | Kollovieh, Ansari, Bohlke-Schneider, Zschiegner, H. Wang, Y. Wang（AWS AI Labs / TUM） | — |
| 年份/来源 | **ICML 2023**（PMLR v202，pp.31016–31029，已核验） | arXiv:2307.11494（NeurIPS 2023 workshop 系，**主会状态未核验**） | — |
| 概率输出 | **样本** | **样本** | 样本 |
| 多变量/多步 | ✅ | ✅ | ✅ |
| 需联合采样 | ✅ **必须** | ✅ 必须 | ✅ 必须 |
| 关键机制 | 非自回归 + future mixup（式14）+ 自回归初始化（式16）；直接预测数据而非噪声 | 无条件训练 + 推理期 self-guidance；refine 用隐式密度作先验 | TimeGrad 为自回归扩散 |
| 校准 | 未核验 | 未核验 | — |
| 关键观察 | 正文明确指出 CSDI 因 transformer 在长多变量序列上**易 OOM**，并批评 TimeGrad 的**误差累积与慢推理** | refine 在数据空间迭代，**计算开销低于完整反向扩散** | — |
| **相关性** | **D**（不推荐第一阶段） | **D** | **D** |
| **证据等级** | **A** | **B** | **C** |
| 代码 | 官方（**未核验 URL**） | ✅ github.com/amazon-science/unconditional-time-series-diffusion（正文声明） | — |

### 4.5 结构化/低秩与 copula（补充）

- **GPVAR**（见 4.1）：低秩 Gaussian copula 的标准参照，$O(Nr^2+r^3)$，$r=10$。
- **Deep generative quantile-copula**（Wen & Torkkola, arXiv:1907.10697，GPVAR 论文引用，**未核验正文**）：非参数 copula，GPVAR 指出其需 $\varepsilon^{-N}$ 分块、难扩展。
- **Multivariate Probabilistic TS with error autocorrelation**（arXiv:2402.01000，**发表状态未核验**）：低秩+对角协方差建模误差自相关，正文称复杂度不随序列数增长、可用于校准。

### 4.6 校准与风险评估方法（多为工具性文献）

| 方法 | 用途 | 是否需概率标签 | 阈值来源 | 指标 | 证据等级 |
|---|---|---|---|---|---|
| Temperature scaling | 软化过度自信 | 需（valid 集） | validation-only | ECE、NLL | B（通用技术） |
| Platt scaling | 二分类/二值化概率校准 | 需 | validation-only | Brier、ECE | B |
| Isotonic regression | 单调校准 | 需 | validation-only | Brier、ECE | B |
| Conformal（ICP） | 区间覆盖保证 | ❌ | calibration split | coverage、interval width | **A**（Stankevičiūtė） |
| Conformal risk control (CRC) | 风险指标控制 | 需 | calibration split | 风险上界 | C（未核验） |
| Energy score / CRPS | 分布型评分 | ❌ | — | CRPS（越小越好） | B |
| Mahalanobis | 多元异常 | ❌ | 需协方差 | 距离 | B |

> ⚠ **关键区分提醒**：**marginal coverage（逐点边缘覆盖）≠ joint coverage（联合覆盖）≠ simultaneous coverage（同时覆盖）≠ conditional coverage（条件覆盖）**。Stankevičiūtė 的 Bonferroni 修正给出的是**逐 horizon 的边缘覆盖**，其式(9)的 $\forall h$ 是逐 horizon 同时成立（在 Bonferroni 修正下），但**不是联合分布覆盖**，也不是 conditional coverage。报告中不得混用。

---

## 5. 代码仓库表

> **核验方式**：由并行子代理用 `gh` API + `raw.githubusercontent` **逐仓库下载源码文件并阅读**；标 ✅ 者已打开源码，未打开者一律降级 ⚠/❌。核验日期 2026-10-10。
> **注意**：GluonTS 默认分支是 **`dev`**，不是 `master`/`main`（早期按 master/main 访问会 404）。

### 5.1 已核验仓库（✅ 已打开源码）

| 方法 | 仓库 | 官方性 | 许可证 | 对应论文 | 分布形式 | 采样 | 校准 | 关键文件（已读） |
|---|---|---|---|---|---|---|---|---|
| **DeepAR** | `awslabs/gluonts` | ✅ 官方（Amazon/作者团队） | Apache-2.0 | Salinas et al., IJF 2020, arXiv:1704.04110 | `DistributionOutput` 可插拔，**默认 `StudentTOutput()`**，可选 Gaussian / NegativeBinomial | ✅ 逐时刻 AR，`num_parallel_samples` 默认 100 | ❌ 无 conformal/Platt/isotonic | `torch/model/deepar/module.py`、`estimator.py`、`lightning_module.py`、`mx/block/quantile_output.py`、`ev/metrics.py` |
| **DeepVAR** | `awslabs/gluonts`(mx) + `zalandoresearch/pytorch-ts` | ✅ 官方 | Apache-2.0 / MIT | arXiv:1910.03002（论文中称 **Vec-LSTM**） | **`LowRankMultivariateNormalOutput(dim, rank)`**，rank 默认 5；可选边缘 CDF 变换 | ✅ **需联合采样**，`sampling_decoder()` 一次抽全维 | ❌ | `gluonts/mx/model/deepvar/_network.py`、`pytorch-ts: pts/model/deepvar/deepvar_network.py` |
| **MQF2** | `awslabs/gluonts` | ✅ 官方收录（非原作者仓库） | Apache-2.0 | Zeng et al. 2019；实为 Kan/Aubet/Januschowski 的 multi-horizon 变体 | **凸网络分位数函数**（`DeepConvexNet + ActNorm`），**天然单调不交叉** | ✅ `rsample()` ICNN 变换 | ❌ | `torch/model/mqf2/{module,distribution,_cpflows,estimator}.py` |
| **TimeGrad** | `zalandoresearch/pytorch-ts` | ✅ 官方（论文正文给出地址） | MIT | Rasul et al., ICML 2021, arXiv:2101.12072 | 条件扩散 / AR EBM（Langevin） | ✅ **昂贵**，逐步 `self.diffusion.sample()` | ❌ | `pts/model/time_grad/{time_grad_network,epsilon_theta,time_grad_estimator}.py` |
| **TSDiff** | `amazon-science/unconditional-time-series-diffusion` | ✅ 官方（Amazon Science） | Apache-2.0 | Kollovieh et al., arXiv:2307.11494 | 无条件扩散 p(y) | ✅ 逐扩散步 | ⚠ 有 inference-time self-guidance（**非校准**） | `src/uncond_ts_diff/model/diffusion/tsdiff.py`、`sampler/observation_guidance.py`、`utils.py` |
| **CQR** | `yromano/cqr` | ✅ 官方（Romano 本人） | MIT（LICENSE 原文；API licenseInfo 为 null） | Romano/Patterson/Candès, NeurIPS 2019, arXiv:1905.03222 | quantile（无参数化分布） | ❌ | ✅ **split conformal**：`run_icp()` fit→calibrate→predict | `cqr/helper.py`（590 行）、`cqr/torch_models.py`、`nonconformist/` |
| **EnCQR**（时序 conformal） | `FilippoMB/Ensemble-Conformalized-Quantile-Regression` | ✅ 官方（Bianchi 本人） | MIT（Copyright 2022 Filippo Bianchi） | Jensen/Bianchi/Anfinsen, IEEE TNNLS, arXiv:2202.08756 | quantile（base model 出 `[α/2, 0.5, 1-α/2]`） | ❌ | ✅ conformal + 在线滚动更新 | `conformal_prediction.py`（86 行）、`data_preprocessing.py`、`main_EnCQR.py` |
| **XGBoost QR** | `dmlc/xgboost` | ✅ 官方 | Apache-2.0 | XGBoost KDD 2016，`objective="reg:quantileerror"` | quantile（`QuantileDMatrix`，一次训练多 alpha） | ❌ | ❌ 库内无 conformal | `src/objective/quantile_obj.{h,cc,cu}`、`src/common/quantile.cc`、`doc/parameter.rst` |
| **TACTiS-2** | `ServiceNow/TACTiS` | ✅ 官方（ServiceNow Research） | Apache-2.0 | Ashok et al., ICLR 2024, arXiv:2310.01327 | **边缘 Deep Sigmoid Flow + attention copula**（参数量对变量数**线性**增长） | ✅ 联合采样 | ❌ | `tactis/model/{marginal,tactis,flow}.py` |
| **UQNO** | `neuraloperator/neuraloperator` | ✅ 官方 | MIT | Li et al., Neural Operator | **主解 + 分位数双头** | ❌ | ❌ | `neuralop/models/uqno.py`、`layers/gno_block.py` |
| **ProbTS** | `microsoft/ProbTS` | ✅ 官方（Microsoft） | ⚠ 未读 LICENSE | 统一 benchmark 工具包 | 集成 TimeGrad/CSDI/TSDiff | 视模型 | — | `probts/model/forecaster/prob_forecaster/*.py` |

### 5.2 官方仓库未找到（❌ 不臆造 URL）

| 方法 | 检索结果 |
|---|---|
| **MQ-RNN** | 三次 GitHub 搜索仅返回 `roi-meir/MQ_RNN`（2★、个人复现、无 license）与 `ericli1123/MQ-RNN`。**官方仓库不存在或未公开** |
| **DiffTime** | GitHub 搜 "DiffTime" total=10 全为同名无关项目（LiDAR、C++ `difftime()`）。**⚠ DiffTime ≠ TimeDiff**（后者 = arXiv:2306.05043, ICML 2023） |
| **TimeWeaver** | `AEElastic/TimeWeaver` 无法解析，用户 404，搜索 total=0。**⚠ 与 ICML 2024 的 "Time Weaver"（Narasimhan 等，arXiv:2403.02682，元数据条件扩散**生成**模型）完全无关** |
| **TimeGrad++** | GitHub 搜 "TimeGrad++" total=4 全不相关；websearch 两轮无命中。**无法定位同名论文** |
| **SACR** | GitHub 搜索 total=0；arXiv API 429。**无法确认该方法存在可核验实现** |
| **TactPFN** | GitHub 搜索 total=0。已改用最接近的 **TACTiS-2** 替代核验（见 5.1） |

### 5.3 ⚠ 两个必须警惕的数据泄漏点（子代理逐行阅读后发现）

1. **EnCQR 仓库不可直接照搬**：`conformal_prediction.py::EnCQR` 用 **train 段标签**计算 conformal 非一致性分数（非独立 calibration 集），且循环末尾 `utils.asym_nonconformity(label=test_y[i,:], ...)` **用 test 标签在线更新 epsilon**。这在严格前向链式在线 conformal 下合法，但按"一次性离线回测"解读时 **coverage 会被历史 test 标签污染**——与本仓库 AGENTS.md 硬规则（风险阈值只能用 validation split，不能用 test label 搜索）**直接冲突**。

2. **TSDiff 的缩放不是 train-split 归一化**：`utils.ScaleAndAddMeanFeature` 用 `mean(abs(target[..., :-prediction_length]))`——窗口内排除最后 `prediction_length` 步求尺度，属 **per-window / per-series 缩放**，不满足本仓库"归一化统计量只能来自 train split"的护栏。

3. ✅ **安全的对照**：GluonTS / pytorch-ts / TACTiS 的 scaler **均只取 context / 历史段**（GluonTS: `self.scaler(context, observed_context)`；pytorch-ts: `past_target_cdf[:, -context_length:]`；TACTiS: `NormalizationIdentity(hist_value)`），且划分走 `ValidationSplitSampler`/`TestSplitSampler`（顺序切分不打乱）。**这三家的归一化与划分设计可直接借鉴。**

### 5.4 可借鉴内容汇总（只提取结构，不复制代码）

| 来源 | 借鉴点 | 对本项目价值 |
|---|---|---|
| GluonTS `DistributionOutput` | 分布与主干**完全解耦**，换分布不改模型；scaler 作为独立可审计对象 | ⭐⭐⭐⭐⭐ 直接支撑"参数化 vs 分位数"公平对照 |
| GluonTS `crps_weights_pwl()` | **加权 pinball loss 可折算为 CRPS** | ⭐⭐⭐⭐⭐ **关键**：让 V3C 现有分位数头**无需采样即可算 CRPS** |
| GluonTS `ev/metrics.py` | 30+ 指标（Coverage、SumQuantileLoss、MSIS、MASE…）定义 | ⭐⭐⭐⭐ 可直接对齐指标口径 |
| pytorch-ts TimeGrad | `unroll_encoder()` 使训练 O(L) 与 AR 采样 O(L·T_diff) **解耦**；`loss_weights = observed_values.min(dim=-1)` 对缺失鲁棒 | ⭐⭐⭐ 与本项目 masked-input 契约高度契合 |
| CQR `run_icp()` | **proper-train / calibration / test 三段显式传入**，`RegressorAdapter` 解耦 conformal 与回归器 | ⭐⭐⭐⭐⭐ 校准层直接范式 |
| CQR `use_rearrangement` | 99 个分位点 + Chernozhukov 保序（**非交叉的经典做法**） | ⭐⭐⭐ 与 V3C 的 cumsum 保序可作对照 |
| MQF2 `DeepConvexNet` | 凸性约束 → 分位数**天然单调不交叉** | ⭐⭐⭐ 另一种非交叉实现 |
| TACTiS 边缘流+copula | 边缘用 Deep Sigmoid Flow 求条件 CDF，依赖交给 attention copula；**参数量对变量数线性增长** | ⭐⭐⭐ 若日后做联合，这是高维友好的两段式 |
| UQNO 双头 | 主任务 + 分位数任务共享 encoder、独立 head | ⭐⭐⭐ 干净的接口范式 |
| XGBoost `QuantileDMatrix` | 一次训练多 alpha，成本几乎不变；树模型不过拟合 | ⭐⭐ 作为 conformal 的 base learner / 低风险 baseline |

---

## 6. 与 Unified V3C 的差异

| 能力 | Unified V3C 当前状态（代码核实） | 候选概率方法 | 需要新增代码 | 影响范围 |
|---|---|---|---|---|
| 历史编码器 | ✅ `UnifiedTemporalEncoder`（逐传感器 + 多尺度 patch） | 全部复用 | 无 | 无 |
| 多变量表示 | ✅ `z_history [B,C,H]` + learned sensor pooling | 复用 | 无 | 无 |
| 多步状态输出（MetroPT） | ⚠ **已有分位数头**（`StateHead`，`quantiles=(0.1,0.5,0.9)`，非交叉）+ 点预测 | 分位数（已有）/ Gaussian / Student-t / 低秩 | Gaussian/Student-t 头（新文件）；低秩需 $\mu,\log\sigma^2,V,d$ 四个输出 | 编码器不动，仅换头 |
| RUL 输出（FD004） | ✅ `RULHead`（`Linear(input_dim,1)` + 分位数 increments，`input_mode="global"`，含 `training_cap`/`rul_over_penalty`） | RUL 分布/区间 | 分位数已有；Gaussian 头新写 | **raw RUL 语义与 cap 隔离必须保持** |
| TEP 分类输出 | ✅ `TEPHead`（detection + diagnosis 双头，`mode="probability"`） | 概率校准 | 校准层（新文件） | 按 run 隔离、事件级指标 |
| 传感器相关性 | 现有**确定性**融合（`cross_channel` 单头 MHA + `sensor_scorer`） | 低秩协方差（可选） | 低秩头（新文件）+ 正定约束 | 第二/三阶段扩展 |
| 不确定性 | ⚠ **当前不足**（无 CRPS/NLL/覆盖率；仅有分位数输出形式） | 分位数/Gaussian/区间 | 指标模块（全新） | `experiments/metrics.py` |
| 风险分数 | ❌ 需扩展 | 由预测分布导出（尾部概率/区间超限/标准化残差/conformal score） | 风险模块（全新） | 阈值只由 validation 拟合 |
| 校准 | ❌ 需要独立实现 | conformal / Platt / isotonic（validation-only） | 校准层（全新） | **严禁 test 调参** |

**直接可接在 V3C 后面的**：分位数头（**已接**）、Gaussian/Student-t 头（仅需替换 `StateHead` 的 `base/increments` 为 $\mu,\sigma$）、conformal 校准层（**完全后处理，不改模型**）。
**需改编码器的**：无（除非做低秩/flow/diffusion 的条件生成）。
**只需新增 decoder 的**：Gaussian/Student-t/低秩头。
**需新增采样器的**：diffusion、flow（**第一版不建议**）。
**会破坏训练速度的**：diffusion（迭代采样）、flow（大似然训练 + 雅可比）。
**最适合最小可行实现**：**conformal 校准层 + 分位数头（已有）+ 概率指标补全**。

---

## 7. 推荐实现路线

| 方案 | 实现复杂度 | 训练稳定性 | 推理速度 | 显存 | 多变量 | 多步 | 风险评估 | 校准能力 | 适合本项目 | 推荐作为第一阶段 |
|---|---|---|---|---|---|---|---|---|---|---|
| **A. 分位数头** | **极低（已实现）** | 高（cumsum 保序） | 快 | 低 | ✅ 逐变量 | ✅ | 分位数距离、区间 | 需外部校准 | ⭐⭐⭐⭐⭐ | ✅ **首选（已存在，只需补指标）** |
| **B. Gaussian/Student-t 头** | 低（2 个 Linear） | 中高（需 $\sigma>0$） | 快 | 低 | ✅ 逐变量（边缘） | ✅ | 尾部概率、标准化残差 | NLL 天然 | ⭐⭐⭐⭐⭐ | ✅ **必须作为 A 的对照** |
| **C. 低秩 Gaussian 头** | 中（$\mu,\sigma,V,d$ + 正定约束） | 中（易非正定/发散） | 中（Cholesky） | 中 | ✅ 联合 | ✅ | Mahalanobis、联合尾部 | 强 | ⭐⭐⭐ | ⛔ 第二阶段（可选扩展） |
| **D. Flow 头** | 高（似然 + 雅可比） | 低-中（小样本不稳） | 慢（采样） | 高 | 需联合构造 | ✅ | 密度比 | 中 | ⭐⭐ | ❌ 不建议第一阶段 |
| **E. Diffusion 头** | **很高** | 中 | **很慢（迭代）** | **高** | ✅ | ✅ | 样本统计 | 弱 | ⭐ | ❌ **不建议**（工业小样本 + 成本） |
| **F. Conformal 区间层** | **极低（后处理）** | **无（不改模型）** | 极快 | 极低 | 逐变量 | ✅ 逐 horizon | 区间外、nonconformity score | **理论保证** | ⭐⭐⭐⭐⭐ | ✅ **强烈推荐与 A/B 并列** |

**推荐组合（阶段化）**：
- **阶段 1（最小可行）**：**A（已有）+ F（新增）+ 概率指标补全**。产出：逐传感器/逐 horizon 分位数 + conformal 区间 + PICP/MPIW/CRPS。
- **阶段 2**：**B（Gaussian/Student-t）作为 A 的对照**，产出 NLL 与解析分位数对照。
- **阶段 3（可选）**：**C（低秩）**，仅当需要联合分布/Mahalanobis 风险时；须证明优于 A/B。
- **D/E 明确不做**（第一版）。

---

## 8. 实验矩阵

### 8.1 点预测 vs 概率预测
| ID | 头 | 分布 | 备注 |
|---|---|---|---|
| P0 | `StateHead` point（`mode="point"`） | — | 现有基线 |
| P1 | `StateHead` quantile（**现有**，`mode="quantile"`） | 分位数 | 默认 (0.1,0.5,0.9)，需扩展到 {0.05,0.1,0.25,0.5,0.75,0.9,0.95} |
| P2 | Gaussian 头 | $\mathcal{N}(\mu,\sigma^2)$ | `softplus` 保 $\sigma>0$ |
| P3 | Student-t 头 | $t(\mu,\sigma,\nu)$ | $\nu$ 可学习或固定 |
| P4（可选） | 低秩 Gaussian | $\Sigma=D+VV^\top$ | 阶段 3 |

### 8.2 条件输入消融
历史状态 / +未来查询时间 / +目标通道身份 / +工况上下文 / 全部 / **打乱查询时间** / **打乱目标通道** / **打乱工况上下文**（后三项为负对照）。

### 8.3 区间预测
直接分位数区间 / 参数分布分位数 / 未校准 / **validation-only conformal 校准** / 多覆盖水平 {80,90,95}% / 逐 horizon / 逐传感器。

### 8.4 风险评估
点预测残差 / 标准化残差 / NLL / 区间超限 / 尾部概率 / conformal nonconformity score / **validation-only 阈值** / 校准后风险分数。

### 8.5 复杂度（必须报告）
参数量、训练时间、单批推理时间、峰值显存、采样时间（若适用）、checkpoint 大小。

### 8.6 多 seed
单 seed 仅方向筛选；正式结论 ≥3 seeds（2026/2027/2028）。

### 8.7 数据集分工（严格遵守任务书 §12）
- **MetroPT-3**：多步概率状态预测。报告 MAE/RMSE、**NLL、CRPS、PICP、MPIW、interval score**、horizon-wise coverage、sensor-wise coverage、false alarm rate、detection delay。
- **FD004**：**单值 RUL 概率预测**。报告 raw MAE/RMSE、**NASA score**、RUL NLL/CRPS、coverage、interval width、per-engine error。**不得虚构多步 RUL；不得用 fault number；不得混淆 training cap 与 raw test target；阈值只用 validation。**
- **TEP**：检测/诊断概率。报告 AUROC/AUPRC/F1、**log loss、Brier score、ECE、reliability diagram**、false alarm rate、event recall、detection delay。**按 run 隔离；validation-only calibration；不得用故障标签做推理条件。**
- **通用数据集（建议补充）**：Electricity/Traffic/Weather 之一，否则只能声称"三类工业任务验证"。

---

## 9. 创新性与审稿风险

### 9.1 成熟组件（不可声称为创新）
- 边缘分位数回归 + pinball（V3C 已有）
- Gaussian/负二项参数化 + NLL（DeepAR，2020）
- 非交叉约束（cumsum / LP）
- 低秩 Gaussian copula（GPVAR，2019）
- conformal 区间 + 有限样本覆盖保证（CQR 2019；Stankevičiūtė 2021）
- temperature/Platt/isotonic 后处理校准
- CRPS/energy score、PICP/MPIW、Brier/ECE

### 9.2 可能形成贡献的组合（须实验支撑）
1. **工业多传感器分位数/参数化预测 + conformal 校准的端到端协议**（含逐传感器/逐 horizon 覆盖率、风险分数、事件级指标）——定位为**工程与协议贡献**，非算法创新。
2. **校准对工况/故障阶段的异质性分析**（校准误差随工况段变化）——可能是有价值的观察。
3. **点/参数/分位数头在三类工业任务上的系统对照**——属实证贡献。

### 9.3 不可使用的表述
- ❌ ~~"提出新的概率预测方法"~~（各组件均成熟）
- ❌ ~~"首次将概率预测用于本工业场景"~~（工业 RUL/异常概率文献众多，本轮未穷尽）
- ❌ ~~"输出区间即方法创新"~~（任务书明确禁止）
- ❌ ~~"多个成熟模块组合即全新模型"~~（任务书明确禁止）
- ❌ ~~"边际覆盖率=联合覆盖率"~~、~~"预测区间=置信区间"~~（§4.6 提醒）
- ❌ ~~"风险分数=故障概率"~~

### 9.4 是否需要通用数据集
**需要**（若要声称普适性）。任务书 §7 明确：无通用数据集则只能写"在多类工业任务上的验证"。

### 9.5 是否应放弃"联合概率建模"的过强标题
**建议放弃**作为第一版主线。标题/摘要建议聚焦"**校准的不确定性量化与风险评估**"，把低秩联合结构作为可选扩展。若一定要写联合，必须同时给出 C（低秩）对照与正定性论证。

### 9.6 不可省略的实验
1. **点 vs 分位数 vs Gaussian 的点精度对照**（概率头不应损害点精度）；
2. **未校准 vs conformal 校准的覆盖率对照**（否则校准层无意义）；
3. **逐 horizon 覆盖率与区间宽度**（不确定性随预测步变化是核心科学问题）；
4. **负对照：打乱工况/查询时间的风险分数应退化**；
5. **多 seed**；
6. **FD004 的 raw RUL 语义隔离测试**（cap vs raw）。

---

## 10. 最终建议（十问）

1. **第一版实现哪一种概率预测头？**
   **不新增头**——V3C 的分位数头已实现且天然非交叉（`heads/state.py`）。第一版只需 (a) 扩展分位数网格、(b) 补全概率指标、(c) 新增 conformal 校准层。**必须同时实现 Gaussian 头作为对照**（阶段 2，可并行）。

2. **是否必须实现联合采样？**
   **不必须**。分位数/Gaussian/低秩 Gaussian 的 NLL 与 CRPS 均可解析计算（低秩需 Cholesky）。采样仅在需要联合样本/生成式时引入。

3. **是否需要低秩相关结构？**
   **第一版不需要**。作为阶段 3 可选扩展，理由：正定约束与数值发散风险、与点预测头语义不一致、N=15–40 时收益有限。

4. **是否需要 flow 或 diffusion？**
   **不需要**。理由：diffusion 必须迭代采样（TimeDiff 正文批评 TimeGrad 的误差累积与慢推理；CSDI 在长多变量序列上 OOM）；flow 似然训练在小样本上不稳。二者均破坏训练速度，不适合作为硕士论文第四章的第一版。

5. **是否应先实现分位数和 Gaussian 对照？**
   **是**。这是阶段 1–2 的核心，且两者都极低成本。Gaussian 提供 NLL（解析）与参数化尾部概率，与分位数（无参数）形成互补对照。

6. **风险校准应如何独立验证？**
   独立于点预测单独验证：(a) 覆盖率对照（未校准 vs conformal vs temperature）；(b) ECE/reliability diagram；(c) 负对照（打乱条件的风险分数应退化）；(d) 事件级评估（TEP 的检测延迟/误报率；MetroPT 的 false alarm）。**阈值只由 validation 拟合**。

7. **三个工业数据集如何分工？**
   MetroPT-3 = 多步概率状态 + 逐传感器/逐 horizon 区间 + 风险分数；FD004 = **单值 RUL** 分布/区间 + NASA score（**不虚构多步**）；TEP = 检测/诊断概率 + 校准 + 事件级指标（**不泄漏故障标签**）。

8. **是否增加 Electricity/Traffic/Weather？**
   **建议增加**（三者之一即可），否则论文只能声称"三类工业任务验证"。

9. **下一步代码实现顺序？**
   1. `experiments/metrics.py` 新增：CRPS（分位数与参数两种）、NLL、PICP、MPIW、interval score（逐 horizon/逐传感器）；
   2. 单元测试：分位数单调性、区间嵌套性（80% ⊂ 90% ⊂ 95%）、指标边界；
   3. 新增 Gaussian 头（`heads/state_gauss.py`），测试形状/有限性/$\sigma>0$；
   4. 新增 conformal 校准层（**只用 validation**，禁止 test）；测试校准集与训练集无重叠；
   5. 跑 P0/P1 单 seed 对照 → 3 seeds；
   6. 补 conformal 对照与负对照；
   7. 再考虑 FD004/TEP 扩展与通用数据集。

10. **未核验的论文、代码和实验结论**
    - **MQ-RNN 的发表信息与官方仓库**：官方仓库**确认不存在/未公开**（GitHub 三次搜索仅得个人复现）；其论文的准确引用**仍未核验**（本轮检索反复失败，标 C 级，引用前必须重新核验）。
    - **TimeGrad++ 与 SACR**：GitHub 搜索 total=0/4 全不相关，arXiv API 429。**无法确认是否存在同名可核验论文与实现**，需提供 DOI 后再核。
    - **DiffTime / TimeWeaver**：论文本体已核实（前者 NeurIPS 2023 Coletta et al.；后者 ICML 2024 Narasimhan et al. 的**生成**模型），但**官方仓库均未找到**。
    - **GPVAR 的"Vec-LSTM"基线数值、DeepAR 论文声明的 ~15% 提升、各论文消融表**——均未核验，禁止引用其数值。
    - **GluonTS DeepVAR 只有 mx 版本，torch 版不存在**（mx 已被 GluonTS 逐步弃用）——若要用需评估迁移成本（子代理核验所得，工程信号）。
    - **各概率方法在 MetroPT-3/FD004/TEP 上的适配性与消融结果完全未核验**（本轮无任何实验）。
    - **normalizing flow 概率预测的具体代表作未检索核验**（§3 该行证据等级 B；仅 MQF2 经代码核验使用凸网络流）。
    - **energy score / CRPS 的原始文献（Gneiting & Raftery）未核验**；已核验的可用替代是 GluonTS `MQF2Distribution.energy_score()` 与 `crps_weights_pwl()`。
    - **TACTiS-2 的 copula 参数量线性增长结论**引自子代理阅读 `marginal.py`，**未在论文正文核验**。
    - **所有 arXiv ID 必须在使用前重新验证**——本轮发现我方对 DeepAR/DeepVAR/MQ-RNN/TimeGrad/DiffTime/TSDiff/conformal 等 ID 的记忆大面积错误（详见 §10.11），这是本轮最重要的方法学教训。

> **§10.11 ID 纠错记录（如实记录，防止后续复用错误引用）**
> 本轮最初记忆中的 ID **全部错误**，实测：
> - DeepAR：记忆 1912.01621（实为天文学论文）→ **正确 1704.04110**
> - DeepVAR/GPVAR：记忆 1904.10340（实为手势相似性音乐学论文）→ **正确 1910.03002**（且非 AAAI 2020）
> - MQ-RNN：记忆 1905.07879（实为低质量天体候选双星）→ **未核验**
> - MQF2：记忆 1909.12169（实为 Majorana 三结物理）→ **未核验**
> - TimeGrad：记忆 2107.03516（实为引力波 ITIC 论文）→ **未核验**
> - DiffTime：记忆 2303.05844（实为类星体质量分布）→ **实为 TimeDiff = 2306.05043**
> - TSDiff：记忆 2305.08758（实为钙钛矿相变）→ **正确 2307.11494**
> - Conformal Time-series：记忆 2102.06924（实为 Apprenticeship Learning）→ **正确为 NeurIPS 2021, hash 312f1ba2a72318edaaa995a67835fad5**
> - TimeFilter（上一轮任务）：记忆 2406.07933（实为 LLM unlearning）→ **正确 2501.13041**
> **结论：任何 arXiv ID 在写入论文前必须通过 arxiv.org/abs 页面或官方 proceedings 页面核验，不得凭记忆使用。**

> **§10.12 代码核验追加的同名混淆对（子代理证实）**
> - **DiffTime ≠ TimeDiff**：前者 = Coletta et al., NeurIPS 2023, 基于 CSDI 的条件扩散；后者 = Shen & Kwok, ICML 2023, arXiv:2306.05043, 非自回归 + future mixup。名字极近但完全不同。
> - **TimeWeaver ≠ Time Weaver**：前者（A. Elastic 系，概率预测，**官方仓库不存在**）与后者（Narasimhan, Agarwal, Akcin, Sanghavi, Chinchali, UT Austin, ICML 2024, arXiv:2403.02682，元数据条件扩散**生成**模型）**完全无关**，后者代码在 `vita-epfl/TimeWeaver-CSDI`。
> - **DeepVAR 的引用陷阱**：GluonTS docstring 明写该模型 "described as **VEC-LSTM** in this paper: arXiv:1910.03002"——即社区口中的 "DeepVAR" 在 GPVAR 论文中叫 Vec-LSTM，而 AAAI 2020 的 "DeepVar" 是生物医学 NER 论文。