# 预测目标条件化依赖选择与稳定融合：文献与代码调研报告

- 调研日期：2026-10-09
- 研究假设：在多变量时序预测中，固定全变量融合可能把无关或冲突信息传播到所有预测目标，造成负迁移。不同预测目标可能依赖不同的源变量、历史片段和时间尺度。可构建"预测目标条件化的依赖选择与稳定融合"模块，根据当前预测目标动态选择跨变量历史信息，并通过自身历史回退分支，在跨变量信息无效时保持稳定预测。
- 结论性质：**纯文献与代码调研**。未运行任何实验，未修改 `code/`、模型或 `result/`。

---

## 1. 执行摘要

**研究问题**：能否让"选择哪些源变量/历史信息"成为**预测目标的函数**，并在跨变量信息无效时自动回退到目标自身历史？

**最相关方法族**（五类，按与假设的贴合度排序）：

| 族 | 代表 | 贴合点 | 缺口 |
|---|---|---|---|
| 门控跨变量融合 | **Gateformer**（ICML Workshop 2025）、**RWKV-CVM**（MDPI 2026） | 自身历史 + 跨变量双分支门控融合 | **不以预测目标为条件**；稠密无稀疏 |
| 变量/时间感知选择 | **TimePro**（ICML 2025）、**LIFT**（ICLR 2024） | 逐变量、逐时间点的自适应选择 | 无自身历史回退；LIFT 非参数估计 |
| 稀疏依赖路由 | **MS-FLOW**（arXiv 2026）、**TimeFilter**（ICML 2025）、**CCM**（NeurIPS 2024） | 过滤无关依赖、抑制伪相关 | 非目标条件化；无回退分支 |
| 跨变量一致性正则 | **CvLoss**（NeurIPS 2026） | 跨变量残差一致性约束 | 损失层面，非模块 |
| 目标/上下文条件化 | **TimeXer**（NeurIPS 2024） | 外生变量作为查询条件 | 条件不是"预测目标"，无回退 |

**直接重合方法**：**无**（在本次检索范围内未发现）。但**部件级直接重合明确存在**——Gateformer 已在代码层实现"自身历史分支 + 跨变量分支 + 门控融合"。

**当前方向的合理创新边界**：**不能主张**"自身历史回退"、"门控融合跨变量信息"、"过滤无关变量依赖"（三者分别被 Gateformer、Gateformer、MS-FLOW/TimeFilter/CCM 占据）。**唯一未被占据的接缝**是：**路由权重由"预测目标查询 q"生成，而非由输入自身（Gateformer）或全局统计（CCM）生成**——即真正的 target-conditioned routing。

**是否值得继续实现**：**有条件值得**。价值不在"门控+回退"本身（已被占），而在于把它重构成"目标条件化"并用受控实验证明"按目标改变源变量选择"比"输入自适应融合"有增量。证据等级：**B**（Gateformer 为 Workshop 论文；MS-FLOW 发表状态未证实；其余为 A 级顶会）。

---

## 2. 检索范围与检索式

### 2.1 数据库与检索日期
- **arXiv**（`export.arxiv.org/api/query` 逐篇核验 ID/标题/作者/发表注释）
- **OpenReview**、**ICLR/ICML/NeurIPS/AAAI 官方 proceedings**
- **Crossref API**（正式 DOI 元数据）
- **GitHub API**（仓库存在性、目录结构、文件内容）
- **Jina Reader**（`r.jina.ai`，agent-reach web 后端）读取正文
- **检索日期：2026-10-09**

### 2.2 时间范围与纳入标准
- **时间范围**：2020–2026；保留必要早期奠基工作。
- **纳入标准**：正文含可定位的公式/结构图/算法/代码文件，能证明"变量选择/路由/门控/条件化"的具体机制。
- **排除标准**：仅用 attention 做可视化解释而无机制描述；仅检索摘要可得的论文（降级为 C 级且不支撑强结论）；第三方博客转述。
- **代码核验标准**：README 与论文一致 + 含训练/评估代码 + 可定位路由实现文件。

### 2.3 实际执行的检索式（节选，均已执行）
- 领域1：`target-conditioned variable selection time series forecasting`、`horizon-aware variable selection forecasting`、`variable-specific attention time series forecasting`、`target-aware cross-variable attention`、`selective dependency modeling multivariate forecasting`、`dynamic variable routing time series forecasting`、`task-conditioned attention multivariate time series`
- 领域2：`dynamic dependency graph multivariate time series forecasting`、`input-dependent graph forecasting`、`selective cross-channel attention`、`adaptive sensor dependency forecasting`、`sparse graph routing time series`、`negative transfer cross-variable forecasting`
- 领域3：`gated cross-variable fusion`、`selective residual fusion time series`、`adaptive skip connection multivariate forecasting`、`own-history fallback multivariate forecasting`、`negative transfer mitigation multivariate time series`、`robust variable interaction forecasting`
- 领域4：`horizon-specific attention time series forecasting`、`horizon-dependent variable selection`、`forecast-step conditioned transformer`、`different variables for different horizons`、`decoder query horizon variable selection`
- 领域5：`condition-aware sensor dependency forecasting`、`regime-aware variable selection time series`、`operating-condition-aware sensor fusion`、`context-conditioned graph forecasting`、`task-conditioned temporal attention`
- 领域6：`industrial sensor dependency forecasting`、`C-MAPSS sensor attention variable selection`、`MetroPT-3 multivariate forecasting attention`、`Tennessee Eastman Process sensor dependency graph`、`predictive maintenance dynamic sensor graph`
- 领域7：`论文标题 + GitHub`、`方法名 + official code`、`目标条件化 forecasting code`

### 2.4 检索局限（如实记录）
1. **三个并行子代理因速率限制全部失败**，领域 1–7 的检索由主代理自行完成，检索广度低于计划。
2. Elsevier/ScienceDirect 被验证码拦截，部分论文仅摘要级。
3. Scopus/WoS/IEEE Xplore 站内检索、知网未访问。
4. 未做引用链完整回溯（backward/forward citation chasing 不完整）。
5. 因此"未发现直接重合"仅指**本次检索范围内未发现**，不是全局首创证明。

---

## 3. 方法分类

| 类别 | 代表方法 | 核心机制 | 与假设关系 |
|---|---|---|---|
| **目标条件化变量选择** | 本次未发现直接方法 | — | **空白区** |
| **动态跨变量依赖** | MS-FLOW、TimeFilter、CCM、TimePro、iTransformer | 稀疏路由/图过滤/聚类/动态卷积 | 部分覆盖"选择性"，无目标条件 |
| **horizon-aware forecasting** | N-HiTS（层次插值）、TimeMixer（FMM 多预测器） | 多速率/多预测器，**非按 horizon 选源变量** | 弱相关（D） |
| **工况/任务条件化** | TimeXer、RevIN、SAN、LIFT 式(7)(9) | 条件作为输入或调制频域滤波 | 条件≠预测目标（B/C） |
| **门控与稳定融合** | **Gateformer**、RWKV-CVM、CvLoss | 自身/跨变量双分支门控；跨变量损失一致性 | **部件级直接重合** |
| **工业多传感器应用** | FD004/MetroPT-3/TEP 上的注意力方法 | 多为可视化解释或故障分类 | 弱（D），见 §4 |

---

## 4. 论文证据表

> 重合程度：**A**=直接重合（按目标+预测步动态选源变量且有正式实验）｜**B**=部分重合（有变量级注意力/动态图/门控回退，但不以预测目标为条件）｜**C**=机制相近（门控/稀疏路由/专家选择，非目标条件化依赖选择）｜**D**=仅表面相似。
> 证据等级：**A**=正文+公式/结构图/代码文件｜**B**=正文片段或仅摘要｜**C**=仅标题线索。

### 4.1 门控与稳定融合（最重要）

| 字段 | Gateformer |
|---|---|
| 标题 | Gateformer: Advancing Multivariate Time Series Forecasting through Temporal and Variate-Wise Attention with Gated Representations |
| 作者 | Yu-Hsiang Lan 等（NYU） |
| 年份 | 2025（v3 2025-07-03） |
| 期刊/会议 | **ICML Workshop on Foundation Models for Structured Data（Workshop，非主会）** |
| DOI/arXiv | arXiv:2505.00307 |
| 任务 | 多变量点预测 |
| 数据集 | 13 个公开基准（论文声明） |
| 目标条件 | ❌ gate 由 `enc_out`（自身）与 `enc_out_cross`（跨变量）共同生成，二者均为输入函数 |
| 是否 horizon-conditioned | ❌ 所有 horizon 共享同一 `enc_out` 与 gate |
| 是否变量选择 | ⚠ 稠密门控，非稀疏选择 |
| 是否时间片选择 | ✅ 先 patching + temporal-wise attention |
| 是否稀疏/Top-K | ❌ |
| **是否有自身历史回退** | ✅ **有**：`enc_out = gate * enc_out + (1-gate) * enc_out_cross`，`enc_out` 即未交互自身历史 |
| 是否使用工况 | ❌ |
| 关键公式/模块 | 代码 `models/Gateformer.py:100-101`：`gate = sigmoid(gate_w3(enc_out) + gate_w4(enc_out_cross))`；`enc_out = gate*enc_out + (1-gate)*enc_out_cross` |
| 实验消融 | 论文声明 13 数据集 SOTA、可插拔提升至多 20.7%（**该数字我未逐表复核，标记未核验**） |
| **重合程度** | **B**（"自身历史回退 + 门控融合"部件直接重合；目标条件化缺失） |
| 证据等级 | **A**（代码文件级证据，`nyuolab/Gateformer`） |
| 主要局限 | Workshop 论文；无目标/horizon/工况条件；门控逐元素而非按源变量选择 |
| 官方代码 | https://github.com/nyuolab/Gateformer（MIT） |

| 字段 | RWKV-CVM |
|---|---|
| 标题 | RWKV-CVM: Gated Cross-Variate Mixing for Multivariate Power Load Forecasting |
| 作者 | Adil Rizki, A. Echchatbi, Hamid Yantour |
| 年份 | 2026 |
| 期刊/会议 | MDPI 期刊（ISSN 2673-4826）vol.7(2):58 |
| DOI/arXiv | MDPI 期刊页（DOI **未核验**） |
| 任务 | 多变量电力负荷预测（**工业应用**） |
| 目标条件 | ❌ 未核验 |
| **是否有自身历史回退** | ⚠ 摘要称 "selectively incorporate inter-variate information while preserving the linear time complexity"，**门控细节未核验** |
| **重合程度** | **B/C**（需正文核验） |
| 证据等级 | **C**（仅摘要 + ResearchGate 条目） |
| 官方代码 | **未核验** |

### 4.2 变量/时间感知选择

| 字段 | TimePro |
|---|---|
| 标题 | TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state |
| 作者 | Xiaowen Ma, Zhenliang Ni, Shuai Xiao, Xinghao Chen（华为诺亚方舟实验室） |
| 年份 | 2025 |
| 期刊/会议 | **ICML 2025（主会）** |
| DOI/arXiv | arXiv:2505.20774 |
| 任务 | 长期多变量预测 |
| 目标条件 | ❌ 无目标查询 |
| 是否变量选择 | ✅ `n_var` 参与状态构造 |
| 是否时间片选择 | ✅ **自适应选择关键时间点**（`state_pro` = DCNv4 动态卷积，在 `(patch_num, n_var)` 网格上） |
| 是否稀疏/Top-K | ⚠ DCNv4 动态卷积，非显式 Top-K |
| **是否有自身历史回退** | ❌ 仅 SSM 内部残差 `y + xs*Ds`，无独立自身历史分支参与预测 |
| 是否使用工况 | ❌ |
| 关键公式/模块 | 代码 `model/TimePro.py`：`h = self.state_pro(h, shape=(self.patch_num, self.n_var))`；`y = h * Cs; y = y + xs * Ds` |
| 消融 | 论文提出 **multi-delay issue**（不同变量在不同时间区间影响目标变量），与假设的"不同源变量延迟不同"高度相关 |
| **重合程度** | **B**（变量感知+时间感知选择已实现；无目标条件、无回退） |
| 证据等级 | **A**（代码级） |
| 主要局限 | Mamba 基座，工业短序列收益未验证；无稀疏/门控回退 |
| 官方代码 | https://github.com/xwmaxwma/TimePro（MIT） |

| 字段 | LIFT |
|---|---|
| 标题 | Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators |
| 作者 | Lifan Zhao, Yanyan Shen |
| 年份 | 2024 |
| 期刊/会议 | **ICLR 2024** |
| DOI/arXiv | arXiv:2401.17548 |
| 目标条件 | **部分**：逐**目标变体** j 选领先指标（式 4 TopK），但条件是"目标变量 j"，非查询向量 |
| 是否稀疏/Top-K | ✅ 式(4) TopK leading indicators |
| **是否有自身历史回退** | ❌ |
| 关键公式 | 式(3) `δ(i,j,t)=argmax|R(τ)|`；式(5) target-oriented shift；式(7)(9) 状态调制频域滤波 |
| **重合程度** | **B**（逐目标选源变量已实现，但为非参数 argmax，非目标查询条件化） |
| 证据等级 | **A** |
| 官方代码 | https://github.com/SJTU-DMTai/LIFT |

### 4.3 稀疏依赖路由

| 字段 | MS-FLOW |
|---|---|
| 标题 | What If We Let Forecasting Forget? A Sparse Bottleneck for Cross-Variable Dependencies |
| 作者 | Fan Zhang, Shiming Fan, Hua Wang |
| 年份 | 2026-05-08 |
| 期刊/会议 | **arXiv 预印本，发表状态未证实**（ICML 2026 主会说法未经 arXiv 注释佐证） |
| DOI/arXiv | arXiv:2605.08289 |
| 目标条件 | ❌ 依赖图由变量级状态生成，非目标条件 |
| 是否稀疏/Top-K | ✅ sparse bottleneck + selective sparse routing |
| **是否有自身历史回退** | ❌ 未提及 |
| 关键机制 | 摘要原文：dense interactions "can amplify spurious correlations and lead to representation over-smoothing"；"replaces fully connected communication with selective sparse routing, retaining only a few critical dependency paths and injecting cross-variable signals under a strict communication budget" |
| **重合程度** | **B**（负迁移动机 + 稀疏选择完全对应；无目标条件、无回退） |
| 证据等级 | **A**（arXiv 摘要原文核验）；机制细节需正文 |
| 官方代码 | https://github.com/Cola-Fsm/MS-FLOW（**未核验**） |

| 字段 | TimeFilter |
|---|---|
| 标题 | TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting |
| 作者 | Y Hu 等 |
| 年份 | 2025 |
| 期刊/会议 | **ICML 2025（主会）** |
| DOI/arXiv | arXiv:2501.13041 |
| 目标条件 | ❌ patch-specific，非目标条件 |
| 是否变量选择 | ✅ 图过滤剔除无关相关 |
| **是否有自身历史回退** | ❌ |
| 关键机制 | 官方描述："segments the input time series into patches and constructs a spatial-temporal graph, then **filters out irrelevant correlations through patch-specific filtering**, preserving only the most critical dependencies" |
| **重合程度** | **B/C** |
| 证据等级 | **B**（ICML 官方 poster 页正文片段） |

| 字段 | CCM |
|---|---|
| 标题 | From Similarity to Superiority: Channel Clustering for Time Series Forecasting |
| 作者 | Jialin Chen, Jan Eric Lenssen, Aosong Feng, Weihua Hu |
| 年份 | 2024 |
| 期刊/会议 | **NeurIPS 2024** |
| DOI/arXiv | arXiv:2404.01340 |
| 目标条件 | ❌ 聚类由通道嵌入生成 |
| 是否稀疏 | ✅ 软聚类 `θ_i = Σ_k p_{i,k} θ_k` |
| **是否有自身历史回退** | ❌ |
| **重合程度** | **C**（机制相近：软稀疏路由，非目标条件化） |
| 证据等级 | **A**（arXiv comment "NeurIPS 2024"） |
| 官方代码 | https://github.com/Graph-and-Geometric-Learning/TimeSeriesCCM |

### 4.4 跨变量一致性

| 字段 | CvLoss |
|---|---|
| 标题 | Multivariate Time Series Forecasting needs Cross Variable Loss |
| 年份 | 2026-09-25 |
| 期刊/会议 | **NeurIPS 2026** |
| DOI/arXiv | arXiv:2608.05742 |
| 目标条件 | ❌ 损失层面，随机采样变量对 |
| 是否有自身历史回退 | ❌ |
| 关键机制 | 代码 README：跨变量 forecast patch 的残差一致性约束 `loss_add = (pred_diff - true_diff).abs().mean()`；`loss = 0.5*loss_tmp + 0.5*loss_add` |
| **重合程度** | **C**（目标相同——约束跨变量交互，但作用在损失而非模块） |
| 证据等级 | **A**（代码级） |
| 官方代码 | https://github.com/Day333/CvLoss |

### 4.5 其他最近邻（简表）

| 方法 | 来源 | 目标条件 | 变量选择 | 稀疏 | 自身回退 | 工况 | 重合 | 等级 |
|---|---|---|---|---|---|---|---|---|
| **iTransformer** | ICLR 2024 | ❌ | variate token attention | ❌ | ❌ | ❌ | B | A |
| **Crossformer** | ICLR 2023 | ❌ | TSA Cross-Dimension | ❌ | ❌ | ❌ | B | A |
| **TimeXer** | NeurIPS 2024 | 外生变量作 query | ❌ | ❌ | ❌ | ✅ | B | A |
| **MTGNN** | KDD 2020 | ❌ | 静态邻接 | top-k | ❌ | ❌ | C | A |
| **Graph WaveNet** | IJCAI 2019 | ❌ | 静态自适应邻接 | ❌ | ❌ | ❌ | C | A |
| **AGCRN** | NeurIPS 2020 | ❌ | 节点专属参数 | ❌ | ❌ | ❌ | C | A |
| **StemGNN** | NeurIPS 2020 | ❌ | 输入依赖邻接 | ❌ | ❌ | ❌ | C | A |
| **Pathformer** | ICLR 2024 | ❌ | 尺度路由 | Top-K | ❌ | ❌ | C | A |
| **ChannelTokenFormer** | ICLR 2026 | ❌ | mask-guided attention | ❌ | ❌ | ❌ | C | A |
| **DeltaLag** | ICAIF 2025 | ❌ | 逐对滞后 | Top-K | ❌ | ❌ | B | A |
| **TFT** | IJF 2021 | 静态协变量 context | 变量选择权重 | ❌ | ❌ | ✅ | C | A |
| **N-HiTS** | AAAI 2023 | ❌ | ❌ | ❌ | ❌ | ❌ | D | A |
| **SAN** | NeurIPS 2023 | ❌ | ❌ | ❌ | ❌ | ❌ | D | A |
| **RevIN** | ICLR 2022 | ❌ | ❌ | ❌ | ❌ | ❌ | D | A |
| **CI/CD trade-off** | TKDE 2024 | ❌ | ❌ | ❌ | ❌ | ❌ | C（反例） | B |
| **TFT/MTGNN 等 15 篇** | — | — | — | — | — | — | 见 §8 | — |

### 4.6 工业多传感器应用（领域 6）

**检索结论（诚实记录）**：本次检索**未找到**在 C-MAPSS / MetroPT-3 / TEP 上实现"预测目标条件化跨变量依赖选择"的论文。工业场景中注意力多用于：
- **故障分类/诊断特征加权**（TEP、FD004 上的 attention 用于提升分类精度，非路由）；
- **可视化解释**（注意力热图展示变量重要性）。

已知的工业先前工作（来自本仓库前序报告核验）：
- **Condition-Aware Dual-Attention CNN-GRU**（Expert Syst. Appl. 2026, PII S0957417426004951）：对 FD002 的 op1/op2/op3 做 k-means 分 regime + 双注意力，**C 级证据**（Elsevier 拦截，仅摘要）；有工况条件化但无变量路由、无回退。**重合 C**。
- **RUL 注意力模型**（IEEE TIE 2020, DOI 10.1109/TIE.2020.2972443）：FD004 上"treat operating conditions as measurement signals"，工况作普通输入。**重合 D**。
- **DSPR**（arXiv:2604.07393）：工业物理-残差双流，提及 variable transport lags。**重合 B（弱）**。

⚠ **任务书要求的区分已严格执行**：预测 / RUL 回归 / 故障检测 / 故障诊断 / 异常检测 / 仅可视化解释——上述工业工作**全部不属于"预测目标条件化依赖选择"**。

### 4.7 反例

**反例 1（CI/CD trade-off, TKDE 2024, DOI 10.1109/TKDE.2024.3400008）**：实证表明**通道独立（CI）策略在分布漂移下常优于通道依赖（CD）**，CD 容量高但鲁棒性弱、易过拟合训练期通道相关。→ **直接质疑"跨变量融合总是有益"**，本方向必须证明目标条件化路由能规避这一风险。

**反例 2（DLinear, AAAI 2023）**：简单线性+分解在 9 个数据集超越所有 Transformer。→ 提醒"复杂路由模块"的增益可能来自容量而非机制。

---

## 5. 代码仓库表

| 仓库 | 官方性 | 许可证 | 关键目录/文件 | 路由实现 | 输入→输出形状 | 可借鉴内容 | 风险 |
|---|---|---|---|---|---|---|---|
| [nyuolab/Gateformer](https://github.com/nyuolab/Gateformer) | 官方 | MIT | `models/Gateformer.py:100-101` | ✅ `gate_sigmoid(gate_w3(enc_out)+gate_w4(enc_out_cross))` | `x_enc [B,L,D]` → `enc_out [B,D,d_model]` → `dec_out [B,pred_len,D]` | **自身历史+跨变量双分支门控的直接实现** | Workshop 论文；门控逐元素非按源变量；无 horizon/工况条件 |
| [xwmaxwma/TimePro](https://github.com/xwmaxwma/TimePro) | 官方 | MIT | `model/TimePro.py`, `selective_scan/` | ✅ `state_pro`(DCNv4) 在 `(patch_num,n_var)` 网格 | `x_enc [B,L,D]` → patch → `enc_out [B,N,P*d]` → `dec_out [B,S,N]` | **变量×时间 2D 自适应选择**；`n_var` 参与状态 | 依赖 DCNv4 自定义 CUDA 算子，**移植成本高**；Mamba 基座 |
| [Cola-Fsm/MS-FLOW](https://github.com/Cola-Fsm/MS-FLOW) | 官方（**未核验**） | 未核验 | 未核验 | ⚠ 稀疏路由 | 未核验 | 稀疏瓶颈 + Top-K 路由思路 | **仓库未核验**；发表状态未证实 |
| [thuml/iTransformer](https://github.com/thuml/iTransformer) | 官方 | MIT | `models/iTransformer.py` | ⚠ variate self-attention | `[B,L,N]` → `[B,N,d]` → `[B,T,N]` | variate token 化的直接实现 | 稠密 attention，无稀疏 |
| [Thinklab-SJTU/Crossformer](https://github.com/Thinklab-SJTU/Crossformer) | 官方 | Apache-2.0 | `cross_models/` | ✅ TSA Cross-Dimension + router | `[B,T,D]`→`[B,D,T/L_seg,d]`→`[B,T_out,D]` | router 降复杂度 `O(2cD)` | 完整模型非模块；2D 表示需改接口 |
| [TROUBADOUR000/TimeFilter](https://github.com/TROUBADOUR000/TimeFilter) | **❌ 非官方（第三方实现）** | **无许可证声明** | 未核验 | ⚠ patch-specific 图过滤 | 未核验 | 图过滤三模块思路（Spatial-Temporal Construction / Patch-Specific Filtration / Adaptive Graph Learning） | **owner 与论文作者 Y Hu 不符，非官方；无许可证，不可直接复用代码**；论文官方仓库未找到 |
| [thuml/TimeXer](https://github.com/thuml/TimeXer) | 官方 | MIT | `models/TimeXer.py` | ⚠ variate-wise cross-attn | `[B,T_in,N]`+`[B,T_in,M]`→`[B,T_out,N]` | 外生变量作 query 的 cross-attn | 需区分内生/外生，V3C 所有传感器平等 |
| [SJTU-DMTai/LIFT](https://github.com/SJTU-DMTai/LIFT) | 官方 | 未核验 | 未核验 | ✅ 逐目标 TopK 领先指标 | `[C,L]`→逐目标 TopK→`[C,K]` | **逐目标选源变量 + 步长** | 非参数 argmax，计算在数据层 |
| [Day333/CvLoss](https://github.com/Day333/CvLoss) | 官方 | 未核验 | `Time-Series-Library/exp/exp_long_term_forecasting.py` | ❌（损失） | `outputs [B,T,D]`→patch→随机变量对 | 跨变量残差一致性损失 | 损失层面，不能替代模块 |
| [Graph-and-Geometric-Learning/TimeSeriesCCM](https://github.com/Graph-and-Geometric-Learning/TimeSeriesCCM) | 官方 | 未核验 | 未核验 | ✅ cluster-aware FF | 未核验 | 通道软聚类 | 聚类数 K 需调；非目标条件 |

---

## 6. 与 Unified V3C 的逐项差异

> V3C 事实经代码核实（`code/kaf_profiti/models/unified/encoder.py`，381 行）。

| 组件 | V3C 当前实现（代码证据） | 文献已有做法 | 计划新增内容 | 是否真正新增 |
|---|---|---|---|---|
| 逐传感器时间编码 | `self.temporal` MLP 作用于 `[B*C,L,F]` + `channel_identity` 参数（`encoder.py:88-91,236-238`） | iTransformer/TimeXer/Gateformer 均逐变量编码 | 复用 | ❌ 已有 |
| 多尺度/有序 patch | v3 系列：`pool_scorers=Linear(h,1)`+softmax，窗口取末端；**v5 系列**：`patch_tokens [B,C,patch_count,H]` 有序 patch + `patch_mixer`(v5b) + `patch_query`(v5c)（`encoder.py:121-142,240-269`） | PatchTST/Gateformer/TimePro 均 patching | 复用 v5 有序 patch | ❌ 已有 |
| 变量内 patch 选择 | `v5c`: `query=patch_query(patch_tokens[:,:,-1:,:])`，逐变量独立注意力（`encoder.py:260-266`） | 类似 TimePro 的时间选择 | 扩展为**跨变量目标条件** | ✅ 新增 |
| 跨传感器交互 | `self.cross_channel = MultiheadAttention(hidden, num_heads=1)` 作用于 `z_history [B,C,H]`（`encoder.py:109,346`） | iTransformer/Crossformer/TimePro/MS-FLOW | 从**稠密单头**改为**目标条件化稀疏路由** | ✅ 新增 |
| **自身历史回退** | ❌ **无**。`z_history = z_history + attended` 是普通残差，无"未交互自身历史分支"结构（`encoder.py:346-347`） | **Gateformer 已实现**（`gate*enc_out+(1-gate)*enc_out_cross`） | 把 Gateformer 式门控与目标条件化结合 | ⚠ **部件被占**，新增的是"条件来源" |
| learned sensor pooling | `sensor_scorer` Linear-Tanh-Linear → softmax over channels → `z_global`（`encoder.py:128-134,359-362`） | TimeXer global token | 复用/可作自身历史分支 | ❌ 已有 |
| 预测头 | `StateHead`: 共享 `Linear(h,pred_len)`（channel-independent，无 horizon 结构）（`heads/state.py:18-22`） | N-HiTS/TimeMixer 有多预测器 | horizon-conditioned 路由 | ✅ 新增（但见 §7 风险） |
| 工况条件 | `context` 存在但仅加性拼接 `z_history += context(...)`（`encoder.py:303-312`） | TFT/TimeXer | 工况→路由权重（而非仅加性） | ✅ 新增（弱） |
| 数据集独立参数 | ✅ MetroPT/FD004/TEP 独立训练 | — | 保持 | ❌ 已有 |
| train-only normalization / validation-only ckpt | ✅ 已实现 | — | 保持 | ❌ 已有 |

**必须避免的"重新命名"陷阱**：
- 不能把 Gateformer 的 `gate*enc_out+(1-gate)*enc_out_cross` 改个名字叫"稳定融合"；
- 不能把 V3C 已有的 `sensor_scorer`（learned sensor pooling）改叫"目标选择"——它是对**全局**表示加权，不是对**某个预测目标的源变量选择**；
- 方法名不得含 MetroPT/FD004/TEP，避免"为三数据集定制"印象。

---

## 7. 创新性与审稿风险判断

### 7.1 五个问题的直接回答

| 问题 | 回答 | 依据 |
|---|---|---|
| **1. "预测目标条件化依赖选择"是否已有直接先例？** | **未发现**（本次检索范围内） | §4 全部 20 篇中无一篇以"预测目标查询"生成路由权重；Gateformer 的 gate 由输入自身决定；LIFT 的"目标"是变体索引而非查询向量 |
| **2. "不同预测步选择不同源变量"是否已有直接先例？** | **未发现** | N-HiTS 是多速率插值，TimeMixer 是多预测器求和，二者均**不按 horizon 改变源变量选择**；未见逐 horizon attention map 的同类工作 |
| **3. "选择性跨变量路由 + 自身历史回退"是否已有直接先例？** | **部分有**：自身历史回退+门控——**Gateformer（代码级 A 级证据）**；选择性路由——MS-FLOW/CCM/TimeFilter；**但两者未被同一论文结合，且均不以预测目标为条件** | §4.1、§4.3 |
| **4. "工况条件改变变量路由"是否已有直接先例？** | **未发现直接先例** | TimeXer 把外生变量作 query 但不改**变量路由**；RevIN/SAN 是归一化；FD004 条件化 PdM 是普通输入（C 级）。**注**：上一轮报告已记录 LIFT 式(7)(9)"状态→调制滞后使用"（B 级），工况**变量**改路由仍空白 |
| **5. 当前方向更适合作为什么？** | **多个机制的组合创新**（需实验支撑），而非全新基础模块 | 三个部件各自被占（Gateformer 回退、MS-FLOW 稀疏、TimeXer 条件化），仅"目标条件化"接缝未被占 |

### 7.2 可写 / 不可写的 claim

✅ **可写**：
- "我们指出固定全量跨变量交互在目标间传播无关信息的问题，并在工业多传感器数据上系统评估其影响。"
- "现有门控融合方法（Gateformer）的门控由输入自身生成；我们提出以预测目标查询生成路由权重的变体。"
- "我们提供受控实验，量化'按目标改变源变量选择'相对'输入自适应融合'的增量。"

❌ **不可写**：
- ~~"首次提出自身历史回退分支"~~ → Gateformer 已实现（代码级证据）
- ~~"首次提出门控跨变量融合"~~ → Gateformer、RWKV-CVM
- ~~"首次提出选择性跨变量依赖"~~ → MS-FLOW、CCM、TimeFilter、LIFT
- ~~"首次发现负迁移问题"~~ → MS-FLOW 摘要明确提出、CvLoss、CI/CD trade-off
- 任何"显著优于 SOTA"的表述（**本轮无任何实验**）

### 7.3 审稿人会指出的最近邻（必须主动引用）

| 审稿人可能质问 | 我方应对 |
|---|---|
| "这不就是 Gateformer 换个 query？" | 承认门控结构相同，差异在条件来源（目标查询 vs 输入自身），**用消融证明差异有效** |
| "为什么不直接用 MS-FLOW？" | MS-FLOW 非目标条件化且发表状态未证实；我方聚焦目标条件化维度 |
| "目标条件化在 N 个变量的 N 个输出上就是 N 次 attention，成本？" | 报告复杂度并给出共享参数的等价实现 |
| "FD004 只有一个 RUL 输出，怎么验证目标条件化？" | **FD004 只有单一 RUL 查询**（见 §10 边界），目标条件化需多目标输出才可验证 → **FD004 不能验证该机制**，只能验证"选择性依赖" |
| "TEP 用故障标签路由就是泄漏" | 严格禁止；只允许历史可得工况 |
| "工业数据量小，复杂路由过拟合" | 参数量匹配对照 + 多 seed |

---

## 8. 推荐方法定义

### 8.1 命名
- **中文名**：目标条件化依赖选择与门控回退融合（Target-Conditioned Dependency Selection with Gated Fallback Fusion）
- **英文名**：**TCDS-GF**（Target-Conditioned Dependency Selection with Gated Fallback）
- **缩写候选**：TCDS / TCDF / TCDR；**推荐 TCDS-GF**（避免含数据集名，避免以 horizon 为唯一定限定词）
- ❌ 禁止：`MetroPT-V3C-X`、`FD004-Router`、`TEP-Select`

### 8.2 数学问题定义

给定历史窗口 $X \in \mathbb{R}^{L \times C}$ 与历史可得上下文 $c \in \mathbb{R}^{d_c}$，对每个预测目标 $(j, h)$（变量 $j$、预测步 $h$）预测 $y_{j,h}$。目标：从 $C$ 个源变量中为每个 $(j,h)$ 选择有效子集 $S_{j,h} \subseteq \{1..C\}$，并与目标自身历史稳定融合。

### 8.3 输入输出接口
- **输入**：$X$ `[B, L, C]`、`M_obs` `[B, L, C]`、`context` `[B, d_c]`（历史可得工况）、`target_idx` `[B]`（目标变量）、`horizon` `[H]`
- **输出**：`y_hat` `[B, H, C]`
- **约束**：仅历史信息；无未来工况；无故障标签；train-only normalization

### 8.4 路由公式

设有序变量—时间表示 $E \in \mathbb{R}^{B \times C \times P \times H}$（V3C v5 的 `patch_tokens`，P=patch 数，H=hidden）。

**（a）目标查询**
$$q_{j,h} = \mathrm{MLP}_q\big(e_j^{\text{target}} \;\Vert\; \mathrm{Emb}(h) \;\Vert\; \mathrm{Emb}(\text{task}) \;\Vert\; c\big) \in \mathbb{R}^{H}$$
其中 $e_j^{\text{target}}$ 为目标变量的历史表示，$c$ 为**历史可得**工况。

**（b）跨变量依赖路由（目标条件化 + 稀疏）**
$$\alpha_{i \to j, h} = \mathrm{softmax}_{i}\left(\frac{\langle q_{j,h},\, k_i \rangle}{\sqrt{H}}\right), \quad k_i = W_k\, \mathrm{pool}(E_i)$$
$$\mathcal{S}_{j,h} = \mathrm{TopK}\big(\alpha_{\cdot\to j,h}\big) \quad \text{（}K \ll C\text{，稀疏预算）}$$

**（c）自身历史分支与门控回退融合**（Gateformer 式结构，但条件来自 $q$）
$$m^{\text{self}}_{j} = W_s\, \mathrm{pool}(E_j), \qquad
m^{\text{cross}}_{j,h} = \sum_{i \in \mathcal{S}_{j,h}} \alpha_{i\to j,h}\, v_i$$
$$g_{j,h} = \sigma\!\big(W_g [q_{j,h} \;\Vert\; m^{\text{self}}_j \;\Vert\; m^{\text{cross}}_{j,h}]\big) \in [0,1]^{H}$$
$$\boxed{\, z_{j,h} = g_{j,h} \odot m^{\text{self}}_{j} \;+\; (1 - g_{j,h}) \odot m^{\text{cross}}_{j,h} \,}$$
$$y_{j,h} = W_o\, z_{j,h}$$

> ⚠ 注意：当 $g\to 1$ 时退化为**纯自身历史预测**——这正是"跨变量信息无效时保持稳定"的机制。**必须报告 $g$ 的分布**（$\mathbb{E}[g]$、$g<0.5$ 的样本比例），否则无法证明回退真的发生。

### 8.5 复杂度
- 路由打分：$O(B \cdot H \cdot C \cdot H_d)$
- 稀疏聚合：$O(B \cdot H \cdot K \cdot H_d)$
- 总计约 $O(BHC H_d)$，与 V3C 单头 MHA 的 $O(BC^2 H_d)$ 相当（C=15–40 时更省）。

### 8.6 训练与推理约束
1. 仅历史输入；`context` 严格取历史可得工况（FD004: cols 3–5；TEP: XMV；**禁止 fault number**）。
2. train-only normalization；validation-only checkpoint selection；test 只评估一次。
3. 按 unit/run 隔离（FD004 按发动机、TEP 按 simulation run、MetroPT 按时间）。
4. 单 seed 仅作方向筛选；多 seed 评估稳定性。
5. TopK 的 K 只在 validation 上选。

---

## 9. 最小实验矩阵

### 阶段 A：机制正确性（合成数据）
| ID | 目的 | 设置 |
|---|---|---|
| A1 | 目标依赖确实存在时能否恢复 | 合成 $y_j = f(x_{j}, x_{i})$，$f$ 仅对 $i\ne j$ 成立 |
| A2 | **无跨变量依赖时必须回退** | 合成 $\mathrm{Corr}(x_i, y_j)=0$，检查 $\mathbb{E}[g]\to 1$ |
| A3 | 置换检验 | 置换源变量顺序，路由应同步改变 |

### 阶段 B：MetroPT-3（horizon-conditioned routing 主战场）
| ID | 对照臂 |
|---|---|
| B0 | V3C（现状，稠密单头 MHA） |
| B1 | **无跨传感器融合**（逐传感器独立，退化版） |
| B2 | 普通全量跨传感器 attention（多头，参数量对齐） |
| B3 | 目标条件化但**不稀疏**（稠密 $\alpha$，只有 $q$ 条件） |
| B4 | **目标条件化选择性路由**（Top-K） |
| B5 | **选择性路由 + 自身历史回退**（完整方法） |
| B6 | 仅任务条件（去掉 $q$ 中的目标/horizon） |
| B7 | 仅预测步条件 |
| B8 | 任务 + 预测步条件 |
| B9 | 参数量匹配的普通 MLP/channel mixer |
| B10 | **源变量置换** |
| B11 | **历史片段置换** |
| B12 | **查询向量置换** |
| B13 | **路由门控强制关闭**（$g\equiv 1$，纯自身历史） |
| B14 | **路由门控强制开启**（$g\equiv 0$，纯跨变量） |
| B15 | K 敏感性（$K\in\{1,3,5,9,C\}$，只在 valid 选） |

### 阶段 C：FD004（**仅选择性依赖，不验证 horizon 条件化**）
- ⚠ **FD004 是单一 RUL 回归目标，没有多预测步标签**。不得虚构多步 RUL。
- 对照：C0 V3C / C1 无融合 / C5 选择性+回退 / B13 门控关闭 / B14 门控开启 / 源变量置换。
- 指标：raw MAE / raw RMSE / **NASA score**；工况=cols 3–5（历史可得，**禁止 fault number**）。
- 目的：验证"选择性依赖"在退化任务上不劣于稠密融合。

### 阶段 D：TEP（检测/诊断任务）
- ⚠ **TEP 不得用故障标签作为推理期路由条件。**
- 对照：D0 V3C / D1 无融合 / D5 选择性+回退；工况仅用历史可得 XMV 操纵量。
- 指标：Macro-F1、Balanced Accuracy、逐类指标、检测延迟、误报率；run-level 结果。
- 目的：验证检测任务与诊断任务**是否形成不同依赖**（若 $q$ 含 task id，两任务路由应不同）。

### 阶段 E：通用数据集（**跨领域泛化必须做**）
- **建议增加 Electricity / Traffic / Weather 之一**，否则无法反驳"仅在工业小数据上有效"。
- 对照：E0 V3C / E1 PatchTST（CI 对照）/ E5 完整方法。

### 阶段 F：多 seed 与成本
- 通过的臂跑 ≥3 seeds（2026/2027/2028）；报告均值±标准差。
- **分别报告**：参数量、可访问历史信息、训练时间、推理时间、峰值显存（**三者不等价**）。

### 固定项（所有实验）
数据划分、seed、history length、prediction length、batch size、训练轮数、early stopping、学习率、normalization、checkpoint 选择规则、test evaluator、缺失/扰动设置、参数预算——全部锁定并写入 manifest。

---

## 10. 最终结论

### 10.1 是否建议继续实现
**建议继续，但必须以"受控实证研究"而非"新机制"定位。** 理由：
- "自身历史回退 + 门控融合"已被 Gateformer 实现（代码级证据），不能作为创新；
- "目标条件化路由"在本次检索范围内未发现直接先例，是唯一可辩护的接缝；
- 但该接缝必须用 B3（目标条件化不稀疏）vs B4（稀疏）vs B5（+回退）的受控消融证明**每一步都有独立贡献**，否则会被判定为 Gateformer 的重新命名。

### 10.2 最小可行版本
**只做 B0 / B1 / B2 / B3 / B4 / B5 / B13 / B14 八臂**，在 **MetroPT-3 单数据集**上完成：
1. 若 B5 > B2（+目标条件化 +稀疏 +回退 优于 稠密 attention）且方向在 3 seeds 一致 → 进入下一数据集；
2. 若 B5 ≈ B2 → **放弃该方向**，回退为"工业数据上跨变量交互的实证比较"；
3. 若 B13（门控关闭=纯自身历史）≈ B5 → **回退机制无效**，说明跨变量信息在该数据上无用。

### 10.3 是否必须增加通用数据集
**必须**（若要写"跨领域泛化"）。否则只能写"在工业多传感器数据上"。

### 10.4 最容易被质疑的点
1. **"这就是 Gateformer"** —— 必须主动引用并用 B3（只有 $q$ 条件、无回退）做隔离对比。
2. **FD004 无法验证 horizon 条件化** —— 单一 RUL 输出，**不得虚构多步 RUL 标签**。
3. **TEP 泄漏风险** —— 故障标签绝不可作路由条件。
4. **工业数据量小 → 过拟合** —— 参数量匹配 + 多 seed。
5. **"选择性"若不稀疏只是门控改名** —— B3 vs B4 必须有差异。

### 10.5 下一步代码与测试顺序
1. 在 `encoder.py` 新增独立变体（不改动 v3c 路径），实现 $q$ 构造 + 路由 + 门控回退；
2. 单元测试：张量形状、有限性、**masked-input invariance**（回退分支必须满足）、K 边界（$K=1$、$K=C$）、`context_dim=0` 退化路径；
3. 合成数据测试 A1/A2（无依赖时 $g\to 1$ 必须成立）；
4. 参数量对齐脚本；
5. 按 B0→B1→B2→B3→B4→B5 顺序单 seed 筛选；
6. 通过后补 B6–B15 消融，再进入 FD004/TEP/通用集。

### 10.6 尚未核验的事项
1. **三个并行子代理因速率限制全部失败**，领域 1–7 检索广度低于计划，**"未发现直接先例"的结论强度受限**；
2. **RWKV-CVM** 仅摘要级（C 级），其门控是否为"自身历史回退"**未核验**；
3. **MS-FLOW 的官方仓库未核验**；**TimeFilter 官方仓库未找到**——`TROUBADOUR000/TimeFilter`（123 stars，2025-01-18 创建）README 自述为论文实现但 **owner 与论文作者 Y Hu 不符、无许可证声明**，按任务书"不要直接复制代码"与官方性要求，**判定为第三方实现，不得当作官方代码引用**；
4. **Gateformer 的 20.7% 提升数字未逐表复核**；
5. **CvLoss（NeurIPS 2026）的完整机制未核验**（仅 README 代码片段）；
6. 未做完整引用链回溯，可能遗漏 2026 年新发表的相关工作；
7. **本轮无任何实验**，所有性能相关表述均无数据支撑。