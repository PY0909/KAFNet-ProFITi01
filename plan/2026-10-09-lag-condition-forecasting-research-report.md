# 文献调研报告：面向工况变化的时滞感知多传感器时序预测方法

- 调研日期：2026-10-09
- 任务书：`plan/2026-10-09-sosr-literature-research-prompt.md` 同目录新任务（面向工况变化的时滞感知多传感器时序预测）
- 结论性质：**纯文献与规划产出**。本轮未运行任何实验，未修改 `code/`、模型或 `result/`。
- 相关前序报告：`plan/2026-10-09-sosr-literature-research-report.md`（SOSR 方向查新），本报告独立成篇，交叉引用处均已注明。

**报告结构对照**（任务书 §9 要求 → 本报告位置）：

| 要求章节 | 位置 | 要求章节 | 位置 |
|---|---|---|---|
| 0 检索方法/工具/证据边界 | §0 | 6 候选定义 V1–V4 评估 | §6 |
| 9.1 执行摘要（≤500 字） | §9.1（499 字） | 7 实验矩阵（≥10 项） | §7（10 臂 + 2 补充对照） |
| 1–3 A/B/C 分组检索证据 | §1、§2、§3 | 9.5 创新性风险审查 | §8（含 §4 四级判定） |
| 9.2 最近邻文献表（≥15 篇 × 12 字段） | §5（33 条） | 9.6 最小可行研究方案 | §9.6 |
| 9.3 方法机制对照表（5 维度） | §9.3（32 行） | 9.7 参考文献与证据索引 | §9.7（R01–R56、D01–D11） |
| 9.4 数据集与协议建议 | §9.4 | 10 科学边界 10 条 | §10 |

---

## 0. 检索方法、工具与证据边界

### 0.1 工具实际使用情况（如实记录）

> 本节于 2026-10-09 复核并**更正**：初稿曾把 `agent-reach`、`paper-spine` 记为"不可用"，原因是 `which agent-reach` 落空且未找到 skill 注册项。经用户提示复核，两者的实际状态见下表——前者是 `PATH` 未包含安装目录，后者是文件存在但未注册进本会话 skill 表。

| 工具 | 实际状态 | 探测依据 |
|---|---|---|
| `agent-reach` CLI | **可用**（v1.5.0），初稿记录有误 | 实际安装于 `/root/.local/bin/agent-reach`（另有 `/root/anaconda3/bin/agent-reach`），因 `/root/.local/bin` 不在 `PATH` 导致 `which` 落空；用绝对路径执行成功：`agent-reach --version` → `Agent Reach v1.5.0`，`agent-reach doctor --json` 正常返回 |
| agent-reach `web` 后端 | **可用**（`active_backend: Jina Reader`） | `doctor --json` → `web: ok`；本轮读取论文正文实际用的就是它推荐的同一通道 `curl https://r.jina.ai/URL` |
| agent-reach `github` 后端 | warn（gh CLI 可执行、已配置认证；Doctor 为避免写 device-id 未实时验证） | `doctor --json` → `github: warn` |
| agent-reach `bilibili` 后端 | ok（B 站搜索 API，仅搜索） | `doctor --json` → `bilibili: ok` |
| `mcporter` / Exa 搜索后端 | **不可用** | `mcporter: command not found`；`doctor --json` → `exa_search: off`（需 `npm install -g mcporter` + 配置 Exa MCP） |
| agent-reach 其他后端 | Twitter / YouTube / Reddit / Facebook / Instagram 均 off | `doctor --json`（未安装 twitter-cli、yt-dlp、Reddit/OpenCLI 登录态等），本任务不需要 |
| `paper-spine` skill | **文件存在但本会话未注册** | 安装于 `/root/.codex/skills/paper-spine/`（SKILL.md 23KB + 90 个 references + scripts + agents），但本会话可加载 skill 清单中无此项，故未能通过 skill 工具调用；本轮改为其规则的人工遵从（见下） |
| 内置 `websearch` | 已使用 | 主要检索手段（约 15 组查询，覆盖 A/B/C/D 四组主题与数据集核验） |
| `gh`（GitHub CLI） | 已使用 | 仅用于确认代码仓库是否存在，**不作为方法证据** |
| Crossref API | 部分可用 | 元数据交叉核验；部分查询解析失败（MRLCD-A、TiVaT、Sensorformer），已逐条标注 |
| ScienceDirect（Elsevier） | **被拦截** | 验证码页面，MRLCD-A、RRGL、Condition-Aware CNN-GRU 等正文仅得摘要级证据，标 C 级 |

**关于 agent-reach 的实际使用声明**：按 skill 要求"声明你在用什么"——本轮通过 agent-reach 的 `web` 后端（Jina Reader）读取论文正文，`github` 后端（gh）确认代码仓库存在；`doctor --json` 已执行并据 `active_backend` 选型。其 `search` 分类所依赖的 Exa（mcporter）通道为 `off`，故全文检索改用内置 `websearch` 补足——这是一条后端级替代，已在 §0.4 局限中体现。

**关于 paper-spine 的证据边界**：虽未能以 skill 形式调用，但其 `references/scientific-evidence-ledger.md` 与 SKILL.md「Authority and boundaries」条款已人工遵从，对本报告有约束力的三条是：① 绝不发明数据、指标、p 值、引用、图表或方法（本轮无任何实验数字）；② 未经核验的记录只能标 `planned`，不得在行文中呈现为已观察到的证据（对应本报告的 A–D 证据分级与"未证实"标注）；③ **通过格式检查/工具成功不等于科学就绪**（对应本报告不以"检索跑通了"替代"方向已被证明空白"）。

### 0.2 检索分组与实际关键词

- **A 组（时滞感知）**：`lag-aware multivariate time series forecasting`、`time-lagged cross-variable attention`、`lead-lag relationship multivariate forecasting`、`learnable lag multivariate`、`deformable offset lead-lag`、`learnable lag selection graph neural network`、`asynchronous multi-sensor delayed sensor forecasting`、`"lag" attention gating regime condition`。
- **B 组（工况/状态条件化）**：`condition-aware multivariate time series forecasting`、`regime-aware time series forecasting mixture of experts operating condition dynamic graph`、`working condition aware attention graph industrial`、`context vector modulate temporal offset forecasting`、`state-dependent lag routing attention`。
- **C 组（工业多传感器）**：`industrial process multivariate forecasting sensor time lag condition change`、`predictive maintenance cross-variable delay model`、`"transport delay" attention industrial process forecasting`、`"lag-aware" industrial MetroPT compressor Tennessee Eastman`。
- **D 组（邻近方法）**：由并行子任务核验 TFT / Informer 系 / PatchTST / TimesNet / iTransformer / TCN / MTGNN / Graph WaveNet / StemGNN / DCRNN / AGCRN / Neural ODE 系 / DTW·互相关·Granger / regime·MoE / 自监督预训练（见 §5 中标注 D 组来源的条目）。
- **数据集**：MetroPT-3 / C-MAPSS FD004 / TEP（多版本）/ SWaT / WADI / PHM 2012 / XJTU-SY / IMS（见 §9.4）。

### 0.3 证据等级定义

- **A 级**：论文正文的公式/章节/算法/实验表定位（我方直接读取正文全文）。
- **B 级**：正文片段（出版社页面或检索返回的正文段落，非全文通读）或正式会议/期刊页面 + 摘要一致性核验。
- **C 级**：仅摘要或检索结果摘要（正文未取得，如 Elsevier 拦截）。
- **D 级**：仅有线索（标题/引用/他人转述），**不足以支撑任何判断**，只登记不采信。

规则：只有 A/B/C 级证据才进入判定；"是否显式 lag"一栏，凡无正文/公式/结构图依据一律写"未证实"。

### 0.4 检索局限（必须随结论一起引用）

1. ScienceDirect 正文不可达 → MRLCD-A、MDA 全文、RRGL、ESWA 条件化 PdM 论文只有摘要级证据。
2. 未系统检索 IEEE Xplore/ACM DL 站内检索（仅通过 DOI/正式页可达条目），Scopus/Web of Science 未访问 → **不能声称穷尽检索**。
3. 中文文献（知网）未检索。
4. Google Scholar 未直接访问，引用数仅在页面出现时记录，不作为判断依据。
5. 检索语言以英文为主，德/日文会议文献未覆盖。
6. 因此，本报告对任何方向的"未覆盖"结论均为**在上述检索范围内的未发现**，不是全局首创证明。

---

## 9.1 执行摘要（≤500 字）

该方向**有条件值得继续，但不能按"首次提出"立项**。

最近邻：① lead-lag 建模——LIFT（按状态调制滞后）、VCformer、DeltaLag（逐对可学习滞后+TopK）、DeformTime/TiVaT、PDFormer；② 滞后检测对齐——MRLCD-A、STLAformer、HMV-Former、LSTE；③ 异步跨传感器——ChannelTokenFormer、Raindrop、GraFITi；④ 工况条件化——TimeXer、KI-TSF、regime-MoE、op1–op3 分 regime 的 C-MAPSS PdM。

**风险**：V1/V2 已被 DeltaLag、LIFT 完整实现（A 级），LIFT 式(7)(9) 已占"状态→调制滞后使用"，V3 仅剩"外部工况作条件+聚合粒度"接缝 → **判 C 级，仅作研究假设，禁"首次提出"，须过容量匹配对照**。**数据集**：MetroPT-3 作主实验（4 起故障无标签、331 处断档跳过）；FD004 仅做工况条件化验证（249/248，无墙钟时间戳，不做时滞主张）；TEP 锁定 Rieth 2017 版后纳入。三者单元格级无原生缺失，注入与断档分开表述。

---

## 1. A 组：时滞感知多变量预测（逐篇证据）

### 1.1 LIFT —— 与 V1/V2 的直接重合基准

- 引用：Lifan Zhao, Yanyan Shen. *Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators*. ICLR 2024. arXiv:2401.17548（v6, 2024-08-13）。OpenReview: JiTVtCUOpS。代码：github.com/SJTU-DMTai/LIFT（arXiv v6 摘要）/ github.com/SJTU-Quant/LIFT（ICLR 2024 页面），两处链接不一致，**以仓库实际存在为准，未核验其可运行性**。
- 证据定位（A 级，正文通读）：
  - §3.2 Lead Estimator：式 (2) 用 FFT/Wiener–Khinchin 一次算出全部互相关；**式 (3)** `δ(i,j,t) = argmax_τ |R(τ)|` 给出逐对、逐时刻的领先步长；**式 (4)** 对每个目标变体 j 取 TopK 领先指标——即"不同目标选不同源 + 不同滞后位置"。
  - §3.3：**式 (5)** target-oriented shift（把源序列按 δ 平移对齐目标）；**式 (7)** 由历史窗口估计变体状态概率 `P_t = softmax(P0 + f_state(X))`；**式 (9)** 频域滤波器 `r = Σ_n p_n · f_n(R_t)` ——**状态（输入条件）直接调制滞后信息的使用权重**。
  - §5.3 消融：移除 influence/difference/filter 后 MSE 变差（图 5(b)）；§5.4 K、N 超参敏感性（图 5(c)）。
- 重合部件：跨传感器路由、逐目标可变滞后、滞后选择消融、**状态条件化调制**。
- 差异：(a) 滞后由非参数 argmax 估计，非端到端可学习偏移；(b) 条件是内部"变体状态"（由历史窗口学出的隐状态），不是外部可观测工况变量（如 FD004 的 op settings、压缩机负载）；(c) 调制发生在**频域预测精修**，不是跨传感器聚合的路由权重；(d) 六个数据集（Weather/Electricity/Traffic/Solar/Wind/PeMSD8）均非工业设备数据。
- 判定：**对 V1/V2 直接重合；对 V3 部分重合（"状态 → 调制滞后使用"已有先例）**。

### 1.2 VCformer —— 全滞后聚合的可学习权重

- 引用：Yingnan Yang, Qingling Zhu, Jianyong Chen. *VCformer: Variable Correlation Transformer with Inherent Lagged Correlation for Multivariate Time Series Forecasting*. IJCAI 2024, pp. 5337–5343. arXiv:2405.11470。代码：github.com/CSyyn/VCformer。
- 证据定位（A 级，正文通读）：
  - §3.1.1 指出逐点注意力"忽略变量间时间延迟"；**式 (4)** `LAGGED-COR(q_i,k_j) = q_i ⊙ ROLL(k_j, τ)`，τ∈[1,T]；**式 (5)** 用可学习 `λ=[λ1..λT]` 聚合各滞后；**式 (6)** softmax 得 N×N 变量相关图。
  - §3.5 FFT 加速至 O(N²T logT)；§4.2 表 2：VCA 可替换进 Non-stationary/Autoformer/Informer；§4.3.1 表 3 消融 VCA/KTD。
- 重合部件：显式枚举滞后、可学习滞后权重、跨变量注意力。
- 差异：(a) λ 为**全局共享的滞后聚合权重**，非逐源-目标对、非随输入变化；(b) 聚合全部滞后而非选择/路由；(c) 无工况条件；(d) 8 个通用数据集（ETT/Weather/Electricity/Traffic/Exchange），非工业。
- 判定：**对 V1 直接重合（枚举滞后）；对 V2 部分重合（缺逐对、逐输入的路由）**。

### 1.3 DeltaLag —— 逐对可学习滞后 + TopK 路由（V2 的最强重合）

- 引用：Wanyun Zhou, Saizhuo Wang, Mihai Cucuringu, Zihao Zhang, Xiang Li, Jian Guo, Chao Zhang, Xiaowen Chu. *DeltaLag: Learning Dynamic Lead-Lag Patterns in Financial Markets*. 6th ACM International Conference on AI in Finance (ICAIF '25), Singapore, 2025-11. DOI: 10.1145/3768292.3770421. arXiv:2511.00390。
- 证据定位（A 级，正文通读）：
  - §4.1：**式 (6)** key 取每个候选源的最后 `l_max` 个时间步；**式 (7)** `A_{u,v,t} = q_{u,t} K_{v,t}^T ∈ R^{1×l_max}`——每个（目标,源）对在滞后维上打分；**式 (9)(11)** TopK 得到逐对滞后 `τ_{v→u} = l_max − j`。
  - §4.2：**式 (13)(14)** 按各自滞后取源的原始特征，softmax 加权聚合后送 MLP。
  - 表 1 明确给出 Lead-lag / Delta(learnable) / Adaptive 三列对照（vs Lag1Net 固定滞后、离线相关图基线）；§6.3 图 3 滞后分布近似均匀。
- 重合部件：**逐源-目标对、可学习滞后位置、稀疏（TopK）跨传感器路由、滞后配置消融**。
- 差异：(a) 金融域收益预测，非工业多传感器；(b) "条件"仅为叙述性市场状态，**没有以可观测工况/协变量作为路由条件**；(c) 预测头是简单 MLP，不含时序骨干的联合训练（检测与预测端到端但结构分离）。
- 判定：**对 V2 直接重合；对 V4 的稀疏化部分直接重合（TopK 选源+滞后）**。

### 1.4 DeformTime —— 可学习偏移的跨变量注意力

- 引用：Yuxuan Shu, Vasileios Lampos. *DeformTime: Capturing Variable Dependencies with Deformable Attention for Time Series Forecasting*. TMLR 2025（OpenReview M62P7iOT7d）。arXiv:2406.07438。代码：github.com/ClaudiaShu/DeformTime。
- 证据定位（A/B 级）：§3.3 V-DAB "cross-attention over embeddings of both endo- and exogenous variables within proximal time steps"，元素 p 可注意 p+Δp（可学习偏移）；§3.4 T-DAB 处理同一变量邻近时刻；输入依赖的 offsets 属"隐式可学习时滞"。
- 差异：不显式枚举/解释滞后位置，无工况条件，数据集含流感等非工业任务。
- 判定：**部分重合（机制层面覆盖"可学习跨变量时间偏移"，但非显式 lag 选择）**。

### 1.5 TiVaT —— 联合轴注意力 + DTV 采样

- 引用：arXiv:2410.01531（v1 标题 *TiVaT: Joint-Axis Attention for Time Series Forecasting with Lead-Lag Dynamics*, 2024-10-02）；期刊版 *TiVaT: A transformer with a single unified mechanism for multivariate time series forecasting*, Applied Soft Computing（PII S1568494626004278，2026），**标题与作者页已核验，正式 DOI 未取得（记"未证实"）**。
- 证据定位（B 级，arXiv HTML 片段）：§3.2 JA attention 中 query q'(t,v) 吸收 lagged points Z(t',v')（t'≠t, v'≠v），引入 deformable attention 风格 offsets + Distance-aware Time-Variable 2D 采样图。
- 差异：滞后关系以偏移+采样图隐式表达，无显式 lag 枚举/选择报告；无工况条件。
- 判定：**部分重合**。

### 1.6 Sensorformer —— 以动态因果滞后为动机的跨 patch 注意力

- 引用：Liyang Qin, Xiaoli Wang, Chunhua Yang, Huaiwen Zou, Haochuan Zhang. *Sensorformer...* arXiv:2501.03284（2025-01-06，18 页）。**正式发表状态未证实**（arXiv 页无 journal/conference 注释，Crossref 查询失败）。
- 证据定位（B 级）：摘要与 §3.3：以"变量间动态因果滞后导致强制融合无关数据"为动机，两阶段 Sensor Attention Block（末 patch 作 Query 压缩 → 二阶段跨变量/跨时刻注意力），复杂度 O(D²·N·d)。
- 是否显式 lag：**未证实**（无 lag 枚举/学习公式；动机是滞后，机制是全 patch 注意力）。
- 判定：**相关但机制不同（动机重合、实现隐式）**。

### 1.7 MRLCD-A —— 滚动互相关滞后检测 + 对齐（插件）

- 引用：D. Sun et al. *MRLCD-A: Lag-aware alignment for multivariate time series forecasting*. Information Processing & Management（2025）。DOI: 10.1016/j.ipm.2025.104191（doi.org 返回 302 可解析；ACM DL 与 ScienceDirect 页面显示同一 DOI）。**正文未取得（Elsevier 验证码），作者全名单未核验**。
- 证据（C 级，摘要/检索片段）："multivariate rolling lag correlation detection-alignment"，用互相关确定滞后步长并对齐后输入标准预测器，plug-and-play。
- 判定：**部分重合**（显式滞后但为外部预处理对齐，非模型内可学习路由；无工况条件）。

### 1.8 MDA —— 滞后变量重要性的动态注意力

- 引用：Sarah Almaghrabi, Mashud Rana, Margaret Hamilton. *Multidimensional dynamic attention for multivariate time series forecasting*. Applied Soft Computing, 2024. DOI: 10.1016/j.asoc.2024.112350（Crossref 核验：期刊、年份、作者）。
- 证据（C 级，摘要片段）："computing lagged variable importance"，注意力刻画滞后变量重要性随时间变化。
- 判定：**部分重合**（显式滞后重要性；正文未取得，无从判断是否逐对选择滞后位置）。

### 1.9 STLAformer

- 引用：Zhao Yuhui, Sun Lijie. *STLAformer: Segment-based Temporal Lag Attention effective for time series forecasting*. DOI: 10.1145/3759928.3759960（Crossref：Proceedings of the 2nd International Conference on Image Pro…（截断），2025）。**会议全称未核验完整**。
- 证据（C 级，ACM 摘要片段）：分段 + delay-aware attention，"explicitly modeling asynchronous lag dependencies in multivariate time series forecasting"。
- 判定：**部分重合**（仅摘要级，机制细节未证实）。

### 1.10 HMV-Former（LCCA）

- 引用：*HMV-Former: multi-view hierarchical patching and lag-coupled channel attention*. Frontiers in Applied Mathematics and Statistics（2026）。DOI: 10.3389/fams.2026.1825555。
- 证据（B 级，出版社正文片段）：§3.3.2 LCCA = Temporal Channel Adapter 分支 + Lag Mixing 分支；在 patch-token 维平均池化得通道嵌入，构造"lag-aware channel representation"后生成动态亲和矩阵（Eq. 13/14/18 片段可见）。
- 差异：滞后混合在通道嵌入层面完成，滞后信息不显式建模为时间位置；无工况条件；未见工业数据集。
- 判定：**部分重合（命名与目标高度接近，须列为必查反例）**。

### 1.11 PDFormer —— 传播时延显式建模（交通域）

- 引用：Jiawei Jiang, Chengkai Han, Wayne Xin Zhao, Jingyuan Wang. *PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction*. AAAI 2024. DOI: 10.1609/aaai.v37i4.25556。arXiv:2301.07945。
- 证据（A/B 级）：正文（AAAI 版）提出 Delay-aware Feature Transformation，"captures the propagation delay from the short-term historical traffic flow of each node"并注入地理空间自注意力的 key；同时用两组图掩码刻画短/长程动态空间依赖。
- 判定：**相关但不同**（显式建模传播时延 + 动态关系，但为交通流、非跨传感器工况条件化；可作基线思想来源）。

### 1.12 其他 A 组登记条目

- **ChannelTokenFormer**（ICLR 2026 正式页；arXiv:2506.08660）：§3 逐通道异步采样建模（相对采样周期 s_i），§4 式 (2)(3) mask-guided self-attention 统一 local/global token，含 LNG 船舶工业数据集与 ETT-practical 异构采样变体。**判定：部分重合**（异步/缺失/依赖统一处理，无显式 lag 枚举）。
- **Raindrop**（Xiang Zhang, Marko Zeman, Theodoros Tsiligkaridis, Marinka Zitnik, ICLR 2022, arXiv:2110.05357）：摘要级——每样本一张传感器图 + 消息传递预测错位读数。**判定：相关但不同**（不规则采样对齐，非显式滞后路由）。
- **GraFITi**（AAAI 2024, DOI: 10.1609/aaai.v38i15.29560）：§6.5 明确指出其在异步时间序列上因图不连通而受限。**判定：相关但不同**。
- **Autoformer 自相关机制**（NeurIPS 2021，经 LIFT/VCformer 正文转述）：基于时延的序列复用（lag reuse）属固定滞后聚合。**判定：对 V1 直接重合的早期形态**（转述级证据，D→B，未直接读 Autoformer 正文，标注为"需直接核验"）。

---

## 2. B 组：工况条件化/状态条件化关系建模（逐篇证据）

### 2.1 已明确把"条件"作为输入的工作

- **TimeXer**（Yuxuan Wang et al., NeurIPS 2024, arXiv:2402.19072）：正文（NeurIPS 版）以 patch-wise self-attention + variate-wise cross-attention 处理内生/外生变量，global endogenous token 桥接外生信息。**意义**：把工况/协变量作为外生 token 输入跨变量交互已是成熟做法；**但 TimeXer 不改变滞后选择，也不产生逐对滞后路由**。判定：**相关但不同（对"工况作输入"属 A 级成熟）**。
- **Context Matters / ContextFormer**（arXiv:2410.12672）：分类+连续 metadata 拼成条件向量经 cross-attention 调制预测。判定：条件化输入成熟（B 级片段）。
- **Condition-Aware AI for Predictive Maintenance: Dual-Attention CNN–GRU**（Aasi et al., Expert Systems with Applications, 2026, PII S0957417426004951）：摘要级——**对 C-MAPSS 的 op1/op2/op3 做 k-means++ 聚类得到 regime，per-regime scaling + 双注意力**，FD002 上报告 RMSE≈13.36。**意义：在 C-MAPSS 上"用 operating settings 作工况条件"已有先例**（C 级，正文被 Elsevier 拦截）。判定：**部分重合（工况条件化 PdM 已有；无 lag）**。

### 2.2 依赖结构随 regime/状态变化的工作

- **KI-TSF**（Zijie Pan, Dongjin Song, *Knowledge Informed Time Series Forecasting*, KDD 2025 论文 PDF）：正文片段——"different regimes of MTS data are often continuously collected under different operational logic"（图 1），Dynamic Graph Inference Module 在**输入窗口粒度**建动态图，配合结构知识的记忆回放防遗忘。**证据 B 级**。判定：**部分重合（regime 粒度动态依赖 + 明确谈工况运行逻辑；无 lag）**。
- **RRGL: Regime-aware Residual Graph Learning**（Expert Systems with Applications, 2026, PII S0957417426023468）：摘要级——"Regime-aware graph banks capture state-dependent dependencies"，7 个长程预测基准。C 级。判定：部分重合。
- **Dynamic TMoE**（Jiawen Zhu et al., ICML 2026, arXiv:2605.20678, 代码 github.com/andone-07/Dynamic-TMoE）：MMD 检测漂移 + 动态实例化/剪枝专家 + 时序记忆路由器。判定：相关但不同（regime 路由成熟，无跨传感器 lag）。
- **Fuzzy-MoE**（arXiv:2608.20761）：模糊路由器做 regime 条件化专家路由，逐变量分派。判定：相关但不同。
- **L-Drive**（Fan Zhang et al., ICML 2026 poster, arXiv:2605.17730）：Latent-Context + gating 调制增量表示以响应 regime 切换。判定：相关但不同。
- **DisenTS**（TPAMI 2024，作者主页 PDF）：Forecaster Aware Gate 依据"forecasters' states + input series' characteristics"生成路由信号 β_i=FAG(x_i,Γ)。判定：相关但不同（路由门控的对象是骨干模型集合，不是滞后位置）。
- **CaReBayes**（PLOS One 2026, DOI 10.1371/journal.pone.0356474）：regime 依赖因果结构发现 + regime 条件贝叶斯预测。判定：相关但不同（因果发现任务，非预测路由）。

### 2.3 LIFT 的状态条件化（对 V3 的关键挤压，重复登记）

LIFT 式 (7)(9) 已实现"由历史窗口估计的状态概率 → 调制滞后相关频域滤波"。**V3 与它的差异只剩三点**：条件来源（外部工况变量 vs 内部隐状态）、调制位置（聚合路由权重 vs 频域精修滤波）、领域（工业三数据集 vs 通用基准）。**这三点必须靠实验区分，不能靠检索宣称。**

### 2.4 B 组检索否定性结论（在 0.4 局限内）

以 `condition-aware lag`、`state-dependent lag routing`、`working condition cross-variable lag attention`、`operating regime adaptive sensor graph lag` 等至少 7 组措辞检索，**未发现**同时满足以下三条的论文正文：(i) 显式工况/协变量向量作为输入；(ii) 该向量调制**逐源-目标滞后位置**的路由权重；(iii) 工业多传感器数据验证。→ 记为"**本次检索范围内未发现**"，不是首创证明。

---

## 3. C 组：工业多传感器/过程工业/设备预测（逐篇证据）

- **Lag-Specific Transfer Entropy (LSTE)**（Sensors 2025, 25(13), 3980, MDPI，DOI 可解析）：正文片段——扫描候选滞后选最大传递熵，同时给出因果强度与**时延估计**，在 **TEP**、三相分离器、高炉线上验证；§4.2.1 TEP 扰动传播描述。**判定：相关但不同**（诊断/根因任务的滞后估计，非预测模型；说明"TEP 上的滞后分析"已有先例）。
- **Data-driven sensor delay estimation ... using multivariate projection methods**（Chemometrics and Intelligent Laboratory Systems, Vol. 246, 2024, art. 105090, PII S0169743924000303）：正文片段——工业产线传感器时延估计与校正、PLS-SEQ 方法，两处工业设施实测。**判定：相关但不同**（经典统计预处理，非神经模型；证明"工业传感器滞后"是公认问题）。
- **Variable time delay estimation in continuous industrial processes**（Cattaldo, Ferrer, Måge, Chemometrics 2024, DOI: 10.1016/j.chemolab.2024.105082）：综述 + 仿真 + 工业实例，指出时延随处理速度变化（**时变滞后在连续工业过程中真实存在**）。判定：相关但不同（方法学综述，可作动机引用）。
- **DSPR: Dual-Stream Physics-Residual Networks ...**（arXiv:2604.07393v3）：正文片段——工业时序预测，关键词含 regime adaptation、dynamic graph learning，明确讨论"variable transport lags driven by flue gas velocity"，使用 **TEP fault-free 训练分区、3 分钟采样**（§4.1.1）+ SCR + 回转窑。判定：**部分重合**（工业 + 工况/regime + 提及可变传输滞后；滞后机制为物理先验而非学习路由）。
- **Machine RUL Prediction via an Attention-Based Deep Learning Approach**（IEEE Trans. Industrial Electronics, DOI: 10.1109/TIE.2020.2972443）：正文片段——FD004 上"**we treat operating conditions as measurement signals**"（op 设置与传感器一同输入），14 个传感器。判定：**对"FD004 工况作历史可得输入"属 A 级先例**（无 lag、无动态关系）。
- **Predicting machine failures from multivariate time series**（Vago, Forbicini, Fraternali, arXiv:2402.17804）：窗口尺寸/预测界标对故障预测影响的实证。判定：相关但不同（任务与协议参考）。
- **Knowledge-Assisted Multi-Graph Dependency Learning for MTAD**（arXiv:2607.15799）：三张图（无先验/传感器组知识/流程知识）+ GAT + TCN 的预测式异常检测，训练数据限定"normal operating conditions"。判定：相关但不同（工业多传感器动态依赖，含过程知识，无显式 lag）。

---

## 4. 四级创新判定（按部件）

| 部件/候选定义 | 四级判定 | 主要依据（最近邻） | 允许的表述 |
|---|---|---|---|
| V1 固定滞后跨传感器交互 | **A（已有成熟方法）** | LIFT 式 (5) 平移对齐；MRLCD-A 对齐；Autoformer 滞后复用（转述级）；DeltaLag 固定滞后基线 Lag1Net | 只能作为基线，不得称创新 |
| V2 可学习滞后路由 | **A（一般 MTS）/ B（工业应用）** | DeltaLag 式 (6)(7)(9)(11)(14)（逐对可学习滞后+TopK+聚合，ICAIF 2025）；LIFT 式 (3)(4)（逐对选择，非参数）；VCformer 式 (4)(5)（可学习聚合）；DeformTime/TiVaT 可学习偏移 | 机制层面不得称"首次"；在工业三数据集上的适配可作工程贡献并明示 |
| V3 工况条件化可学习滞后路由 | **C（尚未找到完全相同方法，证据不足）** | LIFT 式 (7)(9)（状态→调制滞后使用）+ TimeXer（外生条件输入）+ Condition-Aware CNN–GRU（op1–op3 聚类 regime）+ DeltaLag（逐对滞后）**各自占据一块，未见三者合一** | 只能写成"研究假设/候选方向"，禁止"首次提出"；需实验证明其优于容量匹配基线 |
| V4 稀疏/低秩工况条件化滞后图 | **C（依赖 V3 成立）** | 稀疏本身有先例：DeltaLag TopK、LIFT TopK（式 4）、TimeFilter 过滤、Top-k 图先验 | 新颖性完全取决于 V3；V3 不成立则 V4 无独立价值 |

**核心结论**：该方向的可发表空间不在"是否显式建模滞后"（已被占据），而在**"外部可得工况变量如何调制滞后路由，且在工业三数据集上是否真的优于同等容量的无滞后/无工况基线"**。

---

## 5. 最近邻方法结构化比较表（9.2，共 33 条）

> 说明："是否显式 lag"只有在正文/公式/结构图有依据才填"是"；否则"未证实"。D 组条目由并行核验任务提供（标注"D 组"），证据等级逐条给出。

**列字段（12 个）**：① 方法 ② 年份/出处 ③ 任务与数据域 ④ 是否工业数据 ⑤ 是否多传感器/跨变量 ⑥ 显式 lag（无正文依据一律"未证实"）⑦ 逐传感器可变 lag ⑧ 工况条件化 ⑨ 跨传感器路由 ⑩ 动态关系/图 ⑪ 公开代码 ⑫ 与目标方向的关键差异 + 证据等级。

| # | 方法 | 年份/出处 | 任务/数据域 | 工业 | 多传感 | 显式 lag | 逐传感可变 lag | 工况条件化 | 跨传感路由 | 动态关系 | 公开代码 | 关键差异 / 证据等级 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **LIFT** | ICLR 2024, arXiv:2401.17548 | 通用 MTS 预测（Weather/Elec/Traffic 等 6 套） | 否 | 是 | **是**（式 3 argmax 互相关） | **是**（式 4 逐目标 TopK 源与滞后） | **部分**（式 7/9 内部状态调制，非外部工况） | 是 | 隐式（逐时刻 δ） | github.com/SJTU-DMTai/LIFT、SJTU-Quant/LIFT（两处不一致，可运行性未核验） | 非参数 argmax 估计、频域精修而非聚合路由、无工业数据 / **A** |
| 2 | **VCformer** | IJCAI 2024, arXiv:2405.11470 | 通用 MTS 预测（ETT/Weather 等 8 套） | 否 | 是 | **是**（式 4 roll τ∈[1,T]） | 否（全局 λ 聚合，式 5） | 否 | 是 | 否 | github.com/CSyyn/VCformer | 聚合全部滞后而非选择、λ 非逐对非逐输入 / **A** |
| 3 | **DeltaLag** | ICAIF '25, DOI 10.1145/3768292.3770421 | 金融收益预测 | 否 | 是 | **是**（式 6–7 滞后维打分） | **是**（式 9/11 TopK 逐对滞后） | 否（市场状态仅叙述） | 是 | **是**（动态逐对滞后） | arXiv:2511.00390（代码未记录） | **V2 最强重合**；无外部工况条件、金融域 / **A** |
| 4 | **DeformTime** | TMLR 2025, arXiv:2406.07438 | 通用+外生 MTS 预测 | 否 | 是 | **是（隐式）**（可学习偏移） | 部分（逐变量 offsets） | 否 | 是 | 输入依赖 offsets | github.com/ClaudiaShu/DeformTime | 不枚举/不解释滞后位置、无工况 / **A/B** |
| 5 | **TiVaT** | arXiv:2410.01531；期刊版 Appl. Soft Comput. 2026（DOI 未证实） | 通用 MTS 预测 | 否 | 是 | **是（隐式）**（JA attention lagged points + deformable offsets） | 部分 | 否 | 是 | DTV 2D 采样图 | 有（arXiv 页链接） | 滞后以偏移/采样图隐式表达、无工况 / **B** |
| 6 | **Sensorformer** | arXiv:2501.03284（发表状态未证实） | 高维 MTS 预测 | 否 | 是 | **未证实**（动机提动态因果滞后，机制为全 patch 注意力） | 未证实 | 否 | 是 | 否 | 未证实 | 动机重合、实现隐式 / **B** |
| 7 | **MRLCD-A** | Inf. Process. Manage. 2025, DOI 10.1016/j.ipm.2025.104191 | 通用 MTS 预测插件 | 否 | 是 | **是**（滚动互相关检测 lag） | 未证实（正文未取得） | 否 | 否（预处理对齐） | 否 | 未证实 | 滞后在预处理对齐、非模型内可学习路由 / **C** |
| 8 | **MDA** | Appl. Soft Comput. 2024, DOI 10.1016/j.asoc.2024.112350 | 通用 MTS 预测 | 否 | 是 | **是**（lagged variable importance） | 未证实 | 否 | 是 | 随时间变化（摘要级） | 未证实 | 正文未取得，逐对选择能力未知 / **C** |
| 9 | **STLAformer** | 2025, DOI 10.1145/3759928.3759960 | 通用 MTS 预测 | 否 | 是 | **是**（segment-based temporal lag attention，摘要） | 未证实 | 否 | 是 | 未证实 | 未证实 | 仅摘要级 / **C** |
| 10 | **HMV-Former (LCCA)** | Front. Appl. Math. Stat. 2026, DOI 10.3389/fams.2026.1825555 | 通用 MTS 预测 | 否 | 是 | **部分**（lag-coupled channel attention，在通道嵌入层混合） | 否 | 否 | 是 | 动态亲和矩阵（Eq.13/14/18 片段） | 未证实 | 命名高度接近，列必查反例；滞后非时间位置 / **B** |
| 11 | **PDFormer** | AAAI 2024, DOI 10.1609/aaai.v37i4.25556 | 交通流预测 | 否（交通） | 是（路网节点） | **是**（delay-aware feature transformation） | 部分（逐节点传播时延） | 否 | 是 | **是**（短/长程双图掩码） | github.com/... （可查证） | 显式传播时延 + 动态关系，但无工况条件、非传感器对 / **A/B** |
| 12 | **ChannelTokenFormer** | ICLR 2026, arXiv:2506.08660 | 含工业（LNG 船舶）异步 MTS | **是** | 是 | **未证实**（异步采样周期 s_i，非 lag 枚举） | 逐通道采样周期 | 否 | 是（式 2/3 mask-guided attention） | 掩码依赖 | 有 | 异步统一处理但无显式滞后位置 / **A/B** |
| 13 | **Raindrop** | ICLR 2022, arXiv:2110.05357 | 不规则采样 MTS（医疗等） | 否 | 是 | 否（对齐不规则读数） | 否 | 否 | 是（传感器图消息传递） | 否 | github.com/imperial-qin/... | 不规则采样对齐，非滞后路由 / **B** |
| 14 | **GraFITi** | AAAI 2024, DOI 10.1609/aaai.v38i15.29560 | 不规则采样 MTS | 否 | 是 | 否 | 否 | 否 | 是（稀疏二部图） | 否 | github.com/microsoft/... | §6.5 自陈异步序列局限 / **A/B** |
| 15 | **Autoformer** | ICLR 2022, arXiv:2106.13008 | 通用长程预测 | 否 | 是 | **是（限定）**（式 5–7、图 2 time delay aggregation） | **未证实**（head 级共享 τ） | 否 | 是 | 否 | github.com/thuml/Autoformer | lag 是周期相位对齐非传感器传播延迟；V1 早期形态 / **A** |
| 16 | **iTransformer** | ICLR 2024, arXiv:2310.06625 | 通用长程预测 | 否 | 是 | **否**（正文承认 systematical time lags、图 17，但用整序列 token 规避） | 否 | 否 | 是 | 否 | github.com/thuml/iTransformer | "承认 lag 却不建模"的最强对照 / **A** |
| 17 | **PatchTST** | ICLR 2023, arXiv:2211.14730 | 通用长程预测 | 否 | 是 | 否 | 否（通道独立） | 否 | **否**（显式关掉） | 否 | github.com/yuqinie98/PatchTST | 跨传感器建模的反例/下界 / **A** |
| 18 | **TFT** | IJF 2021, DOI 10.1016/j.ijforecast.2021.03.012 | 含协变量预测 | 否 | 是 | 否（式 27 仅为事后解释） | 否（式 6 为同时刻变量权重） | **部分**（静态协变量 context，regime 为事后识别） | 是 | 否 | 第三方 pytorch-forecasting | 关系=同时刻变量权重、条件为时间不变静态 / **A** |
| 19 | **MTGNN** | KDD 2020, DOI 10.1145/3394486.3403118 | 交通图预测 | 否 | 是 | 否 | 否 | **否**（正文明确拒绝动态图） | 是（静态可学邻接） | **否** | github.com/nnzhan/MTGNN | 静态邻接 + 无时间偏移轴；图 6(b) 有领先现象未建模 / **A** |
| 20 | **Graph WaveNet** | IJCAI 2019, arXiv:1906.00121 | 交通图预测 | 否 | 是 | 否 | 否 | **否**（正文批评输入条件化图"unstable"） | 是（式 5 静态自适应邻接） | 否 | github.com/nnzhan/Graph-WaveNet | 可学≠动态；无时滞维度 / **A** |
| 21 | **StemGNN** | NeurIPS 2020, arXiv:2103.07719 | 通用 MTS | 否 | 是 | 否 | 否 | 部分（式 3 图随输入窗口变，无工况输入） | 是 | **是（数据依赖）** | github.com/microsoft/StemGNN | 动态性来自数据本身而非工况 / **A/B** |
| 22 | **DCRNN / AGCRN** | ICLR 2018 / NeurIPS 2020 | 交通图预测 | 否 | 是 | 否（扩散=同刻空间 hop） | 否（AGCRN 为节点专属参数） | 否 | 是 | 否（训练期静态式 6） | 各自官方仓库 | 空间扩散与时间递归解耦 / **A** |
| 23 | **TCN / WaveNet** | NeurIPS 2018 / 2016 | 通用/音频 | 否 | 部分 | 否（感受野隐式，源出 TDNN） | 否 | WaveNet **是**（式 2.5 外生 global/local conditioning） | 否 | 否 | locuslab/TCN（WaveNet 无官方代码） | 滞后=卷积位置权重，条件只调制输出 / **A** |
| 24 | **连续时间模型**（ODE-RNN/Latent ODE/mTAN） | NeurIPS 2019 / ICLR 2021 | 不规则采样 MTS | 否 | 是 | **未证实**（连续时间间隔≠跨传感器 lag） | mTAN **是（限定）**（式 2 逐变量时间核） | 否 | 否 | 否 | latent_ode 官方；mTAN 官方未证实 | 解决"何时观测"非"传播延迟" / **B** |
| 25 | **Granger/互相关/DTW** | Econometrica 1969 / IEEE TASSP 1978 | 统计基线 | 通用 | 是 | **是** | **是**（逐对统计量） | 否（时不变） | 否 | 否 | 大量实现 | 线性/时不变、无概率输出与校准 / **A** |
| 26 | **regime switching / MoE / gating**（Hamilton 1989、MoLE、FEDformer Eq.10、TFT 4.1） | 各出处 | 计量/长程预测 | 否 | 部分 | 否 | 否（MoLE 每通道不同权重≠lag） | **是**（本组核心） | 否 | 条件作用于权重 | MoLE 官方未证实 | 条件作用于网络权重而非"条件→关系图" / **A/B** |
| 27 | **CoST / TS2Vec / TimesURL** | ICLR 2022 / AAAI 2022 / AAAI 2024 | 自监督表征 | 否 | 是 | 否（**CoST 正文明言放弃直接建模 time-lagged relationship**） | 否 | 否 | 否 | 否 | salesforce/CoST、yuezhihan/ts2vec | 通用表征目标，不建时滞关系 / **A** |
| 28 | **TimeXer** | NeurIPS 2024, arXiv:2402.19072 | 含外生变量预测 | 否 | 是 | 否 | 否 | **是**（外生 token cross-attention） | 是 | 否 | 有 | 工况作输入已成熟，但不改变滞后选择 / **A** |
| 29 | **KI-TSF** | KDD 2025 | 通用 MTS | 否 | 是 | 否 | 否 | **部分**（regime 粒度动态图，正文谈 operational logic） | 是 | **是（输入窗口粒度）** | 有 | 无 lag / **B** |
| 30 | **LSTE / 工业传感器时延估计** | Sensors 2025 / Chemometrics 2024 | 工业根因诊断 / 时延估计 | **是** | 是 | **是**（扫描候选 lag 选最大 TE / PLS-SEQ） | 是（逐传感器对） | 否 | 否（非预测模型） | 否 | 未证实 | 经典统计，非端到端预测 / **B** |
| 31 | **DSPR** | arXiv:2604.07393 | 工业预测（TEP、回转窑） | **是** | 是 | **部分**（物理先验的 variable transport lags） | 未证实 | **是**（regime adaptation） | 是（动态图） | **是** | 未证实 | 滞后为物理先验而非学习路由 / **B** |
| 32 | **Condition-Aware Dual-Attention CNN-GRU** | Expert Syst. Appl. 2026, PII S0957417426004951 | C-MAPSS FD002 PdM | **是** | 是 | 否 | 否 | **是**（op1–op3 k-means 分 regime + per-regime scaling） | 是 | 部分 | 未证实 | **工况作条件在 C-MAPSS 已有先例**；无 lag / **C** |
| 33 | **RUL 注意力模型** | IEEE TIE 2020, DOI 10.1109/TIE.2020.2972443 | C-MAPSS RUL | **是** | 是 | 否 | 否 | **是**（"treat operating conditions as measurement signals"） | 是 | 否 | 未证实 | FD004 工况作输入的 A 级先例 / **B** |

**表格读法提示**：第 3、1、15、33、32 号是判定的关键锚点——它们分别占据了"逐对可学习滞后""状态调制滞后使用""固定滞后聚合""FD004 工况作输入""C-MAPSS 工况分 regime"四个位置；V3 要主张的差异只剩"**外部工况变量 → 直接调制滞后位置上的路由权重**"这一处接缝。

---

## 9.3 方法机制对照表（5 维度）

**图例**：● 是（正文/公式定位）｜◐ 部分/限定｜○ 否或仅隐式｜△ 仅摘要级（C 级证据）｜— 不适用。工业验证列写数据集。

| 方法 | ① lag（滞后建模） | ② cross-sensor routing（跨传感器路由） | ③ condition-aware（工况条件化） | ④ dynamic graph（关系随状态变化） | ⑤ industrial validation（工业数据验证） |
|---|---|---|---|---|---|
| LIFT (ICLR'24) | ● 式 3 argmax 滞后 | ● 逐目标 TopK 源+滞后 | ◐ 式 7/9 内部状态（非外部工况） | ● 逐时刻 δ 变化 | ○ 无 |
| VCformer (IJCAI'24) | ● 式 4 roll τ | ● 全局 λ 聚合 | ○ | ○ | ○ 无 |
| DeltaLag (ICAIF'25) | ● 式 6–11 逐对滞后 | ● TopK 稀疏路由 | ○（市场状态仅叙述） | ● 滞后随输入变 | ○ 金融 |
| DeformTime (TMLR'25) | ◐ 可学习偏移 | ● V-DAB 跨变量 | ○ | ◐ 输入依赖 offsets | ○ 无 |
| TiVaT ('24) | ◐ JA lagged points + offsets | ● | ○ | ◐ DTV 图 | ○ 无 |
| Sensorformer ('25) | △ 未证实（动机提 lag） | ● 跨 patch 注意力 | ○ | ○ | ○ 无 |
| MRLCD-A (IPM'25) | △ ● 滚动互相关检测 | ○（预处理对齐） | ○ | △ 窗口滚动 | ○ 无 |
| HMV-Former (Frontiers'26) | △ ◐ LCCA 通道层 | ● 动态亲和矩阵 | ○ | △ 动态 | ○ 无 |
| PDFormer (AAAI'24) | ● 传播时延 | ● 空间注意力 | ○ | ● 短/长程双图 | ○ 交通 |
| ChannelTokenFormer (ICLR'26) | ○ 未证实（异步采样周期） | ● mask-guided attention | ○ | ◐ 掩码依赖 | **● LNG 船舶工业数据集** |
| Raindrop (ICLR'22) | ○ | ● 传感器图消息传递 | ○ | ○ | ○ 医疗等 |
| GraFITi (AAAI'24) | ○ | ● 稀疏二部图 | ○ | ○ | ○ 无 |
| Autoformer (ICLR'22) | ● 周期性 top-k 滞后（式 5–7） | ● Roll+聚合 | ○ | ○ | ○ 无 |
| iTransformer (ICLR'24) | ○ 明确承认但规避 | ● 变元维整序列注意力 | ○ | ○ | ○ 无 |
| PatchTST (ICLR'23) | ○ | ○ 通道独立 | ○ | ○ | ○ 无 |
| TFT (IJF'21) | ○（式 27 事后解释） | ◐ 同时刻变量权重 | ◐ 静态协变量 context | ○ | ○ 无 |
| MTGNN (KDD'20) | ○ | ● 静态可学邻接 | ○ **正文明确拒绝** | ○ **正文明确拒绝** | ○ 交通 |
| Graph WaveNet (IJCAI'19) | ○ | ● 式 5 静态自适应 | ○ **正文批评条件化图** | ○ | ○ 交通 |
| StemGNN (NeurIPS'20) | ○ | ● | ○ | ● 式 3 输入依赖邻接 | ○ 无 |
| DCRNN / AGCRN | ○（扩散=同刻 hop） | ● | ○ | ○（训练期静态） | ○ 交通 |
| TCN / WaveNet | ○（感受野隐式） | ○ | WaveNet ● 外生条件注入 | ○ | ○ 无 |
| ODE-RNN / mTAN | ○／mTAN ◐ 逐变量时间核 | ○ | ○ | ○ | ○ 医疗等 |
| Granger/互相关/DTW | ● 逐对 lag | ○ | ○（时不变） | ○（需滚动重估） | ◐ 有工业应用（PLS-SEQ 等） |
| MoE / regime / gating | ○ | ◐ MoLE 每通道权重 | ● 本组核心 | ◐ 权重随状态 | ◐ FDM-MoE 等（另一任务） |
| CoST/TS2Vec/TimesURL | ○ **CoST 正文放弃** | ○ | ○ | ○ | ○ 无 |
| TimeXer (NeurIPS'24) | ○ | ● 外生 token cross-attn | ● 外生变量作输入 | ○ | ◐ 多域 |
| KI-TSF (KDD'25) | ○ | ● | ◐ regime 粒度 | ● 输入窗口粒度动态图 | ○ 无 |
| LSTE (Sensors'25) / 时延估计 | ● 扫描候选 lag | ○（非预测模型） | ○ | ○ | **● TEP、高炉、三相分离器** |
| DSPR ('26) | ◐ 物理先验传输时滞 | ● 动态图 | ● regime adaptation | ● | **● TEP、回转窑** |
| Condition-Aware CNN-GRU (ESWA'26) | ○ | ● | ● **op1–op3 分 regime** | ◐ | **● C-MAPSS FD002** |
| RUL 注意力 (TIE'20) | ○ | ● | ● FD004 工况作输入 | ○ | **● C-MAPSS FD004** |
| **目标 V3（待验证）** | 计划：逐对可学习滞后 | 计划：聚合处路由 | **计划：外部工况变量作条件** | 计划：随工况重算 | 计划：MetroPT-3 / FD004 / TEP |

**五维扫描结论**：①+② 已被 LIFT/DeltaLag/VCformer 占据；③+④ 已被 KI-TSF/regime-MoE/TFT 占据；⑤ 已被 DSPR/LSTE/条件化 PdM 占据。**没有任何一行同时具备 ①②③⑤**——最接近的 LIFT 缺 ⑤ 与外部条件，DeltaLag 缺 ③⑤，条件化 PdM 缺 ①。这一格是本方向唯一可能的立足点，但按 §0.4 的检索局限，它只能记作"本次检索范围内未发现"。

---

## 6. 目标方法的可行技术定义（V1–V4 逐条评估）

### V1 固定滞后跨传感器交互

1. **是否已有完全相同/高度相近方法**：是——LIFT 的 target-oriented shift（式 5）即"按固定/估计滞后取源序列"；MRLCD-A 做外部对齐；Autoformer 自相关复用滞后序列（转述级）。**A 级。**
2. **与最近邻的差异**：几乎无机制差异，仅工程位置（模型内 vs 预处理）。
3. **是否需要新理论**：否，纯工程实现。
4. **是否适用三类工业任务**：可用作任何任务的组件，但无创新价值。
5. **最小实现/最小消融**：按候选滞后集合 L 移位源特征后做跨传感器混合；消融 = 取消移位（L={0}）。
6. **目前不能声称**：不能声称任何新颖性；不能把学到的滞后称为物理传播延迟。

### V2 可学习滞后路由

1. **已有方法**：DeltaLag（逐对滞后打分 + TopK + softmax 聚合，式 6–14）**直接覆盖**；LIFT 式 (3)(4) 覆盖逐对选择（非参数）；VCformer 式 (5) 覆盖可学习滞后聚合。**A 级（机制）/B 级（工业场景空缺）。**
2. **差异**：候选差异点=端到端可学习滞后（vs LIFT 非参数）、工业数据、与预测损失联合训练；但 DeltaLag 已端到端、已消融固定 vs 可学习滞后。
3. **需要新理论**：否；需要的是工业数据上的对照证据。
4. **适用性**：三类任务均可实现（同一历史窗口内移位/选择）。
5. **最小实现**：对每个 (目标,源) 对在滞后维 softmax，Top-1 或加权聚合；**最小消融 = 与固定滞后、与全局滞后权重（VCformer 式 5 形态）对比**。
6. **不能声称**："首次提出可学习滞后路由"；不能报告注意力权重为因果。

### V3 工况条件化可学习滞后路由

1. **已有方法**：**未发现完全相同**；最近的三个近邻各占一块（LIFT 状态调制、TimeXer 外生条件、DeltaLag 逐对滞后），条件化 PdM 在 C-MAPSS 已用 op1–op3（ESWA 2026）。→ **C 级。**
2. **技术差异（须在论文中逐条对照）**：(a) 条件向量来自**历史可得的工况变量/窗口统计量**（FD004 的 op settings、MetroPT-3 的负载/启停状态、TEP 的操纵变量），非隐状态聚类；(b) 条件调制**滞后位置上的路由权重**（在聚合处），而非频域精修或外生 token 融合；(c) 逐传感器对的滞后选择与工况条件在**同一处**结合。
3. **需要新理论**：否；需要的是公平实现与容量匹配对照（见 §7），以及"条件是否真的改变路由"的可视化/探针。
4. **适用性**：MetroPT-3（需自造工况信号：压缩机启停 MPG/COMP、负载段）、FD004（op1–op3 直接可用）、TEP（操纵变量/扰动前后段，须先锁版本）。
5. **最小实现**：条件向量 c=MLP(window stats ⊕ 可得工况) → 生成逐对滞后偏置 b_{ij}(c) 加到滞后 logits 上；**最小消融 = (i) 去掉 c；(ii) c 置换（打乱样本间的 c）；(iii) 无滞后（V2-free）；(iv) 容量匹配无条件版。**
6. **不能声称**：不能写"首次提出"；不能称学到的滞后为真实物理延迟；不能用工况以外的未来信息/故障标签。

### V4 稀疏工况条件化滞后图

1. **已有方法**：稀疏/TopK 机制被 DeltaLag（式 9）、LIFT（式 4）、TimeFilter、Top-k 图先验占据；**V4 的全部新颖性都继承自 V3**。→ C 级。
2. **差异**：可主张的只有"稀疏结构 + 工况条件"的组合与低秩约束的专门设计（若实现）。
3. **需要新理论**：低秩/稀疏约束的推导可为工程，不需要新理论；但需要证明稀疏化不是单纯正则化收益（对照容量匹配 dense 版）。
4. **适用性**：同 V3。
5. **最小实现**：在 V3 的滞后 logits 上加 Gumbel-TopK 或 entmax；**最小消融 = dense vs top-k vs 稀疏熵正则，参数量对齐**。
6. **不能声称**：不能把稀疏图解释为传感器物理连接或因果结构。

---

## 7. 最小实验矩阵（不依赖未来信息）

**统一协议（所有臂共享）**：仅用历史窗口与历史可得工况变量；同一 chronological 划分（train/valid/test，比例见 §9.4）；归一化统计量只来自 train；同一训练预算（epochs/早停准则/超参搜索预算）与同一 seed 集（≥3 seeds 才能谈稳定性，单 seed 只作诊断）；指标在 test 上报告，**方向选择只允许看 valid**。

| # | 实验臂 | 输入是否仅历史 | 需额外标签 | 划分 | 主要指标 | 验证的科学问题 | 混杂因素 |
|---|---|---|---|---|---|---|---|
| 1 | V3C 当前统一基线（现行模型，无 lag 模块） | 是 | 否 | chronological | MAE/RMSE（总/逐传感器/逐 horizon/逐工况段） | 参照系：一切增益的底线 | 现行模型容量本身较大 |
| 2 | 参数量匹配的普通跨传感器交互（同一注入点、同等参数的 channel mixing/注意力，无滞后维） | 是 | 同上 | 同上 | 同上 + 参数量/训练时长/显存 | 增益是否只是"多了参数/多了一层交互" | 参数量对齐精度（±5% 内） |
| 3 | 固定滞后交互（V1：候选滞后集 {0..K} 移位后聚合） | 是 | 否 | 同上 | 同上 | 显式滞后本身是否有价值 | 滞后集先验选择偏差 |
| 4 | 可学习滞后交互（V2：逐对滞后 softmax，无工况条件） | 是 | 否 | 同上 | 同上 + 滞后分布可视化 | 可学习滞后 vs 固定滞后的差异 | 可能过拟合到训练期滞后模式 |
| 5 | 仅工况条件化交互（条件向量调制跨传感器权重，无滞后维） | 是（工况须历史可得） | 否 | 同上 | 同上 + 分工况段误差 | 工况条件本身的价值 | 工况变量与目标的高度共线（如 op 设置直接进输入） |
| 6 | **V3：工况条件化可学习滞后交互（完整）** | 是 | 否 | 同上 | 同上 + 路由热图（按工况分组） | 核心假设：条件改变滞后路由且带来增益 | 与 4/5 的差异可能极小 |
| 7 | 参数量匹配 MLP / channel mixer 对照（同输入、同输出，无跨传感器结构） | 是 | 否 | 同上 | 同上 + 参数量 | 任何跨传感器结构是否优于简单混合 | MLP 容量调参不公 |
| 8 | 去除跨传感器交互的退化版本（逐传感器独立预测） | 是 | 否 | 同上 | 同上 + 逐传感器方差 | 跨传感器信息的真实贡献 | channel-independent 基线本身很强（PatchTST 类） |
| 9 | 候选滞后数量敏感性（K ∈ {1,3,5,9,17}，臂 4/6 各扫一遍） | 是 | 否 | 同上（只在 valid 上选 K） | valid 曲线 + test 单点报告 | 滞后候选集的影响与饱和点 | 多重比较导致的 valid 过拟合 |
| 10 | 无跨传感器依赖的合成数据控制实验（构造 cross-channel lag=0 与 lag=常数 两组合成数据；对齐打乱的通道置换） | 是（合成标签由历史生成） | 合成目标 | 合成集内自建切分 | MAE + 滞后估计恢复度 | (a) 无跨传感器依赖时模型是否退化到基线（假阳性检查）；(b) 已知真实滞后时能否恢复 | 合成分布与真实工业数据的域差 |

**补充必查对照（由 §8 风险审查导出）**：
- **LIFT 式机制复现对照**（非参数滞后估计 + 平移聚合）：用于回答"V2/V3 相对最近邻的增量"；若工程量过大，至少实现其核心（argmax 互相关选源与滞后）作为对照臂。
- **工况条件置换检验**（arm 6 中打乱 c）：排除"条件只是额外噪声正则"的解释。

**指标（任务书 §7 要求逐项落实）**：总体 MAE/RMSE；逐传感器 MAE/RMSE；逐 horizon 误差；分工况/分 regime 误差；RUL 任务 raw MAE/RMSE/NASA score；故障任务事件级检测延迟、召回率、误报率（若纳入）；参数量、训练时间、推理时间、峰值显存。

**停止条件（见 §9.6）**：若 arm 6 在 valid 上相对 arm 4/5 的增益 < 1% 且在 3 seeds 上方向不一致 → 放弃 V3 主线。

---

## 8. 创新性风险审查（逐项标记）

### 8.1 可能重复的已有方法

- "显式滞后枚举/聚合" → **VCformer 式 (4)(5)**、Autoformer 滞后复用：重复，不能写。
- "逐源-目标可学习滞后 + 稀疏选择" → **DeltaLag 式 (6)–(14)**：重复，不能写。
- "逐目标选源与滞后" → **LIFT 式 (3)(4)**：重复，不能写。
- "状态/条件调制滞后信息的使用" → **LIFT 式 (7)(9)**：部分重复，写"差异在条件来源与调制位置"。
- "工况变量作为输入" → **TimeXer、RUL 注意力（TIE 2020, FD004 op 输入）**：重复，不能写。
- "C-MAPSS 上以 op settings 聚类 regime 做条件化" → **ESWA 2026 双注意力 CNN–GRU**：部分重复（C 级证据），须在论文中引用。
- "regime 粒度动态图" → KI-TSF、RRGL：重复，不能写。
- "滞后检测插件（互相关对齐）" → MRLCD-A、工业传感器时延估计、LSTE：重复，不能写。

### 8.2 仅属工程组合的部分

- 把 V2 的滞后模块挂到现有 V3C 编码器上、按数据集调 K：**工程组合**，除非证明优于容量匹配基线。
- FD004 的 op1–op3 直接送入条件向量：**工程组合**（先例见 TIE 2020、ESWA 2026）。
- 三数据集统一跑分：**实验覆盖度贡献**，不是方法创新。

### 8.3 需要新实验才能成立的主张

- "工况变化时最优滞后分布确实发生迁移"（需：分时段滞后估计对比 + 统计检验，仅用历史数据）。
- "条件化路由优于同容量无条件路由"（需 arm 4 vs 6 vs 容量匹配对照）。
- "显式滞后优于普通跨传感器交互"（需 arm 2 vs 3/4）。
- "在工业三数据集上有效"（需先解决 §9.4 的协议与缺失语义问题）。

### 8.4 目前不能写进论文的表述

- 任何"首次提出/据我们所知是第一"（检索范围见 0.4，不足以支撑）。
- "注意力权重揭示了传感器间因果/物理传播延迟"（最多称预测性依赖或模型路由）。
- "MetroPT-3/FD004/TEP 天然异步、存在真实缺失"——**三者单元格级均无原生缺失（UCI 明标 `Has Missing Values? No`、FD004 实测 0 NaN、TEP 为仿真完整数据）；MetroPT-3 存在 331 处记录级断档（37.9 天，机制未证实），须与"人为注入缺失"分开表述（详见 §9.4.1、§9.4.3-6）**。
- "FD004 可用于传感器间时滞分析"——**否：每行为一个 flight cycle 快照、无墙钟时间戳，亚循环时滞不可分析；FD004 只承担工况条件化验证角色（§9.4.1-2）**。
- "PHM 2012 航空发动机数据集"——**该官方数据集不存在**：IEEE PHM 2012 = FEMTO/PRONOSTIA 轴承；发动机数据应写 C-MAPSS FD 或 PHM08（§9.4.2）。
- 把人工注入缺失或人为移位当作"真实工业异步"。
- 任何未运行的数字：本报告未产生任何实验数值；论文中不得出现未溯源 run_id 的指标。
- 用测试集选择滞后数 K、工况分段或路由结构。

---

## 9.4 数据集与协议建议

> 本节数据集属性由并行核验任务联网取证，关键结论再经**本地原始文件实测复核**（仓库 `dataset/CMAPSSData`，2026-10-09）。证据标注沿用 `[论文正文证实] / [官方页面证实] / [官方文件实证] / [二手来源] / [未证实]`，并映射到 §0.3 的 A–D 等级。

### 9.4.1 三主数据集核验结论

#### （1）MetroPT-3 —— **建议继续使用，作为唯一主实验**

| 项目 | 核验结论 | 证据 |
|---|---|---|
| 来源 | UCI Dataset 791，DOI `10.24432/C5VW3R`；正典引文为 **Davari et al., DSAA 2021, DOI 10.1109/DSAA53316.2021.9564181** 与 **Davari et al., ACM SAC 2024, DOI 10.1145/3605098.3635906** | `[官方页面证实]` |
| ⚠ 引文纠错 | **"Pietro et al., IEEE Access 2023" 未证实**——多轮检索未发现该文；仓库内也无此引文（已 grep 核对）。勿写入论文 | `[未证实]` |
| ⚠ 易混淆 | Veloso et al., *Scientific Data* 2022（DOI 10.1038/s41597-022-01877-3）描述的是 MetroPT / MetroPT-2（2022 年、3 次灾难性故障），**不是 MetroPT-3（2020 年数据）** | `[论文正文证实]` |
| 变量 | 15 信号 = 7 模拟（TP2/TP3/H1/DV_pressure/Reservoirs/Oil_temperature/Motor_current）+ 8 数字（COMP/DV_eletric/Towers/MPG/LPS/Pressure_switch/Oil_level/Caudal_impulses）+ index + timestamp | `[官方页面证实]` |
| 规模 | 1,516,948 行（UCI 与本地一致；Kaggle 镜像写 15,169,480 为 10 倍错标） | `[官方文件实证]` |
| ⚠ 采样间隔自相矛盾 | 官方既写 "logged at 1Hz" 又写 "collected at 0.1Hz"；实测 index 步长恒 10、时间戳差中位数 10 s（9–13 s 抖动）→ **有效采样 0.1 Hz（10 s），1 Hz 降采样而来** | `[官方页面证实 + 官方文件实证]` |
| 缺失语义 | UCI 明确 `Has Missing Values? No`、逐列 Missing 均为 no；实测单元格 NaN=0。**但记录级断档真实存在：331 处 >30 s 断档、累计 37.9 天（占 213 天的 17.8%），最长 172,918 s**；官方未说明机制 → **机制未证实** | `[官方页面证实 + 官方文件实证]` |
| 故障事件 | **仅 4 条故障报告**（全为 Air leak / High stress），官方明示 "**dataset is unlabeled**" | `[官方页面证实 + 官方文件实证]` |
| 工况变量 | **无显式 setpoint 列**；可用作工况的是 8 个数字状态量，以及 Motor_current 分档（0A 关机 / 4A 卸载 / 7A 带载 / 9A 启动）派生的运行状态 | `[官方页面证实]` |
| regime 丰富度 | 关机–卸载–带载循环、双塔干燥切换、APU 压力建立/消耗、4 段漏气期、37.9 天记录断档 = **丰富 regime 切换** | `[官方文件实证]` |
| 时滞窗口 | 10 s × 151.7 万点，24 h = 8,640 点 → **分钟–小时级传感器时滞分析极佳** | `[官方文件实证]` |
| 官方划分 | "**The first month can be used for training the model, and the rest is for testing**"（时间顺序，前 1 月训、余测） | `[官方页面证实]` |
| 泄漏风险 | ① 事件极稀疏（6 个月 4 起，且按官方划分训练月(2 月)内 0 起故障）② 滑窗重叠 ③ 故障报告边界粗糙 ④ **断档两侧不可拼接成窗** | `[综合]` |
| 分工况报告 | **非常适合**（关机/卸载/带载/塔切换/漏气期分层） | 综合判断 |

**建议**：继续使用。主任务用**预测**（点预测为主，与本报告 §9.6 一致），不把稀疏故障标签当训练目标；工况条件向量由**历史可得**的数字状态量 + Motor_current 分档构造；切分遵循官方"前 1 月"或仓库既定 chronological 协议，**须跳过记录断档**（窗口不得跨越 >10 s 的空洞）。任何"缺失鲁棒"主张必须标注"人为注入 + 附带真实记录断档"两类来源分开统计。

#### （2）C-MAPSS FD004 —— **建议保留，但用途受限：只作工况条件化的验证载体，不作时滞分析主载体**

| 项目 | 核验结论 | 证据 |
|---|---|---|
| 来源 | NASA PCoE / Open Data Portal；原始论文 Saxena, Goebel, Simon, Eklund, PHM08 2008（PDF 随 `CMAPSSData.zip` 分发） | `[官方页面证实 + 论文正文证实]` |
| 变量 | 26 列 = unit# + time(cycles) + **3 operational settings** + **21 sensor measurements** | `[官方页面证实]` |
| ⚠ **采样不规则（关键）** | 每行 = 一个 flight cycle 快照，**无墙钟时间戳**，cycle 时长不固定 → **亚循环传感器时滞不可分析** | `[官方页面证实 + 推断]` |
| **248/249 核验（已本地实测）** | `train_FD004.txt` **61,249 行 / 249 台**；`test_FD004.txt` **41,214 行 / 248 台**；`RUL_FD004.txt` **248 行**（每测试机一行，独立反证）。**NASA readme 写 "Train 248 / Test 249" 与数据不符，属官方文档笔误**（同一 readme 内 FD001/2/3 均正确，仅 FD004 互换；且把第 26 列误写为 sensor 26，实际只有 21 个传感器）。旁证：Wu et al.（NASA 合作）"249 training units and 248 test units"；PMC10439180 Table 3 | **`[官方文件实证，本地复测 + 官方文件 + 论文正文三方一致]`** |
| ⚠ **仓库文档冲突** | `plan/2026-10-05-dataset-specific-ch3-ch5-experiment-plan.md:18` 写 "FD004 contains **248** training engines, **249** test engines"，**与实测相反**。按 AGENTS.md「文档与可执行内容冲突时以可执行内容为准并同步修正文档」，**该行待修正**（本轮未改，见 §9.7 末尾待办） | `[官方文件实证]` |
| 工况变量 | **cols 3–5 = operational settings**（altitude 0–42K ft、Mach 0–0.84、TRA 20–100，Saxena 2008 正文给量纲）；cols 6–26 = 测量量 | `[论文正文证实 + 官方页面证实]` |
| regime 结构 | setting1≈6 档、setting2≈4 档、setting3∈{60,100} → **6 个 operating regimes**；**249 台训练发动机每台都遍历全部 6 个 regime**（轨迹内反复切换） | `[官方文件实证]` |
| 缺失 | 全部 26 列合法数值，NaN/空值 = 0；官方仅称 "contaminated with sensor noise"（噪声≠缺失） | `[官方文件实证]` |
| 序列长度 | train 每台 128–543 cycles（均值 246）；test 19–486；RUL 6–195 | `[官方文件实证]` |
| 官方划分 | 官方 train/test 文件按 **unit** 切分（非随机）；RUL 真值单独提供 | `[官方页面证实]` |
| 泄漏风险 | ① 同机相邻窗口大量共享 cycle ② **piecewise-linear RUL 截断阈值（125/130）是社区惯例、非官方规定 → 来源未证实** ③ test RUL 若在调参中反复使用即泄漏 ④ 跨 regime 归一化用全数据会从 test 泄漏 | `[综合]` |
| 分工况报告 | **非常适合**：6 regimes × 2 fault modes 是标准口径 | 综合判断 |

**建议**：继续使用，但**只承担两个角色**——(a) **工况条件化的验证**（op1–op3 是本课题最干净的历史可得工况变量，且 6 regime 在轨迹内切换，正好构成"工况变化"场景）；(b) RUL 任务的跨域佐证。**不得**把它当作传感器时滞分析的主载体（无墙钟时间戳），也不得把 `unit`/故障模式当工况条件。引用时写 **249 train / 248 test** 并注明 NASA readme 数字互换。

#### （3）TEP —— **建议保留，但必须先锁定版本；推荐 Rieth et al. 2017 版**

| 版本 | 分发 | 规模 | 采样 | 故障注入 |
|---|---|---|---|---|
| (A) Downs & Vogel 1993 / Braatz–Ricker 档案 | 仓库本地 `dataset/tennessee-eastman-profBraatz/`（`teprob.f`、`d00.dat…d21.dat`）；UW Ricker 档案 | 52 列 = 41 XMEAS + 11 XMV（仿真器内部另有 XMV(12)） | **180 s（3 min）**，README 原文 "all data are recorded every 180 seconds" | 21 个 IDV（IDV1–15 有定义，16–20 官方标 Unknown）；**测试文件注入点实测在第 161 个样本**（前 160 ≈ 8 h 正常）；**故障训练文件(480 行)注入时刻未证实** |
| **(B) Rieth et al. 2017（推荐）** | Harvard Dataverse，DOI `10.7910/DVN/6C3JR1` | 55 列 = faultNumber + simulationRun + sample + 52 变量；**simulationRun 1–500，train/test 种子不重叠** | "sampled every 3 minutes"；训练 500 样本（25 h）、测试 960（48 h） | "faults were introduced 1 hour and 8 hours into the Faulty Training and Faulty Testing datasets"（训练第 20 样本、测试第 160 样本后注入） |
| (C) "TensorFlow Datasets 版" | **未证实存在**——TFDS 目录页 `tensorflow.org/datasets/catalog/tep` 返回 404，代码库无对应条目 | — | — | **论文中不得引用"TFDS 版 TEP"** |
| (D) Eastman 仿真器本体 | Fortran 源经 UW/Braatz 档案分发；Eastman 官网独立下载页**未证实** | 同 (A) | 可自设 | 使用者自行生成 |

- **采样间隔结论**：**3 min（180 s）是所有主流版本一致口径，5 min 无任何官方来源**。
- 工况变量：**XMV(1–11/12) = 操纵/设定量**，XMEAS(1–41) = 测量量（其中 39–41 为组分分析采样型）。
- regime：基线单一稳态；IDV1–7 step、IDV8–12 random、IDV13 drift、IDV14–15 sticking **本身就是 regime 切换/故障注入**；另有 Ricker 1995 的 6 个最优稳态工况表。
- 缺失：仿真数据，**无缺失** `[官方文件实证]`。
- 任务：故障检测/诊断（21 类）、异常检测、过程监控——**不是 RUL**。
- 划分：按 simulation run 划分（(B) 版明确 500 种子、train/test 不重叠）→ "按轨迹划分 + 轨迹内时间顺序"。
- 泄漏风险：① 同 run 滑窗重叠 ② **若按行随机划分，同一 run 的相邻 3 min 样本跨集合 → 严重泄漏** ③ 故障后样本是否同时用于训练须写明口径。
- 分工况报告：适合按 fault type / step vs random 分层；**"工况维度"较窄**（基线单一稳态）。

**建议**：继续使用，**先锁定 Rieth 2017 版（DOI 可引用、注入时刻明确、按 run 划分防泄漏）**；若使用仓库本地 (A) 版，须注明其故障训练文件注入时刻未证实。TEP 承担**分钟级传播时滞 + 故障条件**的验证角色；若版本未锁定就跑，结论无法复现 → 列为停止条件 3。

### 9.4.2 候选数据集逐一评估（含不推荐标记）

| 数据集 | 变量 | 采样/规则性 | 真实缺失 | 工况变量 | regime | 任务 | 官方划分 | 泄漏风险 | 分工况报告 | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| **SWaT** | 51（25 传感 + 26 执行） | 1 s（官方页未写，来自论文）；连续 11 天 | 官方未声明机制；官方曾删除前 30 min 维护段 | 无 setpoint，26 执行器=控制量 | 弱–中（6 工艺段 + 41 攻击） | 异常/入侵检测 | 7 天正常训 / 4 天攻击测（按时间）+ 官方标签 | 按行随机划分跨事件；事件稀疏 41 起 | 适合（按工艺段） | **备选**（需申请、任务为检测非预测） |
| **WADI** | 123（论文写 127，以官方 123 为准） | 1 s（官方页未写）；16 天 | **官方声明：因 plant unstable 删除受影响读数 → 记录级空洞（机制部分未证实）** | 无 setpoint | 弱–中 | 异常/入侵检测 | 14 天正常 / 2 天攻击 + 官方 label 文件(-1/1) | 非随机删除可能与攻击相关 → 偏置；15 起攻击 | 适合 | **备选**（同上） |
| **PHM 2012** | — | — | — | — | — | — | — | — | — | **⚠ 不推荐且须改名**：**不存在官方名为"PHM 2012 航空发动机"的数据集**。IEEE PHM 2012 = **FEMTO/PRONOSTIA 轴承**（NASA PCoE 条目 10）；发动机数据应写 **C-MAPSS FD** 或 **PHM08 Challenge**（后者 true RUL 不公开、当前下载受限）。仓库内无此引文（已 grep） |
| **XJTU-SY** | 2（水平/垂直振动） | 25.6 kHz、1.28 s/帧、**帧周期 1 min（帧间不连续）** | 官方未声明 | 无 setpoint；工况由轴承组固定（3 组转速×载荷） | 组间 3 工况、组内恒定 | RUL / 诊断 | **官方无划分协议（未证实）** | 同轴轨迹内随机划分=严重泄漏 | 适合（按 3 组） | **不推荐作主载体**：仅 2 通道且同步采集，**无传感器间传播时滞的物理意义** |
| **IMS** | 4（Set2/3）/8（Set1） | 20 kHz、1 s 快照 / 10 min（Set1 前 43 文件 5 min） | 官方未声明；采集间隔不一致 | **无（2000 rpm、6000 lb 恒定）** | **无** | 早期检测 / RUL | 官方无划分（通行按试验轮次） | 轨迹内切窗；**无逐样本故障起点标签**；每轮 1–2 个失效 | **不适合（工况唯一）** | **不推荐**：既无工况变化，帧间隔 10 min 对时滞也过粗 |

### 9.4.3 协议建议与泄漏防护（跨数据集通用）

1. **切分**：一律 chronological（MetroPT-3 按官方"前 1 月"或仓库既定 50/20/30；FD004 按 unit；TEP 按 simulation run）。**禁止按行随机切分**（TEP 3 min 相邻样本、MetroPT-3 10 s 相邻样本跨集合即泄漏）。
2. **窗口构造**：MetroPT-3 窗口不得跨越记录断档；FD004 同一台发动机的相邻窗口视为相关样本，评测时按 unit 分组统计。
3. **归一化**：仅用 train 统计量（AGENTS.md 硬规则）；FD004 跨 regime 时须用**训练期每个 regime 内**的统计量或全局 train 统计量，不得用全数据。
4. **工况变量白名单**（本课题核心）：FD004 → cols 3–5（历史可得，按 cycle 读取）；MetroPT-3 → 8 个数字状态量 + Motor_current 分档（历史可得）；TEP → XMV 操纵量（历史可得）。**禁止**：unit 编号、fault number、故障标签、未来工况、test 集统计量。
5. **时滞分析载体分工**：分钟–小时级时滞 → MetroPT-3（10 s）与 TEP（3 min）；**FD004 不承担时滞主张**（无墙钟时间戳）。
6. **缺失主张**：三主数据集**单元格级均无原生缺失**（MetroPT-3 有记录级断档、WADI 有官方删除），因此任何"真实异步/真实缺失"表述必须区分「注入缺失」「记录断档」「官方删除」三类，并注明来源与占比。
7. **单 seed 仅作诊断**（AGENTS.md 稳定性护栏）；分工况分层报告的每一格须同时给出样本量，避免稀疏格（如 MetroPT-3 漏气期）被过度解读。

---

## 9.6 最小可行研究方案（MVP）

- **模型模块**：(i) 历史编码器（复用现有统一模型的逐传感器/逐 patch 表示，不改动其骨干）；(ii) 滞后路由头：逐 (目标,源) 对在 K 个候选滞后上打分；(iii) 工况条件向量 c（历史窗口统计量 ⊕ 数据集原生工况变量）→ 对滞后 logits 加性偏置；(iv) 聚合后注入既有预测头。
- **输入输出**：输入 = 历史窗口（多传感器）+ 历史可得工况（FD004: op1–op3；MetroPT-3: 由历史构造的启停/负载段；TEP: 操纵变量，版本待锁）；输出 = 未来 horizon 的逐传感器预测（点预测为主，概率预测不作本章主线）。
- **训练约束**：仅历史信息；无故障标签进入条件；归一化仅 train；相同划分/预算/seed；路由稀疏度只在 valid 调。
- **消融矩阵**：§7 的 10 臂 + LIFT 式对照 + 条件置换检验。
- **指标**：§7 所列全套，重点看分工况段误差与逐传感器误差。
- **图表清单**：
  1. 方法总览图（条件 → 滞后偏置 → 聚合 → 注入点）；
  2. 滞后分布热图（按工况分组，训练期 vs 测试期，仅历史数据计算）；
  3. 主结果表（arm 1–8，含参数量列）；
  4. 消融表（arm 4/5/6/9 + 置换检验）；
  5. 合成数据控制实验图（arm 10）；
  6. 逐传感器/逐 horizon 误差曲线；
  7. 训练-推理成本对比图。
- **停止条件**：
  1. arm 6 相对 arm 4 与 arm 5 的 valid 增益 < 1% 或 3 seeds 方向不一致 → 放弃 V3 主线，回退到"工业适配 + 实验覆盖"的 B 级贡献；
  2. 合成控制实验中 lag=0 场景仍报告显著"最优非零滞后" → 模型存在结构性假阳性，先修模型再谈结论；
  3. 任一数据集协议（§9.4）被判定为不可靠且无替代 → 该数据集退出，不得强行纳入。

---

## 9.7 参考文献与证据索引

> 每条给出：标准引用、DOI/稳定链接、代码链接（若有）、关键证据位置、证据等级、检索日期（均为 2026-10-09）。

> 全部条目检索/核验日期均为 **2026-10-09**。等级 = §0.3 的 A–D。"支持"列指该条目是否**直接支持本报告的判定结论**（是/部分/否，否 = 仅作背景）。

### 9.7.1 A 组（时滞感知）与直接最近邻

| ID | 标准引用（作者, 年份, 会议/期刊） | DOI / 稳定 URL | 公开代码 | 关键证据定位 | 等 | 支持 |
|---|---|---|---|---|---|---|
| R01 | Zhao & Shen, ICLR 2024 (LIFT) | arXiv:2401.17548；OpenReview JiTVtCUOpS | SJTU-DMTai/LIFT、SJTU-Quant/LIFT（不一致，未核可运行性） | §3.2 式(2)(3)(4)；§3.3 式(5)(7)(9)；图 5(b)(c) | A | 是 |
| R02 | Yang, Zhu & Chen, IJCAI 2024 (VCformer), pp.5337–5343 | arXiv:2405.11470 | CSyyn/VCformer | §3.1.1；式(4)(5)(6)；§3.5；表 2、表 3 | A | 是 |
| R03 | Zhou et al., ACM ICAIF '25 (DeltaLag), Singapore | DOI 10.1145/3768292.3770421；arXiv:2511.00390 | 未记录 | §4.1 式(6)(7)(9)(11)；§4.2 式(13)(14)；表 1；图 3 | A | 是 |
| R04 | Shu & Lampos, TMLR 2025 (DeformTime) | arXiv:2406.07438；OpenReview M62P7iOT7d | ClaudiaShu/DeformTime | §3.3 V-DAB、§3.4 T-DAB | A/B | 是 |
| R05 | TiVaT，arXiv 版 2024 | arXiv:2410.01531；期刊版 PII S1568494626004278（**DOI 未证实**） | 有（arXiv 页链接） | §3.2 JA attention lagged points | B | 部分 |
| R06 | Qin et al., arXiv 2025 (Sensorformer)（**发表状态未证实**） | arXiv:2501.03284 | 未证实 | 摘要、§3.3 Sensor Attention Block | B | 部分（仅动机） |
| R07 | Sun et al., Inf. Process. Manage. 2025 (MRLCD-A) | DOI 10.1016/j.ipm.2025.104191 | 未证实 | 仅摘要（Elsevier 拦截） | C | 部分 |
| R08 | Almaghrabi, Rana & Hamilton, Appl. Soft Comput. 2024 (MDA) | DOI 10.1016/j.asoc.2024.112350 | 未证实 | 仅摘要 | C | 部分 |
| R09 | Zhao & Sun, 2025 (STLAformer) | DOI 10.1145/3759928.3759960（会议全称未核验完整） | 未证实 | 仅摘要 | C | 部分 |
| R10 | HMV-Former, Front. Appl. Math. Stat. 2026 | DOI 10.3389/fams.2026.1825555 | 未证实 | §3.3.2 LCCA；式(13)(14)(18) 片段 | B | 是（必查反例） |
| R11 | Jiang et al., AAAI 2024 (PDFormer) | DOI 10.1609/aaai.v37i4.25556；arXiv:2301.07945 | 有 | Delay-aware Feature Transformation；短/长程双掩码 | A/B | 部分 |
| R12 | ChannelTokenFormer, ICLR 2026 | arXiv:2506.08660 | 有 | §3 异步采样周期 s_i；§4 式(2)(3) | A/B | 部分 |
| R13 | Zhang et al., ICLR 2022 (Raindrop) | arXiv:2110.05357 (v2) | 有 | 摘要（传感器图 + 消息传递） | B | 否（背景） |
| R14 | GraFITi, AAAI 2024 | DOI 10.1609/aaai.v38i15.29560 | 有 | §6.5 异步序列局限 | A/B | 部分 |
| R15 | Wu, Xu, Wang & Long, ICLR 2022 (Autoformer) | arXiv:2106.13008 | thuml/Autoformer | §3.2 式(5)(6)(7)、图 2 time delay aggregation | A | 是（V1 早期形态） |

### 9.7.2 B 组（工况/状态条件化）

| ID | 标准引用 | DOI / 稳定 URL | 代码 | 关键证据定位 | 等 | 支持 |
|---|---|---|---|---|---|---|
| R16 | Wang et al., NeurIPS 2024 (TimeXer) | arXiv:2402.19072；NeurIPS 正式 PDF | 有 | 变量级 cross-attention + global endogenous token | A | 是（工况作输入已成熟） |
| R17 | Context Matters / ContextFormer, 2024 | arXiv:2410.12672 | 未证实 | §条件化问题定义（分类+连续 metadata 条件向量） | B | 部分 |
| R18 | Aasi et al., Expert Syst. Appl. 2026 (Condition-Aware Dual-Attention CNN-GRU) | PII S0957417426004951 | 未证实 | 仅摘要（FD002 上 op1–op3 k-means 分 regime） | C | 是（工况条件先例） |
| R19 | Pan & Song, KDD 2025 (KI-TSF) | KDD'25 论文 PDF | 有 | Dynamic Graph Inference；"different regimes ... operational logic" 图 1 | B | 是 |
| R20 | Regime-aware Residual Graph Learning (RRGL), Expert Syst. Appl. 2026 | PII S0957417426023468 | 未证实 | 仅摘要 | C | 部分 |
| R21 | Zhu et al., ICML 2026 (Dynamic TMoE) | arXiv:2605.20678 | andone-07/Dynamic-TMoE | 摘要（MMD 漂移检测 + 动态专家） | C | 否（背景） |
| R22 | Fuzzy-MoE, 2026 | arXiv:2608.20761 | 未证实 | 摘要 | C | 否（背景） |
| R23 | Zhang et al., ICML 2026 poster (L-Drive) | arXiv:2605.17730 | 未证实 | 摘要（latent context gating） | C | 否（背景） |
| R24 | DisenTS, IEEE TPAMI 2024 | 作者主页 PDF | 未证实 | FAG 模块 β_i=FAG(x_i,Γ) | B | 否（背景） |
| R25 | Mossop & Choudhury, PLOS One 2026 (CaReBayes) | DOI 10.1371/journal.pone.0356474 | 未证实 | 摘要+正文片段 | C | 否（背景） |

### 9.7.3 C 组（工业多传感器 / 时延）

| ID | 标准引用 | DOI / 稳定 URL | 关键证据定位 | 等 | 支持 |
|---|---|---|---|---|---|
| R26 | Lag-Specific Transfer Entropy (LSTE), Sensors 2025, 25(13):3980 | MDPI（可解析） | §4.2.1 TEP 扰动传播；滞后扫描选最大 TE | B | 是（TEP 滞后分析先例） |
| R27 | Data-driven sensor delay estimation ... multivariate projection methods, Chemometrics, Vol.246 (2024), art.105090 | PII S0169743924000303 | 摘要+正文片段（两处工业设施实测） | B | 是（工业滞后问题公认） |
| R28 | Cattaldo, Ferrer & Måge, Chemometrics 2024 (Variable time delay estimation ...) | DOI 10.1016/j.chemolab.2024.105082 | 综述+仿真+工业实例（时延随处理速度变化） | B | 是（时变滞后真实存在） |
| R29 | DSPR: Dual-Stream Physics-Residual Networks, 2026 | arXiv:2604.07393 (v3) | §4.1.1 TEP fault-free 3 min 分区；关键词含 regime adaptation / dynamic graph；"variable transport lags" | B | 是（部分重合） |
| R30 | Machine RUL Prediction via an Attention-Based Deep Learning Approach, IEEE TIE 2020 | DOI 10.1109/TIE.2020.2972443 | "we treat operating conditions as measurement signals"（FD004，14 传感器） | B | 是（FD004 工况作输入先例） |
| R31 | Vago, Forbicini & Fraternali, 2024 (Predicting machine failures ...) | arXiv:2402.17804 | 窗口尺寸/预测界标影响实证 | B/C | 否（协议参考） |
| R32 | Knowledge-Assisted Multi-Graph Dependency Learning for MTAD, 2026 | arXiv:2607.15799 | 三图结构 + GAT + TCN；训练限于 normal operating conditions | C | 否（背景） |

### 9.7.4 D 组（邻近方向，由并行核验任务取证）

| ID | 标准引用 | DOI / 稳定 URL | 关键证据定位 | 等 | 支持 |
|---|---|---|---|---|---|
| R33 | Lim et al., Int. J. Forecasting 37(4):1748–1764, 2021 (TFT) | DOI 10.1016/j.ijforecast.2021.03.012；arXiv:1912.09363 | §7.2 + 式(14)(19)(27) 仅为事后解释；式(6) 变量选择；§4.3 静态 context | A | 是 |
| R34 | Zhou et al., AAAI 2021 (Informer) | DOI 10.1609/aaai.v35i12.17325；arXiv:2012.07436 | §3 ProbSparse（`lag/delay` 零命中） | C/B | 否 |
| R35 | Zhou, Ma, Wang & Long, ICML 2022 (FEDformer) | arXiv:2201.12740 | §3.2/3.3；§3.4 式(10) MoE 仅用于趋势分解 | B | 否 |
| R36 | Nie et al., ICLR 2023 (PatchTST) | arXiv:2211.14730 | 摘要+§1 channel-independence 原文 | A | 是（反例/下界） |
| R37 | Wu et al., ICLR 2023 (TimesNet) | arXiv:2210.02186 | 式(1)(2) Period；§3.1/3.2 共享 inception | B/C | 否 |
| R38 | Liu, Wang & Long, ICLR 2024 (iTransformer) | arXiv:2310.06625 | §2/3.1 "systematical time lags"；图 17；§4.3 消融 | A | 是（最强对照） |
| R39 | Bai, Kolter & Koltun, NeurIPS 2018 (TCN) | arXiv:1803.01271 | 图 1(a)；§3.2 TDNN 同源；§3.5 "no ... conditioning" | A | 是 |
| R40 | van den Oord et al., 2016 (WaveNet) | arXiv:**1609.03499**（注：1609.04632 为另一篇论文） | §2.5 global/local conditioning 激活式 | A | 否（背景） |
| R41 | Wu et al., KDD 2020 (MTGNN) | DOI 10.1145/3394486.3403118；arXiv:2005.11650 | §4.2 式(1)–(6) + "extremely hard to converge" 拒绝动态图；§4.4 式(11)；图 6(b) | A | 是 |
| R42 | Wu et al., IJCAI 2019 (Graph WaveNet) | arXiv:1906.00121 | 式(5) 静态自适应邻接；§2.1 批评输入条件化图 | A | 是 |
| R43 | Cao et al., NeurIPS 2020 (StemGNN) | arXiv:2103.07719 | 式(3) 输入依赖邻接 | A/B | 部分 |
| R44 | Li et al., ICLR 2018 (DCRNN) | arXiv:1707.01926 | §2.2 式(2) 扩散卷积；§4.1 "VAR ... lags set to 3" | A | 是 |
| R45 | Bai et al., NeurIPS 2020 (AGCRN) | DOI 10.5555/3495724.3497218 | 式(6) 训练期静态图；NAPL 节点专属参数 | A | 是 |
| R46 | Chen et al., NeurIPS 2018；Rubanova et al., NeurIPS 2019；Shukla & Marlin, ICLR 2021 (mTAN) | arXiv:1806.07366 / 1907.03907 / **2101.10318** | Latent ODE §3.5；mTAN 式(2) | B/C | 部分 |
| R47 | Granger, Econometrica 37(3):424–438, 1969 | DOI 10.2307/1912791 | 摘要 "Measures of causal lag and causal strength" | A | 是 |
| R48 | Sakoe & Chiba, IEEE TASSP 26(1):43–49, 1978 | DOI 10.1109/TASSP.1978.1163055 | 摘要（warping 消除 timing differences） | A | 否（背景） |
| R49 | Hamilton, Econometrica 57(2):357–384, 1989 | DOI 10.2307/1912559 | 摘要（离散 Markov 状态驱动 AR 参数） | A | 否（背景） |
| R50 | MoLE, 2023 | arXiv:2312.06786（**官方代码未证实**） | §4.1 timestamp embedding router；channel-wise weights；"regimes" | B | 部分 |
| R51 | Woo et al., ICLR 2022 (CoST) | arXiv:**2202.01575** | §2 明言放弃 "directly model the time-lagged relationship"；图 3b | A | 是（否定性证据） |
| R52 | Yue et al., AAAI 2022 (TS2Vec) | arXiv:2106.10466 | §2.3/2.4（`lag` 零命中；跨变量细节未证实） | C | 否 |
| R53 | Liu & Chen, AAAI 2024 (TimesURL) | DOI 10.1609/aaai.v38i12.29299；arXiv:2312.15709 | FTAug / double Universum | C | 否（背景） |
| R54 | Chen et al., NeurIPS 2024 (CCM) | arXiv:2404.01340 | 通道聚类插件 | B | 否（背景） |
| R55 | TimeFilter, ICML 2025 | PMLR 267:24893 | patch-specific 图过滤 | B | 否（背景） |
| R56 | MillGNN, 2025 | arXiv:2509.03852 | §3.3 FFT cross-correlation；§4.2.1 lead-lag 图 | B | 部分 |

### 9.7.5 数据集引文

| ID | 数据集/引文 | DOI / 稳定 URL | 关键证据定位 | 等 |
|---|---|---|---|---|
| D01 | MetroPT-3, UCI Dataset 791 | DOI 10.24432/C5VW3R | 页面变量表、Failure Information 表、"first month ... training" 段；包内 `Data Description_Metro.pdf` | A（官方文件实证） |
| D02 | Davari et al., DSAA 2021 | DOI 10.1109/DSAA53316.2021.9564181 | UCI 列为 introductory paper | B |
| D03 | Davari et al., ACM SAC 2024 | DOI 10.1145/3605098.3635906 | 仅页面摘要（ACM 反爬） | C |
| D04 | Veloso et al., Scientific Data 2022（MetroPT-**2**，非 -3） | DOI 10.1038/s41597-022-01877-3 | PMC9747912 正文 | A |
| D05 | Saxena, Goebel, Simon & Eklund, PHM08 2008 (C-MAPSS) | PDF 随 NASA `CMAPSSData.zip`；NASA PCoE 条目 6 | six flight conditions（altitude/Mach/TRA 量纲）；**本地实测**：train 61,249 行/249 台、test 41,214 行/248 台、RUL 248 行；readme 写 248/249 与数据不符 | A |
| D06 | Downs & Vogel, Comput. Chem. Eng. 17(3):245–255, 1993 (TEP) | DOI 10.1016/0098-1354(93)80018-I | 本地档案 README "recorded every 180 seconds"；测试文件注入点实测第 161 样本 | A |
| D07 | Rieth et al., 2017 (TEP 推荐版) | DOI 10.7910/DVN/6C3JR1 | dsDescription："sampled every 3 minutes"、"faults were introduced 1 hour and 8 hours"、500 种子不重叠 | A |
| D08 | SWaT 官方特性页 / WADI 官方特性页 | sutd.edu.sg/itrust/...（dataset-characteristics/swat、wadi） | 51/123 变量、7+4 天与 14+2 天、41/15 攻击、官方标签；WADI "readings have been deleted" | A（官方页面） |
| D09 | XJTU-SY, Wang et al., IEEE Trans. Reliability 69(1):401–412, 2020 | DOI 10.1109/TR.2018.2882682；biaowang.tech/xjtu-sy-bearing-datasets | 采样设置段（25.6 kHz、1.28 s/帧、1 min 周期、3 组工况） | A |
| D10 | IMS / NASA PCoE 条目 4；FEMTO, Nectoux et al., IEEE PHM'12 | data.nasa.gov/dataset/ims-bearings；NASA PCoE 条目 10 | 采样率来自二手论文（PCoE 页未写）→ 部分 B/C | B/C |
| D11 | PHM08 Challenge, NASA PCoE 条目 7 | PCoE 页面 | true RUL 不公开、下载受限 | B |

### 9.7.6 主张 → 证据映射（追索表）

| 本报告主张 | 支撑证据 | 边界（不建立什么） |
|---|---|---|
| V1 已有成熟方法 | R01 式(5)、R07、R15 式(5)–(7) | 不建立"我们的固定滞后实现有创新" |
| V2 机制层面已被占据 | R03 式(6)–(14)、R01 式(3)(4)、R02 式(4)(5) | 不建立"工业场景已解决"——工业三数据集上仍无对照（R29/R30 无 lag 路由） |
| V3 只能作研究假设 | R01 式(7)(9) + R16 + R18/R30 + R03 各占一角；§2.4 的 7 组否定性检索 | **不建立"未被提出"**——检索范围见 §0.4，Elsevier/IEEE/ACM 站内检索未覆盖 |
| 工况作条件在 C-MAPSS 有先例 | R18（C 级）、R30（B 级） | 不建立"我们的条件构造是新的" |
| 三数据集单元格级无原生缺失 | D01、D05、D06/D07 官方文件实证 | 不建立"不存在记录级断档/官方删除"（D01 331 处断档、D08 WADI 删除） |
| FD004 是 249/248 | D05 本地实测 + 官方 readme 反证 + 3 篇论文旁证 | 不建立"NASA 数据被改动"（同 readme 其余子集正确，属文档笔误） |
| TEP 采样 3 min | D06 官方 README、D07 dsDescription | 不建立"存在 5 min 官方版本"（未证实） |
| "PHM 2012 航空发动机"不存在 | D10/D11 检索否定 | — |

### 9.7.7 本轮遗留待办（未执行，留待确认）

1. `plan/2026-10-05-dataset-specific-ch3-ch5-experiment-plan.md:18` 的 FD004 台数表述与本地实测相反（现写 248 train / 249 test，实为 **249 train / 248 test**）。按 AGENTS.md 文档一致性规则应修正该行，但本轮任务边界限定为"报告写入 `plan/` + 登记 progress"，故**未改动，提请确认后再改**。
2. README.md:95 提到"官方 248 台 test engine"——**与实测一致，无需修改**。

---

## 10. 科学边界声明

1. 本报告中所有 attention/路由权重只被描述为**预测性依赖或模型路由**，不作因果解释。
2. 所有"学到的滞后"只被描述为**模型选择的时间偏移**；除非另有领域证据，不称为真实物理传播延迟（C 组的工业时延文献只证明该问题存在，不证明本模型恢复了它）。
3. 不使用故障类别、故障模式或测试标签作为推理条件（FD004 明确禁用 fault number）。
4. 不使用未来目标、未来工况或测试集统计量；归一化只来自训练集。
5. 所有模型比较保持相同划分、训练预算与评价协议。
6. 单 seed 结果只作诊断，不作稳定性结论。
7. 本报告未把搜索结果摘要当作论文证据：C 级条目均已显式标注，且不用于支撑"直接重合"以外的强结论。
8. 不声称"首次提出"；检索范围与限制见 §0.4。
9. 无法证实之处一律写"未证实"或"需要进一步核验"。
10. 本报告未运行任何实验、未修改 `code/` 与 `result/`，不含任何实验结果数字。
