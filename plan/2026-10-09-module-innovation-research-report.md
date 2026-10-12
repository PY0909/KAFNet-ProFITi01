# 模块级联网调研：从通用时序预测顶刊顶会方法中寻找可验证的模块创新

- 调研日期：2026-10-09
- 任务书：本消息内嵌的调研任务（面向计算机硕士论文第四章）
- 结论性质：**纯文献与规划产出**。本轮未运行任何实验，未修改 `code/`、模型或 `result/`。
- 相关前序报告：`plan/2026-10-09-lag-condition-forecasting-research-report.md`（时滞感知方向）、`plan/2026-10-09-sosr-literature-research-report.md`（SOSR 方向）。

---

## 0. 检索方法、工具与证据边界

### 0.1 工具实际使用情况

| 工具 | 状态 | 说明 |
|---|---|---|
| `agent-reach` CLI | **可用**（v1.5.0） | 安装于 `/root/.local/bin/agent-reach`（不在 `PATH`，需绝对路径）；`doctor --json` 正常 |
| agent-reach `web` 后端 | **可用**（Jina Reader） | `curl https://r.jina.ai/URL` 读 arXiv HTML / 会议正式页正文 |
| agent-reach `exa_search` 后端 | **不可用** | `mcporter: command not found`，`doctor` 显示 `exa_search: off` |
| `paper-spine` skill | **文件存在但本会话未注册** | 位于 `/root/.codex/skills/paper-spine/`（SKILL.md + 90 个 references）；其证据规则人工遵从（见 §0.2） |
| 内置 `websearch` | 已使用 | 主要全文检索手段 |
| `gh`（GitHub CLI） | 已使用 | 仅确认代码仓库存在，**不作为方法证据** |
| arXiv API | 已使用 | 逐篇核验 arXiv ID / 标题 / 发表注释（`export.arxiv.org/api/query`） |
| Crossref API | 部分使用 | 对 arXiv 优先的论文命中率低，仅用于有正式 DOI 的会议/期刊论文 |

### 0.2 paper-spine 证据边界（人工遵从）

虽未以 skill 形式调用，其 `references/scientific-evidence-ledger.md` 与 SKILL.md「Authority and boundaries」条款已人工遵从，对本报告有约束力的三条：
1. 绝不发明数据、指标、p 值、引用、图表或方法（本轮无任何实验数字）；
2. 未经核验的记录只能标 `planned`，不得在行文中呈现为已观察到的证据；
3. 通过格式检查/工具成功不等于科学就绪——不以"检索跑通了"替代"方向已被证明空白"。

### 0.3 证据等级定义

- **A 级**：论文正文的公式/章节/算法/实验表定位（我方直接读取正文全文，或 arXiv API 核验身份 + 子代理全文取证经我复核）。
- **B 级**：正文片段（出版社页面或检索返回的正文段落）或正式会议/期刊页面 + 摘要一致性核验。
- **C 级**：仅摘要或检索结果摘要。
- **D 级**：仅有线索（标题/引用/他人转述），**不足以支撑任何判断**，只登记不采信。

### 0.4 子代理证据的复核声明

本轮检索由三个并行子代理分头完成（问题1+2 / 问题3+4 / 问题5+6+7），我对其产出做了以下独立复核：
- **全部 25 篇关键论文的身份经 arXiv API 或 OpenReview/会议官方页逐篇核验**；
- **修正 5 处 arXiv ID 错误**：Crossformer（2209.03915 实为生物学论文，正确为 ICLR 2023 OpenReview vSVLM2j9eie）、Pyraformer（2106.13008 实为 Autoformer，正确为 ICLR 2022 Oral OpenReview 0EXmFzUn5I）、RevIN（2105.14415 实为物理论文，正确为 ICLR 2022 OpenReview cGDAkQo1C0p）、SAN（2306.01918 实为物理论文，正确为 NeurIPS 2023 DOI 10.52202/075280-0628）、Dish-TS（2211.12599 实为天体物理论文，正确为 AAAI 2023 DOI 10.1609/aaai.v37i6.25914，arXiv 2302.14829）；
- **修正 1 处期刊错误**：CI/CD trade-off 论文子代理标注 PVLDB 2024，实为 **TKDE 2024**（DOI 10.1109/TKDE.2024.3400008）；
- **3 篇发表状态标注为"待核验"**：MS-FLOW（arXiv 2605.08289，2026-05，无会议接收注释，子代理称 ICML 2026 主会——**未证实**）、TimeSqueeze（arXiv 2603.11352，2026-03，子代理称 NeurIPS 2025 Workshop——**未证实**）、S-Mamba（arXiv 2403.11144，子代理称 Neurocomputing 2025——**未证实**）。

### 0.5 检索局限

1. ScienceDirect（Elsevier）被验证码拦截，部分论文正文仅得摘要级。
2. 未系统检索 IEEE Xplore/ACM DL 站内检索；Scopus/WoS 未访问；知网未检索。
3. arXiv API 对 2026 年新论文（MS-FLOW、TimeSqueeze）的身份核验只能到"arXiv 存在且标题匹配"，无法确认会议接收。
4. 因此，本报告对任何方向的"未覆盖"结论均为**在本次检索范围内的未发现**，不是全局首创证明。

---

## 0.5 基线 Unified V3C 代码核实（任务书要求）

**核实对象**：`code/kaf_profiti/models/unified/encoder.py`（UnifiedTemporalEncoder）、`code/kaf_profiti/models/unified/model.py`、`code/kaf_profiti/models/unified/config.py`、`code/kaf_profiti/models/unified/heads/state.py`、`code/tests/test_unified_model.py`。

| 交接记录描述 | 代码实证 | 精度修正 |
|---|---|---|
| 逐传感器历史编码 + learned embedding | `self.temporal` MLP 作用于 `[B*C, L, F]`（channelwise），加 `channel_identity` [input_dim, hidden_dim] 参数 | ✓ 一致 |
| 多尺度窗口 | `patch_lens=(12,24,48)`，窗口取自序列末端 `temporal[:, :, -min(p,L):, :]` | ✓ 一致 |
| "多尺度时间 attention pooling" | `pool_scorers[s](window).squeeze(-1)` + softmax | ⚠ **实为 `Linear(hidden,1)` 线性打分池化，无 Q/K 交互、无 value 变换**，严格说不是 attention pooling |
| 保留尺度级 patch tokens | `patch_tokens = torch.stack(summaries, dim=2)` → `[B, C, S+1, H]`（S+1=4） | ⚠ **是每尺度每通道一个摘要 token（共 4 个），不是保留有序时间位置的 patch 序列**；且 StateHead **完全不消费它**，只写入 metadata（`model.py:75-77`） |
| 跨传感器 MHA | `nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)` 作用于 `z_history` [B, C, H] | ✓ 一致；注意 **num_heads=1（单头）** |
| learned sensor pooling | `sensor_scorer` = Linear→Tanh→Linear，softmax over channels → `z_global` | ✓ 一致（v3c 默认 "attention" 模式） |
| masked-input invariance | `safe_values = values * observed` 多处 mask 乘法 + masked softmax（`torch.where(has_observation, weights, 0)`） | ✓ 一致；对任意有限填充值不变 |
| v3c 额外机制 | `_channel_trend_features`（masked least-squares slope + recent shift）、`trend_projection`(2→H)、`scale_scorer`（z_history = Σ softmax(scale_logits)·patch_tokens） | 交接记录未提及，属 v3c 特有 |
| 预测头 | StateHead：`base = Linear(hidden_dim, pred_len)` + `increments = Linear(hidden_dim, pred_len*(Q-1))`，softplus 累加分位数 | ⚠ **channel-independent（所有传感器共享同一组权重）**；quantile 模式输出分位数，point 模式取中位数 |

**代码级可测量弱点（候选模块的靶点）**：

| 编号 | 弱点 | 代码证据 | 对应研究问题 |
|---|---|---|---|
| W1 | 跨传感器交互发生在**时间池化之后**（先压缩时间、再交互变量），池化丢失细粒度时间对齐 | `pool_scorers` 池化 → `cross_channel` MHA | 问题3 |
| W2 | 尺度集 `patch_lens=(12,24,48)` **固定**，不随变量、时间片段、预测步变化 | `config.py:32` 硬编码默认值 | 问题5 |
| W3 | 单头通道注意力（num_heads=1）**无负迁移防护**，所有传感器对强制交互 | `encoder.py:106` | 问题2 |
| W4 | `patch_tokens` 生成但**未被预测头消费**（闲置） | `model.py:75-77` 仅写 metadata | 问题1/3 |
| W5 | 预测头 **channel-independent**，所有传感器共享权重，不按 horizon 组织 | `heads/state.py:18-22` | 问题6 |

---

## 1. 执行摘要

**推荐方向：跨尺度跨变量联合交互模块（Segment-level Cross-Dimension Interaction, SCDI）**——将 V3C 当前的"先时间池化、再单头通道 MHA"替换为**在段级别（segment level）同时建模跨时间与跨变量依赖**的两阶段交互，直接针对代码核实的 W1/W3 弱点。

**主要风险**：
1. **原创性证据不足**——最近邻 Crossformer（ICLR 2023）已提出几乎相同的 DSW+TSA 机制，本方向属**组合设计/工程适配**，按任务书标准不得判为算法创新；
2. **DLinear 反例**（AAAI 2023）证明简单线性+分解在 9 个数据集上超越所有 Transformer，复杂交互机制并非总是有益；
3. **CI/CD trade-off**（TKDE 2024）证明通道依赖（CD）策略在分布漂移下可能不如通道独立（CI），跨变量融合可能引入噪声；
4. **工业数据特殊性**——10 s–3 min 采样、15–40 传感器、工况切换，与公开基准（ETT/Traffic/Weather）差异大，文献收益不一定迁移。

**是否值得继续**：**有条件值得**。作为"组合设计 + 工业数据验证"可实现、研究问题清晰、工作量充分；但**必须收窄贡献**为"工业多传感器数据上池化前 vs 池化后跨变量交互的实证比较"，不得声称首次提出。若坚持原创性，应更换问题（回到上一轮时滞感知方向，或寻找其他空白）。

---

## 2. 文献机制表（18 篇候选，深读 8 篇）

> 字段：① 失败模式 ② 核心模块（形状/公式/复杂度）③ 可学习/输入依赖/时变 ④ 消融/负面结果 ⑤ 工业迁移风险 ⑥ 与 V3C 重复/可替换位置/接口冲突 ⑦ 证据等级 ⑧ 身份核验。

### 深读 8 篇（★）

#### ★ Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
- **身份**：Yunhao Zhang, Junchi Yan. ICLR 2023（notable top 5%）. OpenReview: vSVLM2j9eie. 代码: Thinklab-SJTU/Crossformer. **arXiv ID 未核验**（子代理未给出，2209.03915 是生物学论文）. **等级 A**（OpenReview 正式页 + 摘要 + 正文引用链核验）
- ① 失败模式：现有多变量 Transformer 将同一时间点所有变量嵌入为 point-wise token，忽略跨维度依赖。
- ② 核心模块：DSW embedding（每变量分段嵌入为 2D 数组 `[D, T/L_seg, d_model]`）+ Two-Stage Attention（Cross-Time self-attn + Cross-Dimension self-attn，c 个 router 将复杂度从 O(D²) 降至 O(2cD)=O(D)）+ Hierarchical Encoder-Decoder。
- ③ 参数可学习（router 数 c、投影矩阵）；注意力权重输入依赖。
- ④ 消融：去 Cross-Dimension Stage ETTh1 MSE 0.386→0.421；router c=10 最优；训练时间约 Informer 的 1.5 倍。
- ⑤ 工业风险：假设变量数固定（默认 7）；传感器间无物理关联时 Cross-Dimension Stage 引入噪声。
- ⑥ **与 V3C 高度重叠**：V3C 的"池化后单头 MHA"可替换为 TSA 的 Cross-Dimension Stage（段级别而非池化后），保留更细时间对齐。接口：需 2D 数组输入。
- ⑦ 这是**主方向 SCDI 的直接最近邻**。

#### ★ TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
- **身份**：Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huaiyu Yu, Lintao Ma, Mingsheng Long. ICLR 2024. arXiv:2405.14616 ✓. 代码: kwuking/TimeMixer. **等级 A**
- ① 失败模式：现有多尺度方法仅从过去多尺度提取信息，未利用多尺度观测的互补预测能力；分解+多周期分析将趋势季节混合。
- ② 核心模块：PDM（多尺度平均下采样 + seasonal bottom-up + trend top-down mixing）+ FMM（多尺度多预测器集成 x̂=Σx̂_m）。复杂度 O(M·P·d)。
- ③ 参数可学习；尺度集 {2^m} 固定。
- ④ 消融：去 FMM SMAPE 11.723→12.503（M4）；尺度数 M 需手工调；FMM 等权求和忽略可靠性差异。
- ⑤ 工业风险：固定二倍下采样树可能不匹配所有工况；突变可能被当作趋势平滑。
- ⑥ 与 V3C 重叠：多尺度窗口概念相似；FMM 可替换 V3C 的单一共享预测头。

#### ★ Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting
- **身份**：Peng Chen, Yingying Zhang, Yunyao Cheng, Yang Shu, Yihang Wang, Qingsong Wen, Bin Yang, Chenjuan Guo. ICLR 2024（arXiv comment: "Accepted by the 12th ICLR"）. arXiv:2402.05956 ✓. 代码: decisionintelligence/pathformer. **等级 A**
- ① 失败模式：现有多尺度模型（Pyraformer、Scaleformer）固定尺度，无法根据不同序列的时序动态自适应选择。
- ② 核心模块：多尺度路由器 R(X_trans)=Softmax(X_trans·W_r+ε·Softmax(X_trans·W_noise))，ε~N(0,1)，Top-K 选择 patch 尺度；聚合器 X_out=ΣR_i·T_i(X_out^i)。复杂度 O(M·P²·d)。
- ③ **输入依赖**——路由器根据输入的 DFT 时序分解动态选择尺度，但 W_r、W_noise 固定学习。
- ④ 消融：去自适应路径（固定尺度）性能显著下降；part-tuning 仅微调路由器减少 52% 训练时间；计算成本高于单尺度。
- ⑤ 工业风险：DFT+移动平均分解在非平稳工况下可能失效；路由器可能过拟合训练集尺度偏好。
- ⑥ 与 V3C 重叠：scale_scorer+softmax 与 Pathformer 路由器功能相同，但 V3C 的 patch_lens 固定不随输入变化。

#### ★ iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- **身份**：Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long. ICLR 2024. arXiv:2310.06625 ✓. 代码: thuml/iTransformer. **等级 A**
- ① 失败模式：传统 Transformer 将同一时间点多变量融合为 temporal token，抹去多变量相关性；置换不变注意力不适合时间顺序。
- ② 核心模块：Inverted Embedding（每变量整条序列嵌入为 variate token `[B, C, d_model]`）+ Variate self-attention + FFN on Time。复杂度 O(C²·d + L·d²)。
- ③ 参数可学习；注意力分数输入依赖。
- ④ 消融：去 variate-level self-attention Traffic MSE 0.395→1.021；去 FFN 性能降约 15%；变量数少时优势不明显。
- ⑤ 工业风险：C=15–40 时 O(C²) 有限；传感器间无显著相关性时引入噪声。
- ⑥ 与 V3C 重叠：逐变量编码与 V3C channelwise MLP 相似；variate-level self-attention 可替换 V3C 单头通道注意力，在完整序列（非池化后）上建模变量交互。**这是主方向的"整序列 token"替代方案**。

#### ★ CCM: From Similarity to Superiority: Channel Clustering for Time Series Forecasting
- **身份**：Jialin Chen, Jan Eric Lenssen, Aosong Feng, Weihua Hu. NeurIPS 2024（arXiv comment: "NeurIPS 2024"）. arXiv:2404.01340 ✓. 代码: Graph-and-Geometric-Learning/TimeSeriesCCM. **等级 A**
- ① 失败模式：CI 忽略通道交互，CD 混合所有通道（含不相关通道）导致过拟合过平滑。
- ② 核心模块：Channel Embedding + Cluster Assignment p_{i,k}=Normalize(c_k^T h_i/(||c_k|| ||h_i||)) + Cluster-aware FF θ_i=Σ_k p_{i,k} θ_k。复杂度 O(C·d + K·d)。
- ③ **输入依赖**——cluster 分配随输入变化。
- ④ 消融：去 clustering 性能降 5-10%；仅增 K 个 cluster 嵌入参数，参数效率极高。
- ⑤ 工业风险：工业变量按物理功能分组但相似性可能随工况变化；K 需调优。
- ⑥ 与 V3C 重叠：CCM 可替换 V3C 的固定通道处理，提供自适应通道分组。**这是备选方向（稀疏/选择性跨变量依赖）的最近邻之一**。

#### ★ MS-FLOW: What If We Let Forecasting Forget? A Sparse Bottleneck for Cross-Variable Dependencies
- **身份**：Fan Zhang, Shiming Fan, Hua Wang. arXiv:2605.08289（2026-05）. **会议接收状态未证实**（子代理称 ICML 2026 主会，arXiv 无注释）. 代码: Cola-Fsm/MS-FLOW. **等级 B**（arXiv 存在且标题匹配，但发表状态待核验）
- ① 失败模式：密集跨变量混合扩散噪声、放大伪相关，导致表示过平滑和不可靠预测。
- ② 核心模块：Temporal Patch Encoding + Patch-wise Temporal Interaction + Sparse Dependency Bottleneck（Variable-level State Compression → Top-K Sparsification → 跨变量信息注入）。复杂度 O(C·K·d)，K<<C。
- ③ **输入依赖**——依赖图从变量级状态学习，随输入变化。
- ④ 消融：去稀疏瓶颈性能降 8-12%；去 Top-K 稀疏化降 5-10%；训练时间约密集方法 60%。
- ⑤ 工业风险：工业变量间依赖可能随工况切换变化，静态稀疏假设可能失效；Top-K 可能忽略弱但关键依赖。
- ⑥ 与 V3C 重叠：MS-FLOW 的 Sparse Dependency Bottleneck 可替换 V3C 单头注意力，提供更可控的跨变量交互。**这是备选方向的最直接最近邻**。

#### ★ SAN: Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective
- **身份**：Zhiding Liu, Mingyue Cheng, Zhi Li, Zhenya Huang, Qi Liu, Yanhu Xie, Enhong Chen. NeurIPS 2023. DOI: 10.52202/075280-0628. OpenReview: 5BqDSw8r5j. 代码: ustc-time-series/SAN. **等级 A**（NeurIPS proceedings 正式页核验）
- ① 失败模式：传统归一化（RevIN）在窗口级别操作，无法处理窗口内不同时间段的非平稳性差异；长序列中窗口前后可能来自不同分布。
- ② 核心模块：Temporal Slice（切分为 S 个 slice）+ Slice-level Statistics（每 slice 独立均值方差）+ Adaptive Normalization/Denormalization。复杂度 O(B·N·T)。
- ③ **输入依赖**——每 slice 统计量随输入动态计算。
- ④ 消融：去 slice 切分 ETTh1 MSE 0.386→0.412；S=10 最优；长序列（T>200）上优势更明显。
- ⑤ 工业风险：工况连续渐变时 slice 级统计量过度平滑；slice 内噪声不均匀时不稳定。
- ⑥ 与 V3C 关系：SAN 是 model-agnostic 插件，可加在 V3C 输入层。**但任务书明确"不以归一化为主线"，故 SAN 仅作机制对照，不作主方向**。

#### ★ DLinear: Are Transformers Effective for Time Series Forecasting?
- **身份**：Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. AAAI 2023（Oral）. DOI: 10.1609/aaai.v37i9.26317. arXiv:2205.13504 ✓. 代码: cure-lab/LTSF-Linear. **等级 A**
- ① 失败模式：Transformer 自注意力置换不变性丢失时序信息；复杂 Transformer 被简单线性层超越。
- ② 核心模块：序列分解（moving_avg 趋势 + 季节）+ 两个线性层 W_t: R^{L×C}→R^{H×C}。复杂度 O(L·C)。
- ③ 参数可学习；分解核大小固定。
- ④ 消融：分解+线性在 9 个数据集超越所有 Transformer 基线；ETTm2 上 MSE=0.563（TimeMixer=0.275），纯线性在复杂多变量数据不足；channel-independent 忽略跨传感器相关。
- ⑤ 工业风险：工况切换导致趋势突变，moving average 分解滞后；channel-independent 忽略传感器耦合。
- ⑥ 与 V3C 关系：**效率基线 + 反例**——DLinear 证明简单线性+分解足以超越 Transformer，提示 V3C 的跨传感器 MHA 必要性需重新审视。

### 其余 10 篇（非深读，等级 B/C）

| 论文 | 身份（核验） | 失败模式 | 核心模块 | 与 V3C 关系 | 等级 |
|---|---|---|---|---|---|
| TimeXer | NeurIPS 2024 ✓ arXiv:2402.19072 | point-wise embedding 信息错配 | Endogenous patch + Exogenous variate token + Cross-Attn | 可替换 V3C 池化后 MHA 为变量级 cross-attn | B |
| N-HiTS | AAAI 2023 ✓ arXiv:2201.12886（comment: Accepted AAAI-2023） | 长 horizon 误差累积 | 多速率采样 + 层次插值 | 可替换 V3C 预测头；但 horizon=24 过参数化 | B |
| S-Mamba | arXiv:2403.11144，**Neurocomputing 2025 未证实** | Transformer 二次复杂度 | 双向 Mamba VC + FFN TD | 可替换 V3C 单头 MHA，O(C²)→O(C) | B |
| TimeMachine | ECAI 2024 ✓ arXiv:2403.09898（comment: ECAI-2024） | 长序列线性与效率难兼得 | 四重 Mamba（全局/局部 × mixing/independence） | 可替换 V3C 编码层；工业短序列可能过参数化 | B |
| MambaTS | Pattern Recognition 2026 ✓ arXiv:2405.16440（comment: Accepted PR 2026） | Mamba 因果卷积对多变量不必要 | VAST（变量感知扫描）+ TMB | 可替换 V3C MHA 为自适应扫描 | B |
| RevIN | ICLR 2022 ✓ OpenReview cGDAkQo1C0p | 实例级分布偏移 | 可逆实例归一化 + learnable affine | model-agnostic 插件；任务书排除归一化主线 | B |
| Non-stationary Transformers | NeurIPS 2022 ✓ arXiv:2205.14415 | 过度平稳化丢失非平稳信息 | Series Stationarization + De-stationary Attention | 可加在 V3C 输入层 | B |
| FreTS | NeurIPS 2023 ✓ arXiv:2311.06184 | 时域 MLP 信息瓶颈 | 频域 FreMLP（channel + temporal） | 可替换 V3C 跨传感器注意力为频域交互 | B |
| LightTS | arXiv:2207.01186，**发表状态未证实** | 复杂架构计算成本高 | 连续/间隔采样 + IEBlock | 可替换 V3C 池化层 | B |
| TimeSqueeze | arXiv:2603.11352，**发表状态未证实** | 固定 patch 尺寸信息丢失/浪费 | 相对偏差动态 patch + SSM | 参考动态 patch 选择；面向基础模型，工业小数据不适用 | B |
| Pyraformer | ICLR 2022 **Oral** ✓ OpenReview 0EXmFzUn5I | 长序列二次复杂度 | 金字塔注意力 PAM，O(L) | 固定尺度；工业短序列退化为单尺度 | B |
| Dish-TS | AAAI 2023 ✓ DOI 10.1609/aaai.v37i6.25914，arXiv:2302.14829 | 变量间分布差异 | Dish Normalization + 域对抗 | 与 V3C train-split 归一化冲突 | B |
| CI/CD trade-off | **TKDE 2024** ✓ DOI 10.1109/TKDE.2024.3400008（子代理误标 PVLDB） | CD 策略分布漂移下不如 CI | 实证分析 ACF 漂移 + 容量-鲁棒性权衡 | **反例**：跨变量融合可能不如通道独立 | B |

---

## 3. 失败模式—现有解法—剩余问题映射表

| 失败模式 | 现有解法（代表论文） | 剩余问题 | 对 V3C 的启示 |
|---|---|---|---|
| 先压缩时间、再交互变量，丢失细粒度对齐 | Crossformer DSW+TSA、TimeXer cross-attn、iTransformer variate token | 工业数据上"池化前 vs 池化后交互"无实证；多尺度变量交互未解决 | **W1 是主方向靶点** |
| 跨变量负迁移：融合噪声/伪相关 | MS-FLOW 稀疏瓶颈、CCM 聚类、LIFT TopK、CI/CD trade-off（反例） | 何时不应融合？工况切换下依赖变化无理论保证 | **W3 是备选靶点** |
| 时间压缩丢失局部顺序/突变 | TimeMixer PDM、LightTS 采样、FreTS 频域 | 工况切换下突变保留与压缩平衡；V3C patch_tokens 闲置 | W4 可利用 |
| 非平稳/工况变化 | RevIN、SAN、FAN、NonStat、Dish-TS | 任务书排除归一化主线；工况切换在线适应未解决 | 仅作机制对照 |
| 尺度固定不随变量/时间/步变化 | Pathformer 路由器、TimeMixer 多尺度 | 层间选择已有，层内逐通道/逐预测步选择未解决 | W2 可改进 |
| 多步预测信息需求不同 | N-HiTS 层次插值、TimeSqueeze 多 horizon 头 | horizon=24 短预测下过参数化；误差形态异方差未建模 | W5 优先级低 |
| 结构效率：精度—成本权衡 | S-Mamba、TimeMachine、MambaTS、DLinear（反例） | 工业小数据上 Mamba 选择性对缺失敏感；稀疏 vs 状态空间 Pareto 未确定 | 可作效率对照 |

---

## 4. 候选模块比较（4 个）

### 候选 A：跨尺度跨变量联合交互模块（SCDI）——**推荐主方向**

1. **可测量问题**：V3C 的跨传感器交互发生在时间池化之后（W1），单头 MHA 无负迁移防护（W3）。**否定观察**：若池化后交互已足够（如 DLinear 所示简单线性即可），则池化前交互无增益。
2. **最近邻**：Crossformer（DSW+TSA，ICLR 2023）已解决"段级别跨维度交互"；TimeXer（变量级 cross-attn）解决"外生变量交互"；iTransformer 解决"整序列 token 交互"。**差异**：V3C 已有多尺度窗口但闲置 patch_tokens；SCDI 将 Crossformer 的段级交互思想**限制在 V3C 已有的多尺度框架内**，不引入完整 Crossformer 架构。
3. **最小模块**：见 §6。
4. **可能有效的原因**：保留细粒度时间对齐后再交互，避免池化丢失；多尺度提供不同感受野。**数据假设**：传感器间存在可被注意力利用的稳定依赖；分布漂移不严重（否则 CI/CD trade-off 警告失效）。
5. **性质**：**组合设计/工程适配**，非新机制。Crossformer 已提出核心思想。
6. **最简对照**：V3C 池化后 MHA vs SCDI 池化前段级 cross-dimension attention（同参数预算）。
7. **成本**：段级 cross-dimension attention 复杂度 O((T/L_seg)²·D + D²·L_seg)；L_seg=12、T=168 时约 14²·D + D²·12，与 V3C 单头 MHA O(C²·H) 相当或略高。实现难度中等（需改 encoder 输入表示）。
8. **额外标签/未来协变量/预训练**：均不需要。

### 候选 B：输入依赖多尺度路由（Pathformer 风格）——**淘汰**

1. **问题**：W2 尺度固定。**否定观察**：若不同时间片段最优尺度相同（如周期性强的数据），自适应路由无增益。
2. **最近邻**：Pathformer（ICLR 2024）路由器。**差异**：V3C 的 scale_scorer 已是简单版路由器，但 patch_lens 固定。
3. **性质**：工程适配。
4. **淘汰原因**：Pathformer 的 DFT+移动平均分解在非平稳工况下可能失效；工业序列短（168 步），路由收益不确定；与主方向 SCDI 相比，SCDI 更直接针对已核实的 W1 弱点。**保留为备选备选**。

### 候选 C：稀疏跨变量依赖瓶颈（MS-FLOW 风格）——**备选**

1. **问题**：W3 单头 MHA 无负迁移防护。**否定观察**：若工业传感器间依赖稠密，稀疏瓶颈丢失有用信息。
2. **最近邻**：MS-FLOW（2026-05，发表状态未证实）、CCM（NeurIPS 2024）、LIFT（ICLR 2024）。**差异**：MS-FLOW 的 Top-K 稀疏路由与 LIFT 的 TopK 选源高度重叠；CCM 聚类是"软稀疏"替代。
3. **性质**：组合设计；MS-FLOW 本身极新，复现风险高。
4. **成本**：O(C·K·d)，K<<C 时比 V3C MHA 更省。实现难度低。
5. **保留为备选**：若主方向 SCDI 在实验中显示跨变量融合引入噪声（即 CI/CD trade-off 场景），则转向 C。

### 候选 D：非平稳适应（SAN/RevIN 风格）——**淘汰**

1. **问题**：工况切换导致分布偏移。**否定观察**：任务书明确"不以归一化为主线"。
2. **最近邻**：SAN、RevIN、FAN、NonStat。
3. **淘汰原因**：任务书排除；且 V3C 已有 masked-input invariance 和 train-split 归一化，归一化插件的边际收益不明确。**仅作机制对照**。

### 候选 E：Horizon-conditioned 解码（N-HiTS 风格）——**淘汰**

1. **问题**：W5 channel-independent 预测头。**否定观察**：horizon=24 短预测下过参数化。
2. **淘汰原因**：N-HiTS 面向长 horizon（>192），工业 horizon=24 无优势；且 W5 优先级低于 W1/W3。

---

## 5. 推荐主方向与备选

### 主方向：候选 A — SCDI（跨尺度跨变量联合交互）

**定位**：组合设计/工程适配，**非算法创新**。贡献收窄为"工业多传感器数据上池化前 vs 池化后跨变量交互的实证比较 + 段级多尺度交互模块的工业适配"。

**理由**：
- 直接针对代码核实的 W1（池化后交互）和 W3（单头无负迁移防护）；
- 研究问题清晰可证伪（§6 核心对照）；
- 可实现（模块级替换，不引入完整 Crossformer）；
- 与最近邻 Crossformer/TimeXer/iTransformer 有明确对照接口；
- 工作量充分（2 个数据集 × 多组对照 × 多 seed）。

**原创性声明**：按任务书标准，本方向**不满足"具有原创性证据"**。Crossformer（ICLR 2023）已提出几乎相同的机制。若论文要求算法创新，应更换问题。

### 备选：候选 C — 稀疏跨变量依赖瓶颈

若主方向实验显示跨变量融合引入噪声（CI/CD trade-off 场景），转向 MS-FLOW 风格的稀疏路由。注意 MS-FLOW 发表状态未证实，复现风险高。

### 其余方向淘汰原因

- B（Pathformer 路由）：非平稳工况下 DFT 分解失效风险；与 SCDI 相比靶点不直接。
- D（归一化）：任务书排除。
- E（N-HiTS）：horizon=24 过参数化。

---

## 6. 推荐方向的最小实现、核心对照、数据选择、指标与图表

### 6.1 最小模块（SCDI）

**输入**：V3C 的逐传感器多尺度表示 `temporal` [B, C, L, H]（池化前）。
**输出**：`z_history` [B, C, H]（替换当前池化后 MHA 的输入）。

**伪代码**：
```python
# 输入: temporal [B, C, L, H], patch_lens=(12,24,48)
# 1. 段级别表示: 将每变量的时间序列分段
#    seg[b,c,s,:] = temporal[b,c, s*L_seg:(s+1)*L_seg, :]  # L_seg=12, S=L/L_seg=14
#    seg_emb[b,c,s,:] = Linear_H(seg[b,c,s,:].mean(dim=1))  # [B, C, S, H]
# 2. Cross-Dimension Attention (TSA 风格):
#    Q = seg_emb[b,c,s,:]; K,V = seg_emb[b,:,s,:]  # 跨变量
#    attn[b,c,s,:] = softmax(Q @ K^T / sqrt(H)) @ V  # [B, C, S, H]
# 3. 多尺度聚合:
#    z_history[b,c,:] = sum_s softmax(scale_scorer(seg_emb[b,c,s,:])) * attn[b,c,s,:]
# 4. 残差 + LayerNorm (与 V3C 一致)
```

**张量形状**：
- 输入：`[B, C, L=168, H=64]`
- 段嵌入：`[B, C, S=14, H=64]`
- Cross-Dimension Attention：`[B, C, 14, 64]`（每段内跨变量）
- 输出：`[B, C, 64]`

**复杂度**：O(B·C·S·C·H) = O(B·C²·S·H)。C=15、S=14、H=64 时约 201K·B，与 V3C 单头 MHA O(B·C²·H) 相当。

### 6.2 核心对照（公平实验要求）

| 对照臂 | 目的 |
|---|---|
| Unified V3C（现状） | 基线 |
| SCDI（完整） | 主方向 |
| 池化前交互 vs 池化后交互（同参数预算） | 验证 W1 |
| 单头 vs 多头 cross-dimension attention | 验证 W3 |
| 稀疏版（Top-K 跨变量） | 对照备选 C |
| DLinear（效率基线） | 反例对照 |
| Crossformer（完整原方法） | 最近邻对照（注意：是完整方法，非模块） |
| 去 SCDI 消融（回退到 V3C） | 验证模块必要性 |

**公平性**：相同输入信息、历史长度、预测长度、划分、归一化、checkpoint 选择、调参预算。参数量、可访问历史信息、计算量三者分别报告。

### 6.3 数据选择

**主预测数据集（2 套）**：
1. **MetroPT-3**（已有）：10 s 采样、15 通道、213 天、官方首月协议、4 起故障无标签、331 处记录断档。
2. **Beijing PM2.5**（UCI 381，新增推荐）：小时级、11 变量、2010–2014、43,824 行、**真实 NA 缺失**、冬季供暖/APEC 等 regime 切换、标准 train/val/test 划分。

**扩展任务（1 套）**：ETTh1（7 通道、12/4/4 月划分、Informer 仓库开放获取）或 TEP（故障检测，3 min 采样）。

**不推荐**：FD004（无墙钟时间戳，不适合时滞/交互分析）、XJTU-SY/IMS（通道同步、帧间隔过粗）。

### 6.4 指标与图表

**指标**：MAE/RMSE（overall/逐传感器/逐 horizon）、参数量、训练时间、推理时间、峰值显存。分工况段误差（MetroPT-3 按启停/负载分段；PM2.5 按季节/供暖期分段）。

**图表**：
1. SCDI 模块结构图（段级 cross-dimension attention）；
2. 池化前 vs 池化后交互的逐传感器误差对比；
3. 稀疏度 K 敏感性曲线；
4. 精度—成本 Pareto 前沿（SCDI vs V3C vs DLinear vs Crossformer）；
5. 分工况段误差热图；
6. 跨变量注意力图（按工况分组）。

---

## 7. 分阶段推进与停止条件

### 阶段 0（1 周）：冻结基线与核验
- 为 MetroPT-3、PM2.5 各选择 V3C checkpoint 和 run identity（沿用 P0 协议）。
- 实现 SCDI 模块的单元测试（形状、有限性、masked-input invariance）。

### 阶段 1（2 周）：单 seed 方向筛选
- 在 MetroPT-3 上跑 SCDI vs V3C vs DLinear（单 seed）。
- **停止条件 1**：SCDI 相对 V3C 的 valid MAE 增益 < 1% 且方向不一致 → 放弃主方向，转向备选 C 或更换问题。

### 阶段 2（2 周）：多 seed 评估 + 对照补齐
- 通过的配置跑 3 seeds。
- 补齐核心对照（池化前/后、单头/多头、稀疏版、Crossformer）。
- **停止条件 2**：3 seeds 上 SCDI 增益方向不一致或均 < 1% → 收窄贡献为"工业数据实证比较"，不声称算法创新。

### 阶段 3（1 周）：第二数据集验证
- 在 Beijing PM2.5 上重复阶段 1–2。
- **停止条件 3**：若 SCDI 在 PM2.5 上无效但在 MetroPT-3 上有效 → 结论限定为"特定数据集有效"，不泛化。

### 阈值说明
- 1% MAE 增益阈值依据：V3C 在 MetroPT-3 上的单 seed 诊断波动约 0.5–1%（见 `plan/2026-10-06` 记录），低于此阈值的增益无法与噪声区分。
- 不任意指定更高百分比：因无先验效应量估计，1% 是"可检测下限"而非"有意义的提升"。

---

## 8. 引文证据索引与待核验事项

### 8.1 引文证据索引（25 篇，按深读/非深读分组）

**深读 8 篇**：

| 编号 | 论文 | 身份（核验） | 代码 | 等级 |
|---|---|---|---|---|
| R01 | Crossformer | ICLR 2023, OpenReview vSVLM2j9eie ✓ | Thinklab-SJTU/Crossformer | A |
| R02 | TimeMixer | ICLR 2024, arXiv:2405.14616 ✓ | kwuking/TimeMixer | A |
| R03 | Pathformer | ICLR 2024, arXiv:2402.05956 ✓ | decisionintelligence/pathformer | A |
| R04 | iTransformer | ICLR 2024, arXiv:2310.06625 ✓ | thuml/iTransformer | A |
| R05 | CCM | NeurIPS 2024, arXiv:2404.01340 ✓ | TimeSeriesCCM | A |
| R06 | MS-FLOW | arXiv:2605.08289, **发表状态未证实** | Cola-Fsm/MS-FLOW | B |
| R07 | SAN | NeurIPS 2023, DOI 10.52202/075280-0628 ✓ | ustc-time-series/SAN | A |
| R08 | DLinear | AAAI 2023, DOI 10.1609/aaai.v37i9.26317 ✓ | cure-lab/LTSF-Linear | A |

**非深读 12 篇**：

| 编号 | 论文 | 身份（核验） | 等级 |
|---|---|---|---|
| R09 | TimeXer | NeurIPS 2024, arXiv:2402.19072 ✓ | B |
| R10 | N-HiTS | AAAI 2023, arXiv:2201.12886 ✓ | B |
| R11 | S-Mamba | arXiv:2403.11144, **Neurocomputing 2025 未证实** | B |
| R12 | TimeMachine | ECAI 2024, arXiv:2403.09898 ✓ | B |
| R13 | MambaTS | Pattern Recognition 2026, arXiv:2405.16440 ✓ | B |
| R14 | RevIN | ICLR 2022, OpenReview cGDAkQo1C0p ✓ | B |
| R15 | Non-stationary Transformers | NeurIPS 2022, arXiv:2205.14415 ✓ | B |
| R16 | FreTS | NeurIPS 2023, arXiv:2311.06184 ✓ | B |
| R17 | LightTS | arXiv:2207.01186, **发表状态未证实** | B |
| R18 | TimeSqueeze | arXiv:2603.11352, **发表状态未证实** | B |
| R19 | Pyraformer | ICLR 2022 Oral, OpenReview 0EXmFzUn5I ✓ | B |
| R20 | Dish-TS | AAAI 2023, DOI 10.1609/aaai.v37i6.25914 ✓ | B |
| R21 | CI/CD trade-off | **TKDE 2024**, DOI 10.1109/TKDE.2024.3400008 ✓ | B |

### 8.2 仍待核验事项

1. **MS-FLOW、TimeSqueeze、S-Mamba、LightTS 的发表状态**——arXiv 存在但会议/期刊接收未证实，需向作者或会议官方确认。
2. **Crossformer 的 arXiv ID**——OpenReview vSVLM2j9eie 已确认，但未找到对应 arXiv 编号（可能未公开挂 arXiv）。
3. **SCDI 模块的实际增益**——本轮未运行任何实验，§6 的最小实现和对照均为规划。
4. **Beijing PM2.5 在 V3C 框架下的适配**——其真实 NA 缺失需要与 V3C 的 masked-input invariance 对接。
5. **子代理证据中个别数字的正文复核**——如 Pathformer 的"36.4% MSE 降低"、MS-FLOW 的"训练时间约 60%"等，我未逐一回原文核对，论文中引用前需再次定位。

### 8.3 本轮遗留待办

- `plan/2026-10-05-dataset-specific-ch3-ch5-experiment-plan.md:18` 的 FD004 台数表述与本地实测相反（前序报告已记录，待确认后修正）。

---

## 9. 科学边界声明

1. 本报告中所有 attention/路由权重只被描述为**预测性依赖或模型路由**，不作因果解释。
2. 所有"学到的滞后/依赖"只被描述为**模型选择的信息通道**；不称为真实物理传播延迟或因果结构。
3. 不使用故障类别、故障模式或测试标签作为推理条件。
4. 不使用未来目标、未来工况或测试集统计量；归一化只来自训练集。
5. 所有模型比较保持相同划分、训练预算与评价协议。
6. 单 seed 结果只作诊断，不作稳定性结论。
7. 本报告未把搜索结果摘要当作论文证据：B 级条目均已显式标注，且不用于支撑"直接重合"以外的强结论。
8. 不声称"首次提出"；检索范围与限制见 §0.5。
9. 无法证实之处一律写"未证实"或"需要进一步核验"。
10. 本报告未运行任何实验、未修改 `code/` 与 `result/`，不含任何实验结果数字。
