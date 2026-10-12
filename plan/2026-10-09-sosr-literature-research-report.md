# SOSR 候选创新联网调研报告

日期：2026-10-09。依据：`plan/2026-10-09-sosr-literature-research-prompt.md`。
性质：**文献查新与实验设计建议，不含任何新实验结果**。本次未修改任何代码、模型、数据划分或 `result/` 产物，未复制外部代码，未使用测试集指标做方向决策，未运行任何训练或评估。

---

## 0. 材料边界与检索方法声明

- **本地源码定位不可作为 Unified V3C 的实现证据**。任务书中提到的旧项目路径仅作为检索线索使用。本报告未把任何旧仓库 README、hash、commit id、绝对路径当作结论依据。
- 六个参考仓库（Raindrop、BiTGraph、S4M、ChannelTokenFormer、CRIB、Merlin）的功能描述已**逐个独立核验到论文正文或正式元数据**，不采信 README 自述（README 仅用于定位代码是否存在）。
- 不假定当前 Unified V3C 已具备"逐 patch、逐传感器 token"或"缺失统计接口"。仅就本地可读代码做事实记录：`code/kaf_profiti/models/unified/encoder.py` 中存在观测 mask、timestamp delta、masked temporal block 等处理（如 `mask`、`delta`、`masked_fill(~valid, ...)`），这只能说明**编码器已在用 mask/delta 输入**，不能说明已具备 SOSR 所需的逐 patch support 统计或路由接口，更不能当作创新证据。
- 检索渠道：会议/期刊正式论文页（ICLR/ICML/KDD/AAAI/NeurIPS/CIKM/PMLR/OpenReview）、arXiv 正文（HTML/PDF 片段）、Semantic Scholar、官方 GitHub。Code Mode 内 shell 不可用，故未运行 `agent-reach` CLI，改用内置 web 检索与页面抓取完成同等覆盖。
- **证据等级**（本报告统一使用）：
  - **A**：正式论文，且已核到正文方法节/公式/图表；
  - **B**：正式论文，仅核到摘要 + 部分正文片段；
  - **C**：预印本正文（未经同行评审或状态未定）；
  - **D**：仅摘要/元数据/代码仓库存在性。

---

## 1. 执行摘要

1. **SOSR 的六个部件全部存在公开先例**，其中部件 5（按局部观测支持融合不同时间尺度）**已被人直接做过且是图网络 + 缺失条件化的形式**：Marisca 等 ICML 2024 的 HD-TTS（"Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling"）明确"按观测与缺失模式条件化的多尺度表示可解释注意力融合"，与 SOSR 第 5 条和候选公式中 "scale fusion 按局部支持学习尺度权重" 属**直接重合**。
2. **没有检索到同时具备"多时间尺度 + 目标缺口门控 + 源可靠性路由"三者的论文或代码**。最接近的两个组合各缺一块：HD-TTS 有"多尺度 + 缺失条件化 + 消息传递"但没有目标节点缺口门控和源可靠性偏置；ChannelTokenFormer（ICLR 2026）有"channel token 跨通道路由 + mask 引导注意力 + 频率动态 patch 多尺度"但门控来自 patch 是否存在，而非目标传感器的 support 缺口与源可靠性。
3. **部件 2（目标缺口门控）是先例最少的一环**，但其可检索的近亲很多：mask 作为注意力偏置（MissTSM、SPIN、SAITS）、接收端条件化的 anisotropic 消息传递、以及图上的可靠性加权消息传递（HU-GNN、FairAC）。它更像"已知机制的重新参数化"，而非新机制。
4. **部件 3（源可靠性 = support + freshness + delta-time + 连续缺失长度）的特征集有直接先例**：GRU-D（Che et al., 2018）的输入/隐状态衰减用 δ（距上次观测时间）+ mask 驱动；DATA-GRU（AAAI 2020）把时间间隔转成权重并做"数据质量注意力"。**"观测比例高 ⇒ 传感器更准确"这一单调假设没有文献支持**，文献反而指出 informative missingness 与低频有效测量的存在。
5. 因此按四级判定规则：**SOSR 当前只能定级为"部分重合 + 组合创新候选（有条件）"**；在没有"相对容量匹配基线的稳定独立收益"之前，**不允许升格为组合创新，更不允许写成首次提出**。若消融不支持，应降级为**面向工业遥测缺失鲁棒性的经验/应用贡献**。
6. 三数据集方面：**MetroPT-3 与 C-MAPSS FD004 官方文档均标注无缺失值，TEP 各主流版本为仿真全采样数据**。三者都不能被称作"天然异步或真实缺失数据"。把第四章改为缺失鲁棒性主线，属于**研究范围调整建议**，必须显式写明"缺失由人工注入、仅作压力测试"。

---

## 2. 目标候选文献证据表（14 篇，其中 6 篇重点精读）

> 表 A：身份与输入表示；表 B：机制重合与判定。字段按任务书 §4 要求。

### 表 A — 身份、链接、任务、输入表示

| # | title / authors / year | DOI 或官方链接 | task / data / 缺失类型 | input representation | temporal scale | 证据等级 |
|---|---|---|---|---|---|---|
| 1★ | Graph-Guided Network for Irregularly Sampled Multivariate Time Series (Raindrop)；Zhang, Zeman, Tsiligkaridis, Zitnik；ICLR 2022 | arXiv:2110.05357 | 时间序列表示/分类（P12/P19/HAR），不规则采样、传感器子集观测（真实不规则） | 观测值 + 时间表示 + 学习到的样本级传感器图；无显式 support 标量 | 连续时间/观测级（非 patch） | A |
| 2★ | Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values (BiTGraph)；Chen, Li, Liu, Li；ICLR 2024 | OpenReview `O9nZCwdGcG`（PDF）；GitHub chenxiaodanhit/BiTGraph | 多变量预测（Metr、Electricity、PEMS、BeijingAir），人工 mask（point missing，ratio 可调） | 历史值 X 与 mask M 同时进入网络（M 随时间卷积一起传播） | 多尺度时间卷积（Multi-Scale Instance PartialTCN） | A |
| 3★ | S4M: S4 for Multivariate Time Series Forecasting with Missing Values；Jing Peng, Meiqi Yang, Qiong Zhang, Xiaoxiao Li；ICLR 2025 | arXiv:2503.00900；proceedings.iclr.cc 2025 | 多变量预测（Electricity/ETTh1/Traffic/Weather），time-point / variable / block 缺失（人工） | mask 矩阵 M^(L) 作为第二输入流 + 原型库表征 | 单尺度 S4（无多尺度路由） | A |
| 4★ | Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework for Dependency, Asynchrony, and Missingness (ChannelTokenFormer, CTF)；2026 | arXiv:2506.08660；OpenReview `r4ZamwBE8P`；GitHub jinkwan1115/ChannelTokenFormer | 多变量预测，通道异步采样 + test-time block 缺失（人工遮挡 + 真实工业数据） | (x, m, s) 三元组：值、mask、采样周期；训练期随机 patch masking | 频率引导的 channel-wise 动态 patch（多尺度 patch 长度） | A |
| 5★ | Revisiting Multivariate Time Series Forecasting with Missing Values (CRIB: Consistency-Regularized Information Bottleneck)；2025 preprint | arXiv:2509.23494 | 多变量预测 + 部分观测，人工 mask | 二值 mask M∈{0,1}^{N×T}，直接从部分观测预测（不插补） | patch（P 步非重叠）+ TCN | C |
| 6★ | Merlin: Multi-View Representation Learning for Robust MTSF with Unfixed Missing Rates；Yu, Wang, Yang, Shao, Sun, Qian, Wei, An, Xu；KDD 2025 | arXiv:2506.12459；DOI 10.1145/3711896.3737046；GitHub ChengqingYu/Merlin | 多变量预测，25/50/75/90% 随机缺失（人工） | 缺失张量 X_M 输入学生网络；教师看完整观测 | 单尺度（STID/MTGNN backbone） | A |
| 7 | Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling (HD-TTS)；Marisca, Alippi, Bianchi；ICML 2024 | PMLR 235:34846-34865；arXiv:2402.10634 | 时空图预测，point/spatial-block/spatiotemporal-block 缺失（人工 + AQI 真实缺失分布） | 二值节点 mask m^i_t∈{0,1}^{d_x}，解码注意力以观测与缺失模式为条件 | **层级时间+空间下采样多尺度** | A |
| 8 | Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN)；Marisca, Cini, Alippi；NeurIPS 2022 | arXiv:2205.13479 | 时空图插补，稀疏观测（block 缺失） | "valid / missing observation encoding"，稀疏时空注意力区分 missing query/key | 单尺度时空注意力 | A |
| 9 | Investigating a Model-Agnostic and Imputation-Free Approach for IMTS Modeling (MissTSM)；Neog 等；TMLR | arXiv:2502.15785 | 不规则多变量序列分类 + 预测，真实/半合成缺失 | mask 通过注意力 logits 的大负偏置 η 强制缺失变元注意力为 0 | 时间步级（variate-first） | A |
| 10 | Irregular Multivariate Time Series Forecasting: A Transformable Patching GNN Approach (t-PatchGNN)；Zhang, Yin, Liu, Zhou, Xiong；ICML 2024 | PMLR 235:60179-60196 | 不规则序列预测（PhysioNet/HAR/USHCN/MIMIC） | 观测点 (t,v) 打包为等时长 patch | 可变观测数 patch（单一时间分辨率） | A |
| 11 | Hi-Patch: Hierarchical Patch GNN for Irregular Multivariate Time Series；Luo, Zhang, Liu, Ma；ICML 2025 | PMLR 267:41494-41519 | 不规则序列预测 + 分类（8 数据集） | 每个观测为节点 | **层级：intra-patch + inter-patch 多尺度图** | A |
| 12 | Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D)；Che 等；Scientific Reports 2018 | DOI 10.1038/s41598-018-24271-9；arXiv:1606.01865 | 临床分类/回归，informative missingness | mask m_t 与时间间隔 δ_t（距上次观测）+ 指数衰减 | 递归（无 patch 多尺度） | A |
| 13 | DATA-GRU: Dual-Attention Time-Aware Gated Recurrent Unit for Irregular MTS；AAAI 2020 | cdn.aaai.org/ojs/5440（AAAI-20 PDF） | 临床死亡率预测，不规则采样 | 时间间隔转权重 w_Δt=1/log(e+Δt)；"数据质量注意力 + 医学知识注意力" | 递归 | A |
| 14 | Hierarchical Uncertainty-Aware Graph Neural Network (HU-GNN)；Choi 等；CIKM 2025 | arXiv:2504.19820 | 通用图分类（非时序），节点不确定性 | 节点不确定性 u_i 参与消息传递加权 | 层级（local/community/global） | B |

★ = 任务书点名的六个参考仓库对应论文（重点精读 6 篇），第 7–14 为检索补充的最近邻。
补充相关但判定为"相关但不同"：SAITS（ESWA 2023, arXiv:2202.08516，mask 驱动的表示加权，插补任务）；GraFITi（AAAI 2024, DOI 10.1609/aaai.v38i15.29560，本项目已有基线）；FairAC/Feature Propagation（图上缺失特征的邻居加权，非时序）。

### 表 B — 机制重合、差异与判定

| # | cross-sensor mechanism | target-deficit gate | source-reliability weighting | scale routing | training objective | closest overlap（具体定位） | key difference（SOSR 仍可能不同的可检验差异） | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 Raindrop | 样本级动态传感器图 + 神经消息传递（观测驱动边权更新） | 无显式目标缺口门；靠"观测到的传感器指示未观测传感器" | 无可靠性偏置；用时间表示与邻近观测 | 无 | 监督分类 | §3 消息传递与边权更新（ICLR 正文）；Zitnik Lab 项目页"observed sensors indicate how unobserved sensors behave" | SOSR 若按 scale/patch 粒度并显式以 support 缺口控制注入强度，可与之区分 | 部分重合 |
| 2 BiTGraph | 图卷积（Biased GCN）在时间融合特征上 | 缺失模式随时间一起卷积传播（**不区分目标/源**） | 无 | 多尺度时间卷积（**不是按 support 选择尺度**） | 监督预测（含 mask 消融，Table 3） | OpenReview PDF §3 问题定义（X 与 M 输入）、§4 biased multi-scale PartialTCN + biased GCN | SOSR 的目标缺口门与源可靠性偏置在 BiTGraph 中不存在，可单独消融 | 部分重合 |
| 3 S4M | 无跨传感器路由（通道间靠 S4 联合序列） | 无（mask 直接进状态方程） | 无 | 无 | 监督预测 + mask/ATPM 消融 | §3.3 Eq.(4)：`h_t = A h_{t-1} + B o_t + E E_m(m_t)`；`y_t = C h_t + D o_t + F E_m(m_t)` | SOSR 是跨传感器路由而非双流 SSM；但"mask 作为条件输入"本身已不新 | 部分重合 |
| 4 CTF | channel token 作为只读全局锚点做跨通道注意力 | 部分：缺失 patch 被移除/遮蔽，注意力跳过空缺（**不是目标 support 缺口门控**） | 无源可靠性（只有"是否存在"） | 频率引导动态 patch + 各采样周期共享解码器（**不是按 support 学权重**） | 监督预测 + 训练期 patch masking 代理 | Fig.2 结构；Fig.3 掩码规则（local↔本通道、channel token↔本通道与跨通道、channel token 不自注意）；§"Training-Time Proxying for Test-Time Missingness" | SOSR 若给出可解释的 support 权重与源可靠性偏置，需证明其优于 CTF 式"存在/不存在"二值处理 | 部分重合（对部件 4、5 构成强压力） |
| 5 CRIB | unified-variate attention（跨变量注意力）+ 一致性正则 | 无 | 无 | 无（单 patch 尺寸） | **信息瓶颈 + 一致性正则（不插补直接预测）** | Eq.(1) IB 目标 `min I(Z;X^o) − β I(Y;Z)`；§3 patching + 一致性/去噪 | SOSR 不含 IB 目标；但"不插补直接预测 + mask"这条路线已是共识，SOSR 不能声称"首次不插补" | 部分重合 |
| 6 Merlin | 无（跨传感器由 backbone 决定） | 无 | 无 | 无 | **离线蒸馏（教师看完整观测）+ 多视图对比** | Eq.(5) 表示蒸馏、Eq.(6) 结果蒸馏 `L_RD = Σ Mean((Y_Teacher − Y_M,i)²)`；§3.5 多视图对比 | SOSR 是结构内门控，不是训练目标；但若第四章要做缺失鲁棒性，**Merlin 提供了"不改架构也能提升"的强对照** | 部分重合（训练目标维度） |
| 7 HD-TTS | 时空消息传递（TMP/SMP，isotropic/anisotropic） | 无显式目标缺口门（注意力以整体 mask 条件化） | 无源级可靠性 | **层级多尺度 + 解码注意力按观测与缺失模式加权（直接重合）** | 监督预测 | 摘要与 §4"weights interpreted as soft attention coefficients that select the appropriate spatiotemporal scale"；§5.3 结合 mask 可视化解码权重 | SOSR 唯一可能差异：把条件化从"整段 mask"细化到"目标传感器 support 缺口 + 源级可靠性"，且在工业遥测上验证 | **直接重合（部件 5）** |
| 8 SPIN | 稀疏时空注意力，valid/missing 编码 | 部分：missing query/key 分支（接收端状态进入注意力） | 无 | 无 | 重构/插补 | Fig.2 "Spatiotemporal point (missing query / valid key / missing key)" 与 valid/missing observation encoding | "按目标是否缺失改变注意力"已有实现，SOSR 需以量化差异（support 标量门 vs 二值分支）证明新增价值 | 部分重合（部件 2 近亲） |
| 9 MissTSM | masked cross-variate attention（同一时刻跨变元） | 部分：缺失变元被强制注意力 0（**目标侧屏蔽而非注入门控**） | 无 | 无 | 监督 + 不插补 | Eq.(4) `A_t = Softmax(Q K^T/√d + η M(t,:))`，η→−∞ 使缺失变元注意力为 0 | SOSR 与之差别在"目标缺口越大注入越多"的**方向性**主张，MissTSM 恰是反向屏蔽，二者不可混称 | 部分重合（部件 2/4） |
| 10 t-PatchGNN | Transformer + 时间自适应图学习（时变自适应图） | 无 | 无 | 可变观测数 patch（单尺度） | 监督预测 | 摘要与方法：transformable patching + time-adaptive GSL | SOSR 需说明与"patch 化 + 动态图"的差异是门控与可靠性，不是路由本身 | 部分重合 |
| 11 Hi-Patch | intra-patch / inter-patch 图层，**scale-specific graph networks** | 无 | 无（按采样密度分层而非可靠性） | **多尺度 + 变量差异化尺度处理** | 监督预测/分类 | 摘要：scale-specific graph networks 逐尺度提取全局时序与跨变量特征 | 与 SOSR 部件 1、5 高度相邻；差异只在"尺度权重是否由局部 support 决定" | 部分重合 |
| 12 GRU-D | 无跨传感器图 | 无 | **δ（距上次观测时间）+ mask 驱动衰减 = 源可靠性特征集先例** | 无 | 监督 | §decay：输入衰减 γ^x、隐状态衰减 γ^h，向经验均值衰减，W_γ 对角 | SOSR 的 freshness/delta/连续缺失长度若直接搬用即为已知特征；需证明在跨传感器权重上的新作用 | 部分重合（部件 3 直接先例） |
| 13 DATA-GRU | 双注意力（数据质量 + 医学知识） | 部分：按数据质量调权重 | **按数据质量给观测分配权重 = 源可靠性加权先例** | 无 | 监督分类 | 正文：w_Δt=1/log(e+Δt) 把时间间隔转权重；dual-attention 结构 | SOSR 的"源可靠性偏置"在结构上与之同型，差别是作用在跨传感器消息而非 RNN 状态 | 部分重合（部件 3） |
| 14 HU-GNN | 不确定性加权消息传递 | 无 | **邻居不确定性/可靠性调制消息** | 层级尺度（非时间尺度） | 图分类 | §3.3/§4.1：每个邻居按其不确定性加权聚合，不确定性逐层重算 | 非时序、非传感器场景；说明"可靠性加权消息传递"在图学习中已是通用做法 | 相关但不同 |

---

## 3. 重点精读 6 篇：公式/章节/代码定位与重合模块

### 3.1 Raindrop（ICLR 2022，证据 A）
- 定位：正文 §3 Raindrop；消息传递小节"Passing messages along sensor dependency graphs"；"Updating sensor dependency graphs"（边权按层更新与剪枝）；arXiv HTML/ar5iv 可核。代码：Zitnik Lab 项目页有官方实现（GitHub 仅证明存在）。
- 重合模块：**动态传感器图 + 观测驱动的消息注入**（对应 SOSR 部件 3、4 的"用观测状态调节跨传感器信息流"）。
- 差异：Raindrop 的图按样本动态、面向表示学习与分类；**没有 per-scale support，没有目标缺口门，没有以 (1−support) 为输入的 MLP 门控**。
- 判定：部分重合。

### 3.2 BiTGraph（ICLR 2024，证据 A）
- 定位：OpenReview PDF `O9nZCwdGcG` 正文 §3（`M` 为 mask，输入 `(X_{t-H:t}, M_{t-H:t})` 预测未来）；§4 多尺度实例 PartialTCN + biased GCN；Table 3 缺失消融；§6 结论"explicitly taking the missing values into consideration"。
- 重合模块：**mask 进入时间多尺度卷积 + 图空间结构**（对应部件 1、4 的"逐尺度、逐传感器处理缺失"）。
- 差异：BiTGraph 的"bias"是缺失模式的卷积传播，**不区分目标缺口与源可靠性**，也没有按支持度做尺度选择。
- 判定：部分重合。

### 3.3 S4M（ICLR 2025，证据 A）
- 定位：arXiv PDF §3.3 **Eq.(4)**：`h_t = A h_{t−1} + B o_t + E E_m(m_t;θ_m)`，`y_t = C h_t + D o_t + F E_m(m_t)`；§3.2 ATPM 原型库；消融表 `S4M (w/o mask)`、`S4M (w/o ATPM)`；官方代码 `github.com/WINTERWEEL/S4M`。
- 重合模块：**把缺失 mask 作为与表征并行的条件输入**（部件 1 的"mask 感知"部分）。
- 差异：单流 SSM，无跨传感器路由、无门控、无尺度路由。
- 判定：部分重合。

### 3.4 ChannelTokenFormer（ICLR 2026，证据 A）
- 定位：arXiv HTML §4 与 Fig.2（结构：local token + channel token 统一序列）、**Fig.3 掩码三条规则**、"Channel-wise Frequency-based Dynamic Patching"、"Training-Time Proxying for Test-Time Missingness via Channel-wise Patch Masking"、问题形式化 `ŷ = f_θ({(x^(i), m^(i), s_i)})`；官方仓库 `github.com/jinkwan1115/ChannelTokenFormer`（含 `run_irregular.py`、practical/missing 脚本）。
- 重合模块：**channel token 跨通道路由 + mask 引导注意力 + 动态 patch 多尺度**（部件 4、5 的核心形态）。
- 差异：缺失处理是"patch 存在与否"的结构遮蔽；**没有 (1−support) 的目标缺口门，没有源级可靠性（freshness/delta/连续缺失长度）偏置，也没有按 support 学尺度权重**。
- 判定：部分重合，**对 SOSR 部件 4/5 构成最强竞争**。若要主张组合创新，必须以"CTF 式二值处理 vs support 连续门控"的直接对照证明差异。
- 备注：作者名单本次仅从 Semantic Scholar 侧核到部分（记为 Jang、Park 等），**以 arXiv 2506.08660 元数据为准，报告定稿前需复核**。

### 3.5 CRIB（2025 预印本，证据 C）
- 定位：arXiv HTML §3；**Eq.(1)** 信息瓶颈目标；Fig.2（数据增强 → patch embedding → unified-variate attention → 一致性正则）；§3 明确"直接从部分观测预测，绕过插补"。
- 重合模块：**不插补直接预测 + mask 输入 + 跨变量注意力**（部件 4）。
- 差异：CRIB 的创新在训练目标（IB + 一致性），不在于门控/可靠性/尺度路由。
- 判定：部分重合；**同时是"不插补"路线的必引对照**。
- 备注：目前为预印本，作者与最终发表状态需复核。

### 3.6 Merlin（KDD 2025，证据 A）
- 定位：arXiv PDF §3.4 **Eq.(5)** 隐表示蒸馏、**Eq.(6)** 结果蒸馏；§3.5 多视图对比学习；§4.6 变缺失率训练；官方代码 `github.com/ChengqingYu/Merlin`（`train_teacher.py` / `train_student_merlin.py`）。
- 重合模块：**缺失鲁棒性这一目标本身**（第四章若转线，必须与之比较）。
- 差异：训练目标层面，不改架构。
- 判定：部分重合（训练目标维度）。**它同时是 SOSR 的方法边界对照：如果"teacher-student + mask 增强"就能拿到 SOSR 的大部分收益，则 SOSR 的结构门控不成立。**

---

## 4. 按 SOSR 部件拆解的重合矩阵

| SOSR 部件 | 已有先例（最强） | 重合度 | 尚未被覆盖的可能差异 | 风险 |
|---|---|---|---|---|
| 1. 每时间尺度、每 patch 计算传感器级观测支持 support=mean(mask) | HD-TTS（多尺度 + mask 条件化）、BiTGraph（多尺度卷积 + mask）、Hi-Patch（层级尺度）、CTF（patch masking） | **高** | "support 作为显式标量接口"本身可解释性更好，但这是工程表述差异 | 把统计量当创新 |
| 2. 目标传感器观测缺口 (1−support) + MLP 门控跨传感器消息注入 | MissTSM Eq.(4) 的负偏置屏蔽、SPIN 的 missing query/key 分支、FairAC/HU-GNN 的接收端条件聚合 | **中—高** | "缺口越大注入越多"的**方向性主张**与 MissTSM 相反，是可证伪的差异点 | 方向性可能被实验证伪 |
| 3. 源可靠性 = support + freshness + delta + 连续缺失长度 调节消息权重 | GRU-D（δ + mask 衰减）、DATA-GRU（数据质量注意力 + w_Δt）、HU-GNN（不确定性加权消息） | **高** | 把这些信号用于**跨传感器消息偏置**并检验单调性 | "support 高 ⇒ 更准"无文献支持，可能误伤低频有效传感器 |
| 4. 传感器关系 + Q/K/V 跨传感器路由 | Raindrop、t-PatchGNN、CTF channel token、Hi-Patch、GAT 系 | **极高** | 无（纯通用机制） | 纯工程 |
| 5. 按局部支持自适应融合时间尺度 | **HD-TTS（ICML 2024）直接重合**、Hi-Patch、BiTGraph | **高（直接）** | 工业遥测场景 + 与目标门控的联合 | **直接重合，不能称首次** |
| 6. 任务与协议不变（MetroPT/FD004/TEP 独立训练） | 本项目既有协议 | — | 应用与协议贡献 | 不能算算法创新 |

**Q3 结论**：截至本次检索，**没有同时具备"多尺度 + 目标缺口门控 + 源可靠性路由"的论文或代码**；但三块各自的最近邻都很近，且"多尺度 + 缺失条件化融合"已被 HD-TTS 直接占据。因此 SOSR 的**实质差异空间只剩部件 2 与部件 3 的联合形态**，且二者都需要独立消融证明其贡献非零。

---

## 5. 最近邻逐篇比较（为什么不是它们，以及差多少）

1. **HD-TTS（ICML 2024）**：比 SOSR 早、级别更高、任务同为"缺失下的时空图预测"。它已经回答了"多尺度表示按观测/缺失模式加权融合"这一问。SOSR 若只把"整段 mask"换成"逐传感器 support"，属于**细化**，不是新机制。
2. **ChannelTokenFormer（ICLR 2026）**：已经把"channel token 跨通道路由 + 掩码引导 + 动态 patch 多尺度 + test-time 缺失"打包成统一框架，并在真实工业数据上验证。SOSR 与其的差异集中在"连续 support 门控 + 源可靠性"两点。
3. **Hi-Patch（ICML 2025）**：多尺度 + 跨变量图，且按变量采样密度分层——已覆盖部件 1、5 的大部分动机。
4. **BiTGraph（ICLR 2024）/S4M（ICLR 2025）**：证明"mask 进入主干"是成熟范式；SOSR 的 mask 感知部分不构成差异。
5. **CRIB（2025）/Merlin（KDD 2025）**：分别给出"不插补直接预测"和"蒸馏+对比"两条**不改结构**的替代路线，是评估 SOSR 结构收益是否必要的关键对照。
6. **GRU-D / DATA-GRU**：源可靠性特征集与质量加权注意力的先例，说明部件 3 的**输入构造**不新。
7. **MissTSM / SPIN**：说明"目标侧是否缺失影响聚合"已有实现路径。
8. **GraFITi（AAAI 2024）**：本项目已有基线，作为缺失图预测的对照位保留。
9. **Raindrop（ICLR 2022）**：动态传感器图 + 消息传递的起点，作为"观测驱动图"的引用锚点。

---

## 6. 创新性四级判定

| 级别 | 归属内容 | 依据 |
|---|---|---|
| **直接重合** | 部件 5（按观测/缺失模式条件化多尺度融合）；部件 4（Q/K/V 跨传感器路由）；部件 1 的 mask 感知输入 | HD-TTS ICML 2024；Raindrop/CTF/Hi-Patch；BiTGraph/S4M |
| **部分重合** | 部件 2（目标缺口门控）、部件 3（源可靠性加权） | MissTSM/SPIN/FairAC/HU-GNN；GRU-D/DATA-GRU |
| **组合创新候选（有条件）** | "多尺度 support + 目标缺口门 + 源可靠性偏置"三者在**同一模型内联合** | 未检索到同时具备三者的公开工作；但**必须先有相对容量匹配基线的稳定独立收益**才允许使用该级别 |
| **经验/应用贡献** | 在 MetroPT-3 / FD004 / TEP 上的缺失鲁棒性评测协议、跨任务统一接口、按传感器/工况分层的可靠性分析 | 三个数据集均非真实缺失，"工业遥测缺失压力测试 + 协议"是更稳的贡献口径 |

**默认定级：部分重合 + 组合创新候选（待验证）**。在实验矩阵（§7）未完成前，论文文本不得出现"首次提出""填补空白"类表述。

---

## 7. 三数据集适配审查

### 7.1 MetroPT-3
- **缺失语义**：UCI MetroPT-3 数据集页明确 "Has Missing Values: No"，字段级 Missing Values 全为 No；配套论文（Davari 等 DSAA 2021 / MetroPT Scientific Data 2022）记录采集为固定频率、**"no missing values (no pre-processing technique was applied)"**。→ **该数据集不存在原生缺失，异步性也不成立**；SOSR 所需 mask 只能来自人工注入。
- **时间戳**：UCI 版 1,516,948 行、0.1 Hz、2020-02 至 2020-08；描述中存在 1 Hz 与 0.1 Hz 两种说法，**行距是否严格均匀需在本地数据上核验**（属于数据核验，不属于创新证据）。
- **划分**：保持官方首月训练、后续测试的时间隔离；首月内部再切 train / validation-selection；归一化只用 train。
- **指标**：raw 单位下 overall 与逐传感器 MAE/RMSE；缺失压力测试须单独列表并标注"人工 mask"。
- **失效情形**：数字传感器（COMP/MPG/LPS 等 0/1 信号）上 mean(mask) 支持度与"传感器准确度"几乎无关；低频但关键的量（如 Oil_temperature）会被 freshness 衰减误伤。
- **禁止**：把人工 mask 压力测试写成"真实部署缺失证据"或"天然异步数据"。

### 7.2 C-MAPSS FD004
- **缺失语义**：NASA 官方说明为 26 列完整逐 cycle 快照，"contaminated with sensor noise"，**未提供缺失机制**；常见预处理（特征筛选、移动中值滤波、插补）是研究者引入的。→ FD004 上的缺失**只能是人为遮挡**。
- **条件结构**：3 个 operational settings、6 种工况、2 种故障模式（HPC + Fan）；单位 unit=发动机、时间单位 cycle。
- **划分**：必须 engine-level 划分（不跨发动机泄漏）；工况归一化按 train 拟合。
- **标签语义**：raw RUL、training cap（常见 125/150，本项目需以实际 config 为准）与 prediction output **三个概念不得混称**；概率/区间产物若用 cap 标签训练，不得声称覆盖 raw RUL 分布。
- **核验点**：NASA readme 与若干二次资料对 FD004 train/test 发动机数记载不一致（248/249 与 249/248 两种写法），**报告定稿前须以本地 `dataset/CMAPSSData` 实际行数为准**。
- **失效情形**：全观测下 support=1、(1−support)=0，SOSR 的门控与可靠性项**自动退化**，模型退化为普通跨传感器注意力——在 FD004 上若不注入缺失，SOSR 与 w/o SOSR 应几乎无差异，这本身是一个**重要的可证伪预测**。

### 7.3 TEP
- **缺失语义**：主流版本（Braatz/Rieth）为仿真数据，训练 500 次 run、每 run 500 个样本（25 h），测试每 run 960 个样本（48 h），3 分钟采样、52 列（含 faultNumber/simulationRun/sample）；**缺失是否出现取决于所用版本与文件协议**（例如某些版本列名/行缺失、RData 与 CSV 差异），不是数据本身的语义。必须逐版本核查后再下结论。
- **划分**：run-level 划分（Rieth 版训练/测试 seed 不重叠，可直接沿用）。
- **输出语义**：必须区分**检测概率**（fault vs normal）、**诊断概率**（故障类别）与**故障延迟**（注入时刻 1 h/8 h 之后的检出时间）；三者不能用同一指标叙述。
- **禁止**：把回归型传感器插补/门控机制的收益直接等同于分类概率质量或检测延迟改善；SOSR 在 TEP 上只能作为扩展，且不得用测试 run 调阈值。
- **失效情形**：TEP 的"缺失"若为人工注入，注入模式（随机 vs 块状）会直接决定 support 统计分布，结论对注入协议敏感。

**三数据集总体结论**：**没有任何一个可以被称为"天然异步或真实缺失"**。若第四章改为缺失鲁棒性主线，属于研究范围调整，须在论文与 `plan/` 中显式声明"缺失为受控人工注入，作为压力测试协议"。

---

## 8. 实验矩阵（含容量匹配基线与逐部件消融）

> 全部为**待执行设计**，不是已有结果。所有臂共享：同一数据协议、split identity、mask 注入协议（固定 seed、训练前固化）、归一化来源、训练预算、checkpoint 只按 validation 选择、**test 只在最终确认时评估一次**。

| 编号 | 变体 | 回答的问题 | 判据 |
|---|---|---|---|
| A0 | **w/o SOSR**（= Unified V3C 原样） | 基准 | — |
| A1 | **mask-only**（mask/δ 作为输入特征，无门控、无路由） | 收益是否只是"看到 mask" | 若 A1≈SOSR，SOSR 门控不成立 |
| A2 | **global mixer**（跨通道全连接/全注意力混合，无条件化） | 是否需要条件化 | 若 A2≈SOSR，路由不成立 |
| A3 | **容量匹配 MLP**（参数量 ±5%，加宽加深至与 SOSR 同量级，结构为普通 MLP 拼接） | 收益是否只是参数变多 | 若 A3≥SOSR−δ，**不升级创新主张** |
| A4 | **w/o target deficit**（去掉 (1−support) MLP 门） | 部件 2 独立贡献 | 需在 ≥2 seed 出现一致退化 |
| A5 | **w/o source reliability**（去掉 support/freshness/delta/连续缺失长度偏置） | 部件 3 独立贡献 | 同上 |
| A6 | **w/o scale routing**（尺度融合改为固定/均值权重） | 部件 5 独立贡献 | 同上 |
| A7 | **SOSR 全量** | 组合效果 | 相对 A0/A1/A2/A3 均有稳定增益 |
| A8（对照） | **shuffled reliability**（源可靠性统计量在传感器间随机打乱后输入） | 单调性/方向性主张是否真的在起作用 | 若 A8≈A7，可靠性偏置未被有效使用 |
| A9（对照） | **Merlin 式蒸馏 + mask 增强**（不改结构的缺失鲁棒基线） | 结构门控是否优于训练技巧 | 若 A9≥A7，SOSR 结构不必要 |
| A10（协议） | **全观测（无注入）** | 全观测下 g=0 是否让 SOSR 退化为 A0 | 用于检验"第三章一致性"主张 |

补充规则：
- **容量匹配**：以参数量与 FLOPs 双指标对齐（±5%），并记录训练/推理开销；报告每臂参数量。
- **缺失注入协议**：point / spatial-block / spatiotemporal-block 三档，注入率 10%/30%/50%；mask 文件训练前固化，同一 seed+rate 复用；**注入只发生在历史窗口，不得读取未来观测构造可靠性**。
- **seed**：单 seed 做方向筛选；通过者进入 ≥3 seed 才谈稳定性。
- **统计单位**：MetroPT 按时间块、FD004 按 engine、TEP 按 run 做配对比较；窗口不视为独立样本。
- **指标**：MetroPT/FD004 raw 单位 MAE/RMSE（FD004 另报 NASA score，明确 raw vs cap）；TEP 报检测/诊断/延迟三类分开的指标。人工缺失下的结果**单列"压力测试"表**，不与主表混排。
- **决策门**：A7 不优于 A3 ⇒ 不得主张组合创新；A4/A5/A6 无一致退化 ⇒ 删掉对应部件再做主张。

---

## 9. 所需图表（待生成，不是已有结果）

| 图/表 | 回答的问题 | 结论边界 |
|---|---|---|
| 图 A 方法结构图 | support、目标门、源可靠性、尺度融合的真实接口与数据流 | 只解释方法 |
| 表 A 实验主表（A0–A10） | 各变体是否真的更好，参数量与开销 | 单 seed 只作筛选 |
| 图 B 消融瀑布图 | 部件 2/3/5 的独立贡献 | 控制变量不足不得归因 |
| 图 C support–收益校准曲线 | "缺口越大注入越多"方向是否成立 | 仅历史输入可得条件 |
| 图 D 可靠性单调性检验（A8 打乱对照） | 源可靠性偏置是否被有效使用 | 打乱对照必须做 |
| 图 E 缺失模式×尺度权重热图 | 尺度选择是否随缺失模式变化 | 需与 HD-TTS 结论对照叙述 |
| 表 B 三数据集缺失语义与任务适配表 | 缺失是真实还是注入、单位、标签语义 | 不得混称 |
| 图 F（可选）TEP 检测延迟分布 | 注入缺失对检测延迟的影响 | 需 run 级统计 |

---

## 10. 任务书 §2 专项假设审查

| 假设 | 审查结论 |
|---|---|
| 观测比例高 ⇒ 传感器准确 | **无文献支持且很可能错误**。GRU-D 明确"informative missingness"，观测缺失本身携带信息；DATA-GRU 用质量注意力而非比例。必须把 support 定义为**可观测性**而非**准确度**，并用 A8 打乱对照检验。 |
| freshness 衰减有益 | **会误伤低频但有效传感器**（如油温、退化相关的慢变量）。建议 freshness 项改为可学习且可关闭的形式，并报告按采样频率分层的消融。 |
| 目标缺失越多 ⇒ 应注入越多消息 | **方向未被文献证明，且 MissTSM/SPIN 是反向（屏蔽）**。这是最可证伪的一条，必须用图 C 直接检验；若方向相反，部件 2 需要重新设计（改为"不确定性驱动"而非"缺口驱动"）。 |
| 所有源都不可用时对无效源做 softmax | **数学上失效**（softmax 会把权重分配给无效源）。必须设计显式回退（如空源集合 → 仅自身/先验嵌入），并新增空源用例测试。 |
| 源可靠性偏置保证单调关系 | **未被任何检索到的文献保证**；HU-GNN 等用可学习不确定性，不承诺单调。若论文要写单调，必须先给出构造性保证（如单调参数化）或改为经验检验。 |
| 完全观测下 g=0 ⇒ 与第三章预测一致 | **只在残差路径关闭时成立**；若改动尺度融合或联合训练，A10 必须实测一致性，不能由 g=0 推出。 |
| 训练/推理不得读取未来观测 | 必须在实现层面保证 support/freshness/delta/连续缺失长度**仅由历史窗口计算**，并写入协议检查项。 |

---

## 11. 不依赖人为缺失的备选方向（防止检索只找支持性证据）

**备选方向（建议）**：以**自然可观测的条件变量**（FD004 的 6 种工况与故障模式、MetroPT 的模拟/数字传感器模态与压缩机运行状态、TEP 的 run 与故障类型）作为条件化信号的多尺度融合与不确定性/校准建模，**全程不注入 mask**，从而：
1. 不需要"天然异步/真实缺失"的表述，避免范围漂移；
2. 条件变量在推理时真实可得，不存在读取未来的问题；
3. 与已有 `plan/2026-10-09-ch4-innovation-and-figure-plan.md` 的条件化思路兼容，可作为 SOSR 不通过时的退路。

（该备选同样是待验证假设，不构成已有贡献。）

---

## 12. 未解决问题

1. ChannelTokenFormer、CRIB 的完整作者名单与最终发表状态需以 arXiv 元数据/OpenReview 复核。
2. FD004 train/test 发动机数在不同来源记载不一致，须以本地数据核验。
3. MetroPT-3 行时间戳是否严格均匀、是否存在采集空档，需在本地数据上统计（不作为创新证据）。
4. TEP 所用具体版本/文件格式决定缺失语义，须先固定版本再写适配结论。
5. 当前 Unified V3C 是否已暴露逐 patch/逐传感器粒度的张量接口，需按代码逐处核对（本次未核到足以断言的证据）。
6. HD-TTS 的解码注意力究竟以"整段 mask"还是"逐节点 mask"为条件，其代码实现细节需在复现阶段核对。
7. 未能访问全部付费/受限全文的部分（如 DATA-GRU 的完整公式编号）已在证据等级中标注。

---

## 13. 参考链接（每条对应上文具体判断）

- Raindrop: https://arxiv.org/abs/2110.05357 ；ICLR 版 PDF https://zitniklab.hms.harvard.edu/publications/papers/raindrop-iclr22.pdf ；项目页 https://zitniklab.hms.harvard.edu/projects/Raindrop
- BiTGraph: https://openreview.net/pdf?id=O9nZCwdGcG ；代码 https://github.com/chenxiaodanhit/BiTGraph
- S4M: https://arxiv.org/abs/2503.00900 ；ICLR 2025 论文页 https://proceedings.iclr.cc/paper_files/paper/2025/hash/7b2f0758334389b8ad0665a9bd165463-Abstract-Conference.html ；代码 https://github.com/WINTERWEEL/S4M
- ChannelTokenFormer: https://arxiv.org/abs/2506.08660 ；OpenReview https://openreview.net/pdf?id=r4ZamwBE8P ；代码 https://github.com/jinkwan1115/ChannelTokenFormer
- CRIB: https://arxiv.org/abs/2509.23494
- Merlin: https://arxiv.org/abs/2506.12459 ；DOI https://dl.acm.org/doi/10.1145/3711896.3737046 ；代码 https://github.com/ChengqingYu/Merlin
- HD-TTS: https://proceedings.mlr.press/v235/marisca24a.html ；https://arxiv.org/abs/2402.10634
- SPIN: https://arxiv.org/abs/2205.13479 ；NeurIPS 2022 页 https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html
- MissTSM: https://arxiv.org/abs/2502.15785
- t-PatchGNN: https://proceedings.mlr.press/v235/zhang24bw.html ；代码 https://github.com/usail-hkust/t-PatchGNN
- Hi-Patch: https://proceedings.mlr.press/v267/luo25r.html
- GRU-D: https://www.nature.com/articles/s41598-018-24271-9 ；https://arxiv.org/abs/1606.01865
- DATA-GRU: https://cdn.aaai.org/ojs/5440/5440-13-8665-1-10-20200511.pdf
- HU-GNN: https://arxiv.org/abs/2504.19820
- SAITS: https://arxiv.org/abs/2202.08516 ；DOI https://doi.org/10.1016/j.eswa.2023.119619
- GraFITi: https://ojs.aaai.org/index.php/AAAI/article/view/29560
- MetroPT-3: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset ；https://www.nature.com/articles/s41597-022-01877-3
- C-MAPSS: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- TEP（Rieth 版）: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1

---

## 14. 最终建议

1. **继续 SOSR 的最小证据**：在同一冻结 V3C 编码器与协议下，A7 在 validation 上同时优于 A0、A1、A2、A3（容量匹配 MLP），且 A4、A5、A6 各自产生方向一致的退化，A8（打乱可靠性）显著差于 A7；以上在 ≥2、正式 ≥3 个 seed 复现。
2. **应删除或改造的部件**：
   - 删除"首次提出按局部支持融合尺度"的表述（HD-TTS 已直接占据），改为"在工业遥测上的条件化尺度融合对照"；
   - 把 support 从"准确度"改述为"可观测性"，去掉单调性硬主张，改为可检验假设；
   - freshness/delta 项改为可学习可关闭，并按采样频率分层报告；
   - 增加空源回退分支，禁止对无效源做 softmax；
   - 部件 4（Q/K/V 路由）作为通用实现说明，不列为创新点。
3. **放弃 SOSR 的触发条件**（满足任一即放弃）：A3 容量匹配 MLP 达到 A7 ≥95% 的效果；A4/A5/A6 无稳定退化；收益只在人工注入缺失下出现、全观测（A10）下无差异且第三章一致性被破坏；或 A9（蒸馏式）以更小代价取得同等收益。
4. **第四章推荐主线（≤40 字，可被实验否证）**：观测支持条件化路由须胜过容量匹配基线（23 字）。
5. **下一步唯一优先实验**：MetroPT-3 单 seed、固定 V3C 编码器与冻结注入协议，先跑 **A0 / A1 / A3 / A7 四臂 validation 对照**，据此决定 SOSR 是否进入完整消融矩阵。
