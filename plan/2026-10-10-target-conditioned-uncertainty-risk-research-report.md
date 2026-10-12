# 预测目标条件化不确定性、Conformal 校准与风险评分：时序预测文献和代码调研

- 调研日期：2026-10-10
- 任务：围绕四个方向（目标条件化不确定性 / 不确定性驱动融合 / 多步 conformal 与多变量覆盖 / 风险分数与事件级校准）做查新
- 结论性质：**纯文献与代码调研**。未运行实验，未修改 `code/`、模型或 `result/`。

---

## 0. 前置声明

- **KST ProbFlow 未实现**，本项目无其代码，也不存在可核验的官方实现。本报告不将其作为既有方法检索。
- 当前代码基线为 **Unified V3C 点预测模型**。
- 不要求联合采样；不把 marginal coverage 说成 joint coverage；不把风险分数说成故障概率；attention 权重不解释为因果关系。

---

## 1. 执行摘要

**代码核验结论（项目代码核实，见 §6）**：任务书对基线的描述**基本准确，但有一处重要偏差**——V3C 已有非交叉分位数与 pinball loss、已有 TEP 的 sigmoid/softmax 概率输出与**按 run 隔离的事件级检测延迟/误报评估**；确实缺失的是 CRPS/NLL/PICP/MPIW/interval score 与 validation-only conformal 校准层。

**四个方向的核心判断**：

| 方向 | 最近邻状态 | 建议定位 |
|---|---|---|
| ① 目标条件化不确定性 | DeepAR/GPVAR 已给**参数化分布**，逐变量、逐 horizon 的异方差是标准做法；但**未见"以预测目标查询生成不确定性"** | 组合候选，证据不足以作算法主张 |
| ② 不确定性驱动融合 | **确有数例**（MoGU/UGGM/CGT/IBN/Under-Cali），但**无一同时满足"逐传感器连续加权 + 跨变量轴 + stop-grad + 硬回退"**；Gateformer 的门控由输入自身生成，非不确定性 | **组合候选**，且必须显式处理退化通道 |
| ③ 多步 conformal | Stankevičiūtė(NeurIPS'21) Bonferroni、AgACI(Zaffran et al. 2022)、CAFHT(AISTATS 2024)、FLIPR(PMLR 2026)、O²CP(2025) 已覆盖多种修正 | **更适合作为校准工具**，非独立第四章 |
| ④ 风险分数与事件级校准 | **Conformal Risk Control (ICLR 2024 Spotlight)** 直接把任意单调损失（如 FNR）控制到期望风险上界 | **最有价值的贡献方向**，但需限定为协议贡献 |

**是否值得继续**：值得，但**应把重心从"概率预测模型"移到"校准与风险评估"**。理由：模型侧组件（分位数、Gaussian、低秩、copula、conformal）全部成熟；未占的接缝是**工业多传感器 + 事件级风险 + 严格协议**这一组合，而它恰是审稿人最难反驳的工程与协议贡献。

**证据等级**：**B**。身份与机制经官方页/代码核验（A 级），但所有工业数据上的适配性均未核验（本轮无实验）。

---

## 2. 检索范围、检索式和证据限制

### 2.1 数据库与日期
- arXiv abs 页（逐篇核验标题/作者/日期）、OpenReview、NeurIPS/ICML(PMLR)/AAAI/IJCAI 官方 proceedings、期刊官网、GitHub API
- **检索日期：2026-10-10**；时间范围 **2020–2026**，保留 CQR(2019)、GPVAR(2019)、Stankevičiūtė(2021) 等奠基工作。

### 2.2 检索式（四个方向）
**方向1**：`target-conditioned uncertainty forecasting`、`horizon-aware uncertainty estimation`、`horizon-specific predictive uncertainty`、`sensor-specific predictive uncertainty`、`variable-conditioned probabilistic forecasting`、`context-conditioned heteroscedastic forecasting`、`task-conditioned probabilistic forecasting`、`conditional prediction interval time series`、`horizon-dependent quantile forecasting`、`target-aware variance forecasting`、`time series uncertainty calibration by horizon`、`heteroscedastic uncertainty head forecasting`

**方向2**：`uncertainty-aware sensor fusion time series`、`uncertainty-guided attention forecasting`、`confidence-aware variable selection`、`risk-aware sensor routing`、`probabilistic cross-channel attention`、`heteroscedastic graph forecasting`、`uncertainty-aware dynamic graph time series`、`uncertainty-guided variable interaction`、`confidence-weighted multivariate forecasting`、`uncertainty-aware feature fusion temporal`、`negative transfer cross-variable forecasting`、`uncertainty-based sensor selection forecasting`

**方向3**：`multivariate conformal forecasting`、`multi-horizon conformal prediction`、`simultaneous coverage time series`、`conditional coverage forecasting`、`conformal prediction under distribution shift`、`grouped conformal forecasting sensors`、`adaptive conformal inference time series`、`conformalized quantile regression forecasting`、`multivariate prediction region forecasting`、`sensor-wise conformal calibration`、`horizon-wise conformal intervals`、`risk-controlling prediction sets`

**方向4**：`predictive distribution based risk score`、`forecast uncertainty anomaly detection`、`probabilistic forecast exceedance risk`、`calibrated predictive maintenance risk`、`event-level risk calibration time series`、`uncertainty-aware fault detection`、`forecast-based anomaly detection multivariate`、`conformal risk control time series`、`predictive maintenance probability calibration`、`risk scoring from prediction intervals`、`probabilistic forecasting event detection`、`distributional forecast anomaly score`

### 2.3 纳入/排除标准
- **纳入**：正文可定位公式/机制，或官方页/代码可核验；arXiv ID 经 abs 页核验标题后才可填。
- **排除**：仅检索摘要可得且无法二次核验（降级并标注）；第三方博客转述（仅作线索）；**不得凭 README 宣称仓库与论文一致，须定位实际代码**。

### 2.4 证据限制（如实记录，必须随结论引用）
1. **Jina Reader 本机已被封禁**（`401 bad IP reputation`）；**arXiv API 限流严重**，大量 ID 改用官方 proceedings 页或 `arxiv.org/abs` HTML 核验。
2. **"未发现某方法"仅限本检索范围**，不等于不存在；未做完整引用链回溯。
3. 未系统检索 IEEE Xplore/ACM DL/Scopus/WoS/知网。
4. **本轮无任何实验**，所有方法在三个工业数据集上的适配性与数值表现均未核验。
5. ⚠ **我方记忆中的 arXiv ID 在此前任务中曾大面积出错**（概率预测轮 9 个 ID 全错），本报告所有 ID 均已逐条核验标题。

---

## 3. 四个方向的方法分类

### 3.1 方向一：预测目标条件化的不确定性估计

| 子类 | 代表 | 条件形式 | 是否真正改变分布参数 | 逐 horizon/变量校准报告 |
|---|---|---|---|---|
| **参数化边缘头** | DeepAR (IJF 2020)、GPVAR (NeurIPS 2019) | 隐含（编码器状态） | ✅ 输出 $\mu,\sigma$ 逐时刻变化 | 部分（论文用 ρ-risk、coverage） |
| **逐 horizon 分位数头** | MQ-RNN/MQF2（**核验受限**）、**V3C 现有 StateHead** | 隐含（`base+pred_len` 每步独立） | ✅ 每步独立分位数 | ❌ V3C **当前无任何校准指标** |
| **条件查询生成不确定性** | 本次**未发现直接先例** | — | — | — |
| **工况/上下文条件化不确定性** | 时间序列上的静态协变量（如 TFT 类） | ⚠ 多作普通输入特征 | 见 §4 逐篇判定 | 少 |

### 3.2 方向二：不确定性驱动的跨变量融合

> ⚠ **本节结论已按子代理深读结果修正**——初稿写"未发现不确定性进入融合的先例"，**这是错的**。确有数例，但都存在实质缺陷。

| 情形 | 代表 | 不确定性参与融合？ | 回退分支？ | 抗"放大尺度逃惩罚"？ |
|---|---|---|---|---|
| ① attention 仅表示交互 | MTGNN、Graph WaveNet、iTransformer、Crossformer | ❌ | ❌ | — |
| ② **不确定性进入融合/路由** | **MoGU**（arXiv:2510.07459）精度加权专家 | ✅ 但融合的是**专家**非变量 | ❌ | **❌ 代码确认无 stop-grad** |
| ② | **UGGM**（arXiv:2603.07753, ICLR'26 Workshop）$A_{ij}\propto e^{q_i^\top k_j/\sqrt d}\cdot G(\sigma_i,\sigma_j)$ | ✅ 但作用在**时间维** self-attention，**非跨变量** | ❌ | **❌ $L_{gate}$ 只约束平滑性，不约束 $\sigma$ 大小** |
| ② | **CGT**（arXiv:2604.17998）目标特定 gate + safety-gated blending | ✅ 且为**变量级**（因果父节点 vs 非因果辅助路径） | ✅ **硬回退 $\gamma_{used}=0$**（式42） | ✅ **三重防护**：stop-grad 隔离（式23/24/26/28/29）+ $L_{res}$ 幅度惩罚（式30）+ 硬回退 |
| ② 但作用域受限 | **IBN**（arXiv:2509.07725）$x̂^{UAI}=\mu/(1+\sigma)$ | ⚠ 只衰减**插补变量**贡献，非真实传感器间仲裁 | ✅ 结构性抑制 | ✅ 除数 $1+\sigma$ 单调，$\sigma\to\infty$ 时贡献趋 0 |
| ② 但路由对象是样本 | **Under-Cali**（arXiv:2605.28603）双专家 | ⚠ 路由**样本/专家**非变量 | ✅ 双专家 | ✅ **最干净**：UE 用归一化 MSE 直接监督，不经 NLL 反推 |
| ③ 不确定性仅预测后报警 | 多数工业 PdM 工作 | ❌ | ❌ | — |
| ④ 自身历史回退分支 | **Gateformer**（Workshop 2025，代码核实） | ⚠ 门控由输入自身生成，**非不确定性** | ✅ | ⚠ 未核验 |

> 🚨 **本方向最重要的实验设计发现——退化通道**：
> MoGU 官方代码 `models/MoE.py:63-65` 为 `inv_var = 1.0/(expert_unc+1e-8)`，权重与被加权的 NLL 项**共享同一个 $\sigma_i^2$**，**无 stop-gradient、无幅度惩罚、无独立分支**。模型可把某专家 $\sigma$ 推高来降低该项 NLL 贡献，而**点估计完全不受影响**——这是一个纯退化解通道。UGGM 式(9) 同理。
> **对项目的意义**：若采纳不确定性加权融合，**必须做"冻结 $\sigma$ 只重训预测头"的对照**，并检查训练后 $\sigma$ 分量与实际残差方差是否强正相关（若强正相关说明 $\sigma$ 在吸收真实方差而非被博弈）。

### 3.2b 方向一补充：时间结构化不确定性

| 方法 | 条件形式 | 机制 | 证据 |
|---|---|---|---|
| **VolDy-VAE**（arXiv:2603.24254，2026-03 预印本） | **预测步 + lookback 波动状态** | location/scale 路径分离；尺度状态从 lookback **迁移**到 horizon 并用 **GRU 递归演化**；adaptive attenuation（高方差观测对 location 影响更低但保留其不确定性）；消融显示 MLP→GRU 在 ETTh2 H=720 CRPS 0.197→0.174 | 正文+表格已核实（PDF 全文） |
| **Rollcast**（arXiv:2609.05561，2026-09 预印本） | **状态（state-dependent）** | 相似历史状态检索残差分布 + rolling anchors 的 state-dependent softmax gate；**诚实报告覆盖缺口**：coverage 86.2%（名义 90%）、91.5%（名义 95%） | 摘要级（数字来自摘要） |
| **DeRegiME**（arXiv:2605.19231，2026-05） | **隐 regime + horizon** | GP regime 取代固定似然，stick-breaking gate 剪枝 regime 数；稀疏变分 GP 后验 + Student-t 似然 | 摘要级 |
| **Climate-Invariant Conformal**（arXiv:2607.11470，2026-07） | **站点/气候分组 + horizon** | 条件进入**分组非对称 split-conformal**，明确声称"**无需逐 horizon 调参**" | 摘要级。⚠ **这是方向一的反例**：证明 horizon 条件化 conformal 有时不如不条件化 |

### 3.3 方向三：多步 conformal 与多变量覆盖

> 覆盖定义严格分列：**M**=边际 / **SH**=同时跨 horizon / **MJ**=多变量联合向量 / **C**=条件 / **LR**=长时平均。

| 方法 | 覆盖目标 | 多 horizon 修正 | 校准集分离 | 交换性假设 |
|---|---|---|---|---|
| **CF-RNN** (Stankevičiūtė, NeurIPS 2021) | **SH**（逐 horizon 同时，式(7) 用 $\alpha/H$） | **Bonferroni** | ✅ | 极弱，仅"最小交换性" |
| **ACI** (Gibbs & Candès, NeurIPS 2021) | **LR** + 漂移小时近似 M | 在线逐点更新 | 在线滚动 | **不要求** |
| **AgACI** (Zaffran et al., ICML 2022) | LR，参数无关 | 在线 + 专家聚合 | 在线滚动 | **不要求** |
| **MWCP** (Wang & Hyndman, 2024) | LR（NexCP 加权） | 缺口由交换分数向量 TV 距离控制 | 在线 | 放弃交换性 |
| **ConForME** (CoPPA 2024 workshop) | SH | 时序依赖建模的联合覆盖 | ✅ | 利用时序内部依赖 |
| **CAFHT** (Zhou et al., 2024) | **SH-M + SH-C**（分别报告） | ACI 自适应逐序列 $\alpha_t$ | ✅（且需额外划分选 $\gamma$） | 多序列联合可交换 |
| **O²CP** (Li et al., 2025 预印本) | **逐 horizon LR 且 $\forall h$ 同时** | 可接受集内跨 horizon 优化 | 在线，无需额外 i.i.d. 轨迹集 | 非 i.i.d. |
| **CopulaCPTS** (Sun & Yu, **ICLR 2024**) | SH，**时间轴**联合 | **copula 联合残差**（非 Bonferroni） | ✅ | 有限样本有效 |
| **MultiDimSPCI** (Xu et al., **ICML 2024**) | **MJ + C**（条件覆盖 gap 界） | 每步一个多维**椭球**区域 | ✅ 测试时逐步重估半径 | 显式非交换（借 SPCI） |
| **MSR** (Braun et al., 2025) | **MJ + C**（充分条件） | 学到局部协方差白化 | ✅ | 交换性仍在 |
| **FLIPR** (English & Lippert, 2026) | **K-family-wise**（$\Pr[V>K]\le\epsilon$） | **K-th order 非一致性分数** + 逐 horizon 标准化 | ✅ | 需校准集 |
| **Vaze 2026** | **同时覆盖 + 同时效率** | ACI 更新 = pinball loss 上的在线梯度下降 | 在线 | 非平稳，含 regret 界 |
| **Conformal Risk Control** (ICLR 2024) | **任意单调损失的期望** | 不适用（不是区间） | ✅ | 有限样本 $O(1/n)$ |

> ⚠ **覆盖概念必须区分**：CF-RNN 给的是"逐 horizon 在 Bonferroni 下同时成立"，**不是**联合分布覆盖；FLIPR 给的是 K-family-wise，$K=1$ 才退化为 simultaneous；**CopulaCPTS 的联合轴是时间维，不是变量维**。
> ⚠ **CF-RNN 的已知缺陷**：CAFHT 正文明确批评"CFRNN 在大 horizon 下 band 宽到无信息量"；MWCP 也承认时序中覆盖率缺口可能很大。

### 3.4 方向四：风险分数与事件级校准

| 风险分数类型 | 是否需要概率 | 阈值来源 | 是否可校准为概率 |
|---|---|---|---|
| 点预测残差 / 标准化残差 (R1) | ❌ | valid 分位数 | ⚠ 只能排序，非概率 |
| Gaussian NLL (R2) | 需要参数化头 | valid | ✅ 若经 **exceedance** 校准（见下） |
| 尾部/超限概率 (R3) | 需要分布 | valid | ⚠ **需 exceedance calibration，ECE 不等价** |
| 区间越限指示 (R4) | 需要区间 | valid 阈值 | ⚠ 同上 |
| Mahalanobis / 多变量距离 (R5) | 需要协方差 | valid + Hotelling T² 渐近精确阈值 | ⚠ 维度升高误差增大 |
| conformal nonconformity / p-value (R6) | ❌ | **calibration split** | ✅ 有限样本保证 |
| 多次采样超限比例 (R7) | 需采样 | valid | ⚠ |
| 分类概率 (R8) | 已有 sigmoid | valid + Platt/isotonic | ✅ **可校准为概率** |
| 事件级累积 (R9) | 需要 | **必须在 valid 固定 persistence rule** | ⚠ **事件率 ≠ 时点概率** |

> 🚨 **本方向最重要的方法学警示（Gneiting, Balabdaoui & Raftery, JRSS-B 2007, DOI 10.1111/j.1467-9868.2007.00587.x）**
> 该文正文 §2.1 用构造性反例证明 **probabilistic calibration（PIT 均匀）与 exceedance calibration 不等价**——其 Example 1 的对偶中，预测器是 probabilistically calibrated 但原文明确"not exceedance calibrated"。
> **后果**：风险阈值/越限判定的数学对象是 $F_t(x)$ 的**上尾质量** $1-F_t(x)$，对应 exceedance calibration，**不是** PIT 均匀性。**任何只报 ECE/PIT/reliability diagram 就把分位数或 $P(y>\tau)$ 当作风险分数的做法，其风险分数本身未经验证。**
> Allen et al. (JASA 2025, arXiv:2407.03167) 进一步证明 probabilistic calibration **不蕴含** tail calibration（或反之），并要求**分别报告** occurrence ratio 与 excess-PIT。

---

## 4. 核心论文证据表

> 重合程度：**直接重合 / 部分重合 / 机制相近 / 背景相关**。证据等级：**正文核验 / 官方摘要 / 仅线索**。

### 4.1 方向一 + 方向二

| 字段 | DeepAR | GPVAR（"DeepVAR"） |
|---|---|---|
| 标题 | DeepAR: Probabilistic forecasting with autoregressive recurrent networks | High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes |
| 作者 | Salinas, Flunkert, Gasthaus, Januschowski | Salinas, Bohlke-Schneider, Callot, Medico, Gasthaus |
| 年份 | 2020（arXiv v1 2017） | 2019（NeurIPS 2019） |
| 状态 | 期刊正式版 | 会议主会 |
| DOI/arXiv | DOI 10.1016/j.ijforecast.2019.07.001；arXiv:1704.04110 | arXiv:1910.03002 |
| 研究问题 | 从相关序列集合学全局概率预测模型 | 高维多变量协方差估计（$O(N^2)$/$O(N^3)$ 不可行） |
| 任务与数据集 | 概率预测；零售/能源/交通 | 概率预测；交通等 6 个真实集 |
| 条件变量 | 隐含（状态 $h_t$） | 隐含 + 边缘非参数 CDF 变换 |
| 概率输出 | 样本/分位数 | 边缘 copula + 联合低秩 Gaussian |
| 不确定性是否影响融合 | ❌ DeepAR 边缘独立 | ⚠ 协方差建模变量依赖，但**不用于控制融合** |
| 回退分支 | ❌ | ❌ |
| 校准 | coverage 曲线（论文） | CRPS/CRPS-Sum（论文） |
| 关键公式 | $Q_\Theta$ 乘性似然因子分解 | $\Sigma_t=L_tL_t^\top+\mathrm{diag}(d_t)$，$O(Nr^2+r^3)$，$r=10$ |
| 重合程度 | 背景相关（参数化头参照） | 部分重合（低秩结构参照） |
| 证据等级 | 正文核验 | 正文核验 |
| 局限 | 自回归误差累积、需迭代采样 | 需正定约束；N 小时收益有限 |
| 代码 | `awslabs/gluonts`（✅已读源码） | 同上（**仅 mx 版**，见 §5） |

| 字段 | Gateformer（方向二核心最近邻） |
|---|---|
| 标题 | Gateformer: Advancing Multivariate Time Series Forecasting through Temporal and Variate-Wise Attention with Gated Representations |
| 作者 | Yu-Hsiang Lan 等（NYU） |
| 年份 | 2025 |
| 状态 | **ICML Workshop on Foundation Models for Structured Data（非主会）** |
| DOI/arXiv | arXiv:2505.00307 |
| 条件变量 | ❌ 门控由 `enc_out`（自身）与 `enc_out_cross`（跨变量）共同生成 |
| 概率输出 | ❌ 点预测 |
| 不确定性是否影响融合 | ❌ **门控不是不确定性驱动** |
| 回退分支 | ✅ `gate*enc_out + (1-gate)*enc_out_cross` |
| 关键公式 | 代码 `models/Gateformer.py:100-101` |
| 置换实验 | ❌ 未核验 |
| 重合程度 | 部分重合（回退分支结构） |
| 证据等级 | 正文核验（**代码级**） |
| 代码 | `nyuolab/Gateformer`（✅已读源码） |

| 字段 | TimePro | LIFT |
|---|---|---|
| 标题 | TimePro: Efficient Multivariate Long-term TS Forecasting with Variable- and Time-Aware Hyper-state | Rethinking Channel Dependence…: Learning from Leading Indicators |
| 作者 | Ma, Ni, Xiao, Chen（华为诺亚） | Zhao, Shen |
| 年份/状态 | 2025，**ICML 2025 主会** | 2024，**ICLR 2024** |
| DOI/arXiv | arXiv:2505.20774 | arXiv:2401.17548 |
| 条件变量 | ❌ 无目标查询 | ⚠ 逐**目标变体 j** 选源（非查询向量） |
| 是否时间点选择 | ✅ `state_pro`=DCNv4 在 `(patch_num, n_var)` 网格 | ✅ 式(3)(4) 选领先步与变量 |
| 回退分支 | ❌（仅 SSM 内部残差 `y+xs·Ds`） | ❌ |
| 关键公式 | `h = self.state_pro(h, shape=(patch_num, self.n_var))` | 式(3) `δ=argmax\|R(τ)\|`；式(7)(9) 状态调制 |
| 重合程度 | 机制相近（变量×时间选择） | 部分重合（逐目标选源） |
| 证据等级 | 正文核验（代码级） | 正文核验 |
| 代码 | `xwmaxwma/TimePro`（✅已读） | `SJTU-DMTai/LIFT` |

**方向一/方向二的否定性结论（已按子代理深读结果修正）**：

- **方向一**：子代理用 6 个独立检索式 x arXiv API + arXiv 全文搜索两种路径，**未发现"预测目标变量条件化不确定性"的时序预测工作**（命中全为无关领域）——这是**明确的检索空白而非检索失败**。但"预测步/状态条件化不确定性"已有 VolDy-VAE、Rollcast、DeRegiME 等。
- **方向二**：**初稿"未发现"的结论是错的**。确有 5 例让不确定性进入融合（MoGU / UGGM / CGT / IBN / Under-Cali），但：MoGU 融合的是**专家非变量**且存在退化通道；UGGM 作用在**时间维非跨变量**且 L_gate 不约束 sigma；CGT 是变量级但为**二元因果结构**且任务是 AD；IBN 只管**插补变量**；Under-Cali 路由**样本非变量**。
> **"用逐传感器预测不确定性连续加权跨变量依赖、且带 stop-grad 与硬回退"这一完整设计，在本次全部检索式下未找到先例。**

### 4.1b 方向一/方向二新增证据表

| 论文 | 状态 | 标识 | 条件变量 | 抗退化 | 与本方向重合 | 证据等级 |
|---|---|---|---|---|---|---|
| **CGT** | 2026-04 预印本 | arXiv:2604.17998 | 目标特定 gate `alpha_i = sigma(l_i)` | 3重：stop-grad 隔离（式23/24/26/28/29）+ `L_res` 幅度惩罚（式30）+ **硬回退 `gamma_used=0`**（式42）+ 非父变量置换实验 | **部分重合（最接近）** | 正文 PDF 全文已核实（式22-45） |
| **MoGU** | 2025-10 预印本 | arXiv:2510.07459 | 专家方差 `w_i = sigma_i^-2 / sum_j sigma_j^-2` | **无**（代码确认无 stop-grad） | 机制相近（专家非变量） | 正文 + 官方代码均已核实（`yolish/moe_unc_tsf`，2星） |
| **UGGM** | **ICLR 2026 Workshop**（已录用） | arXiv:2603.07753 | `A_ij ∝ exp(q_i^T k_j / sqrt(d)) * G(sigma_i, sigma_j)` | **无**（`L_gate` 只约束平滑性 `||g_t - g_{t-1}||^2`） | 机制相近（时间维非跨变量） | 正文 PDF 全文已核实（式6-13、31-36） |
| **IBN** | 2025-09 预印本 | arXiv:2509.07725 | `x_hat^UAI_t = mu_t / (1 + sigma_t)` | 有（除数单调，sigma→∞ 时贡献趋 0） | 部分重合（仅插补变量） | 正文 + 代码双重核实（`IBN_Cell.py:45`） |
| **Under-Cali** | 2026-05 预印本 | arXiv:2605.28603 | 双阈值 `tau_alloc / tau_trig` 路由**样本** | **最干净**：UE 用归一化 MSE 直接监督，不经 NLL 反推 | 机制相近 | 正文 PDF 已核实 |
| **VolDy-VAE** | 2026-03 预印本 | arXiv:2603.24254 | **预测步 + lookback 波动状态** | — | 机制相近 | 正文+表格（PDF 全文） |
| **Rollcast** | 2026-09 预印本 | arXiv:2609.05561 | 状态（检索残差分布） | — | 机制相近 | 摘要级 |
| **DeRegiME** | 2026-05 预印本 | arXiv:2605.19231 | 隐 regime + horizon | — | 机制相近 | 摘要级 |
| **TimeGrad** | **NeurIPS 2021** | arXiv:2101.12072 | RNN 隐状态（历史上下文） | — | 背景相关 | 正文 PDF 已核实 |

### 4.4b 方向一/方向二的两个反例

**反例 3（Halo, arXiv:2609.10589，2026-09 预印本）**——**直接反驳"更精细的目标条件化 QuantileHead 就能改善预测"这一论证路线**。
它给已有深度预测器加一个 scale 输出头（Gaussian/Laplacian 匹配似然），在 3 架构 x 5 电力市场 x 2 指标共 30 组比较中 28 组改善（平均 MSE 降 2.6%-16.5%）。但论文明确说：scale 头是双头还是并行网络 "matters far less than **whether the network estimates scale**"，两种架构差异 <1.5% 且方向一致；且**为点估计基线调好的超参直接可用，重调在 2 个市场有效、3 个市场有害**。
> **对本课题的含义**：收益来自"**是否存在尺度头**"这个二值判断，**不是条件化结构的精巧程度**。任何"我们的条件化不确定性头比普通分位数头更好"的主张，**必须在同为异方差头的基线上做消融**，而非对比确定性基线。

**反例 4（TimeCorr, Sensors MDPI 2026, DOI 10.3390/s26196238, published-online 2026-10-01）**——**最容易误判为"不确定性驱动融合"的类型**。
它构建 Dynamic Correlation Module（带符号动态跨变量相关权重）+ Heterogeneous Correlation Module（正负极性分别校准）+ Selective Correlation Module（稀疏关系 mask）三个模块，在 8 个 benchmark x 55 个评估情形取得最低误差。**但全文所有权重都来自特征相似度/极性/稀疏学习，没有任何预测不确定性参与。** 相关权重刻画的是"变量是否统计相关"，不是"这条变量当前有多可信"。
> 更关键：它引用的整条文献链（LIFT、Crossformer、SageFormer、TimeFilter、DADiffNet、transfer-entropy graph 等 43 篇）**没有一篇做不确定性驱动路由**——这是一篇 2026-10 的最新综述式工作，其参考网络**整体上不落入本方向**。

### 4.2 方向三：多步 conformal

| 字段 | CF-RNN | AgACI | CAFHT | FLIPR |
|---|---|---|---|---|
| 标题 | Conformal Time-series Forecasting | Adaptive Conformal Predictions for Time Series | Conformalized Adaptive Forecasting of Heterogeneous Trajectories | FLIPR: FLexible and Interpretable Prediction Regions for time series |
| 作者 | Stankevičiūtė, Alaa, van der Schaar | Zaffran, Dieuleveut, Féron, Goude, Josse（EDF R&D 等） | Zhou et al. | English, Lippert |
| 年份/状态 | **NeurIPS 2021** | 2022 预印本（EDF R&D） | **AISTATS 2024**（PMLR v235） | **PMLR 2026**（Third Conference on Parsimony and Learning, CPAL 2026） |
| 标识 | NeurIPS hash `312f1ba2…` | **arXiv:2202.07282**（已核验标题/作者/日期） | PMLR `v235/zhou24l` | PMLR `v328/english26a` |
| 覆盖定义 | 逐 horizon **同时**（Bonferroni $\alpha/H$） | 长期**边际**（在线自适应） | **simultaneous marginal**（整条轨迹） | **K-family-wise**（$\Pr[V>K]\le\epsilon$） |
| 多 horizon 修正 | **Bonferroni** | 逐步误差更新 | ACI + split conformal 混合 | **K-th order 非一致性分数** + 逐 horizon 标准化 |
| 交换性假设 | 条件独立 + 近似可交换 | **不要求** | 校准轨迹与测试可交换 | 需校准集 |
| 校准集分离 | ✅ | 在线滚动 | ✅（并需**额外划分** $\mathcal{D}_{cal}^1/\mathcal{D}_{cal}^2$ 选 $\gamma$） | ✅ |
| 多变量 | ❌ 论文自陈 $d=1$ | 未核验 | ✅ 扩展到 $d>1$ | ✅ 残差矩阵 $H\times d$ |
| 区间宽度 | 偏保守（Bonferroni） | 未核验 | 比 Bonferroni 更窄 | 逐 horizon 平衡（Property 3） |
| 逐 horizon/变量报告 | ✅ 逐 horizon | 在线 | 平均宽度 | **强调逐 horizon 平衡** |
| 关键公式 | 式(9) + Bonferroni $\alpha/H$ | ACI 更新 | 定理 1（有限样本 simultaneous） | $\alpha_i^K=\text{K-max}(\|R_{h,j}-\mu_{h,j}\|/\sigma_{h,j})$ |
| 重合程度 | **部分重合**（可直接作为 V3C 后处理层） | 机制相近 | 机制相近 | **部分重合** |
| 证据等级 | 正文核验 | **正文核验**（abs 页） | 正文核验 | 正文核验 |
| 代码 | `vanderschaarlab/conformal-rnn` | 论文声明代码公开（**URL 未核验**） | **未核验** | **未核验** |

**附加**：**O²CP**（Optimization-Based Online CP，arXiv:2508.13362，**发表状态未核验**）——两层设计：可接受集 + 跨 horizon 优化，目标为**逐 horizon 长期边际覆盖**，声称比联合覆盖更不保守，在自动驾驶/气候/公共卫生/电力 5 个数据集上降低 calibration score 达 57.2%（**该数值来自摘要，未核验**）。

### 4.3 方向三/四共核：Conformal Risk Control 与风险控制工具链

| 论文 | 年份/状态 | 标识 | 风险定义 | 阈值来源 | 对本项目 |
|---|---|---|---|---|---|
| **Conformal Risk Control (CRC)** | **ICLR 2024 Spotlight** | arXiv:2208.02814 ✅ | R6 + 任意**单调**损失 $\ell$（例：漏报比例） | 校准集上二分搜索 $\hat\lambda$ | ✅ **把误报/漏报变成可控期望风险**；代码 `aangelopoulos/conformal-risk`，`get_lhat.py` **仅 5 行** |
| **RCPS** (Bates et al.) | 2021, **JACM 68(6) Art.43** | arXiv:2101.02703；DOI 10.1145/3478535 | R6，集合 $T_\lambda(X)$ + 自定义损失 | 校准集 | CRC 的前身（高概率保证，比 CRC 保守） |
| **non-X CRC** (Farinhas et al.) | **ICLR 2024 主会** | ICLR 2024 proceedings | R6 加权（非交换） | 按与测试样本相关性加权校准 | ✅ 处理漂移；有时序实验（多标签+电力监测） |
| **CRC under Non-Monotone Losses** | 2026 预印本 | arXiv:2604.01502 | R6 非单调 | 网格化 + 二分 | 机制相近 |
| **Adaptive Conformal Anomaly Detection (TSFM)** | **ICLR 2026 主会** | arXiv:2604.20122 | R6：$A_{n+1}=\mathbb{1}[S_{n+1}>\hat q_\alpha^w]$，**学习权重 $w$** | 学习而非固定 | ✅ 分数直接解释为 p-value/误报率；Prop 3.1 给误报上界 |
| **Conformal ML for industrial CPS** | 2026, **RESS**（期刊） | S0951832026002334 | R6 + TQA（时间分位调整）+ 滑动校准 + reject | 滑动校准集 | ✅ **工业多变量 + 漂移 + FAR 正式保证**（仅摘要级） |
| **Distribution-Free Process Monitoring** | 2025 预印本 | arXiv:2512.23602 | R6 + 多变量 p-value chart | 校准集 + 用户选 $\alpha$ | ✅ 把 SPC 重述为异常检测 |
| **CP p-value 离群检测** (Bates et al.) | 2023, **Ann. Statist. 51:149-178** | DOI 10.1214/22-AOS2244 | **R6**：conformal p-value | 分位数水平 | ✅ 证明 BH 程序控制 FDR；作者明言"relatively weak marginal sense" |
| **CADES** (Conformal AD in Event Sequences) | **ICML 2025 主会** | PMLR v267:77177-77193 | R6 + 时标重标定定理 | 校准集 | ✅ **连续时间事件序列** FPR 控制 |
| **TRACE-CRC** | 2026 预印本 | arXiv:2608.27124 | R5/R6 + Frobenius 球 | trajectory-level CRC | ✅ "至少一帧未覆盖"的**轨迹级风险**，FWER=$\delta$ |
| **RUL UQ + calibration (BDNN)** | 2026, **RESS** | DOI 10.1016/j.ress.2026.112763 | R2/R4：dropout 集成 + isotonic + split conformal | **校准集一分为二（各 10%）**：一半 isotonic，一半 nonconformity | ⭐ **流程最贴近本项目**；报 ECE |
| **SVGD-Bayesian RUL** | 2024 预印本 | arXiv:2402.01098 | R2/R3：后验 + $p_{\text{late}}$（迟报概率） | 迟报概率 | 部分重合：$\mu^*=\hat\mu-k(1-p_{\text{late}})$；**未报 ECE/Brier** |
| **因果约束概率预测 AD** | 2026 预印本 | arXiv:2604.17998 | R2 + **阴影辅助路径**捕获非因果残差 | 未核验 | ⚠ **用不确定性隔离非因果通道**（回退思想的变体）；⚠ 根因归因侧用因果父掩码 |
| **TRACE-C**（多流遥测） | 2026 预印本 | arXiv:2608.21251 | R6 上尾秩 p-value | 经验秩计数 | ⭐ **最有价值的负面证据**：作者自陈"empirical rank counts are diagnostics, **not coverage or false-discovery proofs**" |
| **Mahalanobis 渐近精确阈值** | 2026 预印本 | arXiv:2607.15637 | R5：$M_\Sigma(X)$；Hotelling T²↔F↔Beta pivot | 渐近精确 + 箱线图规则 | ⚠ 维度升高误差增大 |

### 4.4 两个关键反例

**反例 1（D1）：Gneiting, Balabdaoui & Raftery (JRSS-B 2007)** —— 见 §3.4 方框。这是"用 ECE/PIT 判合格 → 直接拿去做风险阈值"的教科书级反例。

**反例 2（C14）：Schlembach, Smirnov & Koprinska, *Conformal Multistep-Ahead Multivariate Time-Series Forecasting* (CoPPA 2022, PMLR v179:316-318, 3 页)**
标题中的 "Multivariate" **在机制上未兑现**：其预测区域写作 $[\hat\mu(x_{n+1})\pm Q_{1-\alpha/h}(\cdot)]^\top$，是 $h$ 个 horizon、每 horizon $d$ 个坐标的**边缘区间笛卡尔积**——对 horizon 轴做了 Bonferroni（$\alpha/h$），**对变量轴完全没做修正**（无 Bonferroni、无 copula、无椭球协方差）。因此它**既不是多变量联合（MJ）覆盖，也不是同时（SH）覆盖**，报告的"coverage"只能读作逐坐标边际（M）。
> **可迁移判据**：判断一篇 conformal 多变量工作是否"真联合"，只需看其 nonconformity score 是**逐坐标标量**（$|\hat y-y|_j$ 类）还是**跨坐标聚合**（Mahalanobis、$\ell_\infty$、copula 变换）。前者即使乘以高维区域形状，也只是边缘区间的乘积。

### 4.5 检索发现的三个真空（可作贡献，但需实验支撑）

1. **"H 步 × d 变量"的完整联合预测区域**（同时覆盖所有 horizon 且所有变量落入区域）**在本轮检索中未找到理论保证 + 实验证据齐全的代表作**。CopulaCPTS 的联合轴是时间维；MultiDimSPCI / MSR 的联合轴是变量维但只有单步；CF-RNN / C14 的变量维只是边缘区间乘积。
2. **事件级（event-level / run-to-failure）覆盖与覆盖率的关系没有任何理论刻画**。RCPS/CRC 控制的是**期望层面**风险 $\mathbb{E}[\ell]\le\alpha$；conformal p-value 工作控制的是**逐点 FPR/FDR**；工业 AD 工作控制的是**长时 FAR**。三者都不是"每个故障事件至少被检测到一次"。
3. **persistence / alarm-merging 规则几乎没有文献**。PHM 的 EoUP/EoL 定义与时序 AD 评测（DQE、LSF$^w$、dT-PAF）把它当**评测假设**，但几乎没有工作把它当作**推理时需在 valid 上固定的超参**并报告敏感性。

---

## 5. 代码仓库核验表

> 均为**实际下载并阅读源码**（上一轮子代理核验 + 本轮我的核验）；未打开源码者明确标注。

| 仓库 | 官方性 | 许可证 | 概率头 | 校准器 | 风险 evaluator | 关键文件（已读） | 协议/隔离 | 泄漏风险 | 可借鉴机制 | 不应复制 |
|---|---|---|---|---|---|---|---|---|---|---|
| `awslabs/gluonts` | ✅ 官方（DeepAR 团队） | Apache-2.0 | ✅ `DistributionOutput`（**默认 `StudentTOutput()`**） | ❌ | `ev/metrics.py` | `torch/model/deepar/module.py`、`mx/block/quantile_output.py`、`ev/metrics.py` | ✅ scaler 只用 context；`ValidationSplitSampler` 顺序切分 | 低 | 分布与主干解耦；**`crps_weights_pwl()` 加权 pinball→CRPS**；30+ 指标定义 | — |
| `yromano/cqr` | ✅ 官方（Romano） | MIT（LICENSE 原文） | quantile + `use_rearrangement` | ✅ `run_icp()` split conformal | ✅ `compute_coverage` | `cqr/helper.py`、`nonconformist/` | ✅ `idx_train`/`idx_cal` 显式三段 | **低**（`AllQNet_RegressorAdapter(test_ratio=0.2)` 是内部 CV，须与最终 test 区分） | conformal 与回归器解耦；Chernozhukov 保序 | — |
| `FilippoMB/Ensemble-Conformalized-Quantile-Regression`（EnCQR） | ✅ 官方（Bianchi） | MIT | quantile `[α/2,0.5,1-α/2]` | ✅ conformal + 在线滚动 | `compute_coverage_len` | `conformal_prediction.py`、`main_EnCQR.py` | ⚠ 见右 | **🚨 高**：用 **train 段标签**算非一致性分数，且循环末 `asym_nonconformity(label=test_y[i,:])` **用 test 标签更新 epsilon** | 分层非一致性分数思想 | **绝不照搬**（与 AGENTS.md 冲突） |
| `aangelopoulos/conformal-risk` | ✅ 官方（Angelopoulos） | **未核验** | ❌（不产出分布） | ✅ 风险控制 | ✅ FNR/F1 等单调损失 | `core/get_lhat.py`（5 行） | ✅ 需独立校准集 | 低（视调用方） | **风险→阈值→期望风险上界的最短路径** | — |
| `nyuolab/Gateformer` | ✅ 官方 | **未核验** | ❌ 点预测 | ❌ | ❌ | `models/Gateformer.py:100-101` | 未核验 | 未核验 | 回退分支门控结构 | — |
| `xwmaxwma/TimePro` | ✅ 官方 | MIT | ❌ | ❌ | ❌ | `model/TimePro.py` | 未核验 | 依赖 DCNv4 自定义 CUDA | 变量×时间 2D 选择 | 移植成本高 |
| `vanderschaarlab/conformal-rnn` | ✅ 官方（NeurIPS'21） | BSD-3（继承） | ❌ 点预测 | ✅ ICP + Bonferroni | ❌ | **文件未打开（⚠ 仅 README）** | ✅ 校准集分离 | 低 | 逐 horizon Bonferroni 实现 | — |
| `Cola-Fsm/MS-FLOW` | **未核验** | 未核验 | 未核验 | 未核验 | 未核验 | ❌ 未打开 | 未核验 | 未核验 | （仅 §4 摘要级记录） | 全部 |
| `amazon-science/unconditional-time-series-diffusion` | ✅ 官方 | Apache-2.0 | diffusion | ⚠ self-guidance（**非校准**） | ❌ | `model/diffusion/tsdiff.py`、`utils.py` | ⚠ 见右 | **🚨 中**：`ScaleAndAddMeanFeature` 用 `mean(abs(target[...,-pred_len:]))` = **per-window 缩放**，非 train-split | 训练/任务解耦 | 缩放逻辑 |
| `zalandoresearch/pytorch-ts` | ✅ 官方（TimeGrad 论文给出） | MIT | EBM | ❌ | `MultivariateEvaluator` | `pts/model/time_grad/*` | ✅ scaler 只用 `past_target_cdf[:, -context_length:]` | 低 | `unroll_encoder` 解耦训练/推理；**`loss_weights=observed_values.min(-1)` 对缺失鲁棒** | — |
| `ServiceNow/TACTiS` | ✅ 官方 | Apache-2.0 | ✅ 边缘 Deep Sigmoid Flow + copula | ❌ | `tactis/gluon/metrics.py` | `tactis/model/{marginal,tactis,flow}.py` | ✅ `NormalizationIdentity(hist_value)` **只对历史归一化** | 低 | 边缘流+copula 两段式，**参数量对变量数线性增长** | — |
| `dmlc/xgboost` | ✅ 官方 | Apache-2.0 | ✅ `QuantileDMatrix` | ❌ 库内无 | ❌ 库内无 PICP/coverage | `src/objective/quantile_obj.*` | 由用户保证 | 视用户划分 | 一次训练多 alpha；不过拟合的 conformal base learner | — |

| **`Rose-STL-Lab/CopulaCPTS`**（ICLR 2024 CopulaCPTS） | ✅ 官方 | **未核验** | copula 联合残差区间 | ✅ split conformal | ✅ | **未核验（⚠ 仅据仓库存在）** | 时间轴 copula 联合 | 联合轴非变量维 |
| **`kamilest/conformal-rnn`**（CF-RNN） | ✅ 官方（⚠ 与 `vanderschaarlab/conformal-rnn` **并存**，见 §5 注） | 未核验 | ❌ 点预测 | ✅ ICP + Bonferroni | ❌ | 未核验（⚠ 未打开源码） | 逐 horizon Bonferroni | — |
| **`Quilograma/AEnbMIMOCQR`**（AEnbMIMOCQR） | ✅ 官方 | 未核验 | MIMO 分位数 | ✅ conformal + ACI | ❌ | 未核验（⚠ 仅仓库存在） | MIMO+ACI 组合 | 自述 preprint/tutorial |
| **`Copula-Fsm/MS-FLOW`** | **未核验** | 未核验 | 未核验 | 未核验 | 未核验 | 未核验（⚠ 仅 §4 摘要级） | 稀疏路由 | 全部未核验 |

**关于 CF-RNN 代码的两个仓库**：子代理记录为 `kamilest/conformal-rnn`（78★），我此前记录为 `vanderschaarlab/conformal-rnn`。**两者是否同一实现（fork/镜像/不同版本）未核验**，论文中引用官方仓库前须确认作者主页指向。

**未找到官方仓库（不臆造 URL）**：MQ-RNN、DiffTime、TimeWeaver、TimeGrad++、SACR、TactPFN、CAFHT、FLIPR、AgACI、MultiDimSPCI、MSR、CADES。

---

## 6. 与 Unified V3C 的代码差异（项目代码核实）

> **任务书描述 vs 实际代码**：逐条核对如下，**以可执行代码为准**。

| 组件 | 任务书描述 | **代码核实结果** | 差异 |
|---|---|---|---|
| MetroPT StateHead | 非交叉分位数 + pinball loss | ✅ **一致**。`heads/state.py:20-33`：`quantiles=(0.1,0.5,0.9)`；`base=Linear(h,pred_len)`；`increments=Linear(h,pred_len*(Q-1))`；`quantiles=cat((base, base+cumsum(softplus(increments))))`；`heads/base.py:38` `pinball_loss` | **无差异**。注意：`mode="point"` 时返回中位数 |
| FD004 RUL head | 点预测/分位数结构，保留 raw RUL 与 training cap | ✅ **一致且更强**。`heads/rul.py:24-33`：`prepare_target` 用 `targets.raw_rul` 回退到 `targets.values`；`_apply_cap` 仅 `clamp_max(training_cap)`；`input_mode ∈ {global, global_trend}`；`over_penalty` | **raw/cap 语义已分离**，实现比描述更完整 |
| TEP head | 故障检测与诊断概率输出 | ✅ **一致且更强**。`heads/tep.py:18-27`：`detection=sigmoid(Linear)`、`diagnosis=softmax(Linear)`，`mode="probability"` 强制 | **无差异** |
| 概率评估 | 缺 CRPS/NLL/PICP/MPIW/interval score | ✅ **一致**。`experiments/metrics.py` 仅 `state_metrics`（MAE/RMSE 逐传感器）、`rul_metrics`、`classification_metrics`；全仓 grep `conformal|brier|ece|crps|picp|mpiw|nll|interval_score` **无真实命中**（仅匹配到 `recent`/`masked` 等子串） | **无差异** |
| validation-only conformal | 无 | ✅ **一致**。全仓无 conformal 实现 | **无差异** |
| 概率风险分数与事件级校准 | 尚无统一流程 | ⚠ **部分已有**：`heads/tep.py:45-80` 已有 `detection_delay()` 与 `evaluate_tep_events()`——**按 `run_ids` 隔离**统计检测延迟、误报、漏报，保留 delay 分布 | **差异**：TEP 事件级评估**已存在**，非"尚无"。但 `threshold=0.5` 为**默认硬编码**，未见 validation-only 阈值拟合 |
| KST ProbFlow | 未实现 | ✅ **一致**。全仓无该名称 | **无差异** |

**结论**：任务书对基线的描述**准确**，唯一需修正的是"尚无统一的风险分数与事件级校准流程"——**TEP 侧已有事件级评估骨架**，缺的是 (a) 从预测分布导出的风险分数、(b) validation-only 阈值/校准、(c) 跨三任务的统一流程。

---

## 7. 最近邻与创新性判断

### 7.1 九个重点问题的回答

| # | 问题 | 回答 | 依据 |
|---|---|---|---|
| 1 | 目标条件化的不确定性是否已有直接先例？ | **本次未发现**（受 §2.4 限制） | DeepAR/GPVAR 是隐含状态条件；LIFT 的"目标"是变体索引非查询向量；未见以目标查询生成 $\mu/\sigma$ 的工作 |
| 2 | 不确定性驱动变量路由是否已有直接先例？ | **部分有，均不完整** | CGT（变量级+stop-grad+硬回退，但是 AD+二元因果结构）、MoGU/UGGM（融合专家/时间维，无 stop-grad）、IBN（仅插补变量）、Under-Cali（路由样本） |
| 3 | conformal 能否作为独立第四章贡献？ | **更适合作为校准工具** | 组件全部成熟（CF-RNN/ACI/AgACI/MWCP/CAFHT/FLIPR/O²CP/CopulaCPTS/MultiDimSPCI/MSR/CRC）；创新应来自**工业多传感器 + 事件级 + 严格协议**的组合 |
| 4 | 事件级风险校准的最近邻？ | ① **CRC (ICLR 2024 Spotlight)** 直接；② **CADES (ICML 2025)** 事件序列 FPR；③ **TRACE-CRC** 轨迹级风险；④ **RUL UQ+calibration (RESS 2026)** 校准集二等分流程最贴近本项目；⑤ **TRACE-C (2026)** 是最有价值的**负面证据**（自陈经验秩计数不构成覆盖证明） | 见 §4.3 |
| 5 | 第四章应聚焦？ | **"校准不确定性 + 风险评估"**，而非"概率预测模型" | 模型侧无空白；校准与事件级协议是审稿人难反驳的贡献 |
| 6 | 哪些方向可跨三任务适配？ | **校准层 + 指标**可跨三任务；**低秩联合/flow/diffusion** 仅适合 MetroPT（多变量多步）；**horizon 条件化** 仅 MetroPT（FD004 单值 RUL 无多步标签） | §8 |
| 7 | 是否需要通用数据集？ | **需要**，否则只能写"三类工业任务验证" | 任务书 §7 明确要求区分 |
| 8 | 最适合 V3C 最小改动的扩展？ | ① **conformal/CRC 校准层**（纯后处理，零模型改动）② **指标补全** ③ **Gaussian/Student-t 头**（只换头） | §9 |
| 9 | 应避免的 claim？ | 见 §7.3 | — |

### 7.2 候选方向定性

| 候选 | 定性 | 理由 |
|---|---|---|
| 分位数/Gaussian 概率头 | **已有成熟组件的合理应用** | V3C 已有；DeepAR/GluonTS 已标准化 |
| conformal 校准层 | **评估协议贡献** + 成熟组件应用 | 组件成熟；工业多变量 + 事件级的组合未被系统做过 |
| **validation-only 事件级风险校准** | **最有价值的贡献方向** | CRC 提供理论工具；工业事件稀疏、误报代价高；V3C 已有事件级骨架可扩展 |
| 目标条件化不确定性 | **组合方法候选**（证据不足） | 组件可拼，但"目标条件"未见直接先例，需实验证明 |
| 不确定性驱动融合 | **组合方法候选**（有先例但均不完整） | 5 例先例各有缺口；**退化通道已定位**（MoGU/UGGM 无 stop-grad）；完整设计未见先例 |
| 低秩/flow/diffusion 联合建模 | **工业场景适配**（非创新） | 组件成熟；工业小样本下成本高 |

### 7.3 不可声称的表述
- ❌ "首次提出目标条件化不确定性估计" / "首次用不确定性控制变量路由"（本次未发现 ≠ 不存在，且检索受限）
- ❌ "边缘覆盖率 = 联合覆盖率"；❌ "预测区间 = 置信区间"
- ❌ "风险分数 = 故障发生概率"（除非有校准证据；且事件级持久化规则会改变语义）
- ❌ "attention 权重揭示因果/物理传播时滞"
- ❌ 任何未运行实验的性能数字；单 seed 不得写成稳定结论
- ❌ 把 Gateformer 的门控说成"不确定性驱动"

---

## 8. 跨三个工业任务的适配边界

| 能力 | MetroPT-3 | FD004 | TEP |
|---|---|---|---|
| 多步概率状态预测 | ✅ 主战场（pred_len=24） | ❌ **单值 RUL，不得虚构多步标签** | ❌ 分类任务 |
| 逐传感器不确定性 | ✅ 15 传感器 | ⚠ 单一 RUL 输出，传感器维度无意义 | ⚠ 分类概率 |
| horizon-wise 校准 | ✅ | ❌ | ❌ |
| sensor-wise 校准 | ✅ | ❌ | ⚠ 诊断类概率 |
| 低秩联合 / copula | ✅ 适用 | ❌ 单输出无联合 | ❌ 类别间非连续 |
| conformal 区间 | ✅ | ✅（RUL 区间） | ❌（分类用预测集而非区间） |
| 事件级风险校准 | ⚠ 故障事件仅 4 起且无标签 | ⚠ 每台发动机 run-to-failure 可作"循环事件" | ✅ **主战场**（按 run 隔离） |
| 目标/预测步条件化 | ✅ | ❌ | ⚠ 仅 task 条件（检测 vs 诊断） |

**结论**：**TEP 是事件级风险校准的主战场**（V3C 已有 run 隔离的事件评估骨架）；**MetroPT-3 是 horizon/sensor 条件化不确定性的主战场**；**FD004 只承担单值 RUL 区间与 cap 语义隔离验证**。

---

## 9. 最小实验矩阵

### 阶段 A：基线与机制正确性
| ID | 臂 | 目的 |
|---|---|---|
| A0 | V3C point（`mode="point"`） | 点预测基线 |
| A1 | **V3C 现有 quantile head**（扩展到 {0.05,0.1,0.25,0.5,0.75,0.9,0.95}） | 分位数基线 |
| A2 | Gaussian 头（$\mu$, $\sigma=\text{softplus}(\cdot)+10^{-3}$） | 参数化对照 |
| A3 | Student-t 头 | 厚尾对照（GluonTS DeepAR 默认） |
| A4 | 未校准区间（由 A1/A2 直接取分位数） | 校准必要性基线 |
| A5 | **validation-only conformal 区间**（ICP/Bonferroni） | 校准层 |
| A6 | **Conformal Risk Control**（控制漏报率/误报率） | 风险控制 |
| A7 | 简单残差风险（\|y−ŷ\| 的 valid 分位数阈值） | 最简风险基线 |
| A8 | 分布导出风险（标准化残差 / NLL / 区间超限 / 尾部概率） | 分布风险 |
| A9 | **负对照**：打乱目标通道 / 打乱工况上下文 / 打乱查询时间 | 证明条件真正起作用 |
| A10 | 目标/horizon 条件化的尺度或分位数（仅在 §4.1 未发现先例的前提下纳入） | 核心机制 |
| A11 | uncertainty-aware fusion（**可选，需先通过 A12 退化检验**） | 变量级不确定性加权 + stop-grad + 硬回退 |
| **A12** | **⚠ 退化通道检验（本方向必做）**：① "冻结 sigma 只重训预测头"对照；② 训练后 sigma 分量与实际残差方差的相关系数（若强正相关说明 sigma 被博弈而非吸收真实方差）；③ 无 stop-grad 版 vs 有 stop-grad 版对比 | **证明不确定性加权未被"放大尺度逃避惩罚"** |
| **A13** | Halo 式对照：**同为异方差头**的基线（Gaussian 头）vs 目标条件化头 | 分离"有尺度头"的收益与"条件化结构"的收益（见反例 3） |
| **A14** | 非父变量/无关源变量**置换**（保持边缘分布、破坏逐样本对齐） | 检验融合权重是否真的用了可靠性信息 |

### 阶段 B：MetroPT-3（多步 + 逐传感器）
指标：MAE/RMSE、**CRPS**（分位数经 `crps_weights_pwl`；参数分布解析）、**NLL**、**PICP**、**MPIW**、interval score、**horizon-wise** coverage/width/NLL、**sensor-wise** coverage、false alarm rate、detection delay。

### 阶段 C：FD004（单值 RUL）
指标：raw MAE/RMSE、NASA score、RUL NLL/CRPS、coverage、interval width、**per-engine error**。
**约束**：不得虚构多步 RUL；`training_cap` 与 `raw_rul` 严格隔离（代码已有 `_apply_cap`，须在 manifest 记录）；阈值只用 validation。

### 阶段 D：TEP（事件级风险）
指标：AUROC/AUPRC/F1、**log loss**、**Brier**、**ECE**、**reliability diagram**、**false alarm rate**、**event recall**、**detection delay**。
**约束**：按 `run_ids` 隔离（代码 `evaluate_tep_events` 已实现）；**故障标签不得进入推理路由**；连续报警 persistence rule **必须在 validation 固定**（当前 `threshold=0.5` 硬编码需改为 valid 拟合）。

### 阶段 E：通用数据集
**建议增加** Electricity / Traffic / Weather 之一（可选 ETTh1）。无此则只能声称"三类工业任务验证"。

### 阶段 F：多 seed 与成本
单 seed 仅方向筛选；正式结论 ≥3 seeds（2026/2027/2028）。
报告：参数量、训练时间、单批推理时间、峰值显存、采样时间（若适用）、checkpoint 大小。

### 冻结项（所有比较）
data split、normalization 来源（**train-only**）、seed、history/prediction length、batch size、训练预算、early stopping/checkpoint selection（**validation-only**）、evaluator、**calibration split**、**threshold 来源（validation-only）**。

---

## 10. 后续实现顺序

1. **`experiments/metrics.py` 新增**：CRPS（分位数折算 + 参数分布解析两种）、NLL、PICP、MPIW、interval score；**逐 horizon / 逐 sensor** 聚合。
2. **单元测试**：分位数单调性、区间嵌套性（80%⊂90%⊂95%）、逐 horizon 覆盖率边界（$h=0$ 与 $h=H-1$）、per-engine 分组无泄漏。
3. **Gaussian 头**（`heads/state_gaussian.py`），测试形状/有限性/$\sigma>0$；与现有 `StateHead` 同一接口，`mode` 区分。
4. **Conformal 校准层**（`calibration/conformal.py`），**只接受 validation split**；单元测试断言"校准集与 train/test 索引无交集"。
5. **Conformal Risk Control 接入**（可参考 `aangelopoulos/conformal-risk` 的 5 行算法，**自行实现不复制**）；用于 TEP 误报率/漏报率控制。
6. **阈值来源改造**：把 `heads/tep.py` 的 `threshold=0.5` 默认值改为 **validation 拟合**，并把 persistence rule 显式化、记录进 manifest。
7. 跑 A0–A8 单 seed → 3 seeds → B/C/D → 负对照 A9 → 通用数据集。

---

## 11. 未核验事项与不可声称的结论

### 11.1 未核验事项
1. **方向一/方向二的并行检索子代理在本报告定稿时仍在运行**（sessionID `ses_edc752568ffeFf5U5iFsoAVPrn`），其结果未并入本版；§3.1/§3.2/§4.1 的"未发现"结论**仅基于我本人与另一子代理的检索**，覆盖面窄于计划。
2. **AgACI / CAFHT / FLIPR / O²CP / MultiDimSPCI / MSR / CADES 的官方代码未找到**，其实现细节不可引用。
3. **CAFHT 的发表状态**：子代理标为"预印本 arXiv:2402.09623"，我另据 PMLR `v235/zhou24l` 推断为 AISTATS 2024，**两者冲突，未二次核验**。
4. **FLIPR 的 PMLR 2026 / CPAL 2026 归属**依据 `v328/english26a` 路径推断，**未二次核验**。
5. **O²CP 发表状态未核验**（arXiv:2508.13362，v3 2026）；其"57.2% 降 calibration score"来自摘要，**未核验**。
6. **MQ-RNN / MQF2 的发表信息与官方仓库**未核验（官方仓库确认不存在）。
7. **各方法在 MetroPT-3/FD004/TEP 上的适配性、消融与数值完全未核验**（本轮无任何实验）。
8. **D8 / D10 / D15 等仅"官方摘要"级证据的工业论文**（RESS、TII）正文未读，其机制描述不可引用。
9. **energy score / CRPS 原始文献**：Gneiting & Raftery 2007 已核验（DOI + 正文反例）；CRPS 原始定义文献仍建议补核。
10. **CF-RNN 官方仓库存在两个候选**（`kamilest/conformal-rnn` 与 `vanderschaarlab/conformal-rnn`），关系未核验；**两者源码均未打开**。
11. **`CopulaCPTS` / `AEnbMIMOCQR` 仓库仅记录存在**，未打开源码。

### 11.2 不可声称的结论
- ❌ "首次提出目标条件化不确定性估计"
- ❌ "首次用不确定性驱动跨变量路由"
- ❌ "我们的 conformal 方法提供联合覆盖保证"（除非明确用 FLIPR 式 K-family-wise 或 CRC）
- ❌ "风险分数等于故障概率"
- ❌ "attention 权重揭示传感器因果关系"
- ❌ 任何未运行的实验数字；单 seed 稳定性结论
- ❌ 把 Gateformer 的门控机制重新命名后声称为创新

---

## 12. 方法学附记（ID 核验）

本轮延续上一轮的教训：**arXiv ID 凭记忆几乎必错**。本报告所有 ID 均经 `arxiv.org/abs` HTML 或官方 proceedings 页面核验标题与作者。已核验的关键 ID：

| 论文 | 正确标识 | 核验方式 |
|---|---|---|
| DeepAR | **arXiv:1704.04110**（v3 2019-02-22） | abs 页标题+作者+日期 |
| GPVAR/"DeepVAR" | **arXiv:1910.03002** | NeurIPS 2019 官方 PDF |
| CF-RNN | NeurIPS 2021, hash `312f1ba2a72318edaaa995a67835fad5` | NeurIPS 官方 PDF + poster 页 |
| AgACI | **arXiv:2202.07282**（2022-02-15, Zaffran 等, EDF R&D） | abs 页标题+作者+日期 |
| Conformal Risk Control | **arXiv:2208.02814**, ICLR 2024 **Spotlight** | ICLR 官方 PDF + iclr.cc poster + GitHub 官方仓库 |
| CAFHT | PMLR `v235/zhou24l`（AISTATS 2024，**归属待二次核验**） | PMLR PDF |
| FLIPR | PMLR `v328/english26a`（CPAL 2026，**归属待二次核验**） | PMLR PDF |
| TimePro | arXiv:2505.20774, ICML 2025 | abs 页 + icml.cc poster |
| Gateformer | arXiv:2505.00307, **ICML Workshop** | abs 页 comment 字段 |
| O²CP | arXiv:2508.13362（**发表状态未核验**） | 摘要 |