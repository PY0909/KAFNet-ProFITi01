# 低复杂度高收益的时序不确定性校准与风险评估创新方向调研

- 调研日期：2026-10-10
- 任务定位：**低实现难度、三数据集共享同一核心模块、可产生实质实验收益**的第四章创新方向查新
- 明确排除（作为主创新）：多步预测、FD004 多步 RUL 标签、完整联合分布、联合采样、normalizing flow / diffusion / 复杂 copula / 大型动态图、把 attention/gating/分位数回归直接包装成创新、把三个工业数据集本身当创新来源
- 结论性质：**纯文献与代码调研**。未运行实验，未修改 `code/`、模型或 `result/`

---

## 1. 执行摘要

**核心问题的直接回答**：

> 在不修改 Unified V3C 主编码器、不引入联合采样、不依赖多步创新的前提下，哪个轻量模块最可能同时改善三个数据集的概率可靠性、风险评估质量和论文贡献完整性？

**答案：validation-only 校准 + 事件级风险接口（方案 B+C+D 组合），不新增概率头作为第一阶段。**

理由基于三条代码实证（§2）：

1. **V3C 的 TEP head 恒定输出概率**（`sigmoid`/`softmax`，且用 BCE/CE 训练）→ Platt/isotonic/temperature 校准是**纯后处理，零重训**。
2. **conformal 校准只需残差**（点预测 + 校准集残差）→ 可直接在**现有 point checkpoint** 上验证，**零重训**。
3. **但 point 训练的 checkpoint 无法产出可信区间**——`StateHead.loss()` 在 point 模式只对 `base` 求梯度，`increments` 分支**零梯度**，切到 quantile 模式得到的是未训练区间。

**最推荐排序**：

| 排名 | 方向 | 定位 | 是否需重训 |
|---|---|---|---|
| **1** | **B+C+D：validation-only 校准 + 事件级风险接口** | **C 类（协议贡献）** | **否**（阶段 0–2 全部免重训） |
| 2 | A：任务条件化异方差头 | **D 类（需消融）** | 是 |
| 3 | ** Halo 式对照 + 可靠性—精度联合评估** | 对照 | 部分 |

**新增关键支撑（第二轮定向检索）**：

| 能力 | 支撑工作 | 需重训 |
|---|---|---|
| validation-only 阈值控制单调损失期望 | CRC (ICLR'24 Spotlight) | ❌ |
| **误报预算 → 分数 p-value 化 → FAR 直接读出 + 漂移自适应** | **arXiv:2604.20122 (ICLR 2026)**，明确 "post-hoc, **without requiring additional fine-tuning**"，model-agnostic | ❌ |
| **工业传感器场景的风险阈值替代"validation 上调阈值调 F1"** | **arXiv:2503.19146 (CSP plants)** | ❌ |
| 先筛选高置信样本、再在被选子集控风险 | SCRC, arXiv:2512.12844 | ❌ |
| 任意有界连续风险 + 信任子集风险控制 | SCoRE, arXiv:2603.24704 | ❌ |
| 分组条件校准（可作为"每数据集一个校准器"的统计依据） | Kandinsky CP, arXiv:2502.17264 | ❌ |

> 🚨 **必须先回答的前置问题（负面证据）**：arXiv:2607.02124 在 19 个 driver / 55,563 条真实遥测上发现，**即便纯后处理的 rolling split conformal 也会因残差自相关违反 exchangeability，产生 15.3% 的误报率**（precision≈0、recall≈0，Ljung-Box p<0.001）。**MetroPT-3 的 stride=60 重叠窗口正是高自相关场景**，故"依赖强度 → 覆盖率崩塌幅度"的定量刻画是任何方案的前置条件。

**应暂时淘汰**：方案 E（不确定性驱动融合）——虽已改判定为"组合候选"，但实现成本、退化风险与本轮"低复杂度高收益"的约束直接冲突（§12）。

**证据等级**：**B**。代码状态全部为本项目自核实（A 级）；文献身份经官方页/arXiv abs 核验（A 级）；但**所有方法在三个工业数据集上的收益完全未核验**（本轮无任何实验）。

---

## 2. 当前 Unified V3C 能力与缺口（项目代码核实）

> 逐条读代码核实，**以可执行代码为准**。

### 2.1 已存在的能力（可直接复用）

| 组件 | 代码位置 | 实证内容 |
|---|---|---|
| MetroPT 分位数输出 | `heads/state.py:11-38` | `quantiles=(0.1,0.5,0.9)`；`base=Linear(h,pred_len)`；`increments=Linear(h,pred_len*(Q-1))`；`quantiles=cat((base, base+cumsum(softplus(increments))))` → **天然非交叉** |
| pinball loss | `heads/base.py:38-45` | 标准 pinball |
| FD004 RUL 分位数 | `heads/rul.py:44-51` | `base=softplus(Linear)`、`increments=softplus(Linear)` + cumsum → **同样非交叉**；`mode="point"/"quantile"` 均支持 |
| **raw RUL / training cap 语义** | `heads/rul.py:24-33, 52` | `prepare_target` 用 `raw_rul` 回退 `values`；`_apply_cap` 仅 `clamp_max`；metadata 记录 `target_convention="raw_output_training_cap_{X}"` → **语义已完整隔离并可审计** |
| NASA score | `experiments/metrics.py:41-42` | `penalty = where(error<0, exp(-error/13), exp(error/10)) - 1` → 标准实现 |
| TEP 检测/诊断概率 | `heads/tep.py:18-27` | `sigmoid(detection)`、`softmax(diagnosis)`，**恒定输出概率**；BCE/CE 训练；`mode="probability"` 强制 |
| TEP 事件级评估 | `heads/tep.py:45-80` | `detection_delay()` / `evaluate_tep_events()`：**按 `run_ids` 隔离**统计误报、漏报、检测延迟，保留 delay 分布 |
| frozen checkpoint 评估 | `experiments/frozen_evaluation.py:40` | `evaluate_frozen_checkpoint(bundle, source_dir, output_dir, ...)` → **支持用已有 artifact 重评估** |
| train-only normalization / validation-only ckpt | 全链路（AGENTS.md + manifest） | 已建立 |

### 2.2 真实缺口（比任务书描述更严重）

| 缺口 | 代码证据 | 影响 |
|---|---|---|
| **无 CRPS / NLL / PICP / MPIW / interval score** | `metrics.py` 仅 `state_metrics`（MAE/RMSE）、`rul_metrics`、`classification_metrics` | 概率评估**完全缺失** |
| **`classification_metrics` 无法接受概率** | `metrics.py:66-67`：`predictions = torch.as_tensor(predictions).long()` → **立即转硬标签** | 🚨 **Brier / ECE / 可靠性图当前根本无法计算**。这是比"缺少 Brier/ECE"更严重的问题 |
| **无 conformal / 校准层** | 全仓 grep `conformal` 无命中 | 区间与概率校准均需新建 |
| **`threshold=0.5` 硬编码** | `heads/tep.py:45, 62`：`evaluate_tep_events(..., threshold=0.5)` | 无 validation-only 阈值拟合 |
| **无 persistence rule** | 全仓 grep `persistence/alarm/consecutive` 无命中 | 连续报警合并规则完全缺失 |
| **per-engine 统计受限** | `metrics.py:47-50`：`len(set(engine_ids)) != len(engine_ids)` 即报错 | 🚨 **要求每台发动机唯一** → 只支持 test 末尾单点 RUL，**无法统计整段 test 的 per-engine 误差分布** |
| **无 exceedance calibration** | 全仓无 | 风险分数合法性无验证手段（§4 反例） |
| **RUL 头只用 `z_global`** | `heads/rul.py:35` | FD004 无逐传感器区间（符合任务语义，但需在论文中说明） |

### 2.3 一个决定实验设计的关键事实

> 🚨 **point 训练的 checkpoint 无法产出可信区间**。
> `StateHead.loss()`（`heads/state.py:39-42`）在 point 模式下 `prediction.point is not None` 成立 → 直接返回 MAE；而 `point` 是 `quantiles[..., center_index]`，即 `base`。**`increments` 分支获得零梯度。**
> 结论：**把 point 训练的 checkpoint 切到 `head_mode="quantile"` 会得到未训练的区间宽度。** 任务书"阶段 0：现有 checkpoint、不重训"中的"区间"部分**不可行**；但 **conformal 后处理（只需残差）与 TEP 概率校准（概率恒定输出）确实零重训可行**。

---

## 3. 低复杂度候选模块分类

| 候选 | 需改主干 | 需重训 | 三数据集适配 | 现状分类 |
|---|---|---|---|---|
| **B. validation-only conformal 区间** | ❌ 否 | **❌ 否** | MetroPT ✅ / FD004 ✅ / TEP ❌（分类用预测集） | **A 类成熟组件**，本项目尚未实现 |
| **C. 预测分布 → 风险分数** | ❌ 否 | ❌ 否 | ✅✅✅（统一接口） | **B 类合理组合** |
| **D. 事件级风险控制（CRC）** | ❌ 否 | ❌ 否 | TEP ✅ / MetroPT ⚠ / FD004 ⚠ | **C 类协议贡献** |
| **A. 任务条件化异方差头** | ⚠ 只改头 | **✅ 是** | MetroPT ✅ / FD004 ✅ / TEP ❌（用分类概率） | **D 类需消融** |
| **E. 不确定性驱动融合** | ✅ 改编码器 | ✅ 是 | ⚠ 需逐任务适配 | **D/E 类**，本轮建议淘汰 |
| 可靠性—精度联合评估 | ❌ 否 | ❌ 否 | ✅✅✅ | **C 类协议贡献** |

---

## 4. 论文证据表

> 重合度：**直接重合 / 部分重合 / 机制相近 / 背景相关**。证据等级：**正文核验 / 官方摘要 / 仅线索**。

### 4.1 核心最近邻（已在前轮逐条核验，本轮直接引用）

| 标题 | 作者/年份/状态 | DOI/arXiv | 类别 | 输入条件 | 输出 | 需重训 | 需联合采样 | 适配三任务 | 校准对象 | 风险对象 | 关键位置 | 重合度 | 等级 | 可借鉴 | 主要风险 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Conformal Risk Control** | Angelopoulos, Bates, Fisch, Lei, Schuster / 2024 **ICLR Spotlight** | arXiv:2208.02814 | D | 校准集 + 任意单调损失 $\ell$ | 阈值 $\hat\lambda$ | ❌ | ❌ | TEP ✅ | — | **期望风险** $\mathbb{E}[\ell]\le\alpha$ | `get_lhat.py`（5 行） | **直接重合** | 正文 | **误报/漏报率变成可控期望风险** | 非事件级保证；要求损失单调 |
| **RCPS** | Bates, Angelopoulos, Lei, Malik, Jordan / 2021 **JACM 68(6)** | arXiv:2101.02703；DOI 10.1145/3478535 | D | 校准集 | 预测集 $T_\lambda$ | ❌ | ❌ | ⚠ | — | 期望损失（**高概率**保证） | Def.1 $(\alpha,\delta)$ | **直接重合** | 正文 | 高概率版本 | 比 CRC 保守 |
| **CQR** | Romano, Patterson, Candès / 2019 **NeurIPS** | arXiv:1905.03222 | B | 校准集 + 任意 QR 模型 | 自适应区间 | ❌ | ❌ | MetroPT ✅ FD004 ✅ | 分位数区间 | — | `cqr/helper.py::run_icp` | **直接重合** | 正文（代码已读） | **三段划分 + `use_rearrangement` 保序** | 仅边缘覆盖 |
| **CF-RNN** | Stankevičiūtė, Alaa, van der Schaar / 2021 **NeurIPS** | hash `312f1ba2…` | B | 校准块 | 逐 horizon 区间 | ❌ | ❌ | ✅ | 逐 horizon 同时 | — | 式(7) Bonferroni | 部分重合 | 正文 | 逐 horizon 区间 | Bonferroni 在大 H 时过保守 |
| **AgACI** | Zaffran, Dieuleveut, Féron, Goude, Josse / 2022 **ICML** | PMLR v162:25834；arXiv:2202.07282 | B | 时间 + 在线滚动 | 在线区间 | ❌ | ❌ | ✅ | 长期平均覆盖 | — | 式(2) $\alpha_t$ 更新 | 部分重合 | 正文 | 漂移下的自适应 | 仅长时平均，非同时 |
| **CACES** | Zhang, Zhou, Liu, Zhang, Lin, Pan / 2025 **ICML** | PMLR v267:77177-77193 | D | 事件时标 | conformal 分数 | ❌ | ❌ | TEP ✅ | — | **事件序列 FPR** | time-rescaling theorem | **直接重合** | 官方摘要 | 连续时间事件 | 事件序列非定长数值时序 |
| **Gneiting, Balabdaoui, Raftery** | 2007 **JRSS-B 69:243-268** | DOI 10.1111/j.1467-9868.2007.00587.x | 校准理论 | 预测 CDF | — | — | — | 全部 | **probabilistic ≠ exceedance** | — | §2.1 反例 | **直接重合** | 正文 | 🚨 **风险分数合法性判据** | — |
| **Halo** | 2026-09 预印本 | arXiv:2609.10589 | A | context | 尺度头 | ✅ | ❌ | ✅ | — | — | 结论章 | **直接重合（反例）** | 正文 | **必须在同为异方差头上消融** | 仅 5 电力市场 |
| **CGT** | 2026-04 预印本 | arXiv:2604.17998 | E | 目标特定 gate | 高斯头 | ✅ | ❌ | ❌（AD 任务） | 推理期校准 | — | 式(22)(28)(29)(42) | 机制相近 | 正文 PDF 全文 | stop-grad + 硬回退 $\gamma_{used}=0$ | 二元因果结构 |

### 4.2 反例（必须写进论文的负面证据）

**反例 A（Gneiting 2007）**：probabilistic calibration（PIT 均匀）**不蕴含** exceedance calibration。**只报 ECE/PIT/reliability 就把分位数当风险分数的做法，其风险分数未经验证。** Allen et al.（JASA 2025, arXiv:2407.03167）进一步证明 tail calibration 也不等价，要求分别报 occurrence ratio 与 excess-PIT。
> **对本项目的直接影响**：`metrics.py` 必须新增 exceedance calibration，否则任何"风险分数已校准"的表述都不成立。

**反例 B（Halo, arXiv:2609.10589）**：30 组比较 28 组改善，但收益来自"**是否存在尺度头**"这一二值判断，架构差异 <1.5%，且为点估计基线调好的超参重调后**3/5 市场变差**。
> **对本项目的直接影响**：方案 A 的消融**必须以 Gaussian 头为基线**，不能对比 point 基线。

**反例 C（Schlembach 2022, CoPPA, PMLR v179:316-318）**：标题 multivariate，机制却只是边缘区间笛卡尔积（变量轴无任何修正）。
> **判据**：nonconformity score 是**逐坐标标量**还是**跨坐标聚合**（Mahalanobis / $\ell_\infty$ / copula）。前者即使形状高维也只是边缘区间乘积。

---

## 5. 开源代码仓库核验表

> 均为**实际下载并阅读源码**；未打开源码者明确标注。

| 仓库 | 官方性 | 许可证 | 概率头 | 校准器 | 风险 evaluator | 关键文件（已读） | 协议/隔离 | 泄漏风险 | 可借鉴机制 | 不应复制 |
|---|---|---|---|---|---|---|---|---|---|---|
| `aangelopoulos/conformal-risk` | ✅ 官方 | **未核验** | ❌ | ✅ 风险控制 | ✅ FNR/F1 等单调损失 | `core/get_lhat.py`（**5 行**） | ✅ 需独立校准集 | 低（视调用方） | **风险→阈值→期望风险最短路径** | — |
| `yromano/cqr` | ✅ 官方（Romano） | MIT（LICENSE 原文） | quantile + `use_rearrangement` | ✅ `run_icp()` | ✅ `compute_coverage` | `cqr/helper.py`（590 行）、`nonconformist/` | ✅ `idx_train`/`idx_cal` 三段 | 低（内部 CV 须与 test 区分） | conformal 与回归器解耦 | — |
| `awslabs/gluonts` | ✅ 官方（DeepAR 团队） | Apache-2.0 | ✅ 默认 `StudentTOutput()` | ❌ | `ev/metrics.py`（30+ 指标） | `torch/model/deepar/module.py`、`mx/block/quantile_output.py` | ✅ scaler 只用 context；顺序 sampler | 低 | **`crps_weights_pwl()` 加权 pinball→CRPS** | — |
| `FilippoMB/Ensemble-Conformalized-Quantile-Regression` | ✅ 官方（Bianchi） | MIT | quantile | ✅ + 在线滚动 | `compute_coverage_len` | `conformal_prediction.py`、`main_EnCQR.py` | ⚠ | 🚨 **高**：用 train 标签算分数 + 循环末用 **test 标签更新 epsilon** | 分层非一致性分数 | **绝不照搬**（与 AGENTS.md 冲突） |
| `nyuolab/Gateformer` | ✅ 官方 | 未核验 | ❌ 点预测 | ❌ | ❌ | `models/Gateformer.py:100-101` | 未核验 | 未核验 | 回退分支门控 | — |
| `yolish/moe_unc_tsf`（MoGU） | ✅ 官方 | 未核验 | 专家方差加权 | 事后 conformal | ❌ | `models/MoE.py:63-65`（已读） | ✅ 校准集独立目录 | ⚠ 退化通道（见 §4.1） | — | — |
| `ServiceNow/TACTiS` | ✅ 官方 | Apache-2.0 | 边缘 Flow + copula | ❌ | `tactis/gluon/metrics.py` | `tactis/model/{marginal,tactis,flow}.py` | ✅ 只对历史归一化 | 低 | 边缘流+copula，参数对变量数线性 | — |

**未找到官方仓库（不臆造 URL）**：MQ-RNN、DiffTime、TimeWeaver、TimeGrad++、SACR、TactPFN、CAFHT、FLIPR、AgACI、MultiDimSPCI、MSR、CopulaCPTS、VolDy-VAE、Rollcast。

---

## 6. 与 Unified V3C 的逐项差异

| 组件 | 当前状态（代码核实） | 候选方向新增 | 需新增代码量 |
|---|---|---|---|
| 点预测 | ✅ 已有 | 作为基线 | 0 |
| MetroPT 分位数 | ✅ 已有 | 校准 + 指标 + 风险接口 | 小 |
| FD004 RUL 分位数 | ✅ 已有（非交叉、cap 隔离） | RUL 区间校准 | 小 |
| TEP 检测/诊断概率 | ✅ 已有 | **Platt/isotonic/temperature 校准** | 小（纯后处理） |
| CRPS / NLL | ❌ 缺 | 分位数折算（GluonTS `crps_weights_pwl`）+ 参数分布解析 | **中** |
| PICP / MPIW / interval score | ❌ 缺 | 逐 horizon / 逐 sensor | 中 |
| Brier / ECE / 可靠性图 | ❌ **且 `classification_metrics` 无法接受概率** | 必须先改造为接受概率 | **小（但阻塞项）** |
| exceedance calibration | ❌ 缺 | Gneiting 判据必需 | 中 |
| validation-only 校准 | ❌ 缺 | conformal + CRC 层 | **中（核心新增）** |
| event-level evaluator | ⚠ **已有 TEP 版**（按 run 隔离） | 扩展到 MetroPT/FD004 + persistence rule + validation 阈值 | 中 |
| persistence rule | ❌ 缺 | validation 固定 + 记入 manifest | 小 |
| per-engine 统计 | ⚠ 受限（要求 engine 唯一） | 支持多 cycle 聚合 | 小 |
| 主编码器 | ✅ 不改 | **全程不改** | **0** |

**结论**：核心新增集中在 **calibration/ + metrics/ + risk/ 三个新模块**，编码器零改动。

---

## 7. 三数据集适配性

| 能力 | MetroPT-3 | C-MAPSS FD004 | TEP |
|---|---|---|---|
| **现有输出可直接校准** | 分位数（quantile ckpt） | RUL 分位数 | ✅ 概率（恒定输出） |
| **conformal 区间** | ✅ | ✅（单值 RUL 区间） | ❌（分类用预测集/风险分数） |
| **风险分数** | ✅ 逐 sensor 标准化残差 / 区间超限 | ✅ 排序分数（per-engine） | ✅ 分类概率 + CRC |
| **事件级评估** | ⚠ 故障仅 4 起且无标签 | ⚠ 每台 run-to-failure 可作循环事件 | ✅ **主战场**（按 run 隔离已有） |
| **多步预测作为创新** | ❌ **明确排除** | ❌ **禁止构造多步 RUL 标签** | ❌ 不适用 |
| **joint coverage 主张** | ❌ 不做 | ❌ 不做 | ❌ 不做 |
| **需重训** | 阶段 0–2 否；方案 A 是 | 同左 | 同左 |

**统一风险接口设计**（三数据集共享同一核心模块）：
```
prediction_output → [风险分数定义] → [validation-only 校准器] → [阈值/persistence] → [事件级评估]
                      ↑ MetroPT: 分位数超限     ↑ conformal / CRC    ↑ val 拟合      ↑ 按 engine/run 隔离
                        FD004:  RUL 分位数超限
                        TEP:    分类概率
```

---

## 8. 创新性和审稿风险

### 8.1 分类结论

| 方向 | 分类 | 理由 |
|---|---|---|
| validation-only conformal 区间 | **A 类**（成熟组件） | CQR/CF-RNN/AgACI 已标准化；本项目尚未实现属工程补全 |
| 预测分布→风险分数 | **B 类**（合理组合） | 组件成熟，但三任务统一接口需自行设计 |
| **validation-only 事件级风险校准** | **C 类（协议贡献）** ⭐ | 组件成熟，但"工业多传感器 + 事件级 + exceedance 验证 + 三任务统一接口"未被系统做过 |
| 任务条件化异方差头 | **D 类**（需消融） | Halo 证明"条件化精巧度"未必带来收益，必须在异方差头上消融 |
| 不确定性驱动融合 | **E 类**（本轮淘汰） | 实现成本高、退化风险、与"低复杂度"冲突 |
| 可靠性—精度联合评估 | **C 类**（协议贡献） | 报告 reliability 曲线而非单点指标 |

### 8.2 审稿风险与应对

| 风险 | 应对 |
|---|---|
| "只是把 conformal 用在工业数据上" | 必须补 **exceedance calibration**（反例 A）+ 事件级指标 + 三任务统一接口；单纯套 conformal 不足以成篇 |
| "风险分数就是阈值划分的产物" | 报 **reliability diagram + ECE + Brier**，并区分 marginal/exceedance |
| "区间方法都是边缘覆盖" | 明确声明只主张 **marginal**，不主张 joint/simultaneous（反例 C） |
| "为什么不改模型" | 正面论证：见 §9 复杂度矩阵——改模型的收益不确定性（反例 B）远大于校准层的确定收益 |
| "事件数太少（MetroPT 仅 4 起故障）" | 以 **TEP 为主战场**；MetroPT 只报健康期误报率与故障前风险分数演化 |
| "单 seed" | 单 seed 仅阶段 0 初筛，正式结论 ≥3 seeds |
| "缺少通用数据集" | 若要通用性主张须加 Electricity/Traffic/Weather；否则只能写"三类工业任务验证" |

### 8.3 论文表述约束（任务书 §8 已逐条采纳）

✅ 允许："构建了面向多任务工业时序预测的校准不确定性评估流程"；"提出了可复用的 validation-only 风险校准接口"；"系统比较了分位数、参数化分布和校准后的风险评分"；"在三类工业任务上验证了预测可靠性与事件级风险指标"
⚠️ 谨慎："目标条件化不确定性估计方法"；"不确定性驱动的传感器融合"；"风险控制方法"
❌ 禁止："首次提出"；"证明完整联合分布"；"预测区间=置信区间"；"边缘覆盖=联合覆盖"；"风险分数=故障概率"；"ECE/PIT 单独证明尾部风险校准"；未运行实验的性能提升；单 seed 稳定性结论

---

## 9. 实现复杂度与预期收益矩阵

> 预期收益栏为**定性判断**，非实验数字；依据是组件成熟度与最近邻的证据强度。

| 方向 | 实现复杂度 | 训练开销 | 显存 | 数值发散风险 | 预期收益 | 证据强度 | 可先用旧 ckpt 验证 |
|---|---|---|---|---|---|---|---|
| conformal 区间（CQR/Bonferroni） | **低**（纯后处理） | **0** | 0 | **无** | 中（覆盖率确定改善，宽度可能变宽） | **强**（A 级，多篇顶会） | ✅ **是** |
| 分类概率校准（Platt/isotonic/temp） | **低** | **0** | 0 | **无** | 中（TEP 校准必然改善） | **强** | ✅ **是**（TEP 概率恒定输出） |
| 事件级风险 + CRC | **中** | 0 | 0 | 无 | **高**（直接产出可控误报预算） | **强**（ICLR'24 Spotlight） | ✅ **是** |
| exceedance calibration 指标 | **中** | 0 | 0 | 无 | 中（使风险分数可验证） | **强**（JRSS-B 2007） | ✅ 是 |
| Gaussian/Student-t 头 | 低 | 需重训 | +小 | 中（$\sigma$ 需约束） | **不确定**（反例 B：可能仅因"有尺度头"而获益） | **中**（GluonTS 已标准化） | ❌ 否 |
| 任务条件化 scale | 中 | 需重训 | +小 | 中 | **低-不确定**（Halo 警告） | **弱** | ❌ 否 |
| uncertainty-aware fusion | **高** | 需重训 | +中 | **高**（尺度退化通道） | **不确定** | 弱（先例均不完整） | ❌ 否 |
| 低秩/copula/flow/diffusion | 高-很高 | 显著 | +高 | 高 | 低（工业小样本） | 中 | ❌ 否 |

---

## 10. 分阶段实验方案

### 阶段 0：已有模型重评估（**零重训**）
- A0 V3C point（MetroPT/FD004）
- A0q V3C **quantile**（须确认 ckpt 为 quantile 模式训练，否则区间无效，见 §2.3）
- A0t TEP 原始分类概率
- 产出：冻结 checkpoint 下的点预测/分位数/概率 artifact（`evaluate_frozen_checkpoint` 已支持）

### 阶段 1：指标补全（**零重训**，阻塞项优先）
| 指标 | 说明 | 阻塞性 |
|---|---|---|
| CRPS | 分位数：GluonTS `crps_weights_pwl` 权重折算；参数分布：解析 | — |
| NLL | 参数分布必需 | — |
| PICP / MPIW / interval score | 逐 horizon / 逐 sensor | — |
| **Brier / ECE / reliability diagram** | 🚨 **必须先改 `classification_metrics` 接受概率** | **是** |
| **exceedance calibration** | Gneiting 判据，风险分数合法性的前提 | **是** |

### 阶段 2：校准层（**零重训**）
- B1 未校准分位数区间 vs B2 **validation-only CQR** vs B3 逐 horizon Bonferroni
- B4 TEP：Platt / isotonic / temperature（validation-only）
- B5 风险阈值：validation 分位数 vs conformal score vs CRC
- B6 **persistence rule**：k ∈ {1,2,3,...} **只在 validation 上选**，记入 manifest
- B7 event-level：CRC 控制漏报率/误报率

### 阶段 3：轻量概率头（**需重训**，作为对照）
- C1 Gaussian 头（$\sigma=\text{softplus}(\cdot)+10^{-3}$）
- C2 Student-t 头
- C3 **参数量匹配**的确定性头
- C4 相同训练预算与 checkpoint 规则
> ⚠ **Halo 式对照必做**：C1/C2 的比较基线必须是 C3，而非 point 基线。

### 阶段 3-bis：若做异方差头，**必须用 Beta-NLL 而非朴素 NLL**

第二轮检索到一篇直接解决"低复杂度"顾虑的工作：

**Faithful Heteroscedastic Regression with Neural Networks**（arXiv:2212.09184, NeurIPS 2022）——用**两个简单优化修改**修掉"log-likelihood 梯度导致均值次优、方差未校准"的问题，且**可证明均值精度不低于同容量的纯 MSE 均值模型**，同时方差校准达到同类最优。另有 **Variational Variance**（arXiv:2006.04910）把方差当变分变量充分正则化，通过 PPC 检查（预测均值/方差校准 + 样本质量）。

> **意义**：若阶段 3 做异方差头，**用 Beta-NLL 两行改动就能同时保住均值精度和方差校准**，直接消除 Halo 反例带来的"是否损害点精度"疑虑。这把方案 A 的成本从"中等"降到"低"。

### 阶段 4：可选创新模块（**仅在阶段 1–3 结果支持时**）
- D1 target/context-conditioned scale
- D2 uncertainty-aware fusion（**需先通过退化检验**）
- D3 risk-controlled alarm policy

### 每阶段统一报告
MAE/RMSE、NLL/CRPS、coverage、interval width、Brier/ECE、false alarm rate、event recall、detection delay、参数量、训练时间、推理时间、显存。

### 冻结项
data split、normalization（**train-only**）、seed、history/prediction length、batch size、训练预算、early stopping/checkpoint（**validation-only**）、evaluator、**calibration split**、**threshold 与 persistence rule 来源（validation-only）**。

---

## 11. 最推荐方向（详细）

**方案 B+C+D 组合：validation-only 校准 + 统一风险接口 + 事件级风险控制**

**为什么是它**：
1. **零编码器改动、零重训**（阶段 0–2 全部）——直接满足"低实现难度"；
2. **三数据集共享同一核心模块**（§7 的统一风险接口）；
3. **收益确定**：覆盖率与校准误差的改善是 conformal 的有限样本保证所**保证**的，不依赖"模型更好"这种不确定增益；
4. **贡献可辩护**：定位为 **C 类协议贡献**而非算法创新——这是任务书排除项之外最安全的定位；
5. **有现成理论工具**：CRC (ICLR'24 Spotlight) + CQR (NeurIPS'19) + CACES (ICML'25)。

**必须补齐的三件事**（否则不成立）：
1. **exceedance calibration**（反例 A）——否则风险分数未经验证；
2. **`classification_metrics` 改造为接受概率**——否则 Brier/ECE 无法计算（代码阻塞项）；
3. **persistence rule 在 validation 上固定并记入 manifest**——否则事件级指标不可复现。

**论文定位建议**：以"面向三类工业任务的校准不确定性评估流程与事件级风险协议"为第四章主线，**不主张方法创新**。

---

## 12. 需要淘汰的方向

| 方向 | 淘汰理由 |
|---|---|
| **E. 不确定性驱动融合** | ① 需改编码器 + 重训，违反低复杂度约束；② 已定位退化通道（MoGU/UGGM 无 stop-grad）；③ 先例均不完整（CGT 是 AD 任务+二元结构）；④ 若采纳必须加 A12 退化检验 + stop-grad + 硬回退，成本远超收益确定性 |
| **任务条件化 scale 作为主线** | Halo 反例表明收益主要来自"有尺度头"而非条件化精巧度；且需重训，与"阶段 0 免重训"冲突 |
| 低秩 Gaussian / copula / flow / diffusion | 任务书已明确排除；工业小样本下成本高、收益不确定 |
| **多步预测相关的一切** | 任务书明确排除；FD004 更禁止构造多步 RUL 标签 |
| 联合覆盖（joint/simultaneous）主张 | 三数据集均无必要；且反例 C 表明极易误用概念 |

---

## 13. 未核验文献、代码和实验结论

1. **本轮未运行任何实验**，§9"预期收益"全为定性判断，非实测。
2. **所有方法在三个工业数据集上的适配性与收益完全未核验**。
3. **方案 A 的直接先例未穷尽检索**：`task-conditioned uncertainty` 类工作的检索由子代理进行但覆盖面有限。
4. **未找到官方仓库**：CopulaCPTS、VolDy-VAE、Rollcast、CACES、AgACI、FLIPR、CAFHT、MultiDimSPCI、MSR。
5. **`aangelopoulos/conformal-risk` 许可证未核验**；其 5 行算法可参考但需自行实现。
6. **CF-RNN 存在两个候选官方仓库**（`kamilest/conformal-rnn` 与 `vanderschaarlab/conformal-rnn`），关系未核验。
7. **persistence rule 在文献中为"最干净的空白"**——第二轮用 5 组检索式（含 `abs:"persistent" AND abs:"conformal"`、`(consecutive AND alarm)`、`(event-level OR run length OR duration) AND conformal`）确认：**没有任何工作把"连续 k 点超限才报警"当作需在 validation 固定的超参并报告敏感性**。最接近的 arXiv:2609.27179 控的是**在线序贯检验的 ARL/误报间隔**，与"事后把逐点越限合并成事件"是**不同数学对象**。
   > **可论证的技术点（子代理判断成立，但无文献支撑）**：定义 persistence rule 后的**事件级误报损失** $L(\lambda)=\mathbb{1}\{\text{合并后事件数}>0\}$ 关于阈值 $\lambda$ **单调不减**，故 CRC 的单调损失假设直接成立，可套 `get_lhat.py`。而现有全部工作（2604.20122 / 2503.19146 / 2605.13642 / 2607.27143）**只用逐点 FAR/FDR，无事件级合并**。
8. **FD004 的 per-engine 统计需求与现有 `rul_metrics` 的唯一性约束冲突**，改动方案未设计。
9. **MetroPT-3 仅 4 起故障且无标签**，事件级评估在该数据集上的统计效力未核验。

---

## 14. 下一步实现顺序

1. **改造 `classification_metrics` 接受概率**（保留硬标签路径，向后兼容）→ 解除 Brier/ECE 阻塞
2. **新增 `metrics/probabilistic.py`**：CRPS（分位数折算 + 参数解析）、NLL、PICP、MPIW、interval score、**exceedance calibration**
3. **单元测试**：分位数单调性、区间嵌套性、逐 horizon 边界、exceedance 反例可复现（用 Gneiting 的构造例）
4. **新增 `calibration/` 模块**：CQR-style split conformal（纯后处理），断言校准集与 train/test 索引无交集
5. **新增 TEP 概率校准**：Platt / isotonic / temperature（validation-only）
6. **改造 `evaluate_tep_events`**：`threshold` 由 validation 拟合；新增 persistence rule 参数并记入 manifest
7. **新增事件级统一接口**：把 TEP 的 run-隔离评估扩展到 MetroPT（按 time block）与 FD004（按 engine）
8. **扩展 `rul_metrics`**：支持多 cycle per-engine 聚合
9. **阶段 0–2 全流程跑通**（零重训），产出第一批指标
10. **阶段 3**（Gaussian/Student-t 对照）仅在阶段 1–2 结果显示"分位数表达力不足"时才启动
11. **阶段 4** 仅在有明确证据时启动

---

## 附：可安全借鉴的仓库分工（不复制代码）

| 需要的能力 | 参考仓库 | 借鉴点 |
|---|---|---|
| conformal 区间 | `yromano/cqr` | 三段划分 + `use_rearrangement` 保序 |
| 风险阈值控制 | `aangelopoulos/conformal-risk` | 期望风险→阈值的 5 行算法 |
| 分布指标与 CRPS 折算 | `awslabs/gluonts` | `crps_weights_pwl()`、30+ 指标定义 |
| 事件级异常分数 | CACES（ICML 2025，**代码未找到**） | 时标重标定思想 |
| 安全门控模式 | `nyuolab/Gateformer` | 回退分支结构（若日后做方案 E） |

> ⚠ **绝不可照搬**：`FilippoMB/Ensemble-Conformalized-Quantile-Regression`（用 test 标签更新 epsilon，与 AGENTS.md 硬规则冲突）。