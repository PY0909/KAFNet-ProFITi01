# Unified Model Reimplementation Progress

### 2026-10-12 Round 1 Phase B：git 身份冻结（commit 完成，push 被网络阻断）

- 服务器 worktree 死指针已备份至仓库外 `/private/tmp/kaf_profiti_unified_v3.git.server-pointer.20261012165127.bak`；`git init -b unified-v3-ch3-seeds` + `remote add origin https://github.com/PY0909/KAFNet-ProFITi01.git`。
- 依赖清单经实际 import 审计写入 `requirement.txt`（torch/numpy/pandas/tqdm/pyreadr/pytest；**scikit-learn 未被任何模块导入，按审计结果不列入**，与规划假设不同）。
- 暂存审查：111 文件、21194 行；`dataset/`、`result/`、指针备份均不在索引（check-ignore 验证通过）。
- commit `e8335cc`（"ch3: freeze unified v3c reproduction snapshot"，265 tests passed）。`git diff --cached --check` 的报警均为 plan 文档中有意的 Markdown 硬换行（行尾双空格），不作为缺陷处理。
- **push 停在门禁**：github.com 443 直连不通（75s 超时 ×2），本机无代理在监听、无全局 git 代理配置；远端分支侦察（ls-remote）同样不可达，故按规划走"独立快照分支"兜底——该分支与远端历史不连通属预期，已在此登记。待用户开启代理/VPN 后执行 `git push -u origin unified-v3-ch3-seeds` 并复核远端是否已有同名分支（若有，先比较不得静默强推）。
- **当日续：push 完成，Phase B 关闭**。用户开启代理后（系统级生效；本机代理端口 7897，直连仍间歇超时，git 可用 `-c http.proxy=http://127.0.0.1:7897` 稳定访问）：ls-remote 确认远端仅有 `main`（7af9d58），无同名分支；push 建立新分支 `unified-v3-ch3-seeds`，复核远端哈希与本地 `7391252` 完全一致。

### 2026-10-12 Round 1 Task A4：provenance 与 TEP 公平性代码改动（TDD 完成）

依据 [Round 1 规划](2026-10-12-round1-v3c-single-seed-reproduction-plan.md) Task A4，先写失败测试（15 个新测试红）再实现，全量回归 **265 passed**（原 250 + 新 15），并用真实数据 FD004/MetroPT unified v3c 1-epoch CPU smoke 复跑确认 manifest 新字段端到端齐全（断言通过；产物在 `/tmp/kaf_v3_local_smoke`，仓库外）。

- `industrial/datasets.py`：新增 `bundle_content_provenance()`——`split_members_sha256`（FD004 engine 集 / TEP dataverse unit 集 / MetroPT 显式划分行区间，与 stride 无关）与 `window_catalog_sha256`（逐窗口身份），六个 DatasetBundle 构造点全部写入；各 adapter metadata 补记 history/pred/stride（MetroPT）、history/sensor_filter/split_seed（FD004）、history/split_seed/valid_fraction/requested_max_runs_per_fault（TEP dataverse）。
- `experiments/entrypoint.py`：manifest 新增 `epochs/batch_size/lr_scheduler/early_stopping_patience/device/amp/head_mode` 与 `target_normalizer_stats_sha256`；`_normalize_bundle` 计算 `normalizer_stats_sha256`（condition 模式含逐工况统计），并对未经 adapter 的 bundle 兜底补齐内容哈希；TEP 任务在 comparison 模型上调用 `set_tep_class_weights` 注入同一组 train-frequency 权重。
- `compare_code/models/base.py`：`BaselineConfig` 增加 `tep_class_weights`（长度校验）；`ForecastBaseline.set_tep_class_weights()`；TEP loss 使用权重 NLL；`predict()` 输出 `detection_probability = 1 - p(class 0)`——基线由此自动接入与 unified 相同的 validation-only 阈值选择和事件级评估（entrypoint 通用消费路径）。
- `compare_code/cli.py`：新增 `--lr` 并显式传给 `run_bundle(learning_rate=...)`。
- 测试：`test_dataset_adapters.py`（3 个：三数据集配置字段+SHA 确定性与敏感性）、`test_experiment_entrypoint.py`（2 个：manifest provenance 全字段+确定性+敏感性；TEP comparison 权重/detection 通路端到端）、`test_compare_models.py`（4 个：五基线 detection_probability、加权/无权 loss 数值对照、config 校验）、`test_compare_cli.py`（2 个：--lr 解析与转发）。偏离说明：`test_cli.py` 未改——unified CLI 已有 `--lr`（既有测试覆盖），provenance 经 run_bundle 层落地，无需 unified 侧改动。
- Git commit 按 Phase B 流程在测试通过后统一冻结（当前本地 `.git` 仍为服务器死指针）。未运行任何正式训练；未修改既有 `result/`。

### 2026-10-12 Round 1 Phase A1–A3：数据拷贝与本地验证

依据 [Round 1 规划](2026-10-12-round1-v3c-single-seed-reproduction-plan.md) 完成 Phase A 的 A1–A3，全部断言式验收通过；A4（provenance/公平性代码改动）未开始，git 未重建，AutoDL 未开机。

- A1 拷贝：`kst_light_v2/dataset` 三目录（CMAPSSData 43M、metropt+3+dataset 208M、dataverse_files 1.3G，共 22 文件）以 APFS clone 拷入 `dataset/`；未覆盖任何代码或结果目录。
- A2 校验：TEP 自带 `SHA256SUMS.txt` 4/4 OK；源/目标全量 SHA-256 清单 22/22 完全一致（含清单行数断言。首次执行因子 shell 无 `shasum` 产生 0 行空清单的 diff 假阳性，被行数断言拦截后以绝对路径复验通过）。
- A3 本地验证：pytest 全量 250 passed（`/opt/anaconda3/envs/kaf_profiti`，torch 2.5.1 CPU）。unified v3c 1-epoch CPU smoke 两项均通过产物断言（manifest/checkpoint/metrics/test_metrics/efficiency/predictions 齐全、身份字段正确、全部指标 finite）：FD004（history 80 / cap 150 / all_windows / condition / standard，1 epoch MAE 51.1 属未训练量级）；MetroPT（official_first_month 168/24/60，1 epoch overall MAE 1.008），且 `split_identity=metropt3_official_first_month_v1_val20pct-units1516948` 与既有冻结 run 一致。smoke 产物位于仓库外 `/tmp/kaf_v3_local_smoke`，不进入论文。
- TEP 本地 smoke 按规划跳过（`datasets.py` 显式加载 Linux `libiconv.so`，macOS 不可用），改在 AutoDL Phase C 环境门禁执行。

### 2026-10-11 Unified/CUDTR 十二项设计问题整改方案

新增 [Unified/CUDTR 十二项设计问题分阶段解决方案](2026-10-11-unified-model-12-defects-remediation-plan.md)。方案明确官方 MetroPT 首月训练、后续月份测试协议保持不变，单独修正 quantile calibration 的 validation 内部隔离；覆盖全缺失源屏蔽、source/base 融合层统一、fallback 诊断、soft-to-hard routing、CUDTR 与全量 attention 消融、多尺度/horizon 解码、FD004/TEP 任务适配、TEP 类别不平衡和 validation-only 阈值校准。该文档为实施方案，尚未修改模型代码或既有 result。

2026-10-11：完成整改 Stage 0 可观测性与回归门禁。CUDTR prediction metadata 和 test prediction artifact 新增逐 route `source_available`；新增无标签诊断汇总，记录 source 选中次数/比例、平均权重、不可用源选中数、fallback 均值/P95/饱和率、route disagreement 和 route inactive 判据。新增 5 个后续阶段的严格 xfail 门禁，覆盖 P1/P2/P4/P5/P7；待对应阶段修复后转为绿色断言。`PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python -m pytest code/tests -q`：234 passed，5 xfailed。未修改既有 result，未启动训练；下一步为 Stage 1 输入有效性、表示边界和 fallback 修复。

2026-10-11：完成整改 Stage 1（P1/P2/P4/P7）。CUDTR 仅允许严格正观测率源进入路由，全缺失源在默认配置下不可用；source 与 target base 统一使用 encoder 的 pre-cross 表示边界；fallback 默认改为 0.5 中性有界初始化并记录饱和率/失活判据；metadata 明确 MetroPT 逐传感器 head 与 FD004/TEP global task head 的表示契约。旧的“fallback 永久接近 1 且严格复现 V3C”测试改为有限值和稳定性断言。`PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python -m pytest code/tests -q`：238 passed，1 xfailed（Stage 2 soft-to-hard routing）。未修改既有 result，未启动训练；下一步为 Stage 2 稀疏融合与 top-k 训练策略。

2026-10-11：完成整改 Stage 2（P5/P6）。训练态 CUDTR 使用 dense soft routing，eval 态才启用 hard top-k；新增 `attention` 与 `cudtr_only` 跨传感器结构模式，后者关闭 encoder 的 dense cross-channel attention，避免把重复叠加结构误当作独立路由收益；结构模式写入 config/manifest/metadata。CLI 增加 `--cross-channel-mode`。`PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python -m pytest code/tests -q`：240 passed，1 xfailed（后续 validation 隔离阶段）。未修改既有 result，未启动训练；下一步为 Stage 3 validation selection/calibration 底层窗口隔离。

2026-10-11：完成整改 Stage 3（P3）。validation selection/calibration 拆分在存在 `window_start/window_end` 时按底层原始区间隔离，删除与 selection 区间重叠的 calibration 窗口；MetroPT 窗口适配器保存原始窗口边界。manifest 记录两段窗口范围、是否完成 overlap 检查和 window gap；旧无边界 synthetic 样本保留兼容路径但标记未检查。`PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python -m pytest code/tests -q`：241 passed，1 xfailed。未修改既有 result，未启动训练；下一步为 Stage 4 多尺度 token 与 MetroPT horizon-specific decoder。

2026-10-11：完成整改 Stage 4（P8/P9）。新增 MetroPT 专用 `state_decoder=scale_horizon`：保留 encoder 的尺度 token，使用 horizon query 生成逐预测步、逐传感器的 point/quantile 输出；默认 global decoder 和 FD004 单值 RUL、TEP 分类路径不变。CLI 增加 `--state-decoder`，config/manifest 保留 decoder 身份。`PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python -m pytest code/tests -q`：243 passed，1 xfailed。未修改既有 result，未启动训练；下一步为 Stage 5 FD004/TEP 任务级 utility、TEP 类别不平衡和 validation-only 阈值。

2026-10-11：完成整改 Stage 5（P10/P11/P12）。TEP 增加 validation-only F1 threshold 选择，固定阈值用于 test event detection，并记录 detection rate、miss rate、false alarm rate 与 delay 分布；TEPHead 支持由 train split 统计提供的 `tep_class_weights`，用于诊断类别不平衡。对比模型无 detection probability 时保持原分类路径，不虚构事件指标；FD004 保持单值 RUL 与任务级 utility 语义。`PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python -m pytest code/tests -q`：245 passed，1 xfailed。未修改既有 result，未启动训练；下一步为 Stage 6 统一消融、敏感性和三数据集验证。

2026-10-11：完成整改 Stage 6 代码与协议收口。TEP 训练入口在 metadata 未提供权重时从 train split 自动计算 balanced class weights；checkpoint/manifest 保存 `cross_channel_mode`、`state_decoder`、`tep_class_weights` 等整改配置；新增 artifact provenance 回归测试。`PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python -m pytest code/tests -q`：247 passed。未启动三数据集长时矩阵训练，未修改既有 result；Stage 6 的正式结论仍需按统一消融矩阵以新 run ID 实验验证。

### 2026-10-10 低复杂度不确定性校准与风险评估方向查新完成

新增 [低复杂度高收益的不确定性校准与风险评估调研报告](2026-10-10-low-complexity-uncertainty-calibration-research-report.md)：14 节齐备，含 V3C 能力/缺口代码核实表、6 类候选模块分类、8 篇核心最近邻、7 个已读源码仓库、复杂度×收益矩阵与五阶段实验方案。**核心结论**：推荐 **B+C+D 组合（validation-only 校准 + 统一风险接口 + 事件级风险控制）**，定位 **C 类协议贡献**，**零编码器改动、阶段 0-2 零重训**。**四条关键代码实证（此前报告未发现）**：① `metrics.py:66-67` 的 `classification_metrics` 开头即 `.long()` 转硬标签，**Brier/ECE/可靠性图当前根本无法计算**（阻塞项）；② `rul_metrics:47-50` 要求 `len(set(engine_ids))==len(engine_ids)`，**只支持 test 末尾单点，无法统计整段 per-engine 误差**；③ `StateHead.loss()` 在 point 模式只对 `base` 求梯度、`increments` **零梯度**，故 **point 训练的 ckpt 切 quantile 模式会得到未训练区间**（任务书"阶段 0 用旧 ckpt 得区间"不可行，但 conformal 与 TEP 概率校准零重训可行）；④ `evaluate_tep_events(threshold=0.5)` **硬编码**，且全仓无 persistence rule。另确认 RUL 头已有非交叉分位数与 raw/cap 语义隔离，TEP 头概率恒定输出，`frozen_evaluation.py:40` 支持已有 checkpoint 重评估。**方法学警告**：Gneiting JRSS-B 2007 证明 probabilistic calibration ≠ exceedance calibration，故必须新增 exceedance 指标否则风险分数未经验证；Halo(arXiv:2609.10589) 证明收益来自"是否存在尺度头"而非条件化精巧度，故方案 A 的消融必须以 Gaussian 头为基线。**淘汰**：方案 E 不确定性驱动融合（需改编码器+退化通道+先例均不完整）、任务条件化 scale 作主线、低秩/copula/flow/diffusion、多步预测相关、联合覆盖主张。本轮仅联网调研与文档产出，未修改代码/模型/result，未运行任何实验。

**第二轮定向检索（选择性预测/弃权、误报预算、persistence 校准、任务条件化、RUL 风险排序、异方差正则）已并入同一报告**，7 条关键 arXiv ID 经主代理独立 curl abs 页复核标题一致。**三项改变结论的新证据**：① **纯后处理路线获得工业场景直接先例**——arXiv:2604.20122（Adaptive Conformal AD with TSFM，ICLR 2026）明确 "post-hoc, without requiring additional fine-tuning"，把异常分数 p-value 化后**误报率可直接读出**且有漂移自适应；arXiv:2503.19146（CSP plants）直接以"validation 上调阈值优化 F1"为被替换 baseline，改为有限样本风险控制 + 弃权转交专家。② **新增必须回答的前置负面证据**——arXiv:2607.02124 在 19 driver/55,563 条真实遥测上发现**即便纯后处理 rolling split conformal 也会因残差自相关违反 exchangeability，产生 15.3% 误报率**（Ljung-Box p<0.001）；MetroPT-3 的 stride=60 重叠窗口正是高自相关场景，故"依赖强度→覆盖率崩塌幅度"须作为任何方案的前置论证。③ **persistence rule 确认为"最干净的空白"**——5 组检索式确认无任何工作把"连续 k 点超限才报警"当作需在 validation 固定的超参并报告敏感性；最接近的 arXiv:2609.27179 控的是在线序贯检验 ARL/误报间隔，与"事后逐点越限合并成事件"是不同数学对象。可论证点（**子代理判断成立但无文献支撑**）：事件级误报损失 $L(\lambda)=\mathbb{1}\{\text{合并后事件数}>0\}$ 关于阈值单调不减，CRC 单调损失假设直接成立可套 get_lhat.py，而现有工作（2604.20122/2503.19146/2605.13642/2607.27143）**全部只用逐点 FAR/FDR，无事件级合并**。另：arXiv:2212.09184（NeurIPS 2022, Faithful Heteroscedastic Regression）证明**两个简单优化修改（Beta-NLL）即可保住均值精度不低于同容量纯 MSE 模型**，把方案 A 成本从"中"降到"低"；arXiv:2502.17264（Kandinsky CP）可作"每数据集一个校准器"的分组条件校准统计依据。**新登记的未核验项**：三数据集统一接口缺深度代表作，FiLM 不在 arXiv 未核验；"多任务共享编码器+per-task 头"的时序代表作未找到；RUL 风险排序方向基本空白（可能是原创空间）；工具链受限（websearch 空返回、Semantic Scholar 持续 429、Jina 已封禁）。

### 2026-10-10 目标条件化不确定性/Conformal/风险评分调研完成

新增 [不确定性、Conformal 校准与风险评分调研报告](2026-10-10-target-conditioned-uncertainty-risk-research-report.md)：四方向（目标条件化不确定性 / 不确定性驱动融合 / 多步 conformal 与多变量覆盖 / 风险分数与事件级校准）共 40+ 篇证据 + 代码仓库核验 + V3C 代码差异 + 分阶段实验矩阵。**代码核实**：任务书描述基本准确，唯一需修正——TEP 侧**已有事件级评估骨架**（`heads/tep.py` 的 `detection_delay()` / `evaluate_tep_events()`，按 `run_ids` 隔离统计检测延迟/误报/漏报），缺的是 validation-only 阈值拟合（当前 `threshold=0.5` **硬编码**）与跨任务统一流程。**方向二结论已按子代理深读修正：确有 5 例不确定性进入融合（CGT/MoGU/UGGM/IBN/Under-Cali），但无一同时满足"逐传感器连续加权+跨变量轴+stop-grad+硬回退"**——CGT 变量级且有三重防护与硬回退 gamma=0，但是 AD 任务+二元因果结构；MoGU 融合专家非变量；**官方代码 models/MoE.py:63-65 确认无 stop-grad，存在"推高 sigma 逃避 NLL"的纯退化通道**；UGGM 作用在时间维非跨变量且 L_gate 只约束平滑性；IBN 只管插补变量；Under-Cali 路由样本非变量。**方向一定性：子代理用 6 检索式 x 2 路径确认"预测目标变量条件化不确定性"是明确检索空白（非检索失败）**。**新增两个反例**：Halo（arXiv:2609.10589）证明收益来自"是否存在尺度头"这一二值判断而非条件化精巧度——故任何条件化头的主张必须在同为异方差头的基线上消融（已加入实验臂 A13）；TimeCorr（Sensors MDPI 2026, DOI 10.3390/s26196238）有 "Selective Correlation Module" 但无任何不确定性参与，且其 43 篇参考链整体不落入本方向。**实验矩阵已扩充 A12 退化通道检验 / A13 Halo 式对照 / A14 无关源变量置换**。**方向三判定为校准工具而非独立贡献**（CF-RNN/ACI/AgACI/MWCP/CAFHT/FLIPR/O²CP/CopulaCPTS/MultiDimSPCI/MSR/CRC 组件全成熟）。**方向四最有价值**：CRC（ICLR 2024 Spotlight，arXiv:2208.02814，官方代码 `aangelopoulos/conformal-risk` 仅 5 行）+ CADES（ICML 2025）+ TRACE-CRC + RESS 2026 的 RUL UQ 校准流程（校准集二等分）构成直接最近邻。**🚨 最重要的方法学发现**：Gneiting et al. (JRSS-B 2007) 证明 **probabilistic calibration ≠ exceedance calibration**——只报 ECE/PIT/reliability 就把分位数当风险分数者，其风险分数本身未经验证（Allen et al. JASA 2025 进一步证明 tail calibration 亦不等价）。**检索发现三个真空**：① "H 步 × d 变量"完整联合预测区域无理论+实验齐全的代表作；② 事件级覆盖与覆盖率关系无理论刻画（CRC 控制的是期望层面风险，非"每个事件至少检测一次"）；③ persistence/alarm-merging 规则几乎无文献。**两个反例**：Gneiting 2007（ECE≠风险分数合法性）、Schlembach 2022（标题 multivariate 但机制只是边缘区间笛卡尔积——判据：nonconformity score 是逐坐标标量还是跨坐标聚合）。结论：第四章重心应从"概率预测模型"移到"校准 + 事件级风险评估"。两路子代理结果已全部并入（本条为最终版）。本轮仅联网调研与文档产出，未修改代码/模型/result，未运行任何实验。

### 2026-10-10 多变量时序概率预测与风险评估调研完成

新增 [概率预测与风险评估调研报告](2026-10-10-probabilistic-forecasting-research-report.md)：11 节齐备，含方法族分类表、论文证据表、代码仓库表、V3C 差异矩阵、五方案（分位数/Gaussian/低秩/flow/diffusion/conformal）对比与分阶段实验矩阵。**关键代码发现**：V3C **已有分位数头**（`StateHead`，`quantiles=(0.1,0.5,0.9)`，`base + cumsum(softplus(increments))` 保证天然非交叉）与 `pinball_loss`，真正缺口是 **`experiments/metrics.py` 无 CRPS/NLL/PICP/MPIW/ECE/Brier** 且**无 validation-only 校准层**——故第一版应"不新增概率头，补指标 + 加 conformal 层"。推荐路线：阶段 1 分位数（已有）+ conformal + 指标；阶段 2 Gaussian/Student-t 对照；阶段 3 低秩可选；**明确不做 flow/diffusion**。代码核验（11 个已打开源码的仓库 + 6 个确认无官方仓库）追加两项：**GluonTS `crps_weights_pwl()` 证明加权 pinball 可折算 CRPS ⇒ 现有分位数头无需采样即可算 CRPS**；**⚠ 两个泄漏风险（子代理逐行阅读发现）**：① EnCQR 仓库用 train 标签算 conformal 分数且循环内用 test 标签更新 epsilon，**与 AGENTS.md 硬规则冲突，不可照搬**；② TSDiff 用 per-window 缩放而非 train-split 归一化。✅ 可安全借鉴：GluonTS/pytorch-ts/TACTiS 的 scaler 均只取 context 段（GluonTS 默认分支为 `dev` 而非 master/main）。引用纠错：**"DeepVAR" 实为 Salinas et al. NeurIPS 2019 arXiv:1910.03002（GP-Copula，论文内称 Vec-LSTM），AAAI 2020 的 "DeepVar" 是生物医学 NER 论文**；同名混淆对：DiffTime≠TimeDiff、TimeWeaver≠Time Weaver。**方法学教训已记录（§10.11）：本轮我方记忆中的 9 个 arXiv ID 全部错误**（DeepAR 1912.01621→1704.04110、DiffTime 2303.05844→实为 TimeDiff 2306.05043 等），任何 ID 写入论文前必须核验 abs 页或官方 proceedings 页。工具受限：arXiv API 与 Jina Reader 本轮中后期对本机 IP 限流/封禁，部分核验改用官方 proceedings 页。本轮仅联网调研与文档产出，未修改代码/模型/result，未运行任何实验。

### 2026-10-09 目标条件化依赖选择与稳定融合调研完成

新增 [目标条件化依赖选择与稳定融合调研报告](2026-10-09-target-conditioned-dependency-research-report.md)：20 篇论文证据表 + 10 个代码仓库核验 + V3C 组件级差异矩阵 + 分阶段实验矩阵（B0–B15 十六臂 + 合成控制 A1–A3）。**核心发现**：① **Gateformer（ICML Workshop 2025，非主会）已在代码层实现"自身历史分支 + 跨变量分支 + 门控融合"**（`models/Gateformer.py:100-101`），故该部件不可作创新；② MS-FLOW/CCM/TimeFilter 已占"选择性依赖"；③ **"目标条件化路由"与"按预测步选不同源变量"在本次检索范围内未发现直接先例**，是唯一可辩护接缝；④ 建议定位为"组合创新 + 受控实证"而非新机制，并须主动引用 Gateformer。V3C 代码核实：v5 变体含有序 patch_tokens 与 patch_query（变量内选择），但**无自身历史回退分支**。工具事实：三个并行子代理**全部因速率限制失败**，检索由主代理自行完成，广度低于计划，已在报告 §10.6 声明。修正 TimeFilter 仓库：官方仓库未找到，`TROUBADOUR000/TimeFilter` 判定为第三方实现（owner 与论文作者不符、无许可证）。本轮仅联网检索与文档产出，未修改代码/模型/result，未运行任何实验。

### 2026-10-09 模块级联网调研完成（通用时序预测顶刊顶会方法）

新增 [模块级联网调研报告](2026-10-09-module-innovation-research-report.md)：基线 Unified V3C 代码核实（5 个可测量弱点 W1–W5，含 patch_tokens 是每尺度摘要 token 非有序 patch 序列、"attention pooling"实为 Linear(h,1) 的精度修正）；七类问题检索 25 篇（深读 8 篇）；推荐主方向 SCDI（跨尺度跨变量联合交互，组合设计/工程适配，**原创性证据不足**，Crossformer 最近邻）；备选 MS-FLOW 风格稀疏跨变量依赖；淘汰 Pathformer 路由/归一化/N-HiTS。数据集推荐 MetroPT-3 + Beijing PM2.5（UCI 381）主预测、ETTh1/TEP 扩展。子代理 5 处 arXiv ID 错误已修正（Crossformer/Pyraformer/RevIN/SAN/Dish-TS），1 处期刊错误（CI/CD trade-off 实为 TKDE 2024 非 PVLDB），4 篇发表状态标"待核验"。本轮仅联网检索与文档产出，未修改代码/模型/result，未运行任何实验。

### 2026-10-09 时滞感知×工况条件化方向联网查新完成

新增 [面向工况变化的时滞感知多传感器时序预测调研报告](2026-10-09-lag-condition-forecasting-research-report.md)：A/B/C/D 四组检索，33 条最近邻结构化比较（12 字段）+ 5 维机制对照表，逐篇给出章节/公式/图号证据定位与 A–D 证据等级；V1/V2 判 A（DeltaLag 式 6–14、LIFT 式 3–9 已完整实现），V3/V4 判 C（仅作研究假设，禁"首次提出"）。附 10 臂实验矩阵、V1–V4 六问评估、三数据集协议核验与 56+11 条证据索引。数据集实测核验：本地 `dataset/CMAPSSData` 复测 FD004 为 **249 train / 248 test**（NASA readme 写反，仓库 `plan/2026-10-05-dataset-specific-ch3-ch5-experiment-plan.md:18` 与实测相反、待确认后修正）；MetroPT-3 UCI 明标无缺失但存在 331 处记录级断档（37.9 天、机制未证实）；TEP 推荐 Rieth 2017 版、采样 3 min；"PHM 2012 航空发动机"官方数据集不存在。工具事实更正：`agent-reach` v1.5.0 实际可用（`/root/.local/bin` 未入 `PATH`），其 Exa 后端因缺 mcporter 为 off，`paper-spine` 文件位于 `/root/.codex/skills/paper-spine` 但本会话未注册、其证据边界改人工遵从。本轮仅联网检索与文档产出，未修改代码/模型/result，未运行任何实验。

### 2026-10-09 SOSR 联网查新完成

新增 [SOSR 候选创新联网调研报告](2026-10-09-sosr-literature-research-report.md)：核验 14 篇最近邻（含任务书点名的 Raindrop/BiTGraph/S4M/ChannelTokenFormer/CRIB/Merlin 六篇精读），给出逐部件重合矩阵、四级创新判定、三数据集缺失语义审查与 A0–A10 消融实验矩阵。结论：部件 5 被 HD-TTS（ICML 2024）直接重合，整体暂定"部分重合 + 组合创新候选（有条件）"，不得称首次提出。本轮仅联网检索与文档产出，未修改代码/模型/result，未运行任何实验。

### 2026-10-09 第四章方向调整与联网调研委托

用户明确第四章不再进行概率区间预测，前述概率方案仅保留为历史提案，不作为后续执行主线。新增 [SOSR 候选创新联网调研任务书](2026-10-09-sosr-literature-research-prompt.md)，将观测支持条件化传感器路由作为待查新假设，要求核验最近邻、旧项目源码引用、人工缺失范围与三任务适配。本轮仅整理委托要求，未完成联网文献核验，未实施模块或启动实验。

### 2026-10-09 第四章贡献与图表重构方案

新增 [第四章创新点与实验图表重构方案](2026-10-09-ch4-innovation-and-figure-plan.md)，以 Unified V3C 为主基线，提出条件化非对称分位数头的待验证假设、概率损失修复前置条件、raw/cap 语义、校准隔离、同预算对照及图表证据矩阵。该文件仅为规划，未修改模型或正文，未启动实验，不代表算法新颖性已完成文献核验或性能改善已证实。

### 2026-10-07 V4-1 shared temporal encoder

新增可选 `v4` 编码器版本：在历史值、时间戳、观测掩码上加入 mask-safe 一阶/二阶差分，经 dilation 1/2/4 的 depthwise residual temporal blocks，并返回逐传感器 `recent_features` 和 `trend_features`。V2/V3 路径保持不变；MetroPT、FD004、TEP 仍使用各自任务头和独立参数。torch23 全量测试 125 passed；三个任务均完成合成数据前向/反向 smoke，CLI、冻结评估兼容和 compileall 检查通过。暂未调整各任务头目标函数、checkpoint 选择指标或校准逻辑，也未运行真实数据训练，不能据此声称指标改善。

## Completed stages

### 2026-10-07 chapter experiment matrix

新增 `plan/2026-10-07-ch3-ch5-dataset-experiment-matrix.md`，逐数据集列明第三至第五章实验类型、指标类别及表图产物，并明确 TEP 第三章分类任务、类别不平衡处理、按 run 统计及章节间指标边界。该文件是待冻结的实验规划，不包含或声称任何新实验结果。

2026-10-07：依据当前 `code/` 实现复核并修订第三至第五章矩阵，区分目标实验与已实现 evaluator/artifact。确认当前 test 指标只覆盖 MetroPT 状态 MAE/RMSE、C-MAPSS RUL MAE/RMSE/NASA score；TEP runner 尚无分类指标输出，五个比较模型尚无 TEP 分类适配；第四章虽有分位数头及部分预测序列化能力，但缺正式 test 概率指标，NLL/CRPS 等不得默认列为可报告结果；第五章风险校准和事件评价尚未端到端接入。补充分阶段准入门槛。仅修改规划文档，未运行实验或改动结果数据。

### 2026-10-06 direction reset

新增执行基线：`plan/2026-10-06-unified-model-target-and-correction-spec.md`。该规格将论文与 unified 代码主线改为规整输入、无缺失建模、变量级共享编码器和数据集特定任务头；列出 P0-P2 缺陷、文件边界、阶段门禁和验收标准。当前仅完成规格文档，代码整改尚未开始。

| Stage | Scope | Commit | Verification |
|---|---|---|---|
| 1 | Package contracts, history-only batch, output identity | `3083ecd` | 8 targeted tests passed |
| 2 | Mask-safe hierarchical temporal encoder | `7c5bd15` | 4 targeted tests passed |
| 3 | MetroPT state, C-MAPSS RUL, TEP detection/diagnosis heads | `a3b7bd1` | 4 targeted tests passed |
| 4 | Dataset-specific model assembly and parameter accounting | `b368734` | 4 targeted tests passed |
| 5 | Generic batch/protocol/manifest/registry contracts | `c030e8e` | 4 targeted tests passed |
| 5a | Deterministic dataset-unit splits and missingness helpers | `7800b21` | 4 targeted tests passed |
| 6 | Native dataset adapters, artifact writer, training runner and CLI | pending | 7 targeted tests; full regression 36 passed |

## Follow-up verification

- Task runner smoke: 1 passed.
- Final regression after all code commits: 29 passed.
- Full test command must be run in conda `torch23` before formal experiments.
- No formal multi-seed result, real-data TEP training result, or paper performance claim is established.
- Raw MetroPT/C-MAPSS/TEP file parsers, runnable dataset-specific CLI, checkpoint writer, and prediction/metrics artifact writer are not implemented yet; Stage 4 is an interface foundation, not an end-to-end experiment pipeline.
- Stage 6 now provides baseline adapters for MetroPT CSV, official C-MAPSS FD004 text files, and numeric TEP `.dat`, plus `code/run_unified_experiment.py`. File naming, release-specific TEP metadata, and full native-data validation remain prerequisites before formal runs.
- Real-data smoke on checked-out files succeeded in a temporary output root: MetroPT datetime CSV, FD004 official text/RUL files, and TEP `d01.dat` all produced finite windows and isolated artifacts. This is an engineering smoke only, not a formal result.

## Boundaries

- `result/` and external project directories were not modified.
- Normalization statistics require `source_split="train"`.
- Test evaluation requires an explicit `evaluate_test=True` call.
# 整改进度

2026-10-10：新增 [CUDTR 模块实现与第四章实验方案](2026-10-10-cudtr-implementation-and-ch4-experiment-plan.md)。方案以反事实效用蒸馏的目标条件化依赖路由为第四章候选核心模块，明确训练期效用教师、推理期无标签路由、top-k 稀疏选择、有界稳定回退、概率输出接口、三数据集适配、对比/消融/敏感性实验和阶段门禁。已注明 TFT、CAMul、MoNIG 等相邻方法及查新边界；当前仅为规划，未实现、未训练、未修改既有 result。

2026-10-10：修订 [第四章任务条件化结构化概率预测模块实现方案](2026-10-10-calibrated-uncertainty-risk-module-implementation-plan.md)。根据章节职责调整为：第四章负责任务条件化概率解码、位置/不确定性双分支、可选 MetroPT 低秩共享因子及原始概率质量评价；第五章独立消费冻结概率产物，负责 conformal、分类概率校准、exceedance calibration、阈值、persistence 和事件级风险评估。明确不以多步预测或联合采样作为创新依据，不构造 FD004 多步 RUL 标签。该文档仅为实现规划，尚未修改模型代码、训练结果或既有 `result/`。

2026-10-06：完成统一模型无缺失主线 Stage 0-2 初步整改。移除 unified 输入掩码和 missingness 包，加入通道级 temporal encoder、非有限输入硬失败、任务头中心量一致性、C-MAPSS raw test RUL 语义、TEP composite-run loader 入口、validation-best checkpoint 恢复和 artifact provenance 字段。torch23 回归测试 36 passed；正式实验尚未运行。
2026-10-06：完成 Stage 3 事件评价与工程门禁。TEP 返回检测/漏报/误报率及延迟分布；manifest 使用可读 provenance，不记录内容摘要；runner 拒绝非有限 loss；registry 仅将 unified 模型标记为 enabled，legacy 标记为 not_implemented。torch23 回归测试 41 passed；正式实验尚未运行。
2026-10-06：训练入口增加 epoch 和 batch 进度条。CLI 默认显示，可用 `--no-progress` 关闭；直接调用 `run_bundle` 通过 `progress` 参数控制。torch23 回归测试通过；正式实验尚未运行。
2026-10-06：训练入口完成 GPU/AMP、批次预缓存、train-only z-score 归一化、向量化通道编码、validation 学习率调度和早停。新增 `--device`、`--no-amp`、`--scheduler`、`--patience` 参数；torch23 回归测试 44 passed，CUDA synthetic smoke 通过；正式实验尚未运行。
2026-10-06：修复 MetroPT stride=1 导致的内存峰值问题。MetroPT CLI 默认使用 `stride=60`，仍可通过 `--stride` 调整；完整数据窗口构造 smoke 得到 train/validation/test 为 12639/5054/7582。正式实验尚未运行。
2026-10-06：补齐第三章 MetroPT test 指标与效率 artifact。新增 overall/逐传感器 MAE、RMSE；CLI 显式 `--evaluate-test` 后仅对 validation-best 模型执行一次 test 前向评估；产出独立 `test_metrics.json`、`efficiency.json`，记录 run/seed/checkpoint/split/schema/normalizer 追溯字段、训练时长、单批推理耗时、参数量及 CUDA 峰值显存。run id 默认唯一，重复指定的目录会拒绝写入；未修改既有结果目录。torch23 全量测试通过，尚未用真实 MetroPT 数据运行，不构成论文性能结论。
2026-10-06：Compare-Code 阶段 A 完成。新增比较模型共同 point/loss/config 接口及 TCN、GRU；输入仅使用历史值，输出统一为 `[batch,horizon,channels]`。torch23 定向测试 5 passed；尚未接入真实数据 runner，未形成性能结果。
2026-10-06：Compare-Code 阶段 B 完成。新增 PatchTST、TimesNet、iTransformer，五个模型统一输出 shape 并支持有限值检查和梯度更新；短历史/patch 与 top-k 边界已覆盖。torch23 定向测试 8 passed，RTX 4060 Ti CUDA 五模型 smoke 通过；尚未接入共享 runner，未形成性能结果。
2026-10-06：Compare-Code 阶段 C 完成。共享 `run_bundle` 支持可注入比较模型工厂，新增 `compare_code.run_experiment` CLI；基线身份进入 manifest/checkpoint config，继续复用 train-only normalizer、validation-best 和显式 test evaluator。定向接入测试 4 passed；尚未运行真实模型矩阵，未形成性能结果。
2026-10-06：Compare-Code 阶段 D 完成。真实 MetroPT TCN 1 epoch smoke 使用新目录 `result/compare_tcn_smoke_20261006/` 成功生成 checkpoint、manifest、test_metrics 和 efficiency；这是工程链路验证，不是正式比较结果。torch23 全量测试 62 passed，compileall、CLI help 和 diff check 通过。正式五模型、多 seed 矩阵尚未运行。
2026-10-06：修复 Unified MetroPT state 目标未归一化导致的通道量纲塌缩。state target 统计量仅由 train split 拟合，训练在标准化空间进行，test 指标和 validation predictions 恢复到 raw 传感器量纲；manifest、test_metrics 和 checkpoint config 记录目标归一化身份与来源 split。50 epoch 单 seed 诊断运行 `result/metropt3_state_seed2026_targetnorm_20261006/` 得到 raw overall MAE 0.5622、RMSE 1.7276，Oil_temperature MAE 2.6895；不作为正式多 seed 论文结论。torch23 全量测试 64 passed，compileall 和 diff check 通过。
