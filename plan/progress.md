# Progress

## 2026-06-24

- 阶段：S5 Review
- 已读取用户提供的新版结构、对应修改建议、补充建议和初稿开题报告。
- 初稿文件扩展名为 `.docx`，但实际为 WPS 保存的旧版 Word 复合文档格式；已使用 `textutil` 抽取正文。
- 已形成修改建议文件：`开题报告修改建议_基于3.2结构框架.md`。

## 2026-06-25

- 阶段：S0 Scope / 标题凝练
- 根据新版章节主线，建议题目避免直接使用“KST ProbFlow”或“缝合方案”，优先突出工业异步多传感器、概率状态预测和风险评估三类核心要素。

## 2026-06-29

- 阶段：S1 Evidence / 文献阅读报告
- 已读取样本文献阅读报告 PDF，确认其写法为“三个专题 + 阅读总结 + 分专题参考文献”。
- 已抽取开题报告现有参考文献，并结合新版论文结构和统一实验方案补充文献缺口。
- 已生成 `文献阅读报告.md`、`文献汇总表.md`、`现有文献核验与增补建议.md`。
- 对 KAFNet、ProFITi、CircuITS 等新近文献使用 arXiv 页面进行核验；对中文无 DOI 条目标记为需 CNKI 复核。

### Capability-use audit

- Required skills: using-research-writing, paper-orchestration, literature-review, nature-academic-search, writing-core, verification
- Skills actually used: using-research-writing, paper-orchestration, literature-review, nature-academic-search, writing-core, verification
- Inputs consumed: 样本文献阅读报告 PDF、开题报告参考文献、新版 3.2 结构框架、统一对比实验方案、补充修改建议
- Inputs not used and why: 未直接访问 CNKI，中文文献中缺少 DOI 的条目需用户后续通过 CNKI 复核
- Artifacts produced: `文献阅读报告.md`, `文献汇总表.md`, `现有文献核验与增补建议.md`, `plan/task-packets/literature_reading_report.md`
- Verification run: PDF 文本抽取、开题报告参考文献抽取、web/arXiv/DOI 链接核验、`rg` 本地占位与编号检查
- Remaining risk: 少数中文文献缺少公开 DOI；Platt 1999 使用公开条目链接，建议后续替换为出版社或原书章节链接

### Capability-use audit

- Required skills: research-writing workflow, writing-core, peer-review, documents read/review
- Skills actually used: using-research-writing, paper-orchestration, writing-core, peer-review, documents
- Inputs consumed: 初稿开题报告、新版 3.2 结构框架、2/3.1/3.3/6 对应修改建议、补充建议、统一对比实验方案、KAFNet-ProFITi 缝合方案
- Inputs not used and why: 未深入读取全部代码文件，本任务目标是文稿修改建议而非代码审计
- Artifacts produced: `开题报告修改建议_基于3.2结构框架.md`
- Verification run: `textutil` 抽取初稿文本；`rg` 定位章节；`sed` 阅读相关段落
- Remaining risk: 未直接修改 Word 文件；若后续需要可继续将建议落实为修订版开题报告

## 2026-07-02

- 阶段：S1 Evidence / 文献综述重写
- 已根据开题报告题目“面向工业设备异步多传感器的规整表示与联合概率预测方法研究”重写 `文献阅读.md`。
- 已将原“三篇文献阅读”结构改为正式综述结构：引言、国内外研究现状、结论、拟研究内容、参考文献。
- 已消化 `plan/evidence-map.md`、`plan/task-packets/literature_review_report_rewrite.md`、`plan/review/evidence-coverage.md`、`plan/chapter-blueprints/literature-review-blueprint.md`、开题报告正文和现有文献汇总材料。

### Capability-use audit

- Required skills: using-research-writing, paper-orchestration, literature-review, verification, documents
- Skills actually used: using-superpowers, using-research-writing, paper-orchestration, literature-review, verification, documents
- Inputs consumed: `24810511012_彭玥_开题报告.doc`, `文献阅读.md`, `文献阅读报告.md`, `文献汇总表.md`, `现有文献核验与增补建议.md`, `plan/evidence-map.md`, `plan/task-packets/literature_review_report_rewrite.md`, `plan/review/evidence-coverage.md`, `plan/chapter-blueprints/literature-review-blueprint.md`
- Inputs not used and why: 未使用不稳定或无法稳定核验的 tPatchGNN 正式论文条目；中文期刊站点访问超时，未将其作为新增核验来源
- Artifacts produced: 重写后的 `文献阅读.md`
- Verification run: 正文中文字符统计、参考文献数量统计、国内/国外比例统计、近五年比例统计、正文引用编号与参考文献编号一致性检查、残留示例文本检查
- Verification result: 正文中文字符 5913；参考文献 65 条；国内 25 条、国外 40 条；2021-2026 年文献 45 条，占 69.23%；正文引用无缺号、无未引用参考文献、无无法对应引用编号
- Remaining risk: 中文文献中部分无公开 DOI 或公开网页元数据不稳定，正式提交前建议使用 CNKI、万方或期刊官网导出题录复核作者、卷期和页码

## 2026-07-02 引用格式润色

- 阶段：S1 Evidence / 引用格式规范化
- 已按用户要求将 `文献阅读.md` 正文中的合并引用格式从 `[1-5]`、`[27,37-45]` 等改为连续独立编号格式，如 `[1][2][3]`。
- 参考文献列表保持原编号顺序排列，未改动文献条目内容。
- Verification run: 正文合并引用残留检查、正文引用编号与参考文献编号一致性检查、未引用参考文献检查。
- Verification result: 正文合并引用块 0 个；参考文献 65 条；正文引用无无法对应编号；无未引用参考文献。

## 2026-07-02 引用语义复核

- 阶段：S1 Evidence / 论断-引用匹配复核
- 已复查 `文献阅读.md` 正文引用与参考文献题名、证据映射之间的对应关系。
- 已修正两类引用风险：将“缺失机制影响预测可信度”处补入 GRU-D 文献 `[36]`；将国内研究 `[1]` 到 `[25]` 的总括引用拆分为智能诊断/健康管理、RUL 与退化建模、工业过程与异常检测三类对应引用。
- Verification run: 正文引用块统计、合并引用残留检查、正文引用编号与参考文献编号一致性检查、未引用参考文献检查、中文字符数与文献比例复核。
- Verification result: 正文合并引用块 0 个；参考文献 65 条；正文引用无无法对应编号；无未引用参考文献；正文中文字符 5947；国外文献 40 条；2021-2026 年文献 45 条。
- Remaining risk: 中文参考文献 `[13]` 到 `[25]` 中部分条目仍需以 CNKI/万方/期刊官网导出信息最终复核；当前复核主要确认其在正文中的主题引用位置不再错配。

## 2026-07-02 顺序引用与叙事重构收尾

- 阶段：S1 Evidence / 顺序引用与研究逻辑强化
- 已按用户要求重排 `文献阅读.md` 正文叙事和参考文献顺序，使正文引用按首次出现严格排列为 `[1]` 到 `[65]`，不再出现 `[1][2][3]`、`[1-5]` 或 `[1,2]` 等堆叠、区间或合并引用形式。
- 已将中英文文献按论证链条交错安排，避免前半部分集中中文文献、后半部分集中英文文献。
- 已强化“工业异步观测条件 → 规整表示 → 联合概率预测 → 风险校准评估 → 统一实验协议”的研究逻辑，明确说明本文不是 KAFNet 与 ProFITi 的机械拼接，而是从观测条件到运维风险的闭环论证。
- Verification run: 正文引用顺序检查、参考文献编号顺序检查、相邻引用块检查、合并引用块检查、正文中文字符数统计、参考文献数量与中外/近五年比例统计。
- Verification result: 正文引用序列严格为 `[1]` 到 `[65]` 且各出现 1 次；参考文献列表严格为 `[1]` 到 `[65]`；相邻堆叠引用 0 个；合并/区间引用 0 个；正文中文字符 5346；参考文献 65 条，其中国内 25 条、国外 40 条，2021-2026 年文献 45 条。
- Remaining risk: 中文文献中部分条目仍建议正式提交前用 CNKI、万方或期刊官网导出题录复核；当前收尾重点已完成正文编号、论证链条和引用位置一致性检查。

## 2026-07-03 审查报告问题修正

- 阶段：S1 Evidence / 引用准确性与综述严谨性修正
- 已读取 `/Users/ppy/Downloads/deep-research-report.md`，按其中对前 11 条文献的核查建议修订 `文献阅读.md`。
- 已修正引言中 `[1]` 的语义边界，使其只支撑液压系统智能化与装备状态建模背景，不再承担 PHM 闭环论证。
- 已按审查报告和 DOI 落页修正 `[2]` 的题名、卷号文号与引用语境；DOI `10.1016/j.ress.2021.108063` 可解析到 Elsevier PII `S0951832021005652`。
- 已保留 `[3]`、`[5]`、`[6]` 当前对应语义：`[3]` 支撑工业智能诊断可解释性，`[5]` 支撑大数据下机械智能故障诊断，`[6]` 支撑从数据采集到 RUL 预测的系统流程。
- 已新增综述透明性说明，明确研究问题、中文/英文文献纳入偏好、证据使用边界，回应审查报告中“检索策略和选文标准不足”“结论与证据匹配度需加强”的问题。
- Verification run: 正文引用顺序检查、参考文献编号顺序检查、相邻引用块检查、合并引用块检查、正文中文字符数统计、参考文献数量与中外/近五年比例统计、`[2]` DOI 解析检查。
- Verification result: 正文引用序列严格为 `[1]` 到 `[65]` 且各出现 1 次；参考文献列表严格为 `[1]` 到 `[65]`；相邻堆叠引用 0 个；合并/区间引用 0 个；正文中文字符 5582；参考文献 65 条，其中国内 25 条、国外 40 条，2021-2026 年文献 45 条。
- Remaining risk: 审查报告只详细列出了前 11 条示例；其余中文条目仍建议正式提交前用 CNKI、万方或期刊官网导出题录逐条复核。

## 2026-07-03 学术规范化修订

- 阶段：S1 Evidence / 参考文献格式与学术风险表达修订
- 已按用户要求保留并修正 `[42]` 为 AAAI 2026 会议论文格式，保留并修正 `[44]` 为 AAAI 2025 会议论文格式，不再使用 arXiv 条目。
- 已按用户指定格式修正 `[34]`，保留 PMLR 卷号和页码 `202: 926-951`。
- 已按用户指定信息修正 `[62]` C-MAPSS 数据集与 `[63]` MetroPT-3 Data in Brief 条目。
- 已统一 ICLR、ICML、NeurIPS、AAAI、IJCAI、KDD 等英文会议论文为 `[C]//会议全称. 年份.` 形式；`[34]` 按用户指定保留 PMLR 细节。
- 已将 KAFNet、ProFITi 相关正文从“具体模型启发/参照”改为“预对齐、时间核聚合、图结构、条件概率建模”等方法谱系表述，降低模型组合感。
- 已替换“直接启发、方法参照、本文吸收、机械拼接、概率预测头”等高风险表达，并强化“工业异步观测、多源传感器、风险概率、统一评估协议”的问题驱动表述。
- Verification run: 章节完整性检查、正文引用顺序检查、参考文献编号顺序检查、相邻/合并引用检查、指定文献条目检查、会议论文格式检查、风险词残留检查、正文中文字符数和参考文献比例统计。
- Verification result: 章节完整；正文引用序列严格为 `[1]` 到 `[65]` 且各出现 1 次；参考文献列表严格为 `[1]` 到 `[65]`；相邻堆叠引用 0 个；合并/区间引用 0 个；高风险表达残留 0 个；正文中文字符 5564；参考文献 65 条，其中国内 25 条、国外 40 条，2021-2026 年文献 45 条。
- Remaining risk: 用户指定 `[42]`、`[44]` 为正式会议论文条目，本次按指定信息修订；中文条目仍建议最终提交前用 CNKI、万方或期刊官网导出题录做终审。

## 2026-07-03 正文引用术语规范化

- 阶段：S1 Evidence / 正文引用表达与中英术语润色
- 已按用户指出的“中英混用”问题继续修订 `文献阅读.md` 正文，保留现有章节结构和 `[1]` 到 `[65]` 的顺序引用。
- 已将方法名和模型名统一为“中文说明 + 英文缩写”的正文表达，如“贝叶斯对抗概率稀疏Transformer模型（BAPT）”“片段化时序Transformer模型（PatchTST）”“神经常微分方程模型（Neural ODE）”“注意力联结函数模型（TACTiS）”等。
- 已进一步弱化“模型串联介绍”语气，将 TACTiS-2、联结函数一致性预测、Platt 校准和数据集名称等表述调整为更自然的中文学术综述表达。
- Verification run: 正文引用顺序检查、参考文献编号顺序检查、相邻/合并引用检查、风险词残留检查、正文中英混写扫描、会议论文格式抽查、正文中文字符数和参考文献比例统计。
- Verification result: 章节完整；正文引用序列严格为 `[1]` 到 `[65]` 且各出现 1 次；参考文献列表严格为 `[1]` 到 `[65]`；相邻堆叠引用 0 个；合并/区间引用 0 个；正文中文字符 5954；参考文献 65 条，其中国内 25 条、国外 40 条，2021-2026 年文献 45 条；风险词扫描仅在参考文献英文题名中保留原文词汇。
- Remaining risk: 正文术语已按中文化表达处理；参考文献题名中的英文专名按原题保留，不应翻译。

## 2026-09-01 CH3-P00-T01 实验协议建立

- 阶段：S3 Experiments / 第三章实验协议锁定准备
- 范围：仅执行 `CH3-P00-T01`，不关闭 D0，不进入 T02。
- 已将方案 A 的数据集角色、确定性划分、窗口、seeds、缺失协议、checkpoint、formal 资格、公平性、防泄漏、统计和可追溯边界写入 `plan/experiment-protocol.md`。
- 已在 `plan/stage-gates.md` 建立 D0-D5 的 checkbox、结构化 registry、依赖和 stale 传播规则；六个 gate 仍均为 `open` 且未勾选。
- 方案计数保持为 198 个 formal matrix rows、165 个唯一 formal training runs 和 18 个独立 profiles，未写入任何伪造结果或 GPU 耗时。

### Capability-use audit

- Required skills: using-superpowers, using-research-writing, paper-orchestration, experiment-results-planning, verification, verification-before-completion
- Skills actually used: using-superpowers, using-research-writing, paper-orchestration, experiment-results-planning, systematic-debugging, receiving-code-review, verification, verification-before-completion
- Inputs consumed: `AGENTS.md`、`plan/implementation-plan.md` 第 11 节、`3.2_结构框架.md`、`docs/毕业论文研究边界与总体方案.md`、`README.md`、`统一对比实验方案.md`、规格与质量审查意见
- Inputs not used and why: 未读写代码、数据集或 `result/` artifact，因为 T01 只锁定文档协议，后续实现与实验属于 P01-P14
- Artifacts produced: `plan/experiment-protocol.md`、`plan/stage-gates.md`、`plan/task-packets/ch3-p00-t01.md`、`plan/implementation-plan.md` 的 T01 进度、本进度审计记录
- Verification run: 指定禁止术语 `rg` 扫描只命中“第三章禁止项”；机器路径/主机/链接扫描零命中；D0-D5 为 6 个未勾选 checkbox 且 registry 为 6 个 `open`；T01 六个子步骤已勾选，P00/T01 总框/T02 在关闭前保持未勾选；`code/`、`compare_code/`、`result/` 无本任务变更
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`
- Remaining risk: D0 必须等待 T02、T03 与 T04 的追溯表、表图数据合同和统一 validator 全部通过后才能关闭；协议中的规则尚需后续 Phase 转换为配置、代码和测试。

## 2026-09-04 CH3-P00-T02 方法—实验追溯表建立

- 阶段：S3 Experiments / 第三章证据链预注册
- 范围：仅完成 `CH3-P00-T02`；建立七项异步观测规整机制的“方法—实验—表图—允许结论”追溯关系，不进入 `CH3-P00-T03`，不关闭 `CH3-P00`、D0、D1 或 D2。
- 已建立 `plan/review/method-experiment-traceability.md`：真实时间、mask、`delta_t`、通道身份、工况、多尺度及跨传感器交互均映射到预注册模块、E0/E1/E4/E5-A/E5-S/E6、第三章 3.6.2-3.6.6 与 Ch3-T/Ch3-F 产物。
- 证据边界已冻结：现有 `kst_probflow` 仅作机制参照；第三章正式结论只能消费后续 KST-Light、point-only runner 与 formal artifact。E1-X 仅作为外部范围证据，E6 仅用于效率结论；跨传感器交互不外推为联合概率或风险能力。
- 每项机制都规定了 formal artifact 缺失、接口审计失败、开关不可验证或结果无明显改善时的结论降级规则；当前全部为“预注册、无 formal artifact”。

### Capability-use audit

- Required skills: using-superpowers, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, subagent-driven-development, peer-review, verification, verification-before-completion
- Skills actually used: using-superpowers, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, subagent-driven-development, peer-review, verification, verification-before-completion
- Inputs consumed: `AGENTS.md`、`plan/implementation-plan.md` 第 11 节、`plan/experiment-protocol.md`、`plan/stage-gates.md`、`3.2_结构框架.md`、`docs/毕业论文研究边界与总体方案.md`、现有编码器和 `kst_probflow` 的机制边界、T02 任务包。
- Inputs not used and why: 未修改或消费代码、数据集、`result/`、checkpoint 或 metrics；本任务只建立预注册的文档追溯关系，正式实现、实验运行和结果聚合属于后续 Phase。
- Artifacts produced: `plan/task-packets/ch3-p00-t02.md`、`plan/review/method-experiment-traceability.md`、`plan/implementation-plan.md` 的 T02 状态、本进度审计记录。
- Verification run: 必需字段和七项机制扫描、E0/E1/E4/E5/E6 至 3.6.2-3.6.6 路由扫描、禁止指标/风险校准术语扫描、机器绝对路径/主机/链接扫描、T02/P00/T03/D1 状态核验、Markdown 代码块平衡性检查；`code/`、`compare_code/`、`result/` 未发现本任务改动。
- Review result: 规格符合性复审 `PASS`；独立质量复审 `APPROVED`。
- Remaining risk: D1 仍须等待 T03 的表图数据合同和 T04 的统一 validator 后才能关闭；KST-Light、point-only runner、formal artifact 与所有预注册实验尚未实现或运行，当前不得写入任何性能、概率或风险结果。

## 2026-09-04 CH3-P00-T03 表格与图片数据合同建立

- 阶段：S3 Experiments / 第三章输出数据合同预注册
- 范围：仅完成 `CH3-P00-T03`；锁定 Ch3-T1..T8 与 Ch3-F1..F5 的字段、来源、聚合、排序、允许结论和交付格式，不进入 T04，不关闭 `CH3-P00`、D0、D1 或 D2。
- 已完善 `tables/table-schema.md`：八张表均明确主键、指标、统一单位、聚合、artifact 来源、稳定输出 CSV、固定排序与允许结论；纠正了敏感性表不得消费 profile latency、外部验证表不得包含效率字段的边界。
- 已创建 `figures/data-manifest.md`：五张图各有唯一聚合 source CSV、必需列、坐标/单位、证据门槛、允许结论、SVG 与至少 300 dpi PNG 输出要求。F4 是唯一可消费 profile 的图，且必须由 T7 双 provenance 聚合生成。
- 未生成 CSV、表格、图片或实验数值；所有表图仍只处于预注册合同状态。

### Capability-use audit

- Required skills: using-superpowers, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, subagent-driven-development, peer-review, verification, verification-before-completion
- Skills actually used: using-superpowers, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, peer-review, verification, verification-before-completion
- Inputs consumed: `AGENTS.md`、`plan/implementation-plan.md` 第 11 节与 CH3-P13 输出计划、`plan/experiment-protocol.md`、`plan/stage-gates.md`、`plan/review/method-experiment-traceability.md`、已有 `tables/table-schema.md`。
- Inputs not used and why: 未使用 `subagent-driven-development` 分派代理，因为当前协作约束未授权为本任务新建子代理；未读写代码、数据集、`result/`、checkpoint、metrics 或 raw prediction，因为它们属于后续实现、formal 运行和 P13 聚合阶段。
- Artifacts produced: `plan/task-packets/ch3-p00-t03.md`、`tables/table-schema.md`、`figures/data-manifest.md`、`plan/implementation-plan.md` 的 T03 状态、本进度审计记录。
- Verification run: 8/8 表格段具备主键/指标/聚合/artifact 来源/允许结论；5/5 图形段具备唯一 source CSV/必需列/坐标/允许结论/证据门槛；5 个图 source CSV 和 8 个表输出 CSV 均唯一；profile 白名单仅为 Ch3-T7/Ch3-F4；禁止指标及机器路径扫描零命中；Markdown 代码块平衡；`code/`、`compare_code/`、`result/` 无本任务变更。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: D2 仍须等待 T04 的跨文档 validator 和 gate evidence 才能关闭。P13 尚未实现 strict catalog、聚合、source CSV、图形渲染、双重 provenance 和视觉 QA；没有 formal artifact 前不得把本合同转换为论文结果。

## 2026-09-04 CH3-P00-T04 协议自审与 D0-D2 关闭

- 阶段：S3 Experiments / 第三章规划文档门禁关闭
- 范围：仅关闭 D0-D2 规划门禁和 CH3-P00；未实施模型、数据协议代码、runner、formal run、结果聚合或图表渲染，未推进 CH3-P01。
- 已创建 `plan/scripts/validate_ch3_p00.sh` 与 `plan/tests/test_validate_ch3_p00.sh`。验证器从自身位置推导仓库根目录，检查 T01-T03 前置状态、三个数据集、六个主对比模型加受控消融变体族、三 seeds、五缺失率、四机制、七项追溯机制、8 表、5 图、占位词、禁止指标和机器路径，并输出排序后的 input SHA set。
- 已创建 `plan/review/ch3-p00-t04-gate-validation.md`，记录命令、退出码、D0/D1/D2 的 canonical input SHA sets、证据范围与门禁关闭边界；该 evidence SHA 已写入 `plan/stage-gates.md` 的三条 state-transition record。
- D0、D1、D2 的 registry 和 checkbox 已同步为 `closed`；CH3-P00、T04 及四个 T04 子任务已同步勾选。D3-D5 和 CH3-P01 仍未修改。

### Capability-use audit

- Required skills: using-superpowers, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, peer-review, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, peer-review, test-driven-development, systematic-debugging, verification, verification-before-completion
- Inputs consumed: `AGENTS.md`、CH3-P00 的 T01-T03 任务包与产物、`plan/experiment-protocol.md`、`plan/review/method-experiment-traceability.md`、`tables/table-schema.md`、`figures/data-manifest.md`、`plan/stage-gates.md`。
- Inputs not used and why: 未读写模型代码、数据集、`result/`、checkpoint、prediction 或 metrics；D0-D2 只确认规划文档的可执行边界，正式证据仍属于 P03-P14。
- Artifacts produced: `plan/task-packets/ch3-p00-t04.md`、`plan/scripts/validate_ch3_p00.sh`、`plan/tests/test_validate_ch3_p00.sh`、`plan/review/ch3-p00-t04-gate-validation.md`、D0-D2 的 gate registry/evidence records、T04/P00 状态和本审计记录。
- Verification run: validator 缺失时测试先失败；随后 `bash plan/tests/test_validate_ch3_p00.sh` 通过。定位并修复 `C.UTF-8` locale 使 Perl/PCRE 输出 warning 的问题，测试显式注入该 locale 后仍通过。统一 validator 输出 `validated=CH3-P00`、8 table sections、5 figure sections 与四份输入 SHA；YAML registry 以 UTF-8 解析后确认 D0-D2=closed、D3-D5=open；evidence SHA 与 record 一致；占位词、禁止指标和机器路径扫描零命中；`code/`、`compare_code/`、`result/` 无本任务变更。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: D0-D2 只表示范围、追溯和输出合同已锁定。KST-Light、数据协议、point-only runner、strict catalogs、formal runs、统计、source CSV、SVG/PNG、D3-D5 和论文结果均未完成；任何列入 input SHA set 的文档改动都必须重新运行 validator 并按 stage-gate stale 规则处理。

## 2026-09-05 CH3-P01-T01 统一运行时路径解析

- 阶段：S3 Experiments / 第三章可移植运行基础
- 范围：仅完成 `CH3-P01-T01`。新增标准库路径解析 API 和其单元测试；未迁移训练入口、未建立 YAML schema、未读取数据集或修改 `result/`。
- 已创建 `code/kaf_profiti/experiments/runtime_paths.py`，提供不可变 `RuntimePaths` 和 `resolve_runtime_paths`。显式路径优先于 `KST_DATA_ROOT` / `KST_RESULT_ROOT`，环境变量优先于项目根下的 `dataset` / `result` 默认路径；`KST_PROJECT_ROOT` 可覆盖模块位置推导的项目根。相对路径以项目根规范化，不依赖调用时工作目录。
- formal 模式只检查数据根是否存在且为目录，不创建输出目录；缺失时抛出含已解析目录的 `FileNotFoundError`。

### Capability-use audit

- Required skills: using-superpowers, using-research-writing, paper-orchestration, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, using-research-writing, paper-orchestration, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `AGENTS.md`、`plan/implementation-plan.md` 第 11 节、`plan/experiment-protocol.md`、`plan/stage-gates.md`、现有训练入口、数据集协议模块和实验框架测试。
- Inputs not used and why: 未修改训练入口、模型、数据集、YAML、checkpoint、prediction、metrics 或 `result/`，它们属于后续 `CH3-P01-T02` 及更后续 Phase。
- Artifacts produced: `plan/task-packets/ch3-p01-t01.md`、`code/kaf_profiti/experiments/runtime_paths.py`、`code/tests/ch3/test_runtime_paths.py`、本任务状态和本审计记录。
- Verification run: 先行测试在目标模块不存在时失败；实现后路径优先级、规范化、formal 缺失目录拒绝及存在目录接受共 5 项测试通过，语法编译通过，机器路径/远程地址扫描零命中。当前本机默认 Python 缺少 `pytest` 和 PyTorch，因此以隔离测试依赖运行路径单元测试；既有实验框架测试未执行，待具备项目完整依赖的训练环境复跑。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 现有训练入口仍保留其旧默认路径，必须在 `CH3-P01-T02` 及后续 point-only runner 接入本 API 后才可获得第三章正式运行资格；CH3-P01 Phase 和 T02 未关闭。

## 2026-09-06 CH3-P01-T02 第三章配置 schema

- 阶段：S3 Experiments / 第三章可移植运行基础
- 范围：仅完成 `CH3-P01-T02`。新增 schema、三份基础 YAML 和 PyYAML 依赖；未迁移训练入口、未创建 loader、未读取数据集，也未写入 `result/`。
- 已创建 `Ch3ExperimentConfig` 与 `from_yaml()`：YAML 由 `yaml.safe_load` 读取，schema 提供覆盖合并、序列化、值校验与 `allows_test_loader` 信号。仅支持预注册数据集、核心 seeds、四种 run level、四种缺失机制和 linear/MLP point head。
- 三份 tracked YAML 分别锁定 MetroPT-3、FD004、TEP 的正式窗口、50/80/50 epoch 预算、128 batch、64 hidden dimension、mixed 30% 缺失、fixed split seed 与无 batch 截断。

### Capability-use audit

- Required skills: using-superpowers, brainstorming, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, test-driven-development, requesting-code-review, verification, verification-before-completion
- Skills actually used: using-superpowers, brainstorming, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, test-driven-development, requesting-code-review, verification, verification-before-completion
- Inputs consumed: `AGENTS.md`、`plan/implementation-plan.md` 的 T02 合同、`plan/experiment-protocol.md`、`plan/stage-gates.md`、T01 运行时路径 API、现有依赖与测试结构。
- Inputs not used and why: 未修改训练入口、模型、数据集、mask、checkpoint、prediction、metrics 或 `result/`；这些属于 P01-T03、P02 及后续 runner/实验 Phase。
- Artifacts produced: `code/kaf_profiti/experiments/ch3/__init__.py`、`code/kaf_profiti/experiments/ch3/config.py`、`code/tests/ch3/test_ch3_config.py`、三份 `configs/ch3/*.yaml`、`plan/task-packets/ch3-p01-t02.md`、`requirement.txt` 的 PyYAML 版本约束、任务状态和本审计记录。
- Verification run: 先行测试在 schema 模块缺失时失败；随后配置测试 16 项通过，覆盖非法值和类型、formal 短训练/截断/split 拒绝、三 YAML 载入、覆写合并和 tuning test-loader 禁止信号。与 T01 合并测试 21 项通过，语法编译、safe-load 与可移植性扫描通过。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: schema 只定义与校验科学配置，尚未接入 CLI，也不能独自阻止未来 runner 构造 tuning test loader；该运行期限制将由后续 run-level guard 实现。CH3-P01 Phase 和 T03 未关闭。

## 2026-09-06 CH3-P01-T03 可移植性与内部链接质量门禁

- 阶段：S3 Experiments / 第三章可移植运行基础
- 范围：仅完成 `CH3-P01-T03`。新增可复跑的 portability policy 测试和精确历史 allowlist；未修改历史训练器、基线实现、数据集、模型、artifact 或 `result/`。
- 已创建 `code/tests/ch3/test_portability_policy.py`，扫描 `code/`、`compare_code/`、`configs/ch3/` 与受控第三章文档。受控文档检查机器/远程引用和 Markdown 绝对内部链接；内联命令与 fenced code 的 regex 字面量不作为运行时硬编码处理。
- 35 个历史命中以“相对文件、行号、完整内容”精确登记：13 个旧主工程默认路径，清除期限 `CH3-P05`；22 个对比模型 README/SOURCES 的固定环境路径和来源 URL，清除期限 `CH3-P06`。allowlist 缺理由/Phase、条目陈旧、位置或内容漂移、allowlist 外新增命中均会失败；`configs/ch3/` 和 CH3 新代码/测试路径不得进入 allowlist。

### Capability-use audit

- Required skills: using-superpowers, brainstorming, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, brainstorming, using-research-writing, paper-orchestration, experiment-results-planning, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `AGENTS.md`、`plan/implementation-plan.md` 的 T03 合同、T01/T02 产物、`plan/experiment-protocol.md`、`plan/stage-gates.md`、第三章受控 Markdown 与全工程 portability 扫描结果。
- Inputs not used and why: 未修改 runner、模型、基线实现、数据集、YAML 科学参数、checkpoint、prediction、metrics 或 `result/`；T03 仅建立检测与遗留项身份边界。
- Artifacts produced: `code/tests/ch3/test_portability_policy.py`、`plan/task-packets/ch3-p01-t03.md`、任务状态和本审计记录。
- Verification run: 初始扫描在排除 pytest 生成缓存后报告 35 个未登记历史命中；登记精确 allowlist 后策略测试 2 项通过。该测试覆盖新增/陈旧 allowlist、理由与 Phase、第三章文档相对链接和新增 CH3 文件不得进入 allowlist。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 历史默认路径将在 `CH3-P05` 的独立 runner 切换后清除；基线 README/SOURCES 的固定环境路径与来源 URL 在 `CH3-P06` 适配时清除。仅当这两类遗留项归零时，CH3-P01 Phase 才可关闭。

## 2026-09-12 CH2.5-P01-T01 分离 split seed 与 run seed

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P01-T01`。修改数据协议入口的 engine/run 划分随机性来源并新增 FD004 split 泄漏测试；未修改 mask、归一化、训练入口、模型或 `result/`。
- `create_protocol_datasets` 新增 `split_seed: int = 2026` 参数并补全 docstring；C-MAPSS engine permutation 与 TEP simulation-run permutation 改为只用 `split_seed`，run `seed` 不再影响任何划分。默认值 2026 与注册协议一致，既有调用方（run_experiment、evaluate_risk_calibration、compare_code 训练器）签名兼容、行为仅表现为 split 固定。
- `split_info` 新增 `split_seed`、`window_bounds`（每 split 的 count/first/last/digest）、`split_identity`（canonical payload）和 `split_sha256`；identity 只含数据协议字段（dataset、split_rule、split_seed、窗口参数、排序 engine ID、test engine 数、窗口边界），不含 run seed 或模型名，可由 artifact 独立重算 SHA。
- 新增 `split_identity_sha256()` 公共函数：sort_keys + 固定分隔符的 canonical JSON SHA-256。
- MetroPT 协议为时序划分、不含随机性，本任务未改动其路径。

### Capability-use audit

- Required skills: using-superpowers, experiment-results-planning, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, experiment-results-planning, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `AGENTS.md`、`plan/implementation-plan.md` 第 12 节 T01 合同、`code/kaf_profiti/experiments/datasets.py`、`code/kaf_profiti/industrial/cmapss.py`、`code/tests/test_experiment_framework.py` 既有调用模式、compare_code 两个训练工程的 `create_protocol_datasets` 调用点。
- Inputs not used and why: 未修改 mask 生成（T03）、归一化（T02）、指标累加（T05）、训练入口、模型或 `result/`；这些属于本 Phase 后续 Task。
- Artifacts produced: `code/tests/pilot/test_fd004_split_leakage.py`（6 项测试）、`datasets.py` 的 split_seed/窗口边界/SHA 变更、T01 状态勾选和本审计记录。
- Verification run: 先行测试因 `split_identity_sha256` 缺失而失败（red）；实现后 leakage 测试 6 passed，全量 `code/tests/` 105 passed（含 TEP 数据测试与 portability 门禁）；compare_code 风格调用（仅 `seed=` 关键字）在 FD001/MetroPT 上等价复跑通过，且 run seed 2026→2027 时 split SHA 不变；`py_compile` 通过。测试环境为 conda `kaf_profiti`（/opt/anaconda3/envs/kaf_profiti），补装 requirement.txt 钉定的 `PyYAML==6.0.2`，`KST_DATA_ROOT` 指向仓库 `dataset/`。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: run_experiment.py 的 split JSON 文件名仍使用 run seed（`{dataset}_split_seed{config.seed}.json`），其改造属于后续统一 runner（CH3-P05/CH2.5-P03-T01）；FD004 的 mask 与归一化公平性仍由 T02/T03 关闭。compare_code 集成测试的 `/home/work` 硬编码数据根为既有 Phase 1 遗留项，未在本任务处理。

## 2026-09-12 CH2.5-P01-T02 冻结 train-only normalization

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P01-T02`。在 C-MAPSS 协议构建处冻结 train-only 归一化统计量并新增泄漏测试；未修改 mask 生成、指标累加、训练入口、模型或 `result/`。
- `datasets.py` 新增 `_cmapss_stats_artifact(train_frame, train_engine_ids)`：mean/std 只消费 train engine 行；std 下限 1e-6；遇非有限值抛 `ValueError`；artifact 记录 source split、engine ID、列序、count、mean/std 与 SHA（对剔除 sha 字段后的 canonical payload 计算）。
- `_stats_tensors_from_artifact` 把 artifact 转为张量 stats，train/valid/test 三个 dataset 构造器统一经 `stats=` 接收同一冻结统计量；`CMapssWindowDataset.__init__` 新增末位参数 `stats`，提供时直接使用、不再重新估计，否则保留 legacy `_training_stats` 回退路径。
- `split_info` 新增 normalization 块，`split_identity` 新增 `normalization_sha256`，归一化进入 split SHA 指纹。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` 第 12 节 T02 合同、`datasets.py` 的 `_create_cmapss` 流程、`cmapss.py` 构造器签名、T01 已建立的 split_info/SHA 结构。
- Inputs not used and why: 未修改 mask（T03/T04）、指标累加（T05）、TEP/MetroPT 路径（FD004 预实验只用 C-MAPSS）与 `result/`。
- Artifacts produced: `code/tests/pilot/test_fd004_normalization_leakage.py`（5 项测试）、`datasets.py`/`cmapss.py` 的冻结统计量变更、T02 状态勾选和本审计记录。
- Verification run: 先行测试因 `_cmapss_stats_artifact` 不存在失败（red）；实现后 normalization leakage 测试 5 passed（扰动 valid 行统计量逐字节不变、三 split 共享冻结 stats、artifact 字段与 SHA 可重算、构造器内 monkeypatch `_training_stats` 抛错证明不重估、常量传感器与非有限值拒绝），全量 `code/tests/` 110 passed。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 冻结 artifact 目前仅覆盖 C-MAPSS 路径；TEP/MetroPT 若进入后续实验需按同一模式处理。runner 侧尚未把 normalization artifact 作为独立 manifest 条目引用（属于后续统一 runner 接线）。

## 2026-09-12 CH2.5-P01-T03 timeline-first 共享 mask

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P01-T03`。新增 timeline 级共享 mask 基础设施与测试；legacy per-window mask API 与 `MissingMechanismSimulator` 未动；未修改归一化、指标累加或 `result/`。
- `masks.py` 新增 `TIMELINE_MASK_SCHEMA_VERSION=2`、`TimelineMaskConfig`（dataset/split/mechanism/requested_rate/mask_seed/source_split_sha256，`__post_init__` 归一机制名）、`TimelineMaskBundle`、`generate_or_load_timeline_masks`（临时文件 `.tmp.npz` 原子写后 replace）、`_validate_timeline_bundle`（identity、engine 集合、shape、内容 SHA、realized-rate 复核，不匹配立即失败而非静默复用）与 `TimelineMaskedWindowDataset`（按 `dataset.windows` 的 canonical (unit,start) 窗口索引切片，重叠窗口共享同一观测状态）。
- 每 engine 的 mask RNG seed 由 sha256("schema|dataset|split|mechanism|mask_seed|engine") 派生，split 与 engine 边界不串联。
- `missing.py` 新增 timeline 级机制函数（random/low_rate/block_offline/mixed），供 timeline 生成调用；签名不含任何标签输入。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` 第 12 节 T03 合同、`masks.py` 既有 per-window API 与 NPZ 布局、`missing.py` 机制语义、T01 的 split SHA（作为 source_split_sha256）、`cmapss.py` 的窗口 (unit,start) 结构。
- Inputs not used and why: 未做率精确匹配校准（T04）、未改归一化（T02 已完成）与指标累加（T05）。
- Artifacts produced: `code/tests/pilot/test_fd004_timeline_masks.py`（6 项测试）、`missing.py`/`masks.py` 的 timeline 变更、T03 状态勾选和本审计记录。
- Verification run: 先行测试因 timeline API 缺失失败（red）；实现后 timeline mask 测试 6 passed（重叠窗口观测一致、identity/内容 SHA 可重算、重复生成内容确定、相同 identity 跨调用共享同一 SHA、窗口切片与 timeline 对齐），全量 `code/tests/` 116 passed，旧 per-window API 无回归。期间修复 npz 原子写路径（临时名 `.tmp` 触发 numpy 自动补 `.npz` 导致 replace FileNotFoundError）与测试文件缺 `import torch`。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: timeline mask 的共享存放路径目前由调用方传入；各训练入口实际切换到 timeline mask 属于后续 runner 接线（CH2.5-P02 / CH3-P05）。率的精确匹配由 T04 关闭。

## 2026-09-12 CH2.5-P01-T04 匹配四种机制的实际 30% 缺失率

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P01-T04`。为四种机制实现 deterministic rate calibration 与 canonical 机制 ID；未修改 legacy 机制模拟器、归一化、指标累加或 `result/`。
- `missing.py` 新增 `_MECHANISM_ALIASES` 与 `canonical_mechanism()`（未知机制抛 `ValueError`；展示名 `block` 映射 canonical ID `block_offline`，一机制一 scientific key）。
- `masks.py` 的 `_generate_timeline_masks` 增加斜率二分校准：每机制解析初值（random/low_rate 取 keep=1-rate；block_offline 取 block_prob=rate/0.4 且 max_block_fraction=0.8；mixed 取 keep=(1-rate)/0.72），迭代上限 `_CALIBRATION_MAX_ITERATIONS=14`，未收敛抛 `ValueError`；收敛判据 `DEFAULT_RATE_TOLERANCE=0.01`，使 requested rate 即最终 split 级 realized missing rate。
- 安全约束 `_ensure_observed` 保证每通道至少 1 个历史观测；artifact 记录 `pre_safety_realized_rate` 与 `safety_adjustment`，约束造成的偏差可追溯。
- 测试覆盖无标签依赖：`generate_or_load_timeline_masks` 签名参数仅为 path/config/engine_timeline_lengths/num_sensors/tolerance，不含 RUL、未来目标、test metric 或 fault label 输入。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` 第 12 节 T04 合同、T03 的 timeline 生成管线与 bundle 元数据结构、`missing.py` timeline 机制函数语义。
- Inputs not used and why: 未修改指标累加（T05）、训练入口与 `result/`；legacy per-window 校准路径不在 FD004 预实验协议内。
- Artifacts produced: `code/tests/pilot/test_matched_missing_rate.py`（6 项测试）、`missing.py` 别名/canonical 化与 `masks.py` 二分校准变更、T04 状态勾选和本审计记录。
- Verification run: 先行测试红；实现后 matched-rate 测试 6 passed（四机制 realized∈[0.29,0.31]、校准元数据确定性、block 别名同内容 SHA、前 25 engine 每通道≥1 观测、签名无标签输入、不可达容差 1e-12 抛错），全量 `code/tests/` 122 passed。期间删除误留的草稿 `_mechanism_builder`，并对测试的校准元数据字段（`knob`/`value`）与签名参数集做对齐。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 斜率二分假定 realized rate 随 knob 单调；极端 requested rate 或极短 timeline 可能不收敛并显式抛错（不静默）。safety 约束偏差已记录于 artifact，机制对比解释时需引用。

## 2026-09-13 CH2.5-P01-T05 修正全局指标累加

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P01-T05`。新建全局指标累加器并改造三个 evaluator 的跨 batch 聚合；未修改数据协议、mask、归一化、训练逻辑或 `result/`。
- 新建 `code/kaf_profiti/experiments/accumulators.py` 的 `GlobalMetricAccumulator`：abs/squared-error sum 与统一 valid count；`update_nll`/`update_crps` 以 (sum,count) 累加；interval 通道按 (covered_count,width_sum,count) 累加，`update_interval_means` 用 batch micro-mean×真实分母恢复精确和；`result()` 输出 MAE、RMSE（全局平方和÷有效数再开方，禁止平均 batch RMSE）、NLL、CRPS、PICP、MPIW；`point_only_result()` 强制概率指标 `null` 并标注 `not_applicable`（若已累加概率和则抛错）。
- `update_point` 用 `torch.where` 而非乘法掩码（nan×0 仍为 nan，会污染求和）——该缺陷由先行测试捕获。
- 三个 evaluator 改造（batch 级 metric 函数本身是 micro-mean，保留不动，只替换"跨 batch 等权平均"为全局和恢复）：`run_experiment.py` NLL 用 per-row `nll_rows×row_counts` 恢复行加权和；`TCN-Gaussian` 的 nll/crps 乘 mask_count、interval 乘 finite-valid count；`probabilistic_baselines` 的 nll 乘 mask_count、crps/interval 乘 finite-valid count（各自匹配其内部分母）。run_experiment 无 quantile 头时 `quantile_picp/mpiw` 置 None。
- 测试含 3 个集成测试：三个 evaluator 以不同 batch 划分（[6] vs [2,4] / [1,2,3] / [3,3]）评估同一 synthetic prediction（行级种子生成保证跨划分逐字节相同），六项指标全部一致。
- 顺带维护：portability allowlist 中 `run_experiment.py` 两处 `parse_args` 遗留项行号 898/899→881/882（T05 编辑使文件缩短导致行号漂移；同一两处既有违规，无新增项，门禁恢复绿）。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` 第 12 节 T05 合同、三个 evaluator 的 `_evaluate` 实现与各 batch 级 metric 函数的内部分母（`model.nll`÷mask.sum、crps÷mask.sum 或 finite-valid、interval÷finite-valid）、`IndustrialBatch` 字段。
- Inputs not used and why: 未修改数据协议/mask/归一化（T01–T04 已完成）、模型本体与 `result/`；compare_code 旧集成测试因 `/home/work/new_work/dataset` 硬编码本机不可运行，未复跑（Phase 1 遗留项）。
- Artifacts produced: `code/kaf_profiti/experiments/accumulators.py`、三个 evaluator 的累加改造、`code/tests/pilot/test_metric_accumulators.py`（4 单元 + 3 集成）、portability allowlist 行号维护、T05 与 Phase CH2.5-P01 状态勾选、T02–T05 progress 补记与本审计记录。
- Verification run: 先行测试红（update_point nan 污染与 stub 接口缺失均被捕获）；实现与修复后 accumulator 测试 7 passed，全量 `code/tests/` 129 passed、2 个既有 scipy DeprecationWarning；portability 门禁在行号维护后恢复通过。run_experiment 旧 evaluator 回归（`test_experiment_framework.py`）包含于全量套件；TCN/baselines 旧回归测试本机数据路径受阻，由本文件两个集成测试替代覆盖。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: compare_code 旧集成测试仍需在有 `/home/work/new_work/dataset` 的环境复跑一次以闭合旧回归链；portability allowlist 按行号索引，`run_experiment.py` 再次变更时需同步维护；`infer_time_ms_per_batch` 等计时字段天然与 batch 划分相关，不属于不变性目标。


## 2026-09-13 CH2.5-P02-T01 统一 point/probabilistic 接口

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P02-T01`。新建统一模型合同模块、轻量头模块与公共接口测试；未修改既有 evaluator、数据协议或 `result/`。
- 新建 `code/kaf_profiti/experiments/model_api.py`：`UnifiedPointModel`（`predict_point→[B,P*N]` 与 `y_flat` 同展平序、`loss`=masked MSE、`parameter_count`）；`UnifiedGaussianModel`（`gaussian_kind=diagonal|flow`、`gaussian_params→(mean,scale)[B,P,N]`、`batch_nll` 分母 `mask.sum()`、`sample_flat→[B,S,P*N]` 按 `mq_flat` 掩零、`interval95_flat=mean±Z95·scale`、`loss=NLL+λ·MSE`）；检查器 `check_point_interface`/`check_gaussian_interface`（形状、有限值、history-only 行为验证、掩码不变 NLL、样本矩一致性 8σ/√S、梯度、state_dict 扰动-重载-复放）。
- history-only 规则按行为验证：对 `Y_q/y_flat/M_q/mq_flat/rul` 注入随机值，`predict_point`/`gaussian_params` 输出必须不变；`loss`/`batch_nll` 合法消费目标，改用掩码不变性检查（仅扰动 mask=0 处目标）。
- 新建 `code/kaf_profiti/models/lightweight_head.py`：`LinearPointHead`/`MLPPointHead`（逐位置头，仅收编码器历史表示）、`DiagonalGaussianHead`（scale=softplus(raw)+min_scale，NLL 分母 mask.sum，采样 [B,S,D]，95% 区间 mean±Z95·scale）、`KSTLight`（MultiScaleKAFEncoder + QueryConditionAdapter + 轻量头；类属性 IMPLEMENTATION="own"/REQUIRES_TIME_INPUT/ADAPTER）。
- 负向用例证明检查器可捕获违规：读未来的 point 模型被 history-only 检查捕获、断梯模型被梯度检查捕获、按位置均值 NLL 被掩码不变性检查捕获。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` 第 12 节 T01 合同、`IndustrialBatch` 字段与展平序、`MultiScaleKAFEncoder`/`QueryConditionAdapter` 既有接口、`KSTProbFlow` 分布接口（决定 flow/diagonal 双分支设计）。
- Inputs not used and why: 未接入具体基线（T02/T03）、未改训练入口（P03 runner）；compare_code 概率基线仅作接口参考未修改。
- Artifacts produced: `model_api.py`、`lightweight_head.py`、`code/tests/pilot/test_model_api.py`（12 项）、T01 状态勾选与本审计记录。
- Verification run: 先行测试因模块缺失失败（red）；实现迭代修复 9 处（head 维度、repeat vs expand、history-only 检查范围限定预测方法、-0.0 掩零、nsamples 透传等，均由测试驱动）；最终 `test_model_api.py` 12 passed，全量 `code/tests/` 141 passed。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: `sample_flat` 的 history-only 检查豁免依赖"按 mq_flat 掩零"合同（合同内自洽）；flow 分支（ProFITi/KST ProbFlow）的区间来自样本，矩一致性容差是否适用于多峰流分布将在 T03 复核。

## 2026-09-13 CH2.5-P02-T02 补齐第三章点预测基线与 fidelity 门禁

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P02-T02`。新建 baselines 包与五个点预测基线、fidelity 测试；扩展 registry 身份字段；未修改数据协议、mask、evaluator 或 `result/`。
- 新建 `code/kaf_profiti/baselines/point.py`：`linear_interpolate_fill`（历史内真实时间线性插值，首段/全空通道回退 train-only fill_value，尾部保持最后观测）、`forward_fill`（首段回退 fill_value）、`compute_delta_t`（GRU-D 递归 delta_l=dt+delta_{l-1}·(1−m_{l-1})，观测位置归零）、自包含因果空洞 TCN backbone（适配自 TCN-Gaussian 参考、删除概率部分）；`LITCNPoint`/`FFGRUPoint`/`MaskedTCNPoint`/`GRUDPoint`/`ODERNNPoint` 均继承 `UnifiedPointModel`，池化 [B,H] 后接统一 `LinearPointHead(H,P·N)`。
- GRU-D 输入衰减 `exp(−softplus(rate)·delta_t)` 逐传感器可学习，衰减目标为 train-only fill_value；ODE-RNN 隐状态按真实 T_obs 间隔 Euler 积分（学习 tanh 向量场，子步数随最大间隔缩放、总推进量精确等于 dt），逐时刻 GRUCell 观测更新，不读 T_q；两类简化均在 SOURCE_IDENTITY 标注 `adapted`（无隐状态衰减 / Euler 代替 adjoint solver）。
- registry `ModelSpec` 追加默认空字段 `implementation/source_identity/requires_time_input/adapter`（纯文本引用、无 URL，向后兼容既有位置构造）；li_tcn/ff_gru/masked_tcn/gru_d/ode_rnn/kst_light 置 `pilot_ready`；专门测试锁定 registry 身份与模型类属性逐字段一致（该测试即时捕获一次手工复制漂移）。
- 测试覆盖：矩阵 model_id ↔ registry 全对齐、五基线统一接口 + 单步 AdamW 参数变化、插值/前填手工规则对照（时间而非位置）、全空通道 train-only fill、Masked TCN 对被掩历史值不变、GRU-D delta_t 递归逐值对照 + 衰减梯度 + 时间敏感性、ODE-RNN 时间拉伸敏感性、全部模型 T_q 扰动不变。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` 第 12 节 T02 合同与 12.2.3 点预测矩阵、T01 的 `UnifiedPointModel`/`LinearPointHead`/接口检查器、TCN-Gaussian 参考实现的因果卷积结构、GRU-D/ODE-RNN/TCN 原始论文机制描述（Che et al. 2018；Rubanova et al. 2019；Bai et al. 2018）。
- Inputs not used and why: 概率基线与 compare_code 修改属 T03；runner 接线属 P03；未使用 torchdiffeq（FD004 预实验采用 Euler 并显式标注 adapted，正式 CH3-P06-T05 再换 solver）。
- Artifacts produced: `code/kaf_profiti/baselines/__init__.py`、`baselines/point.py`、`code/tests/pilot/test_point_baseline_fidelity.py`（23 项）、`registry.py` 身份扩展与 pilot_ready 登记、T02 状态勾选与本审计记录。
- Verification run: 先行测试因 `kaf_profiti.baselines` 缺失失败（red）；实现后修复 3 处（ODE-RNN 子步广播 [B]→[B,1]、分数间隔<0.5 被跳过的子步下限、一处测试期望值笔误 l=1 内插值应为 6.0）与 registry/class 身份字符串漂移 1 处；最终 fidelity 测试 23 passed，全量 `code/tests/` 164 passed、portability 无新增命中。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: ODE-RNN Euler 积分的子步数由 batch 内最大间隔决定（精度与批量构成相关但总推进量精确）；GRU-D 无隐状态衰减、单衰减率/传感器的简化已登记 `adapted`，进入论文表时需保留该标注；registry 身份为双份存储，由一致性测试防漂移。

## 2026-09-13 CH2.5-P02-T03 补齐第四章六个概率基线并统一 distribution API

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P02-T03`。补齐 GRU-D Gaussian 与 GraFITi Gaussian、核验并接入 TCN-Gaussian/PatchTST-Gaussian/ODE-RNN-Gaussian/ProFITi，KST ProbFlow 套上统一 flow 适配层；未修改数据协议、mask、训练入口或 `result/`。
- 统一 flow 合同（`model_api.py` 新增 `UnifiedFlowModel`）：`batch_nll` 将 flow 的 per-row joint NLL 按行有效数重加权到统一有效位置分母；`predict_point` 为同一 flow 的种子采样均值（`point_seed` 固定，确定性可复现）且用全 1 掩码保持 history-only；`sample_flat` 按 `mq_flat` 掩零；`interval95_flat` 由 256 个种子样本 2.5%/97.5% 分位数给出；`loss=NLL+λ·MSE`。两个 flow 头（`ProFITiFlowHead.sample`/`LowRankCopulaFlowHead.sample`）补 `generator` 参数（向后兼容，默认 None）。
- compare_code 接入：`BaseGaussianForecastModel` 改继承 `UnifiedGaussianModel` 并加 `gaussian_params`（forward 保持 legacy 签名供独立 trainer，nll/mse/sample legacy 名不冲突）；`ProFITiGaussian` 改 `(UnifiedFlowModel, BaseGaussianForecastModel)` MRO，统一路径每次重算 `_hidden`（不依赖 legacy `_last_hidden` 缓存，专门测试锁定）；PatchTST/ODE-RNN/ProFITi 加四元身份属性。
- TCN-Gaussian 保持 standalone（不 import kaf_profiti），在类内复制统一数学（softplus+min_scale、mask.sum 分母、[B,S,P*N] 采样掩零、mean±Z95·scale 区间、loss=NLL+0.1·MSE），由公共接口检查器行为验证；登记 `faithful`（本仓库自研对照实现、无简化）。
- 新建 `baselines/grud.py`：`GRUDGaussian` 复用与 point 版完全相同的 `GRUDEncoder`（由 point.py 抽取共享，GRUDPoint 参数名变为 encoder.*，机制不变），池化表示 repeat 到 [B,P·N,H] 接 `DiagonalGaussianHead`。新建 `baselines/grafiti.py`：`GraFITiGaussian`（Yalavarthi et al. 2024）逐传感器特征 [X·M, M, context] 投影、可学习传感器邻接 [N,N] softmax、按真实 T_obs 间隔的指数衰减传播 `state=tanh(exp(−softplus(rate)·dt)·(A@state)+proj)`、逐步掩码传感器均值 → GRU readout → 统一 diagonal 头；登记 `adapted`（静态邻接+衰减传播代替原稀疏二部图边权预测）。
- 新建 `baselines/probabilistic.py` 工厂：repo 相对 sys.path 接入两处 compare_code 目录（无机器路径），`create_probabilistic_baseline` 装配概率矩阵全部 7 个模型；`UnifiedKSTProbFlow` 包装 `KSTProbFlow`（flow_hidden=distribution，NLL/sample 委托 flow_head），registry 保持 `kst_probflow` 状态 `enabled`（`create_model` 供既有 run_experiment 测试使用）仅补身份字段 `own`。
- registry：tcn_gaussian/patchtst_gaussian/gru_d_gaussian/ode_rnn_gaussian/grafiti_gaussian/profiti 六项 `pilot_ready` + 四元身份（引用核验：Bai 2018、Nie 2023、Che 2018、Rubanova 2019、Yalavarthi 2024×2，与 docs/文献汇总表 一致，纯文本无 URL）；identity 一致性测试扩展到 7 个概率模型。
- 测试覆盖（`test_probabilistic_baseline_fidelity.py` 26 项）：矩阵↔registry 身份对齐、7 模型统一接口、5 个 diagonal 模型 batch_nll 与手写高斯公式逐值一致、ProFITi 同源 flow（NLL=flow 重加权、predict_point 确定性、_flow_sample 形状）+ 不依赖缓存、GRU-D 衰减梯度与时间敏感性、GraFITi 邻接/衰减参数与 T_obs 拉伸敏感性、ODE-RNN 时间敏感性、7 模型单步 AdamW 参数更新。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` 第 12 节 T03 合同与 12.2.4 概率矩阵（7 个 model_id）、T01/T02 的统一合同与检查器、`configs/pilot/fd004/probabilistic_matrix.yaml`、compare_code 两个模型文件、`docs/文献汇总表.md`/`docs/文献阅读.md` 的基线引用（修正了一处早期设计稿中 GraFITi 作者误记为 Zeng，实为 Yalavarthi et al. 2024 AAAI）。
- Inputs not used and why: 未接入 pilot runner（P03）；未改 `train_baseline.py`/`train_tcn_gaussian.py` 训练入口（legacy 路径保持原样，统一路径绕过 `_last_hidden` 缓存）；tPatchGNN/KAFNet-Gaussian 不在本轮矩阵（后续消融），仅在 compare 文件内随基类获得统一能力未登记。
- Artifacts produced: `code/tests/pilot/test_probabilistic_baseline_fidelity.py`、`baselines/grud.py`、`baselines/grafiti.py`、`baselines/probabilistic.py`、`model_api.py` 的 `UnifiedFlowModel`、`point.py` 的 `GRUDEncoder` 抽取、两个 flow 头的 generator 参数、compare 两个模型文件的统一接入与身份、registry 七项身份、`test_experiment_framework.py` 一处过期断言更新（tcn_gaussian not_implemented→pilot_ready，create_model 门禁断言保留）、T03 与 Phase CH2.5-P02 状态勾选、本审计记录。
- Verification run: 先行测试因 `kaf_profiti.baselines.probabilistic` 缺失失败（red）；实现后修复 1 处（GraFITi 特征拼接维度 [B,L,N] vs [B,L,N,1]）；最终 fidelity 测试 26 passed，全量 `code/tests/` 190 passed（2 个既有 scipy DeprecationWarning）；portability 扫描唯一命中为 `test_experiment_framework.py:23` 既有 `/root/autodl-tmp` 默认数据根（非本次改动行，allowlist 已覆盖路径）。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: KST ProbFlow 的 flow NLL（Student-t 边际）与采样（Gaussian+低秩 copula）分布不一致为已知 Phase-3 缺口，统一接口按"同一 flow_head 提供两者"处理、风险留给正式消融阶段；ProFITi 的 `_last_hidden` legacy 缓存仍服务于旧独立 trainer，若旧 trainer 后续下线可删除；GraFITi/GRU-D-Gaussian/ODE-RNN-Gaussian 为 `adapted` 实现，进入论文表需保留标注；flow 的 interval95 依赖种子样本分位数，矩一致性容差对重尾分布的适用性已在 T03 用 S=128/256 实测通过。

## 2026-09-13 CH2.5-P03-T01 统一 pilot matrix runner

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P03-T01`。新建 runner 库、CLI 与测试；未执行任何完整训练，未修改数据协议或既有 evaluator。
- 新建 `code/kaf_profiti/experiments/pilot_runner.py`：`load_matrix`（从 tracked YAML 读取并 SHA-256 指纹，按 matrix_id 后缀推断 point/probabilistic track）；`expand()` 按 YAML 模型优先级 × 条件顺序展开 scientific key `dataset|track|model_id|head_type|condition_id|seed`（键由矩阵推导，非用户指定）；`dry_run()` 输出有序 keys、模型顺序、矩阵文件 SHA 与 expected_new；`execute()` 支持 continue-on-error（单 key 失败记录类型+消息并继续）、resume（仅当 manifest status=completed、shared_artifacts SHA 完全一致、且全部 artifact 相对路径存在时跳过）、baseline-first gate（每次从展开矩阵的 key 集合重算 manifest 证据，ours key 在任一基线 key 缺失时拒绝调度——伪造 run ID 无法绕过）；manifest 只写 result-root 相对路径。
- 统一协议经 `RealProtocolProvider`：`create_protocol_datasets(async_mode="none")` + `generate_or_load_timeline_masks`（train/valid/test 三 split 共享 timeline mask bundle，identity 含 source_split_sha256）+ `TimelineMaskedWindowDataset`；protocol fingerprint = split SHA + normalization SHA + 三 bundle content SHA，smoke 逐模型记录。
- smoke 执行器逐模型记录检查项与预算诊断（batch_time_sec/peak_memory_mb/失败消息），point 组含 history-only 与 feature-only 推理；probabilistic 组含 NLL 有限、mask-invariant 分母、[B,S,P·N] 展平序+掩零、区间有序、scale/flow 梯度、checkpoint 扰动-恢复-复放、flow 模型同源分布与 interval 种子确定性；`validate_smoke_report` 强制 ready/failed 计数与 test_metrics=null。
- 默认完整训练器 `pilot_train_and_evaluate`（validation MAE/CRPS 选 checkpoint、test 一次、GlobalMetricAccumulator 聚合）已实现供 P04 使用；本轮测试以注入 fake trainer 验证 runner 编排，完整训练路径留待 T05/AutoDL 单 batch 首跑。
- 新建 `code/run_pilot_matrix.py` CLI：`--mode dry-run|smoke|full`、`--matrix point|probabilistic|all`、`--group`；路径全部经 `resolve_runtime_paths`。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` T01 合同、`configs/pilot/fd004/*.yaml`（49 行矩阵）、`runtime_paths.resolve_runtime_paths`、`datasets.create_protocol_datasets`/`masks.generate_or_load_timeline_masks`（CH2.5-P01 已建协议机器）、`accumulators.GlobalMetricAccumulator`、`model_api.assert_history_only`。
- Inputs not used and why: 未修改 `run_experiment.py`（其 enabled-only create_model 与 pilot_ready 基线不兼容，runner 自建模型工厂并复用统一接口模型）；未执行完整训练（P04，须在 AutoDL）；tPatchGNN/KAFNet 不在矩阵。
- Artifacts produced: `pilot_runner.py`、`run_pilot_matrix.py`、`code/tests/pilot/test_pilot_runner.py`（13 项：展开/顺序/49 行唯一性、dry-run、manifest 相对路径、resume 跳过与重跑、continue-on-error、gate 阻塞与反 ID 伪造、三组 smoke + validator 拒绝用例）、T01 状态勾选与本审计记录。
- Verification run: 先行测试 red（模块缺失）→ 实现迭代修复 5 处（provider 注入点、skipped 计数语义、smoke pred_len 随 provider、时长测量占位、掩码布尔索引维度），全绿后实测全矩阵 dry-run `expected_total=49, expected_new=49`、模型顺序 11 基线在前；全量 `code/tests/` 203 passed；portability 无新增命中。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 默认完整训练器未在真实数据上首跑（P04 AutoDL 首个 run 即首次执行，需先以 T05 单 batch 环境 smoke 验证）；smoke 的 peak_memory_mb 取进程累计峰值（macOS/Linux 单位已区分），仅作预算参考；键格式与 manifest schema 将被 P04 validate-only 复核。

## 2026-09-13 CH2.5-P03-T02 第三章五基线 smoke

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P03-T02`。本机对 5 个点预测基线执行单批 smoke；未生成 test 指标，未触碰正式矩阵结果。
- 执行 `run_pilot_matrix.py --mode smoke --group point_baselines`：真实 FD004 + 混合@0.30 timeline mask（协议主设置），每模型一个 train batch（loss 有限+单步 AdamW 参数变化）、一个 valid batch（predict_point 有限）、一个 test feature-only batch（仅前向有限性，test_metric_count=0）。
- 5/5 ready：li_tcn/ff_gru/masked_tcn/gru_d/ode_rnn 全过 loss_finite、params_changed、valid/test 有限、history-only；全部模型共享同一 protocol fingerprint（split 61c7db91…、norm 2cb43b19…、train/valid/test mask bundle SHA 一致）。
- 修复 smoke 内存诊断单位（macOS ru_maxrss 为字节、Linux 为千字节），修正后峰值为 459–630MB 进程累计峰值。

### Capability-use audit

- Required skills: executing-plans, verification, verification-before-completion
- Skills actually used: executing-plans, verification, verification-before-completion
- Inputs consumed: T01 runner 与 validator、FD004 原始数据（KST_DATA_ROOT）、点矩阵 YAML、CH2.5-P01/P02 的协议与统一接口。
- Inputs not used and why: 未跑完整 epoch（smoke 仅接口验证）；未保存 checkpoint/prediction（smoke 不产生 run 级 artifact）。
- Artifacts produced: `result/pilot/fd004/smoke/point_baselines_smoke.json`（run_level=smoke）、T02 状态勾选与本审计记录。
- Verification run: validator 输出 `ready=5,failed=0,test_metrics=0`；报告内 5 模型 protocol SHA 完全一致。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 单批 smoke 不证明 80 epoch 稳定性或显存上界（完整预算以 P04 实测为准）。

## 2026-09-13 CH2.5-P03-T03 第四章六基线 smoke

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P03-T03`。本机对 6 个概率基线执行单批 smoke；未生成 test 指标。
- 执行 `run_pilot_matrix.py --mode smoke --group probabilistic_baselines`：6/6 ready（tcn_gaussian/patchtst_gaussian/gru_d_gaussian/ode_rnn_gaussian/grafiti_gaussian/profiti），各 11 项检查（profiti 13 项含 flow 同源分布与 interval 确定性）全过：NLL 有限且 mask-invariant、samples [B,S,P·N] 有限且掩零、区间有限有序、head/flow 参数非常零梯度、checkpoint 扰动-恢复-同种子采样复放一致。
- 修复概率组 batch 时长测量占位（perf_counter()-0.0 → 实际训练步计时）；重跑后正常。
- 预算诊断：ProFITi 单 batch 10.6s、进程峰值 15.1GB（flow 逆映射按 B·S 行展开所致），为 P04 显存/时长预算的重要输入；其余 5 模型 0.03–0.7s。

### Capability-use audit

- Required skills: executing-plans, verification, verification-before-completion
- Skills actually used: executing-plans, verification, verification-before-completion
- Inputs consumed: T01 runner、概率矩阵 YAML、CH2.5-P02-T03 的统一 distribution 合同与 7 模型工厂、FD004 原始数据。
- Inputs not used and why: 未计算 test 指标（合同禁止）；未对 flow 模型跑 T_q 扰动不变检查（ODE-RNN/ProFITi 合法消费 T_q 预报起点，该检查不适用，已在 P02 记录）。
- Artifacts produced: `result/pilot/fd004/smoke/probabilistic_baselines_smoke.json`、时长测量修复、T03 状态勾选与本审计记录。
- Verification run: validator 输出 `ready=6,failed=0,test_metrics=0`；6 模型 protocol SHA 一致（与 point 组同 split）。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: ProFITi 的 15GB 峰值为进程累计值且 batch_size=128/nsamples=16 条件下测得，P04 前须按正式 nsamples 复核显存；flow 逆映射行展开是已知的可优化点。

## 2026-09-13 CH2.5-P03-T04 本文模型 smoke

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P03-T04`。11 个基线 smoke 全部 ready 后对本文三配置执行单批 smoke；未生成 test 指标，不构成模型优劣结论。
- 执行 `run_pilot_matrix.py --mode smoke --group ours`：入口 gate 校验两组基线报告 ready=11/failed=0 后调度 kst_light|linear、kst_light|mlp、kst_probflow。
- 3/3 ready：KST-Light 两头 history-only（统一扰动检查器）+ 单步参数变化；KST ProbFlow NLL/sample/interval 全有限、flow_hidden 两次调用逐值一致（NLL 与 sample 同源）、interval95 种子确定性复现（PICP/MPIW 来源固定）。
- 协议一致性：ours 3 项与 11 个基线共享同一 split/normalization/mask SHA；batch 0.30–0.89s、峰值 1.1GB，预算健康。

### Capability-use audit

- Required skills: executing-plans, verification, verification-before-completion
- Skills actually used: executing-plans, verification, verification-before-completion
- Inputs consumed: T01 runner 的 ours gate、两组基线 smoke 报告、`KSTLight`/`UnifiedKSTProbFlow` 统一接口、FD004 原始数据。
- Inputs not used and why: 未调 P04 完整训练；未读取任何 test 指标。
- Artifacts produced: `result/pilot/fd004/smoke/ours_smoke.json`、T04 状态勾选与本审计记录。
- Verification run: validator 输出 `ready=3,failed=0,test_metrics=0`。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: smoke 只证明代码可运行（计划原文）；KST ProbFlow 的 Student-t 边际 NLL 与 Gaussian copula 采样的分布差异仍是 Phase-3 缺口，本 smoke 验证的是"同源调度"而非"分布同一"。

## 2026-09-13 CH2.5-P03-T05 同步版本并执行 AutoDL 环境预检

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：仅完成 `CH2.5-P03-T05`。TDD 新建环境预检脚本与测试、形成 clean commit 并推送、在 AutoDL（无卡模式 + 开卡模式）完成环境核验与跨机比对；未启动任何 P04/P05 正式 run。
- 新建 `code/check_pilot_environment.py` + `code/tests/pilot/test_pilot_environment.py`（7 项）：报告 schema `pilot-environment-preflight-v1`，记录 git commit/clean、FD004 三文件 SHA、两个矩阵 SHA、协议 mask bundle SHA（result-root 相对）、code 目录树指纹、依赖版本、Python/torch/CUDA/GPU/磁盘段；`--compare` 递归 diff 身份段（矩阵/协议/code/数据集），忽略环境名与解析路径；`--smoke` 对 li_tcn 单批设备链路验证（不产 test 指标）；`--require-gpu` 供最终门禁；无 GPU 机器 GPU 字段记 null 不失败。
- 版本同步：本机全量 210 tests + 三组 smoke 全绿后 clean commit `67f39df` 推送 origin；T05 执行中发现比对范围过严（依赖版本被误当阻断项），按计划 12.5 "除环境和解析路径外" 原文修正为四身份段，作为 commit `4e68ac6` 推送，AutoDL pull 后重跑预检，本机重跑比对。
- AutoDL 无卡模式（14 vCPU Xeon 6330）：pilot+ch3 测试子集绿；全量 pytest 在根目录数据测试（MetroPT 原始 CSV 整帧加载）处 OOM 被杀（75% 处，pilot/ch3 已全部通过）——处置为无卡阶段只跑 pilot+ch3 子集，重型数据测试由本机同一 commit 覆盖，跨机一致性由 SHA 比对保证；GPU 预检与设备 smoke 留到开卡。已向用户说明 OOM 原因与加 swap 的可选方案。
- AutoDL 开卡模式（RTX 3090 24135MB）：`--smoke --require-gpu` 通过——CUDA 可用、li_tcn 单批 CUDA loss 有限+参数变化、test_metric_count=0；最终比对 `identity_sections_match`（矩阵/协议 mask bundle/code 指纹/数据集 SHA 逐项一致）。归档三份报告：`result/pilot/fd004/environment/` 下 local-preflight.json、autodl-preflight-nogpu.json、autodl-preflight-gpu.json（前两份 commit=67f39df，GPU 版=4e68ac6；nogpu 与 gpu 版身份段一致）。
- 其余说明：AutoDL 镜像 `OMP_NUM_THREADS` 非法值触发 libgomp 一次性警告，不影响执行（smoke 通过为证）。

### Capability-use audit

- Required skills: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: using-superpowers, executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: `plan/implementation-plan.md` T05 合同、`pilot_runner` 的 RealProtocolProvider/build_model、requirement.txt 锁定依赖、AutoDL 实例（无卡 + RTX 3090 两模式）、GitHub origin 仓库。
- Inputs not used and why: 未在 AutoDL 复跑全量 pytest（OOM 后按"本机同一 commit 已全绿 + SHA 比对保证一致"处置，属计划未要求的附加验证）；未启动 P04/P05 训练（预检报告通过是其前置，本轮到此为止）。
- Artifacts produced: `check_pilot_environment.py`、`test_pilot_environment.py`（7 项）、commit `67f39df` 与 `4e68ac6`（已推送）、三份预检报告、T05 与 Phase CH2.5-P03 状态勾选、本审计记录。
- Verification run: 先行测试 red → 实现 7 tests 全绿（含 GPU 段 monkeypatch 无 CUDA 用例、compare 漂移三用例、CLI 端到端、真实 li_tcn 单批 smoke）；AutoDL 无卡 pilot+ch3 子集绿、开卡 `--smoke --require-gpu` ok、跨机比对 `identity_sections_match`；比对脚本修正后 `test_pilot_environment.py` 7 passed。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 全量 pytest 在小内存无卡实例不可复现（MetroPT/TEP 帧加载 ~数 GB），若 CH3 需要在 AutoDL 复跑全量须先加 swap 或用大内存实例；P04 正式 run 启动前 ProFITi 显存预估（smoke 峰值 15.1GB）需在 3090 24GB 上以正式 nsamples 实测确认；AutoDL checkout 此后不得直接编辑，任何代码变更必须走本机 commit→push→AutoDL pull。

## 2026-09-14 CH2.5-P04-T01/T02 42 个点预测 pilot run（含执行中两轮修复）

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：完成 `CH2.5-P04-T01`（30 基线）与 `CH2.5-P04-T02`（12 KST-Light），含执行中由用户发现的两处 runner 缺陷修复（commit `c7b9114`、`19461c0`）。T03 汇总未开始。
- 执行环境：AutoDL RTX 3090 24GB，commit `19461c0`，单轮 `run_pilot_matrix.py --mode full --matrix point`，42/42 completed、0 failed；总 wall-clock 约 14.4 小时。
- 执行中修复一（用户发现：nvidia-smi 无进程）：默认训练器写死 `torch.device("cpu")`、`build_model` 未 `.to(device)`、`PilotRunner.device` 未送达 trainer——中断无效轮次，修复为三处设备接线 + optimizer 移出 epoch 循环（AdamW 动量此前每 epoch 被重置）+ 补 `batch_nll_rows` API（`_test_metrics` 此前引用不存在的方法），commit `c7b9114`。
- 执行中修复二（用户诊断：DataLoader num_workers=0 + 无 pin_memory；复核中发现更严重问题）：`execute()` 从未按 run 播种科学 seed，先前执行的 run 协议无效——修复为每 run `torch.manual_seed(spec.seed)`（测试以 `torch.initial_seed()==2026` 锁定）+ DataLoader num_workers auto（GPU 4）/pin_memory/persistent_workers + `non_blocking` 拷贝，commit `19461c0`，重跑预检身份比对通过后才重启正式执行。
- 本机 validate-only（runner `_verified_specs` 对同步回的 artifact 副本）：`expanded=42, unique=42, verified=42, baseline=30, ours=12（linear 6 + mlp 6）, duplicate=0, missing=0, fairness_mismatch=0`；全部 `status=completed, device=cuda, test_metric_count=1`；42 份 metrics MAE/RMSE 全有限非负、checkpoint 齐全、history 80 epoch 连续。
- 有效性证据：`pilot_point_run.log` 恰好 42 条 `elapsed_sec` 完成事件，证明 42 个 run 全部在 `19461c0` 单轮执行中产生，无任何 run 经 resume 继承自种子修复前的无效轮次；执行顺序为 30 基线全部完成后才出现 KST-Light 事件（gate 生效的运行时证据）。
- 效率观察（供 T03 效率视图）：li_tcn ~1200s、ff_gru ~1170s、masked_tcn ~1160s、gru_d ~1180s、ode_rnn ~1710s（Euler 子步随 batch 最大间隔缩放）、kst_light|linear ~1910s（多尺度 KAF 编码器更重）/mlp 相当。

### Capability-use audit

- Required skills: executing-plans, verification, verification-before-completion, debugging
- Skills actually used: executing-plans, verification, verification-before-completion, debugging
- Inputs consumed: P03 runner/预检闭环、两个 tracked 矩阵、FD004 数据、AutoDL GPU 实例、用户提供的 nvidia-smi/pgrep 诊断与 DataLoader 分析、42 run 事件日志与 artifact 副本。
- Inputs not used and why: 未读取任何 run 的 test 指标做模型间比较或选择（T03 之前禁止形成排序结论）；未将 result/ 入库（.gitignore 约定，manifest SHA 核对代替）；未勾 Phase CH2.5-P04 框（T03 未完成）。
- Artifacts produced: AutoDL 42 run 目录（manifest/history/metrics/checkpoint）同步至 `result/pilot/fd004/runs/`、commit `c7b9114` 与 `19461c0`（已推送，含 15+7 项 runner/预检测试更新）、T01/T02 状态勾选与本审计记录。
- Verification run: 两轮修复各自先 red 后绿（种子断言 {2026}、meta 设备断言）；本机 validate-only 42/42 全绿（上）；`grep -c elapsed_sec = 42`。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: manifest 未记录协议 SHA 字段（公平性由共享 mask bundle 路径 + shared_artifacts SHA 结构性保证，T03 汇总时从 bundle 文件补记 provenance）；单种子结果不得进入论文主表；P05 ProFITi 显存须以正式 nsamples 实测。

## 2026-09-14 P04 条件轴失效复盘与修复（首轮 42 run 作废）

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：用户审阅指标发现首轮 42 run 数据无效；完成根因定位、复盘文档、TDD 修复与防护测试。T01/T02 勾选撤销，runs 已删除，待重跑。
- 数据无效判定（用户观察，逐条核实）：5 基线跨 6 条件指标 bit 级相同（kst_light 仅第 4 位小数且方向随机 = CUDA 非确定性噪声）；valid_count 恒 5,703,810 满额；protocol 目录仅一套 mixed_0.30 bundle；RMSE≈0.99/MAE≈0.83 处于标准化数据均值预测水平。
- 根因 A：`_build_provider` 硬编码 smoke 常量 mixed@0.30，`spec.missing_mode/target_missing_rate` 从未进入数据路径——六条件数据零差异。根因 B：`TimelineMaskedWindowDataset` 只掩蔽历史段，查询段从未切片到 `M_q`——valid_count 与缺失率无关。
- 复盘：`plan/review/p04-condition-axis-invalid-review.md`——防线逐层失守分析（fake provider 测试零断言真实工厂接线、smoke 只测主条件、一致性检查检测不了系统性错误、mask bundle 数量等免费信号被忽略）+ 五项整改。
- TDD 修复（先 red 后 green）：`_build_provider` 改用 spec 条件；新增 `_build_smoke_provider` 钉住 smoke 主条件；`TimelineMaskedWindowDataset.__getitem__` 将查询段切片到 `M_q`（Y_q 保持原值，缺失由统一 mask 合同排除；查询切片越界显式报错）；`execute()` 运行时断言 provider 指纹 mechanism/rate 与 spec 一致（未来接线错误在第一个 run 即失败）；manifest 新增 `protocol_sha`（split/normalization/三 split mask SHA + 条件身份）。
- 防护测试：新增 `code/tests/pilot/test_pilot_condition_axis.py`（7 项）——两工厂接线（monkeypatch 记录）、合成 timeline 的查询段掩蔽与越界拒绝、execute 条件失配拒绝、manifest protocol_sha、真实 FD004 上四条件（random 0/30/70 + low_rate 30）bundle SHA 互异 + `M_q` 等于 timeline 查询切片 + valid 比例 `1.0 > ~0.7 > ~0.3` 单调。旧测试 `test_timeline_masked_window_dataset_slices_canonical_windows` 的 "targets stay fully observed" 断言恰为根因 B 的固化，按新契约更新为查询切片断言。
- 验收条款修订：P04-T01/T02 validate-only 增加 `condition_axis_effective` 必查项；dry-run 人工复核增加 mask bundle 数 = 条件组合数。

### Capability-use audit

- Required skills: using-superversers, executing-plans, test-driven-development, verification, verification-before-completion, debugging
- Skills actually used: executing-plans, test-driven-development, verification, verification-before-completion, debugging
- Inputs consumed: 用户提供的四条指标异常观察（本次无效判定的唯一来源）、42 run artifact 副本、mask bundle 文件、`masks.py`/`pilot_runner.py`/`cmapss.py` 源码、FD004 原始数据。
- Inputs not used and why: 未尝试从旧 run 挽救任何结论（数据层无效，不可修补）；未修改矩阵 YAML（矩阵设计正确，错在接线）；未勾任何完成框（T01/T02 撤销，Phase 框未勾）。
- Artifacts produced: 复盘文档、修复三文件（`masks.py`、`pilot_runner.py`）、新测试文件（7 项）+ 旧测试契约更新、计划撤销注记、runs 目录删除、本审计记录。
- Verification run: 新测试先 red（ImportError/_build_smoke_provider 缺失、M_q 断言失败）后 green；全量 `code/tests/` 219 passed。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 重跑前 AutoDL 侧 runs 目录必须删除（否则 resume 误判旧 manifest 已验证跳过重跑）；预检两份报告须在 pull 后重新生成比对（code 指纹已变化）；训练动力学异常（valid 自 epoch 3 恶化）修复后重看，T03 必须如实呈现。

## 2026-09-14 P04 复盘修订：查询段掩蔽判定为误判（用户改判）

- 阶段：S3 Experiments / CH2.5 FD004 单种子预实验
- 范围：修订上一条审计的根因结论。真正的缺陷只有根因 A（`_build_provider` 硬编码 mixed@0.30）；原根因 B（查询段未掩蔽）为误判——人工缺失协议只作用于历史输入 `M_obs/X_obs`，`Y_q/M_q` 与 valid count 跨条件恒定是统一未来目标评测的公平性必需，valid_count 恒 5,703,810 是正确信号而非异常。
- 代码修订（用户完成，本条目核验通过）：查询段掩蔽回退（`TimelineMaskedWindowDataset` 只替换历史段）；timeline mask schema 升 v3 且文件名加 `v3_` 前缀（旧 v2 bundle 不被静默复用、预检只指纹活跃 schema）；`loss.detach().item()`；防护测试改为双向断言——不同条件 history mask SHA 互异 + 历史可观测率 1.0 > 030 > 070 单调 + `M_q/Y_q` 与 query mask 逐字节跨条件恒定；计划 T01/T02 的 `condition_axis_effective` 定义同步补全为双向。
- 本核验补充修正：复盘文档 §2 一行残留旧口径（"不同 valid_count"）已改为"history mask SHA 互异 + query 侧逐字节恒定"；`check_pilot_environment.py` import 归位。
- 有效信号重新解读：首轮 42 run 中基线跨条件 bit 级相同 + protocol 目录仅一套 bundle 仍是根因 A 的确凿证据；RMSE≈0.99/MAE≈0.83 均值预测水平与 valid 自 epoch 3 恶化（best checkpoint 在 epoch 3）仍是 T03 须如实呈现的训练动力学问题。

### Capability-use audit

- Required skills: verification, verification-before-completion
- Skills actually used: verification, verification-before-completion
- Inputs consumed: 用户完成的八文件修订、`git diff` 全量核对、FD004 防护测试。
- Inputs not used and why: 未改动用户的核心改判（评测语义决定权在协议 owner）；未重跑 42 run（修复验证留待 AutoDL）。
- Artifacts produced: 复盘文档一行修正、import 归位、本审计条目、全量测试复跑。
- Verification run: 全量 `code/tests/` 218 passed（含修订后双向防护测试）。核验中发现并处置一处陈旧 artifact：15:20 生成的三个 v3 mixed bundle 携带中间代码状态下的 split SHA（4c62…，与确定性重算的 61c7… 不符），占据确定性路径导致 mixed@0.30 provider 构建被严格校验硬拒——确认 split SHA 两次重算完全一致（排除非确定性）后删除，重建 bundle 携带正确 SHA 且 realized≈0.29 达标。
- Review result: 规格符合性复审 `PASS`；质量复审 `APPROVED`。
- Remaining risk: 同上一条——AutoDL 侧 runs 删除 + 重跑预检 + 重跑 42 run；valid 恶化现象在修复后首轮重看。

## 2026-09-15 CH34-S00-T01 冻结 FD004 路线并登记诊断证据

- 阶段：S4 Reconstruction / 第 13 节（MetroPT-3 重启前冻结）。
- 内容：**CH2.5-P04~P07 已冻结，由第 13 节替代。** FD004 已完成的 2 个连续传感器预测 run 一并登记为 `diagnostic_only`，不进入任何第三/四章 formal 结论或模型排序。
- 判定：代码链路有效（连续性 32/32、engine 隔离、窗口对齐、train-only 归一化、timeline mask 可复现均通过），但 FD004 裸传感器回归任务不适配（21 通道 cycle 级 lag-1 自相关≈0，学习模型未稳定超过零/常数预测；持久性 test MAE≈1.02 因未来工况切换而劣化）。文档：`plan/review/fd004-task-suitability-review.md`。
- 保留物证：`result/pilot/fd004/runs/` 下点预测 random@0% 与 random@30% 两个 run（manifest/metrics/history/checkpoint 原样，未手工修改）。

### Capability-use audit

- Required skills: executing-plans, verification, verification-before-completion
- Skills actually used: executing-plans, verification, verification-before-completion
- Inputs consumed: 两个 FD004 run 的 manifest/metrics/checkpoint（只读）、§13.1.1 诊断事实、诊断脚本与既有 ch3 可移植策略测试。
- Inputs not used and why: 未改写任何 `result/` artifact；未重建 MetroPT-3 协议（CH34-S01）。
- Artifacts produced: `plan/review/fd004-task-suitability-review.md`、本进度条目、计划 CH34-S00-T01 勾选。
- Verification run: review 文档数值与 manifest 逐项核对；`result/` 未被写入。
- Review result: 规格符合性 `PASS`。
- Remaining risk: FD004 未来复用须另立 RUL/风险协议与 new SHA（§13.1.2 注记）；MetroPT-3 协议重建属 CH34-S01，未在本阶段展开。

## 2026-09-15 CH34-S01-T01 MetroPT-3 私有协议 catalog

- 阶段：S4 Reconstruction / MetroPT-3 协议重建。
- 范围：完成 T01 的私有 catalog 层；未注册 `metropt3_chrono_502030_v2` 公开入口，未触碰旧 MetroPT 协议与 `create_protocol_datasets`，T02/T03/T04/T05 保持未开始。
- 设计：保留 `source_row_id` 并按 `(timestamp, source_row_id)` 稳定排序；按累计行数 50%/70% 目标在 timestamp-group 合法边界中最近切分（平局取早边界）；split 内按 `gap > 3 × train median interval` 生成 segment；segment 仅作为 mask 适配层的 `timeline_key`，窗口使用不可变 `WindowRecord`，`sample.unit_id` 语义留给后续 T02。
- 实现：`load_metropt_frame_v2`、`median_interval_seconds`、`split_chronological_by_timestamp_group`、`segmentize`、`build_window_catalog`；新增 raw→partition→timeline→window_catalog 四层 SHA，window ID 只引用 timeline SHA + 自身 forecast/query/source-row 字段，避免循环依赖。
- 防护测试：新增 `code/tests/pilot/test_metropt_protocol.py` 10 项——重复 timestamp 组不跨 split、边界平局取早、31 秒大间隔不进窗口、WindowRecord 单一投影、source row 唯一/稳定排序、重复 timestamp hard-fail、真实 MetroPT 1,516,948 行集成不变量和 SHA 稳定性。

### Capability-use audit

- Required skills: executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: 第 13 节 CH34-S01-T01 合同、用户关于 v2 入口/SHA 分层/segment 语义/source_row_id/切分规则的修订、现有 `metropt.py`/`datasets.py`/mask window 契约、真实 MetroPT CSV。
- Inputs not used and why: 未注册公开 v2 入口（normalization/protocol SHA 尚未在 T04 完成）；未实现 7/8 target/context、真实时间、归一化、mask 和 learnability gate（分别属于 T02–T05）；未触碰 GPU。
- Artifacts produced: `WindowRecord` 与私有 catalog helper、`code/tests/pilot/test_metropt_protocol.py`、T01 计划勾选、本条目。
- Verification run: 新测试先 red（`WindowRecord` import 缺失）后 green；T01 测试 `10 passed`；真实 CSV 集成不变量通过；全量 `code/tests/` `240 passed`；`git diff --check` OK。
- Review result: 规格符合性复审 `PASS`；旧协议回归 `PASS`。
- Remaining risk: T01 helper 尚未接入 provider，不能启动 MetroPT 训练；T02 必须在此 catalog 上补齐 7 连续 target/8 binary context；T04 才能生成 normalization/protocol SHA 并注册 v2。

## 2026-09-15 CH34-S01-T02 MetroPT-3 连续目标与二值历史 context

- 阶段：S4 Reconstruction / MetroPT-3 协议重建。
- 范围：完成 T02，在 T01 catalog 上实现 `MetroPTChronoDataset`，用 7 连续目标 + 8 二值历史 context 填全 `__getitem__`；尚未接入 provider、真实时间（T03）、归一化/mask（T04）、learnability/风险门禁（T05）。
- 实现：`METROPT_CONTINUOUS_COLUMNS`(7)、`METROPT_BINARY_CONTEXT_COLUMNS`(8) 常量；`MetroPTChronoDataset` 暴露 `_units`(segment->frame)、`windows`((segment_id,start))、`window_ids`，`__getitem__` 返回 `X_obs/Y_q/M_obs/M_q` 末维 7、`context` 末维 8（origin 前最后观测）、`unit_id=0`；构造期校验列固定顺序/缺失/重复与二值 0/1（报告列名+source_row_id）。
- 防护测试：6 项 T02 用例——shape、context=last history、query 段 8 状态量扰动不改 X/M/T/context、列顺序/缺失 hard-fail、非 0/1 报列名+source_row_id、window 投影一致。全量 `test_metropt_protocol.py` 16 passed。

### Capability-use audit

- Required skills: executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: T01 catalog、`MetroPTWindowSample`/`IndustrialCollator` 契约、§13.2 列分离合同。
- Inputs not used and why: 未接入 provider 或 `create_protocol_datasets`（T04）；未实现真实时间（T03）；未做归一化/mask（T04）；未触 GPU。
- Artifacts produced: `MetroPTChronoDataset`、6 项 T02 测试、T02 计划勾选、本条目。
- Verification run: T02 测试先 red（`MetroPTChronoDataset` 缺失 + object-dtype 修复 + torch import/测试用例修正）后 green；全量 `code/tests/` 246 passed；真实 MetroPT 冒烟 `X_obs(168,7)/Y_q(24,7)/context(8)`、unit_id=0。
- Review result: 规格符合性 `PASS`；旧协议回归 `PASS`。
- Remaining risk: 真实时间仍为 arange（T03 替换）；归一化/mask 未接入（T04）；真实故障风险标签 `rul` 已按 fault-window 交集计算，但正式风险口径在 T05 复核。

## 2026-09-15 CH34-S01-T03 MetroPT-3 真实时间与数值稳定缩放

- 阶段：S4 Reconstruction / MetroPT-3 协议重建。
- 范围：完成 T03，`MetroPTChronoDataset.__getitem__` 弃用 `arange`，改为段起点秒数 ÷ train median interval 的真实缩放时间；提供 `metropt_time_scale_artifact`（unit/source/median/sha）。尚未接入 provider/归一化/mask（T04）与 learnability/风险门禁（T05）。
- 实现：`_real_time` 以 `segment` 时间戳相对段起点换算秒、除以 `median_interval`，保留真实抖动（真实数据 diffs 出现 0.9/1.0，非等距）；`metropt_time_scale_artifact` 只从 train frame 派生并记录 sha256。
- 防护测试：4 项 T03 用例——合成 [0,10,21,31]s 段以 median=10 缩放后 diffs==[1.0,1.1,1.0]、`T_q[0]>T_obs[-1]` 且 origin→query 间距 (44−31)/10=1.3、artifact 记录原始单位+稳定 sha、GRU-D/ODE-RNN/KST 在合法时间间隔变化（值不变）下输出有限变化。全量 `test_metropt_protocol.py` 20 passed。

### Capability-use audit

- Required skills: executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: T01 catalog 的窗口时间戳、T02 dataset、`compute_delta_t`(GRU-D)/ODE-RNN Euler 间隔/KST `encoder(x,T_obs,m,ctx)` 的时间消费点、§13.2 时间缩放合同。
- Inputs not used and why: 未把时间尺度 artifact 写入 provider fingerprint（T04）；未实现归一化/mask（T04）；未触 GPU。时间敏感性测试按 GRU-D 全观测 mask 下 delta_t 恒 0 的机制特性使用稀疏 mask 构造，以真实暴露时间依赖。
- Artifacts produced: `_real_time`、`metropt_time_scale_artifact`、4 项 T03 测试、T03 计划勾选、本条目。
- Verification run: 新测试先 red（median 断言与合成偏移不自洽）修正后 green；T03 全文件 20 passed；真实 MetroPT 冒烟 T_obs diffs=[1.0,0.9,1.0,...] 非等距、T_q[0]>T_obs[-1]、全量 250 passed。
- Review result: 规格符合性 `PASS`；旧协议回归 `PASS`。
- Remaining risk: 时间尺度 sha 尚未进入 provider fingerprint（T04 落地）；T04 须以 `reject_duplicate_timestamps=True` 构建正式 v2 入口并注册 `metropt3_chrono_502030_v2`。

## 2026-09-15 CH34-S01-T04 MetroPT-3 归一化冻结、条件轴与 v2 公开入口

- 阶段：S4 Reconstruction / MetroPT-3 协议重建。
- 范围：完成 T04——train-only 归一化（7 连续通道）、timeline-first 条件轴（segment 即 timeline unit）、完整分层协议身份，并注册公开入口 `metropt3_chrono_502030_v2`。T05（可学习性/风险门禁）未开始。
- 实现：`metropt_v2_stats_artifact`（只由 train source_row_id 行计算 mean/std，二值 context 不进归一化）；`MetroPTChronoDataset` 新增 `stats`（仅作用 X_obs/Y_q）；`_create_metropt_chrono_502030_v2` 串联 raw→partition→timeline→window_catalog + normalization/time_scale → `split_sha256`（protocol 层），`segmentize` 强制 `reject_duplicate_timestamps=True`；`create_protocol_datasets` 注册 v2 分派。
- provider 集成实测：`RealProtocolProvider` 直接消费 v2——mask 三 split bundle 生成、loader batch `[32,168,7]`、li_tcn loss 有限；`num_sensors=7/context_dim=8`。
- 防护测试（`test_metropt_condition_axis.py` 4 项）：归一化 SHA 对 valid/test 行扰动不变、对 train 行扰动必变；4 条件 bundle SHA 互异且 random 0/30/70 可观测率 1.0>~0.7>~0.3 严格递减、mixed 与 random 在同 30% 实际率一致；条件间 query 侧逐字节不变（Y_q/M_q/context/window 投影/valid count）且 X_obs==base×M_obs、两机制 30% 的 history mask 必不相同；公开入口身份字段（7/8 维、分层 SHA、context policy、masked channels、window bounds）。

### Capability-use audit

- Required skills: executing-plans, test-driven-development, verification, verification-before-completion
- Skills actually used: executing-plans, test-driven-development, verification, verification-before-completion
- Inputs consumed: T01 catalog/T02 dataset/T03 时间、`TimelineMaskedWindowDataset` 的 segment 契约、`RealProtocolProvider` 指纹链、§13.2 冻结合同、用户关于占位 SHA/分层 SHA/公开入口时机的修订。
- Inputs not used and why: 未修改 `pilot_runner.py`（provider 指纹结构对 v2 已够用，layered SHA 经 split_identity 进 split_sha256）；未修改 `masks.py`（segment-as-unit 直接满足契约）；未触 GPU；未开始 T05。
- Artifacts produced: `metropt_v2_stats_artifact`、`_create_metropt_chrono_502030_v2` + 分派、`MetroPTChronoDataset.stats`、`test_metropt_condition_axis.py`（4 项）、T04 勾选、本条目。
- Verification run: 测试先 red（artifact 缺失 + v2 未注册 + 路径嵌套/`.windows` 代理两处用例修正）后 green（4 passed）；provider 端到端实测通过；全量 `code/tests/` 254 passed；`git diff --check` OK。
- Review result: 规格符合性 `PASS`；旧协议回归 `PASS`（旧 `metropt3_chrono_502030` 及全部既有断言不动）。
- Remaining risk: v2 的 49-run 矩阵与 runner profile 化属 CH34-S02；learnability/风险标签门禁属 T05；当前窗口计数（train≈1.2 万窗）仅为协议产物，尚未经 learnability 门禁确认可学习。
