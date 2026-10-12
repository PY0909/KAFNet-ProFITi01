# SOSR 候选创新的联网调研任务书

> 用途：将本文件全文复制给负责联网检索的 agent。目标不是证明 SOSR 已经创新，而是系统检索最接近的方法，判断哪些部件已被提出、哪些组合仍有研究空间。

## 1. 研究背景与候选机制

当前论文主模型是 Unified V3C，第四章不再以概率区间预测为主线。待检索候选方向暂命名为 **SOSR（Scale-Observation Support Conditioned Sensor Routing，尺度—观测支持条件化传感器路由）**：

1. 在每个时间尺度和 patch 上计算传感器级观测支持量；
2. 用目标传感器的观测缺口控制跨传感器消息注入强度；
3. 用源传感器的局部可靠性（support、freshness、最近观测间隔、连续缺失长度）调节消息权重；
4. 用传感器关系和 query/key/value 进行跨传感器路由；
5. 根据局部观测支持自适应融合不同时间尺度；
6. 目标任务仍是 MetroPT 状态预测、FD004 RUL、TEP 故障检测/诊断，不改变统一编码器和数据集独立训练原则。

候选公式可概括为：support = mean(mask)；source reliability 由 support、freshness、delta-time 和连续缺失长度构成；target deficit gate 由 (1-support) 与 MLP 门控构成；routed message 使用可靠性偏置的跨传感器 attention；scale fusion 按局部支持学习尺度权重。

不要预设它是“首次提出”。重点检验：这些部件是否已有先例；完整组合是否只是已有缺失感知图、通道 token、多尺度融合的拼接；组合是否带来可独立验证的技术差异。

## 2. 必须回答的研究问题

先核验材料边界：附件中的本地源码定位来自旧项目，不是当前 Unified V3C 的实现证据；六个参考仓库及其功能描述均是检索线索，须独立验证。不要假定当前模型已经有完整的逐 patch、逐传感器 token 或缺失统计接口。

专门审查以下假设：观测比例高不代表传感器准确，freshness 衰减也可能误伤低频但有效的传感器；目标缺失越多不保证应注入越多消息；所有源都不可用时需要显式回退而不是对无效源做 softmax；源可靠性偏置必须保证所声称的单调关系；完全观测下 g=0 仅保证新增残差路径关闭，不自动保证改过尺度融合或联合训练后与第三章预测一致。训练和推理均不得读取未来观测来构造可靠性。

此前人工缺失仅作为可选压力测试。若第四章改为缺失鲁棒性主线，需明确这是研究范围调整建议，不得默默把三个数据集称为天然异步或真实缺失数据。另给出一个不依赖人为缺失的备选方向，以免检索只为 SOSR 寻找支持。

请按以下顺序回答：

1. 最近邻方法是谁：缺失感知跨变量图/消息传递、多尺度时间序列 patch 或 channel token、观测条件化动态传感器图、目标节点缺口门控、源节点可靠性加权、缺失模式驱动的尺度选择、不插补预测、蒸馏或一致性训练。
2. SOSR 的每个部件是否已有公开先例？逐部件列出，而不是只列总体名称。
3. 有没有已经同时具备“多时间尺度 + 目标缺口门控 + 源可靠性路由”的论文或代码？列出重合模块、差异和是否足以构成实质创新。
4. 候选创新属于新模块、已知模块的新组合、面向工业异步/缺失遥测的新应用，还是主要是工程实现？
5. 最小可辩护贡献是什么？如果不能声称新算法，改写为可验证的经验贡献或协议贡献。
6. 哪些实验可以区分 SOSR 与简单拼接？必须检查 mask-only、global mixer、无目标缺口、无源可靠性、无尺度路由等消融。
7. 三个数据集是否真的适配？分别说明 MetroPT、FD004、TEP 的缺失语义、时间单位、任务输出和可能失效情形。

## 3. 检索范围和检索词

优先检索 2019 年至今的正式论文、预印本和官方代码。至少使用英文同义词，不要只搜 SOSR 这个自定义名称。

- 缺失与观测支持：irregular multivariate time series missingness sensor reliability forecasting；observation mask aware multivariate time series forecasting；missingness aware sensor graph message passing；observation support dynamic graph time series；freshness time since last observation sensor forecasting；target node missingness gate source node reliability。
- 多尺度与通道 token：multi-scale channel token time series forecasting missing data；channel-wise patching irregular time series；multi-resolution sensor representation learning；patch-level missingness multivariate forecasting。
- 动态图与路由：dynamic sensor graph conditioned on observations；adaptive inter-variable attention missing time series；reliability weighted graph neural network sensors；selective message passing missing nodes time series。
- 方法边界与对照：Raindrop irregular clinical time series graph；BiTGraph missing multivariate time series；S4M missing time series representation；ChannelTokenFormer missing patch；CRIB partial observation forecasting；Merlin teacher student missing time series。

检索至少覆盖 Google Scholar 或 Semantic Scholar、arXiv、Papers with Code/官方 GitHub。GitHub 只能证明代码实现存在，不能单独证明论文创新性。

## 4. 每篇候选文献的证据记录格式

| 字段 | 要求 |
|---|---|
| title/authors/year | 从论文首页或正式元数据核对 |
| DOI/arXiv/official URL | 至少一个可复核链接 |
| task/data | 任务、数据集、缺失类型 |
| input representation | 是否使用 mask、delta-time、freshness、support |
| temporal scale | 单尺度、patch、多尺度或连续时间 |
| cross-sensor mechanism | attention、graph、message passing、imputation 等 |
| target-deficit gate | 是否明确控制目标节点缺口 |
| source-reliability weighting | 是否明确调节源节点可靠性 |
| scale routing | 是否按局部支持选择尺度 |
| training objective | 监督、重构、蒸馏、一致性等 |
| closest overlap | 与 SOSR 重合的具体模块/公式/代码位置 |
| key difference | SOSR 仍可能不同的可检验差异 |
| evidence quality | 正式论文全文、预印本、代码或仅摘要 |
| verdict | 直接重合/部分重合/相关但不同/不相关 |

论文中的公式、方法图或代码路径必须给出页码、章节、图表或文件/函数定位。不能用“看起来相似”代替证据。

## 5. 创新性判定规则

使用四级判定，不使用“绝对原创”：直接重合；部分重合；组合创新候选；经验/应用贡献。只有当最近邻工作没有同时覆盖关键机制，且候选模块相对于容量匹配基线有稳定独立收益时，才允许建议“组合创新候选”。否则降级为应用或经验贡献。

## 6. 三数据集适配审查

### MetroPT-3

区分真实缺失与人工缺失；检查时间戳是否不规则；保持首月训练、后续测试的时间隔离；报告状态预测的逐传感器 MAE/RMSE。不能把人工 mask 压力测试写成真实部署缺失证据。

### C-MAPSS FD004

检查传感器是真实缺失还是人为遮挡；明确 setting、engine、cycle 的条件结构；保持 engine-level 划分；raw RUL、training cap、prediction output 不能混称。

### TEP

检查缺失是否由版本/文件协议决定；保持 run-level 划分；区分诊断概率、检测概率和故障延迟；不要把回归传感器插值机制直接等同于分类收益。

## 7. 输出格式

请输出 Markdown 报告：执行摘要；目标 12 篇候选文献证据表（其中重点精读 6 篇最接近方法）；按 SOSR 部件拆解的重合矩阵；最近邻逐篇比较；创新性分级；三数据集适配结论；包含 w/o SOSR、mask-only、global-mixer、w/o target deficit、w/o source reliability、w/o scale routing、容量匹配 MLP 的实验矩阵；所需图表；未解决问题；参考链接。每条链接必须对应报告中的具体判断。有效来源不足时如实报告，不凑数；未做新实验时只能判断新颖性风险和提出待验证贡献，不能声称实验支持已经存在。

## 8. 禁止事项

不要把 README 自我描述当作创新证明；不要把组合已有模块写成首次提出；不要复制外部代码或修改当前项目；不要使用测试集指标决定方向；不要把未运行实验写成结果；不要混淆三数据集的缺失、异步和标签语义；不要使用 hash、commit id 或机器绝对路径作为结论依据；无法访问全文时标注证据等级。

## 9. 最终建议格式

报告末尾必须给出：继续 SOSR 的最小证据；修改 SOSR 应删除或改造的部件；放弃 SOSR 的触发条件；不超过 40 字且可被实验否证的第四章推荐主线；下一步唯一优先实验。
