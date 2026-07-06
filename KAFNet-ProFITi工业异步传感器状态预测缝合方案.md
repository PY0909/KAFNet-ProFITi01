# KAFNet-ProFITi 工业异步传感器状态预测缝合方案

## 1. 方案定位

总题目可表述为：

**面向工业设备异步传感器状态预测的规整表示与联合概率预测方法研究**

本方案的目标不是把 KAFNet 和 ProFITi 两个完整模型顺序串起来，而是拆出二者最适合复用的部分：

- KAFNet 负责工业异步多传感器历史观测的规整表示学习。
- ProFITi 负责未来多传感器状态的条件联合概率分布建模。
- 额外增加工业状态风险头，将概率样本转化为可用于预警的风险分数、预测区间和状态概率。

推荐融合结构：

```text
工业事件序列
  -> IndustrialEventCollator
  -> KAFNetEncoder
  -> QueryConditionAdapter
  -> ProFITi NormalizingFlow
  -> Point / Interval / Risk outputs
```

其中 KAFNet 代码库的 `KAFNet/model/KAFNet.py` 已经包含三个可复用部件：`TTKMN` 时间核池化、变量维度 `FreqBlock`、时间嵌入 `_TE`。ProFITi 代码库的 `profiti/model.py` 和 `profiti/core/flow_layers.py` 已经提供 `NormalizingFlow`、joint NLL、marginal NLL、采样、CRPS 和 energy score 等概率预测接口。

## 2. 预测对象与工业场景定义

工业设备异步传感器状态预测可以拆成三个预测对象，方案实施时需要固定其中一个为主任务：

| 预测对象 | 输出形式 | 适合数据集 | 说明 |
|---|---|---|---|
| 关键传感器未来值 | 多变量连续值分布 | MetroPT-3, C-MAPSS, TEP | 例如压力、油温、电流、振动未来若干步联合分布 |
| 设备健康指标 | HI/RUL 的概率分布 | C-MAPSS | 适合退化过程明确的数据 |
| 状态风险 | 正常/预警/异常概率 | MetroPT-3, TEP | 需要故障记录、状态标签或阈值规则 |

建议主场景选 **MetroPT-3 空气压缩机预测性维护**。它更贴近工业设备多传感器运行状态，包含压力、油温、电流、阀门等信号，并有故障记录，适合从连续预测进一步构造风险预警任务。

## 3. 两个代码库的接口事实

### 3.1 KAFNet 可复用接口

KAFNet 的 `forecasting(tp_pred, X, tp_true, mask)` 接收：

```text
tp_pred: 预测时间点，形状通常为 (B, Lp)
X: 历史观测值，形状为 (B, L, N)
tp_true: 历史时间点，形状为 (B, L) 或 (B, L, N)
mask: 历史观测掩码，形状为 (B, L, N)
```

当前输出是点预测：

```text
y_pred: (1, B, Lp, N)
```

源码中的关键步骤：

- `TTKMN` 对每个变量沿时间进行 K 个高斯核池化，并拼接是否观测标志。
- `pre_conv` 先对每个变量的时间序列做轻量卷积。
- `_TE` 为历史和查询时间生成时间嵌入。
- `FreqBlock` 在变量维度上做频域线性注意力。
- `decoder` 把变量表示和查询时间嵌入拼接后输出点预测。

融合时保留 `TTKMN + feat_proj + FreqBlock + var_agg`，但不直接使用原始 `decoder` 作为最终输出头。

### 3.2 ProFITi 可复用接口

ProFITi 使用事件式批次：

```text
tx: 历史观测时间
cx: 历史观测通道 ID
x: 历史观测值
mx: 历史观测 mask
tq: 查询时间
cq: 查询通道 ID
y: 查询目标值
mq: 查询目标 mask
```

它的训练过程是：

```python
model.distribution(tx, cx, x, mx, tq, cq, mq)
loss = model.compute_njnll(y, mq).mean()
```

`distribution` 先得到 hidden states，`NormalizingFlow` 再对目标向量 `y` 计算联合负对数似然。其 flow 接口要求：

```text
y: (B, Kq)
hidden_states: (B, Kq, D)
mask: (B, Kq)
```

这里的 `Kq` 是一次预测中所有有效查询项的展平长度，例如 `预测步数 Lp x 传感器数 N`。

## 4. 总体缝合策略

### 4.1 不推荐的串联方式

不建议：

```text
KAFNet 点预测值 -> ProFITi 再预测
```

原因：

- KAFNet 点预测会压缩掉历史观测中的不确定性。
- ProFITi 的 flow 需要条件 hidden states，不是只需要一个点预测输入。
- 这种方式难以解释联合分布来自哪里，也难以做严格消融。

### 4.2 推荐的结构化融合方式

推荐：

```text
KAFNetEncoder(history) -> query-wise hidden states -> ProFITiFlow(target)
```

也就是把 KAFNet 从点预测模型改造成 **条件编码器**，把 ProFITi 从完整模型改造成 **条件 flow 解码器**。

核心新增模块：

| 模块 | 功能 | 来源 |
|---|---|---|
| `IndustrialEventCollator` | 把工业传感器表格转成事件序列和密集张量两种格式 | 新增 |
| `KAFNetEncoder` | 从历史异步观测得到变量级隐表示 `z_var` | 改造 KAFNet |
| `QueryConditionAdapter` | 将变量级表示扩展到查询项级别 `h_query` | 新增 |
| `ProFITiFlowHead` | 计算联合 NLL、采样和概率指标 | 复用 ProFITi |
| `RiskScoringHead` | 将预测样本转为阈值越界概率、状态概率、预警分数 | 新增 |

## 5. 张量协议设计

### 5.1 原始工业事件格式

每条设备样本统一为：

```text
record_id
device_id
operation_context: 工况向量，例如负载、转速、启停状态、环境温度
events:
  - time: float
    channel: int
    value: float
    miss_type: observed | natural_gap | sensor_offline | abnormal_dropout
target_window:
  - time: float
    channel: int
    value: float
    state_label: optional
```

### 5.2 给 KAFNet 的密集格式

```text
X_obs: (B, L, N)
T_obs: (B, L)
M_obs: (B, L, N)
T_q:   (B, Lp)
Y_q:   (B, Lp, N)
M_q:   (B, Lp, N)
C_ctx: (B, C)
```

`X_obs` 中没有观测的位置填 0，真实缺失由 `M_obs` 表示。不能把填充值当成真实观测。

### 5.3 给 ProFITi flow 的展平格式

```text
y_flat: (B, Kq)
mq_flat: (B, Kq)
h_query: (B, Kq, D)
```

其中：

```text
Kq = Lp * N
y_flat = flatten(Y_q)
mq_flat = flatten(M_q)
```

查询项顺序固定为：

```text
[(t1, ch1), (t1, ch2), ..., (t1, chN), (t2, ch1), ...]
```

这个顺序会影响 ProFITi 的 triangular attention 所学习到的联合依赖。工业场景下建议优先按“时间优先、通道次之”排序；如果更关注同一传感器的时间连续性，可在消融中加入“通道优先、时间次之”。

## 6. 模型改造细节

### 6.1 KAFNetEncoder

从 `KAFNet/model/KAFNet.py` 拆出编码部分：

```python
class KAFNetEncoder(nn.Module):
    def forward(self, X, T_obs, M_obs, context=None):
        # X: (B, L, N)
        # T_obs: (B, L)
        # M_obs: (B, L, N)
        # context: (B, C), optional
        # return z_var: (B, N, D)
```

编码流程对应 KAFNet 原始 `forecasting` 中 decoder 之前的部分：

```text
X, T_obs, M_obs
  -> per-variable pre_conv
  -> time embedding
  -> TTKMN kernel pooling
  -> feat_proj
  -> variable positional encoding
  -> FreqBlock x nlayer
  -> z_var: (B, N, D)
```

需要对原始 KAFNet 做两处修改：

1. 把 `forecasting` 中第 155 到 189 行左右的编码逻辑提取成 `encode(...)`。
2. 在 `encode(...)` 中加入工况条件 `context`，可采用 FiLM 或拼接投影：

```python
gamma, beta = context_proj(context).chunk(2, dim=-1)
z_var = z_var * (1 + gamma[:, None, :]) + beta[:, None, :]
```

这样工作量 1 中的“工况感知规整表示”可以落到具体模块，而不是只在输入端增加工况字段。

### 6.2 QueryConditionAdapter

KAFNetEncoder 输出的是变量级表示：

```text
z_var: (B, N, D)
```

ProFITi flow 需要查询项级表示：

```text
h_query: (B, Kq, D)
```

因此新增：

```python
class QueryConditionAdapter(nn.Module):
    def forward(self, z_var, T_q, channel_ids, context=None):
        # z_var: (B, N, D)
        # T_q: (B, Lp)
        # channel_ids: (Kq,)
        # return h_query: (B, Kq, D)
```

构造方式：

```text
h_query(t, c) =
  Linear([
    z_var[:, c, :],
    time_embedding(t),
    channel_embedding(c),
    context_embedding
  ])
```

这一步是 KAFNet 与 ProFITi 的核心接口。它把 KAFNet 的变量级规整表示，转化为 ProFITi 每个未来查询项的条件 hidden state。

### 6.3 ProFITiFlowHead

从 ProFITi 复用：

- `NormalizingFlow`
- `compute_jnll`
- `compute_mnll`
- `samples`
- `mse`
- `crps`
- `energy_score`

建议新增包装类：

```python
class ProFITiFlowHead(nn.Module):
    def __init__(self, latent_dim, f_layers, marginal_training, device):
        self.flow = NormalizingFlow(f_layers, latent_dim, marginal_training, device)
        self.jnll = compute_jnll()
        self.mnll = compute_mnll()

    def nll(self, y_flat, h_query, mq_flat):
        z, ldj = self.flow.forward(y_flat, h_query, mq_flat)
        return self.jnll(z, mq_flat, ldj) / mq_flat.sum(dim=-1).clamp_min(1)

    def sample(self, h_query, mq_flat, nsamples):
        z = torch.randn(B * nsamples, Kq, device=device)
        y, log_det = self.flow.inverse(z, repeated_h_query, repeated_mask)
        return y.reshape(B, nsamples, Kq)
```

注意：ProFITi 原始 `ProFITi` 类内部有 `conditioning_module = GraFITi(...)`。融合模型中不再使用该 conditioning module，而是直接把 KAFNet 生成的 `h_query` 送入 `NormalizingFlow`。

### 6.4 RiskScoringHead

从 flow 采样得到：

```text
samples: (B, S, Lp, N)
```

可构造三类工业风险输出：

```text
P_upper(b, t, n) = P(sample[b, :, t, n] > upper_limit[n])
P_lower(b, t, n) = P(sample[b, :, t, n] < lower_limit[n])
P_any_risk(b) = P(any sensor crosses threshold within horizon)
```

状态分类可以定义为：

```text
normal: P_any_risk < tau_1
warning: tau_1 <= P_any_risk < tau_2
critical: P_any_risk >= tau_2
```

如果数据集有故障标签，则用标签监督风险头；如果没有故障标签，只能报告阈值越界概率和预测区间，不能声称完成故障预警。

## 7. 三个工作量的具体落地

### 工作量 1：工业异步传感器数据的事件化建模与工况感知规整表示

目标：解决工业多传感器采样频率不同、缺失机制不同、直接插值会产生伪观测的问题。

代码落点：

```text
industrial/
  datasets/
    metropt3.py
    cmapss.py
    tep.py
  collate.py
  missing_simulator.py
  context_encoder.py
models/
  kafnet_encoder.py
```

关键实现：

1. `IndustrialEventDataset` 读取原始表格，生成 `events`。
2. `IndustrialEventCollator` 同时输出 KAFNet 密集张量和 ProFITi 展平目标。
3. `MissingMechanismSimulator` 支持三种缺失构造：
   - 低频采样：不同传感器设置不同保留周期。
   - 传感器离线：连续时间段整段缺失。
   - 异常中断：故障前后提高缺失概率。
4. `ContextEncoder` 编码负载、转速、启停状态等工况变量。
5. `KAFNetEncoder` 在变量表示上注入工况条件。

可写成创新点：

**提出面向工业异步多传感器的工况感知规整表示方法，在避免固定插值伪观测的同时保留变量间时序相关性。**

实验验证：

| 对照方法 | 目的 |
|---|---|
| Linear interpolation + Transformer | 验证固定插值是否引入误差 |
| Forward fill + GRU/TCN | 验证简单填补策略 |
| GRU-D | 验证缺失衰减机制 |
| KAFNet without context | 验证工况条件 |
| KAFNetEncoder with context | 验证本工作量 |

### 工作量 2：KAFNet-ProFITi 联合概率状态预测模型

目标：从点预测扩展到未来多传感器联合概率分布，建模变量之间的共同不确定性。

代码落点：

```text
models/
  kaf_profiti.py
  query_condition_adapter.py
  profiti_flow_head.py
train_industrial_kaf_profiti.py
evaluate_industrial_kaf_profiti.py
```

推荐总模型：

```python
class KAFProFITi(nn.Module):
    def __init__(self, args):
        self.encoder = KAFNetEncoder(args)
        self.adapter = QueryConditionAdapter(args)
        self.flow_head = ProFITiFlowHead(args)

    def distribution(self, batch):
        z_var = self.encoder(
            batch.X_obs,
            batch.T_obs,
            batch.M_obs,
            batch.context,
        )
        h_query = self.adapter(
            z_var,
            batch.T_q,
            batch.query_channel_ids,
            batch.context,
        )
        self.hidden_states = h_query

    def loss(self, batch):
        self.distribution(batch)
        return self.flow_head.nll(
            batch.y_flat,
            self.hidden_states,
            batch.mq_flat,
        ).mean()

    def sample(self, batch, nsamples=100):
        self.distribution(batch)
        return self.flow_head.sample(
            self.hidden_states,
            batch.mq_flat,
            nsamples,
        )
```

训练目标：

```text
L = L_joint_nll
  + lambda_point * L_point
  + lambda_calib * L_calib
```

其中：

- `L_joint_nll`：ProFITi joint normalized negative log-likelihood。
- `L_point`：样本均值与真实值之间的 MAE/MSE，稳定训练。
- `L_calib`：预测区间覆盖率或分位数校准损失，可选。

建议第一阶段先只使用：

```text
L = L_joint_nll + 0.1 * L_point
```

可写成创新点：

**构建规整表示驱动的条件流概率预测模型，实现工业设备多传感器状态的联合分布预测与不确定性估计。**

消融实验：

| 模型 | 说明 |
|---|---|
| KAFNet point | 原始 KAFNet 点预测头 |
| ProFITi original | 原始 GraFITi conditioning + flow |
| KAFNetEncoder + Gaussian head | 只做高斯概率输出 |
| KAFNetEncoder + marginal flow | 独立变量概率预测 |
| KAFNetEncoder + joint flow | 完整融合模型 |
| KAFNetEncoder + context + joint flow | 加入工况条件的完整模型 |

### 工作量 3：面向工业运维的概率风险预警与鲁棒性评估

目标：把未来多变量分布转成可解释的设备状态风险，而不是只报告误差。

代码落点：

```text
industrial/
  risk.py
  thresholds.py
  calibration.py
  early_warning.py
```

风险分数：

```text
risk_sensor[n] = P(Y_{t:t+H,n} > upper_n or Y_{t:t+H,n} < lower_n)
risk_device = 1 - product_n(1 - risk_sensor[n])
```

如果数据集包含故障时间 `t_fault`，提前预警定义为：

```text
预测窗口 [t, t + H] 中 risk_device >= tau，且 t < t_fault
lead_time = t_fault - t
```

可写成创新点：

**设计面向工业设备状态预测的概率风险评估机制，将联合分布预测结果转化为可解释的状态预警指标。**

评估指标：

| 指标 | 含义 |
|---|---|
| MAE/RMSE | 样本均值点预测误差 |
| NLL | 概率密度质量 |
| CRPS | 分布预测质量 |
| PICP/MPIW | 预测区间覆盖率与区间宽度 |
| ECE | 概率校准误差 |
| F1/AUROC/AUPRC | 状态预警性能 |
| Lead time | 提前预警时间 |

## 8. 数据集适配方案

### 8.1 MetroPT-3

角色：主数据集。

任务设置：

- 输入：过去 24h 或 48h 的压力、油温、电流、阀门等传感器事件。
- 输出：未来 1h/6h/12h 的关键传感器联合分布。
- 风险标签：根据故障记录构造故障前窗口，或根据传感器安全阈值构造风险状态。

异步构造：

- 压力类传感器保留较高采样率。
- 温度类传感器降采样。
- 电流类传感器加入短时离线片段。
- 故障前一段时间加入异常中断缺失机制。

### 8.2 C-MAPSS

角色：退化预测验证数据集。

任务设置：

- 输入：历史多传感器运行序列和 operating conditions。
- 输出：未来传感器值或 RUL/HI 分布。
- 工况变量：C-MAPSS 中的 operating settings 可直接作为 `context`。

注意：

- C-MAPSS 是仿真退化数据，工业真实性和 MetroPT-3 不同。
- 适合验证工况条件和概率 RUL，不宜单独支撑“真实工业故障预警”。

### 8.3 Tennessee Eastman Process

角色：过程工业故障状态验证数据集。

任务设置：

- 输入：过程变量和操纵变量历史窗口。
- 输出：未来过程变量联合分布或故障状态。
- 标签：TEP 有多类故障，可用于故障检测/分类。

注意：

- TEP 多为规则采样仿真数据。
- 需要人工构造传感器异步和缺失机制。

## 9. 实验矩阵

### 9.1 主实验

| 数据集 | 任务 | 预测长度 | 指标 |
|---|---|---|---|
| MetroPT-3 | 传感器值 + 风险预警 | 1h/6h/12h | MAE, RMSE, NLL, CRPS, AUROC, Lead time |
| C-MAPSS | RUL/HI 概率预测 | 10/20/30 cycles | MAE, RMSE, NLL, CRPS, PICP |
| TEP | 过程变量预测 + 故障状态 | 1/5/10 steps | MAE, RMSE, NLL, F1, AUROC |

### 9.2 异步缺失鲁棒性实验

| 缺失机制 | 设置 |
|---|---|
| 随机缺失 | 10%, 30%, 50%, 70% |
| 低频采样 | 不同传感器使用不同采样间隔 |
| 连续离线 | 随机选择传感器连续缺失 1h/6h |
| 故障相关缺失 | 故障前后提高缺失概率 |

### 9.3 消融实验

| 消融项 | 对应问题 |
|---|---|
| 去掉工况条件 | 工况感知是否有效 |
| 去掉 KAFNetEncoder，换 GraFITi | KAFNet 表示是否有效 |
| 去掉 joint flow，换 Gaussian head | flow 是否必要 |
| joint flow 改 marginal flow | 联合分布是否必要 |
| 改变查询排序 | triangular attention 对变量依赖是否敏感 |
| 不同缺失机制单独测试 | 方法适用于哪类异步缺失 |

## 10. 代码实施路线

### 阶段 A：最小可运行融合

目标：在 ProFITi 代码库里复用 KAFNetEncoder 和 NormalizingFlow，先跑通一个公开数据集。

步骤：

1. 将 KAFNet 的 `TTKMN`、`FreqBlock`、`KAFNet` 编码逻辑复制或封装到 `profiti/models/kafnet_encoder.py`。
2. 新增 `models/kaf_profiti.py`。
3. 写 `IndustrialBatch`，包含：

```python
X_obs, T_obs, M_obs, T_q, Y_q, M_q, context
y_flat, mq_flat, query_channel_ids
```

4. 使用 `NormalizingFlow.forward(y_flat, h_query, mq_flat)` 计算 joint NLL。
5. 用样本均值计算 MAE/RMSE，用 ProFITi 原有 CRPS 逻辑计算分布指标。

### 阶段 B：工业数据集适配

目标：完成 MetroPT-3、C-MAPSS、TEP 的统一数据协议。

步骤：

1. 每个数据集写独立 reader。
2. 统一输出事件序列。
3. 写缺失构造器，保证 train/valid/test 使用固定随机种子。
4. 保存异步构造日志，包括每个传感器实际保留率。

### 阶段 C：风险预警层

目标：把概率预测转为工业状态风险。

步骤：

1. 定义传感器上下限阈值或从训练集分位数估计阈值。
2. 从 `samples` 计算越界概率。
3. 如果有故障标签，计算 AUROC/AUPRC/F1 和提前预警时间。
4. 如果没有故障标签，只报告风险分数和区间覆盖，不写故障检测结论。

## 11. 预计文件改造清单

建议以 ProFITi 为主工程，因为它已经有概率训练和评估框架；将 KAFNet 作为编码器移入。

```text
ProFITi/
  profiti/
    models/
      kafnet_encoder.py              # 新增：KAFNet 编码器
      query_condition_adapter.py     # 新增：查询条件适配
      kaf_profiti.py                 # 新增：融合模型
    industrial/
      datasets/
        metropt3.py                  # 新增
        cmapss.py                    # 新增
        tep.py                       # 新增
      collate.py                     # 新增
      missing_simulator.py           # 新增
      risk.py                        # 新增
  train_industrial_kaf_profiti.py    # 新增
  eval_industrial_kaf_profiti.py     # 新增
  configs/
    metropt3.yaml                    # 新增
    cmapss.yaml                      # 新增
    tep.yaml                         # 新增
```

如果后续真正编码，建议不要在 Windows 不区分大小写挂载目录里直接开发 ProFITi。当前克隆时出现过 `PROFITI.cpython-310.pyc` 与 `profiti.cpython-310.pyc` 的大小写碰撞警告，正式实现应放在 Linux 原生区分大小写路径中。

## 12. 开题创新点表述

### 创新点一

**提出工况感知的工业异步多传感器规整表示方法。**

该方法将工业设备观测建模为包含变量 ID、时间戳、观测值、缺失类型和工况标签的事件序列，基于 KAFNet 的时间核池化与变量频域注意力形成可学习规整表示，避免固定插值引入伪观测。

### 创新点二

**提出基于 KAFNet-ProFITi 融合的联合概率状态预测模型。**

该模型以 KAFNetEncoder 生成变量级历史表示，通过 QueryConditionAdapter 转换为未来查询项级条件 hidden states，并接入 ProFITi 条件 normalizing flow，对未来多个传感器状态进行联合分布建模。

### 创新点三

**提出面向工业运维的概率风险预警机制。**

该机制从联合预测样本中估计未来窗口内关键传感器越界概率、设备级风险概率和提前预警时间，使模型输出从误差指标扩展到可用于运维分析的风险指标。

## 13. 主要风险与边界

| 风险 | 影响 | 处理方式 |
|---|---|---|
| 公开数据多数为规则采样 | 异步缺失创新可能被质疑 | 实验协议中明确构造异步缺失，并报告构造机制 |
| ProFITi joint flow 维度过高 | 训练不稳定、显存压力大 | 控制预测窗口和传感器数量，或按传感器组分组建模 |
| 没有故障标签 | 无法验证故障预警 | 只做连续状态概率预测和阈值风险，不写故障检测结论 |
| KAFNet 与 ProFITi 接口不同 | 工程改造复杂 | 以 ProFITi 为主工程，KAFNet 只作为 encoder 迁入 |
| 工况变量不可得 | 工况感知模块无法验证 | 用 C-MAPSS operating settings 或 TEP operating modes 验证 |

## 14. 最小可行实验配置

第一轮实验建议只做：

```text
数据集：MetroPT-3
输入窗口：24h
预测窗口：1h / 6h
传感器数：先选 8-15 个关键传感器
缺失设置：随机缺失 30% + 传感器离线 1h
训练目标：joint NLL + 0.1 * MSE
评估：MAE, RMSE, NLL, CRPS, PICP, AUROC
```

第一轮模型：

```text
1. Forward fill + Transformer
2. GRU-D
3. KAFNet point
4. ProFITi original
5. KAFNetEncoder + Gaussian head
6. KAFNetEncoder + ProFITi joint flow
7. KAFNetEncoder + context + ProFITi joint flow
```

如果第 6、7 项相对第 3、4、5 项在 NLL/CRPS/PICP 上有稳定提升，再展开 C-MAPSS 与 TEP。

## 15. 方案结论边界

本方案能够形成一个完整的开题研究框架，但有两个前提需要在正式开题前固定：

1. 主预测对象必须明确：关键传感器值、RUL/HI、故障状态三者不能混写。
2. 异步缺失必须有可复现实验协议：如果公开数据是规则采样，需要明确人工异步构造方式、随机种子和缺失机制。

在这两个前提成立时，KAFNet + ProFITi 的缝合不是简单模块叠加，而是形成了清晰的三层贡献：

```text
规整表示层：KAFNetEncoder
联合概率层：ProFITiFlowHead
工业风险层：RiskScoringHead
```



| 名称 | 依据 |
|---|---|
| **FreqFlow** | 编码端有 `FreqBlock` / `FreqLinearAttention`，输出端是 normalizing flow |
| **QueryFlowNet** | `QueryConditionAdapter` 将查询时间、通道 id、context 条件化后送入 flow |
| **VarFlow** | 主体是变量级表示 `z_var` 加联合流建模 |
| **TriFlowNet** | flow head 里核心相关结构是 `TriangularAttention` 和 lower-triangular transformation |
| **SpectraFlow** | 频域注意力 + 概率流，名称更偏论文风格 |
| **CoraFlow** | 可解释为 Conditioned Correlated Flow，强调条件化联合分布 |
| **VQ-Flow** | Variable-Query Flow，贴合变量级编码和 query-conditioned head |
| **DySpecFlow** | Dynamic Spectral Flow，突出时间查询和频域块 |



数据集
TEP：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1
CMAPSS：https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
MetroPT-3：https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset

## 16. 当前代码实现版说明

当前可运行工程位于：

```text
/home/work/new_work/code
```

统一训练入口为：

```text
/home/work/new_work/code/run_experiment.py
```

当前启用的两个模型名：

| 模型名 | 状态 | 说明 |
|---|---|---|
| `kaf_profiti_joint` | enabled | 早期 KAFNetEncoder + ProFITi joint flow 实现 |
| `kst_probflow` | enabled | 当前主模型，使用 MultiScaleKAFEncoder、动态 sensor graph、low-rank probability head、quantile head、risk head |

当前建议以 `kst_probflow` 作为论文主模型。其结构可以写为：

```text
X_obs, T_obs, M_obs, context
  -> MultiScaleKAFEncoder
  -> DynamicSensorGraphBlock
  -> QueryConditionAdapter
  -> LowRankCopulaFlowHead
  -> QuantileHead
  -> RiskHead
```

与早期方案相比，当前实现已经不再使用独立 `TemporalPatchEncoder`，而是把多尺度 patch 信息合入 `MultiScaleKAFEncoder`。这样可以避免“编码器堆叠过多”的问题，并保留长历史、多尺度趋势和缺失统计信息。

## 17. 模型架构及相关原理

本节对应当前代码中的 `kst_probflow` 实现，主要文件为：

```text
/home/work/new_work/code/kaf_profiti/models/kst_probflow.py
/home/work/new_work/code/kaf_profiti/models/kafnet_encoder.py
/home/work/new_work/code/kaf_profiti/models/query_condition_adapter.py
```

### 17.1 总体数据流

模型输入是经过数据集 reader 和 `IndustrialCollator` 整理后的批次：

```text
X_obs:   (B, L, N)       历史观测值，缺失位置填 0
T_obs:   (B, L)          历史时间索引
M_obs:   (B, L, N)       历史观测 mask
T_q:     (B, Lp)         未来查询时间
Y_q:     (B, Lp, N)      未来监督目标
M_q:     (B, Lp, N)      未来目标 mask
context: (B, C)          工况或上下文变量
y_flat:  (B, Kq)         展平后的未来目标
mq_flat: (B, Kq)         展平后的未来目标 mask
```

其中：

```text
B  = batch size
L  = history_len
Lp = pred_len
N  = sensor 数量
C  = context 维度
Kq = Lp * N
```

模型内部流程为：

```text
历史观测与缺失 mask
  -> MultiScaleKAFEncoder
  -> z_var: (B, N, D)
  -> DynamicSensorGraphBlock
  -> z_graph: (B, N, D)
  -> QueryConditionAdapter
  -> h_query: (B, Kq, D)
  -> LowRankCopulaFlowHead / QuantileHead / RiskHead
```

这里 `z_var` 是变量级历史表示，`h_query` 是面向每个未来查询项的条件表示。模型的核心思想是先把不规则、缺失的历史窗口压缩为每个传感器的规整表示，再根据未来时间和通道 ID 生成查询条件，最后输出未来多传感器的概率分布、预测区间和风险分数。

### 17.2 MultiScaleKAFEncoder：多尺度 KAF 历史编码器

`MultiScaleKAFEncoder` 是当前模型的主干编码器，来源于 KAFNet 编码思想，但做了两点改造：

1. 将原 KAFNet 的全局时间核池化扩展为全局 + 局部多尺度 patch 编码。
2. 将 patch 统计量直接并入 KAF 编码器，删除独立 `TemporalPatchEncoder`。

编码器输入：

```text
X_obs:   (B, L, N)
T_obs:   (B, L) 或 (B, L, N)
M_obs:   (B, L, N)
context: (B, C)
```

输出：

```text
z_var: (B, N, D)
```

#### 17.2.1 TTKMN 时间核池化

TTKMN 可以理解为一组可学习的时间核池化器。它在归一化时间轴上设置 `K` 个可学习核中心 `c_k`，并学习每个核的尺度 `alpha_k`。对每个传感器的历史序列，TTKMN 根据时间距离给观测点分配权重：

```text
w_{l,k} = exp(-0.5 * (t_l - c_k)^2 / alpha_k^2) * m_l
```

其中 `m_l` 是观测 mask。经过归一化后：

```text
a_{l,k} = w_{l,k} / sum_l w_{l,k}
```

每个核得到一个时间池化值：

```text
p_k = sum_l a_{l,k} * x_l
```

最后拼接该传感器在窗口内是否存在观测的标志位：

```text
TTKMN(x, t, m) = [p_1, ..., p_K, observed_flag]
```

这个设计的作用是：不用固定插值，也能把不同时间位置、不同缺失模式下的观测压缩为固定长度表示。mask 直接参与权重计算，缺失位置不会作为真实观测进入池化。

#### 17.2.2 预卷积与时间嵌入

在时间核池化前，模型先对每个传感器的历史序列做轻量 `Conv1d`：

```text
X -> Conv1d(1, preconv_dim, kernel=3) -> ReLU -> Conv1d(preconv_dim, 1)
```

这一步用于提取短期局部变化，例如压力突变、温度缓慢漂移、电流短期波动等。随后将时间嵌入投影到一维并加到观测值上：

```text
X_enhanced = Conv1d(X) + Linear(TimeEmbedding(T))
```

当前时间嵌入由线性项、正弦项和余弦项组成：

```text
TE(t) = [linear(t), sin(W_s t), cos(W_c t)]
```

它的作用是让模型区分同样的观测值出现在历史窗口早期还是近期。

#### 17.2.3 全局表示与局部多尺度表示

全局分支在完整历史窗口上做一次 KAF 池化：

```text
z_global = KAFPool(X_obs, T_obs, M_obs)
```

局部分支按照 `patch_lens` 切分历史窗口，例如 MetroPT 默认：

```text
patch_lens = [12, 24, 48]
```

对每个 patch，模型同时提取两类 token：

```text
local_kaf = KAFPool(patch_x, patch_t, patch_m)
stats = [mean, std, slope, missing_ratio, time_span]
```

其中：

| 统计量 | 含义 |
|---|---|
| `mean` | patch 内有效观测均值 |
| `std` | patch 内有效观测标准差 |
| `slope` | patch 内相对时间上的线性趋势 |
| `missing_ratio` | patch 内缺失比例 |
| `time_span` | patch 覆盖的时间跨度 |

局部 token 由两部分相加后归一化：

```text
token_patch = LayerNorm(local_kaf + MLP(stats))
```

多个 patch token 通过 gated pooling 聚合。模型为每个 patch token 计算分数：

```text
score_i = Linear(token_i)
weight_i = softmax(score_i)
z_local = sum_i weight_i * token_i
```

这一步的作用是：让模型在不同传感器和不同样本上自动选择更有用的局部尺度。例如压力类变量可能更依赖短 patch，温度类变量可能更依赖长 patch。

#### 17.2.4 全局-局部门控融合

全局表示和局部表示不直接拼接，而是用门控残差融合：

```text
gate = sigmoid(MLP([z_global, z_local]))
z = gate * z_global + (1 - gate) * z_local
```

其中 `gate` 的形状为 `(B, N, D)`，因此每个样本、每个传感器、每个隐维度都可以自适应地选择全局或局部信息。这个设计对应当前问题中的一个关键现象：长历史有助于降低 RMSE，但少量局部异常窗口仍会造成大误差，因此编码器需要同时保留长周期趋势和局部状态变化。

#### 17.2.5 变量频域交互 FreqBlock

融合后的变量表示进入 `FreqBlock`。`FreqBlock` 的输入和输出均为：

```text
(B, N, D)
```

它先在变量维度上加入位置编码，再用频域线性注意力建模跨传感器依赖。其核心思想是把变量表示映射到频域后，用随机傅里叶特征近似注意力计算：

```text
Q, K, V = Linear(FFT(z))
phi(Q), phi(K) = random Fourier features
Attention(Q,K,V) ≈ phi(Q) * [phi(K)^T V] / [phi(Q) * sum(phi(K))]
```

相比普通全量注意力，这种线性注意力可以降低变量交互计算开销，同时保留变量间相关性。对工业设备而言，不同传感器通常存在同步变化或滞后耦合，例如压力、电流、阀门状态、温度之间的关系。`FreqBlock` 用于在变量级表示中显式建模这类跨通道依赖。

#### 17.2.6 工况条件 FiLM

如果数据集提供工况变量，例如 C-MAPSS 的 operating settings 或 MetroPT 中的阀门/压缩机状态，编码器会通过 FiLM 方式注入 context：

```text
gamma, beta = Linear(context).chunk(2)
z = z * (1 + gamma) + beta
```

这种方式不是简单把工况拼到输入值上，而是在变量表示层调制每个传感器的隐状态。它适合 C-MAPSS FD002/FD004 这类多工况数据，因为同一个传感器值在不同 operating condition 下可能代表不同健康状态。

### 17.3 DynamicSensorGraphBlock：动态传感器图

编码器输出 `z_var` 后，模型使用动态图模块进一步建模传感器之间的状态依赖：

```text
z_graph = DynamicSensorGraphBlock(z_var)
```

动态图邻接矩阵由两部分组成：

```text
A_dynamic = softmax(Q z * (K z)^T / sqrt(D))
A_static  = softmax(A_learnable)
A         = 0.5 * A_dynamic + 0.5 * A_static
```

其中：

| 矩阵 | 含义 |
|---|---|
| `A_dynamic` | 每个样本自适应生成的传感器关系 |
| `A_static` | 训练得到的全局传感器关系 |
| `A` | 动态关系和静态关系的融合 |

图消息传递为：

```text
message = A * V(z)
gate = sigmoid(Linear([z, message]))
h = gate * z + (1 - gate) * message
z_graph = LayerNorm(h + z)
```

该模块的作用是补充 KAF 编码器中的变量交互：`FreqBlock` 偏向全局变量频域混合，动态图模块偏向样本级 sensor dependency。对于工业系统，故障或异常通常不会只体现在单个传感器上，而是多个变量共同偏移；动态图模块用于捕捉这种联动关系。

### 17.4 QueryConditionAdapter：从变量表示到未来查询表示

`MultiScaleKAFEncoder` 和动态图模块输出的是每个传感器一个表示：

```text
z_graph: (B, N, D)
```

概率头需要的是每个未来查询项一个条件 hidden state：

```text
h_query: (B, Kq, D)
```

`QueryConditionAdapter` 将变量表示扩展为查询表示：

```text
h(t, c) = MLP([
  z_graph[:, c, :],
  TimeEmbedding(t),
  ChannelEmbedding(c),
  ContextEmbedding(context)
])
```

查询顺序默认是时间优先、通道次之：

```text
(t1, ch1), (t1, ch2), ..., (t1, chN),
(t2, ch1), (t2, ch2), ..., (t2, chN)
```

这种排序的含义是：每个未来时刻内先排列所有传感器，再进入下一个未来时刻。它适合联合建模同一时刻不同传感器之间的相关性，也与 `y_flat = flatten(Y_q)` 的监督目标保持一致。

### 17.5 LowRankCopulaFlowHead：边际分布与低秩联合相关

当前 `kst_probflow` 没有继续使用早期 full triangular flow 作为默认概率头，而是使用更稳定的低秩概率头。原因是 MetroPT 中：

```text
Kq = pred_len * num_sensors = 24 * 15 = 360
```

如果直接使用 full triangular flow，逆变换阶段可能出现数值放大和 NLL 长尾。低秩概率头把概率预测拆成两个层次：

1. 每个查询点的边际位置和尺度。
2. 通过低秩共享因子注入查询项之间的相关性。

概率头从 `h_query` 输出：

```text
loc:     (B, Kq)
scale:   (B, Kq)
factors: (B, Kq, R)
```

其中 `R = copula_rank`，默认 `32`。边际分布采用 Student-t 形式，默认自由度 `df=6`，其负对数似然为：

```text
residual = (y - loc) / scale
log_prob = log StudentT(residual; df) - log(scale)
nll = -sum(log_prob * mask) / sum(mask)
```

Student-t 相比 Gaussian 有更厚的尾部，对工业传感器中的尖峰和异常波动更稳健。

采样时，模型同时生成独立噪声和低秩共享噪声：

```text
eps    ~ N(0, I), shape = (B, S, Kq)
shared ~ N(0, I), shape = (B, S, R)
correlated = shared @ factors^T
noise = eps + correlated
y_sample = loc + scale * noise
```

这样得到的样本仍然是 `(B, S, Kq)`，但不同查询项之间会通过 `shared` 和 `factors` 产生相关性。该设计保留联合概率建模能力，同时避免 full triangular inverse 在高维查询下的数值不稳定。

模型还设置：

```text
sample_clip = 30.0
min_scale >= 0.05
```

它们用于限制标准化空间中的极端采样值，避免少量异常样本污染 MAE、RMSE、CRPS、PICP 等指标。

### 17.6 QuantileHead：分位数区间预测

除采样分布外，模型还并行输出分位数：

```text
quantiles = [0.025, 0.1, 0.5, 0.9, 0.975]
```

`QuantileHead` 的输出形状为：

```text
(B, 5, Kq)
```

为了保证分位数单调，模型先输出一个 base，再用 `softplus` 生成非负增量：

```text
q_1 = base
q_i = q_{i-1} + softplus(delta_i)
```

训练损失使用 pinball loss：

```text
L_q = mean(max(q * error, (q - 1) * error))
```

其中：

```text
error = y - q_pred
```

分位数头的作用是直接优化预测区间覆盖和校准。实际评估中，`PICP/MPIW` 可以来自采样分布，也可以来自 quantile head。当前代码会同时记录：

```text
sample_picp, sample_mpiw
quantile_picp, quantile_mpiw
picp, mpiw
```

其中最终 `picp/mpiw` 会结合 validation conformal 校准结果。

### 17.7 RiskHead：辅助风险评分

风险头不是主预测目标，而是把预测分布转成窗口级风险概率。输入包括：

```text
z_summary      = mean(z_var over sensors)
mean_abs       = mean(abs(predicted_mean))
variance_mean  = mean(predicted_variance)
quantile_width = mean(q_0.975 - q_0.025)
```

模型先估计窗口正常概率：

```text
normality = sigmoid(MLP([z_summary, mean_abs, variance_mean, quantile_width]))
```

再得到风险分数：

```text
risk = 1 - normality
```

风险标签根据数据集构造：

| 数据集 | 风险标签 |
|---|---|
| MetroPT-3 | 未来窗口是否落入故障报告区间或预警窗口 |
| C-MAPSS | `RUL <= risk_threshold` |
| TEP | fault label 或故障注入后的窗口 |

风险损失采用 focal BCE：

```text
L_risk = alpha * (1 - p_t)^gamma * BCE(risk, label)
```

当前默认：

```text
gamma = 2.0
alpha = 0.75
```

这是因为工业故障样本通常稀少，普通 BCE 容易被大量正常窗口主导。focal loss 会提高难分类样本和正类风险窗口的权重。

正式评估时，风险阈值不使用 test label 搜索。当前默认采用 validation 正常样本分位数，也就是“报警预算”：

```text
risk_threshold = quantile(validation_normal_risk_scores, 0.95)
```

该设定表示：在 validation 正常窗口上允许约 5% 的窗口触发报警。这样做可以避免用 test label 选择阈值造成数据泄露。

### 17.8 训练目标

当前 `kst_probflow` 的总损失为：

```text
L = L_nll
  + lambda_point * L_point
  + lambda_quantile * L_quantile
  + lambda_risk * L_risk
```

默认配置：

```text
lambda_point = 0.5
lambda_quantile = 0.2
lambda_risk = 0.05
```

各项含义如下：

| 损失项 | 作用 |
|---|---|
| `L_nll` | 优化未来目标在预测分布下的概率密度 |
| `L_point` | 用预测均值的 MSE 稳定点预测 |
| `L_quantile` | 直接约束预测区间和分位数校准 |
| `L_risk` | 辅助训练窗口级风险分数 |

其中 `L_nll` 和 `L_point/L_quantile` 面向 sensor forecasting 主任务，`L_risk` 是辅助任务。论文表述中应保持这个主次关系：模型首先是概率预测模型，风险头是为了把预测分布转成工业预警信号。

### 17.9 推理输出

模型推理时输出三类结果：

| 输出 | 形状 | 文件 |
|---|---|---|
| 预测均值 | `(num_windows, Kq)` | `mean_seed{seed}.npy` |
| 预测样本 | `(num_windows, nsamples, Kq)` | `samples_seed{seed}.npy` |
| 风险分数 | `(num_windows,)` | `risk_seed{seed}.npy` |

评估指标从这些输出得到：

```text
mean -> MAE, RMSE
samples -> CRPS, sample PICP/MPIW
quantiles -> quantile PICP/MPIW
risk -> AUROC, AUPRC, F1, ECE
```

若 test split 没有风险正类，`AUROC/AUPRC/F1` 应视为不可用；此时只能分析预测指标和风险分数趋势，不能声称完成故障预警评估。

## 18. 如何训练

### 18.1 MetroPT-3 风险可评估协议

`metropt3_chrono_502030` 是 chronological 50/20/30 划分。该协议不称为 UCI 官方协议，而是为了保证 test split 中存在故障正类，便于评估风险预警指标。

```bash
cd /home/work/new_work

TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 /root/anaconda3/bin/conda run -n torch23 python -u /home/work/new_work/code/run_experiment.py \
  --dataset metropt3_chrono_502030 \
  --model kst_probflow \
  --seed 2026 \
  --missing-rate 0.3 \
  --missing-mode mixed \
  --history-len 168 \
  --pred-len 24 \
  --stride 60 \
  --data-root /home/work/new_work/dataset \
  --output-dir /home/work/new_work/result \
  --run-id metropt3_chrono502030_mskaf_$(date +%Y%m%d_%H%M%S) \
  --epochs 50 \
  --batch-size 128 \
  --max-train-batches 0 \
  --max-eval-batches 0 \
  --nsamples 100 \
  --device cuda \
  --hidden-dim 64 \
  --te-dim 10 \
  --kernel-count 4 \
  --n-layers 2 \
  --n-heads 2 \
  --preconv-dim 16 \
  --patch-lens 12,24,48 \
  --graph-layers 1 \
  --copula-rank 32 \
  --lambda-point 0.5 \
  --lambda-quantile 0.2 \
  --lambda-risk 0.05 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --sample-clip 30.0 \
  --attention-diag-floor 0.05
```

训练过程中每个 epoch 会输出一行 JSON，例如：

```json
{
  "event": "epoch",
  "epoch": 10,
  "train_loss": 0.31,
  "valid_nll": 0.44,
  "valid_mae": 0.61,
  "valid_rmse": 1.27,
  "valid_crps": 0.49,
  "valid_picp": 0.93
}
```

默认保存两个 checkpoint：

```text
checkpoint_seed2026.pt       # 最后一个 epoch
checkpoint_seed2026_best.pt  # validation CRPS 最优
```

最终 test 默认使用 `checkpoint_seed2026_best.pt`。

### 18.2 C-MAPSS FD004

FD004 复杂度最高，包含 6 个 operating settings 工况组合和 2 种故障模式，适合验证工况感知和跨工况退化预测。

```bash
cd /home/work/new_work

TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 /root/anaconda3/bin/conda run -n torch23 python -u /home/work/new_work/code/run_experiment.py \
  --dataset cmapss_fd004 \
  --model kst_probflow \
  --seed 2026 \
  --missing-rate 0.3 \
  --missing-mode mixed \
  --history-len 50 \
  --pred-len 10 \
  --stride 1 \
  --data-root /home/work/new_work/dataset \
  --output-dir /home/work/new_work/result \
  --run-id cmapss_fd004_mskaf_$(date +%Y%m%d_%H%M%S) \
  --epochs 80 \
  --batch-size 128 \
  --max-train-batches 0 \
  --max-eval-batches 0 \
  --nsamples 100 \
  --device cuda \
  --hidden-dim 64 \
  --te-dim 10 \
  --kernel-count 4 \
  --n-layers 2 \
  --n-heads 2 \
  --preconv-dim 16 \
  --patch-lens 5,10,20 \
  --graph-layers 1 \
  --copula-rank 32 \
  --lambda-point 0.5 \
  --lambda-quantile 0.2 \
  --lambda-risk 0.05 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --risk-threshold 30 \
  --sample-clip 30.0 \
  --attention-diag-floor 0.05
```

C-MAPSS 的风险标签由 RUL 构造：

```text
risk = 1 if RUL <= risk_threshold else 0
```

默认 `risk_threshold=30`。

### 18.3 使用旧 checkpoint 重新评估

如果已经训练完成，只想重新评估：

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 /root/anaconda3/bin/conda run -n torch23 python -u /home/work/new_work/code/run_experiment.py \
  --dataset metropt3_chrono_502030 \
  --model kst_probflow \
  --seed 2026 \
  --missing-rate 0.3 \
  --missing-mode mixed \
  --history-len 168 \
  --pred-len 24 \
  --stride 60 \
  --data-root /home/work/new_work/dataset \
  --output-dir /home/work/new_work/result \
  --run-id metropt3_chrono502030_reeval_$(date +%Y%m%d_%H%M%S) \
  --epochs 0 \
  --batch-size 128 \
  --max-eval-batches 0 \
  --nsamples 100 \
  --device cuda \
  --checkpoint /home/work/new_work/result/某次实验/checkpoints/metropt3_chrono_502030/kst_probflow/checkpoint_seed2026_best.pt
```

### 18.4 只比较风险阈值和校准

不重训、不改预测结果，只比较风险阈值策略：

```bash
TMPDIR=/tmp /root/anaconda3/bin/conda run -n torch23 python -u /home/work/new_work/code/evaluate_risk_calibration.py \
  --run-dir /home/work/new_work/result/metropt3_chrono502030_mskaf_20260609_084658 \
  --data-root /home/work/new_work/dataset \
  --device cuda \
  --batch-size 128
```

输出文件：

```text
calibration/{dataset}/{model}/risk_calibration_e1_seed{seed}.json
```

## 19. 结果保存位置

每次实验保存在：

```text
{output_dir}/{run_id}/
```

例如：

```text
/home/work/new_work/result/metropt3_chrono502030_mskaf_20260609_084658/
```

主要文件：

| 文件 | 作用 |
|---|---|
| `metrics/{dataset}/{model}/metrics_seed{seed}.json` | 最终 test 指标 |
| `training_history/{dataset}/{model}/history_seed{seed}.json` | 每个 epoch 的 train/valid 过程 |
| `checkpoints/{dataset}/{model}/checkpoint_seed{seed}.pt` | final checkpoint |
| `checkpoints/{dataset}/{model}/checkpoint_seed{seed}_best.pt` | best-valid checkpoint |
| `splits/{dataset}/{model}/{dataset}_split_seed{seed}.json` | 数据划分事实 |
| `masks/{dataset}/{dataset}_missing_{rate}_seed{seed}.npz` | 固定异步缺失 mask |
| `predictions/{dataset}/{model}/mean_seed{seed}.npy` | 预测均值 |
| `predictions/{dataset}/{model}/samples_seed{seed}.npy` | 预测样本 |
| `predictions/{dataset}/{model}/risk_seed{seed}.npy` | 风险分数 |
| `configs/{dataset}/{model}/config_seed{seed}.yaml` | 实验配置 |
| `calibration/{dataset}/{model}/calibration_seed{seed}.json` | validation 校准信息 |

## 20. 如何判断指标优劣

### 20.1 主预测指标

| 指标 | 优劣方向 | 解释 |
|---|---|---|
| MAE | 越低越好 | 预测均值的平均绝对误差，代表整体点预测偏差 |
| RMSE | 越低越好 | 对大误差更敏感；若明显高于 MAE，说明存在少量大误差窗口 |
| NLL | 越低越好 | 真实值在预测分布下的负对数似然；需检查是否存在长尾爆炸 |
| CRPS | 越低越好 | 分布预测综合质量，通常比 NLL 更稳健 |
| PICP | 接近目标覆盖率更好 | 95% 区间实验中目标接近 0.95 |
| MPIW | 在 PICP 达标时越低越好 | 区间越窄越好，但不能牺牲覆盖率 |

优先判断顺序建议为：

```text
1. 数值是否 finite：nonfinite_sample_rows/mean/risk 必须为 0 或有明确过滤说明
2. 主预测是否改善：MAE、RMSE、CRPS 越低越好
3. 概率分布是否合理：NLL 不爆炸，PICP 接近 0.95
4. 区间是否过宽：PICP 达标后再比较 MPIW
```

如果出现：

```text
RMSE >> MAE
```

说明平均表现可以，但少量窗口误差很大，应查看：

```text
top_error_rmse_mean
top_error_rmse_max
```

### 20.2 风险预警指标

| 指标 | 优劣方向 | 解释 |
|---|---|---|
| AUROC | 越高越好 | 风险分数排序能力 |
| AUPRC | 越高越好 | 正类稀少时比 AUROC 更重要 |
| F1 | 越高越好 | 阈值下的 precision/recall 折中 |
| Precision | 越高误报越少 | 适合报警成本高的场景 |
| Recall | 越高漏报越少 | 适合漏检成本高的场景 |
| ECE | 越低越好 | 风险概率校准误差 |

风险阈值不能用 test label 搜索。当前代码默认采用“报警预算”方式：

```text
risk_threshold = quantile(validation_normal_risk_scores, 0.95)
risk_threshold_source = validation_normal_quantile
calibration_uses_test_labels = false
```

这表示 validation 正常窗口中约 5% 会被允许触发报警。`validation_best_f1` 和 `test best_f1` 只作为诊断字段，不能作为正式结果。

如果 `AUROC/AUPRC` 很高但 `F1` 很低，通常说明风险排序可用，但阈值迁移或报警预算设置不合适。此时应比较：

```text
raw_validation_normal_q90
raw_validation_normal_q95
platt_validation_normal_q95
```

其中 Platt 校准只使用 validation 拟合，可降低 ECE，但不能用 test label 调参。

### 20.3 MetroPT 协议解释

| 协议 | 用途 | 风险指标解释 |
|---|---|---|
| `metropt3` | UCI 推荐扩展原则：first-month train, remaining-months test | 分布漂移强，适合作压力测试 |
| `metropt3_chrono_602020` | chronological 60/20/20 无泄露划分 | 当前 test 无故障正类时，风险指标不可算 |
| `metropt3_chrono_502030` | chronological 50/20/30 风险可评估补充协议 | test 有故障正类，可报告 AUROC/AUPRC/F1/ECE |

论文中应明确：`metropt3_chrono_502030` 是补充协议，不是 UCI 官方协议。

### 20.4 C-MAPSS 指标解释

C-MAPSS 当前主任务仍是未来 21 个传感器值的概率预测，RUL 只用于辅助风险标签。因此：

```text
MAE/RMSE/NLL/CRPS/PICP/MPIW
```

评价的是 sensor forecasting。

```text
AUROC/AUPRC/F1/ECE
```

评价的是由 `RUL <= risk_threshold` 构造的辅助风险识别。论文中应避免把该风险头表述为直接 RUL 预测模型，除非后续增加专门的 RUL head。
