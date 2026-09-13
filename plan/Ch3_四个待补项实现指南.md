# Ch3 四个待补项实现指南（编码器表示质量验证）

> 本文档对应 Ch3「轻量头独立检验表示质量」所需的 4 个代码缺口。
> 文档只给实施边界与代码；按用户要求**不修改任何代码文件**，由你照此落地。
> 配套说明：`3.2_结构框架.md`、`统一对比实验方案.md`、对话中的「Ch3 实验—指标地图」。

---

## 0. 必读：现有管线的两个硬约束

### 0.1 活路径（不要改错地方）
训练真正走的是：
```
run_experiment.py
 └─ create_protocol_datasets        (experiments/datasets.py)
     └─ MetroPTWindowDataset        (industrial/metropt.py)   ← 样本在此构造，T_obs 在 metropt.py:194
 └─ MaskedWindowDataset            (experiments/masks.py)     ← 用 run 级 mask 覆盖 M_obs/X_obs
 └─ IndustrialCollator             (industrial/batch.py)      ← 产出 IndustrialBatch（含 y_flat）
```
- `MaskedWindowDataset.__getitem__`：`replace(sample, M_obs=mask, X_obs=sample.X_obs*mask)`
  —— **最终喂给模型的 M_obs 是 MaskedWindowDataset 那一刻的 mask**，不是 metropt 里算的。
  ⇒ 凡依赖「最终缺失」的派生量（如 delta_t），必须在 `masks.py` 里从 `mask` 算，而不是在 metropt 里。

### 0.2 评估函数对概率头的硬依赖（重要）
`run_experiment.py` 的 `_evaluate` 写死：
- `model.distribution(batch)`（第 537 行）
- `model.flow_head.nll / .sample / .crps`（538–569）
- `model.flow_head.sample_clip / attention_diag_floor`（645–647）
- `model.quantile_head` / `model.risk_head`（hasattr 分支）

⇒ 一个纯点预测的「轻量头模型」**不能**走 `run_experiment.py`：会直接 AttributeError，且会写出假的 NLL/CRPS/PICP。
⇒ **Ch3 必须新增一条独立评估脚本**（见任务 3 的 `evaluate_ch3.py`），不要复用 `run_experiment.py`，也不要为了复用而给轻量模型塞假的 flow_head。

### 0.3 诚实风险（贯穿四项）
`T_obs` 当前是均匀网格 `[0,1,...,167]`，所有通道共用；所谓「异步」目前**只体现在 M_obs 的逐通道缺失模式上**，MetroPT CSV 本身约 1 分钟均匀采样、没有真异步时间戳。
⇒ 论文里要么把「异步」明确定义为「由缺失模拟造成的逐通道采样模式差异」，要么补真时间戳（见任务 1）。**不要声称有真实异步多传感器时间戳而数据其实没有**。

---

## 任务 1：真·异步时间戳 T_obs

### 现状
`metropt.py:194`：`T_obs = torch.arange(self.history_len, dtype=torch.float32)`
→ 均匀整数网格，全通道共享。

### 怎么做（两档，按诚实度选）

**档 A（推荐，0 代码，必做叙事）**
不改代码。论文 3.1/3.5 把「异步」显式定义为：
> 「不同传感器具有不同的采样频率与离线模式，由缺失机制模拟器（random / 低采样 / 连续离线）在通道轴上施加，使观测过程呈现异步事件流；时间轴 T_obs 为相对网格索引。」

**档 B（可选，1 行增强）**
用 CSV 里真实相对时间替代整数索引（仍是全通道共享，但变成真实秒数，更贴近「时钟时间」）。
在 `metropt.py:__getitem__`，把
```python
T_obs=torch.arange(self.history_len, dtype=torch.float32),
```
改为
```python
T_obs=torch.tensor(hist["relative_time"].to_numpy(), dtype=torch.float32),
```
编码器内部会对 T 做归一化（`kafnet_encoder.py:_kaf_pool` 第 264–265 行），真实秒数可直接吃，无需改编码器。

**档 C（不推荐，易过度宣称）**
构造真·逐通道 3D 时间戳 `[L, N]`。但 MetroPT 没有真实逐通道时刻，等于用模拟器再造一份「异步」——和已有 M_obs 模拟等价，且风险是被评审认为「伪造时间戳」。除非你有真实异步数据源，否则不做。

> 结论：先落档 A（叙事），可选档 B（1 行）。档 C 暂不做。

---

## 任务 2：时间间隔 delta_t（时间自上次观测的间隔）

### 现状
`IndustrialBatch` 与 `MetroPTWindowSample` 都没有 delta_t 字段；编码器 `encode()` 不接收。

### 关键：delta_t 必须从「最终 mask」算
`MaskedWindowDataset` 才持有最终 M_obs，所以 delta_t 在 `masks.py` 里算，metropt 里只给占位字段。

### 2.1 `industrial/metropt.py` —— 给样本加占位字段
`MetroPTWindowSample`（第 41 行）增加：
```python
delta_t: Tensor
```
`__getitem__` 的 `return MetroPTWindowSample(...)` 里补一个占位（会被 masks.py 覆盖）：
```python
delta_t=torch.zeros(self.history_len, len(METROPT_SENSOR_COLUMNS)),
```
（仅占位；真实值来自 MaskedWindowDataset。）

### 2.2 `experiments/masks.py` —— 从最终 mask 算 delta_t
`MaskedWindowDataset.__getitem__`（第 102–105 行）改为：
```python
def __getitem__(self, index: int):
    sample = self.dataset[index]
    mask = torch.tensor(self.masks[index], dtype=torch.float32)
    delta_t = self._time_since_obs(mask)          # 新增
    return replace(
        sample,
        M_obs=mask,
        X_obs=sample.X_obs * mask,
        delta_t=delta_t,                            # 新增
    )

@staticmethod
def _time_since_obs(mask: torch.Tensor) -> torch.Tensor:
    # mask: [L, N]；观测位=1。返回每个位置距上次观测的步数(+1)，观测位为 1。
    length, num_sensors = mask.shape
    delta = torch.zeros_like(mask)
    gap = torch.zeros(num_sensors)
    for t in range(length):
        observed = mask[t] > 0.5
        gap = torch.where(observed, torch.ones_like(gap), gap + 1.0)
        delta[t] = gap
    return delta
```

### 2.3 `industrial/batch.py` —— 透传字段
`IndustrialBatch`（第 9 行）增加：
```python
delta_t: Tensor
```
`to()`（第 23 行）增加：`delta_t=self.delta_t.to(device),`
`IndustrialCollator.__call__`（第 48–76 行）增加：
```python
delta_t = torch.stack([sample.delta_t for sample in batch])
```
并在 `return IndustrialBatch(...)` 里加 `delta_t=delta_t,`。

### 2.4 `models/kafnet_encoder.py` —— 可选接入（让「时间间隔」消融有意义）
给 `KSTProbFlowConfig` / 编码器加 `use_delta_t: bool = False`（默认关，行为不变）。
`MultiScaleKAFEncoder.__init__` 增加（默认 None，关时不创建）：
```python
self.delta_proj = nn.Linear(1, hidden_dim) if use_delta_t else None
```
`encode()` 签名改为 `encode(self, X, T_obs, M_obs, context=None, delta_t=None)`，
在得到 `z`（传感器级聚合后）前，若启用则注入：
```python
if delta_t is not None and self.delta_proj is not None:
    d_feat = delta_t.mean(dim=1, keepdim=True)      # [B, N, 1] 时间维取均值
    z = z + self.delta_proj(d_feat)                  # [B, N, hidden]
```
`KSTProbFlow.encode_variables`（kst_probflow.py:265）改为透传 `delta_t=batch.delta_t`。

> 默认 `use_delta_t=False` ⇒ 现有 KSTProbFlow 行为完全不变；Ch3 轻量模型可开 `use_delta_t=True` 并据此做「去时间间隔」消融。

---

## 任务 3：轻量头 LightweightHead + Ch3 点预测模型

### 目标
独立于 Ch4 概率头，用一个 Linear/MLP 轻量头消费编码器表示 z，只预测未来传感器轨迹（`y_flat`），仅用 MAE/RMSE 评分 —— 解耦「表示质量」与「概率头能力」。

### 3.1 新增模型（加到 `models/kst_probflow.py` 或新建 `models/kst_light.py`）
```python
class LightweightHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, mlp: bool = True):
        super().__init__()
        if mlp:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.net = nn.Linear(hidden_dim, out_dim)

    def forward(self, z: Tensor) -> Tensor:
        z_pool = z.mean(dim=1)          # [B, N, hidden] -> [B, hidden]
        return self.net(z_pool)         # -> [B, pred_len * num_sensors]


class KSTProbFlowLight(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = MultiScaleKAFEncoder(
            num_sensors=config.num_sensors, hidden_dim=config.hidden_dim,
            kernel_count=config.kernel_count, time_dim=config.te_dim,
            n_layers=config.n_layers, n_heads=config.n_heads,
            preconv_dim=config.preconv_dim, patch_lens=config.patch_lens,
            context_dim=config.context_dim,
            use_delta_t=getattr(config, "use_delta_t", False),
        )
        self.graph = DynamicSensorGraphBlock(
            num_sensors=config.num_sensors, hidden_dim=config.hidden_dim,
            graph_layers=config.graph_layers,
        )
        self.head = LightweightHead(
            config.hidden_dim, out_dim=config.pred_len * config.num_sensors, mlp=True
        )
        self._hidden = None

    def encode_variables(self, batch):
        z = self.encoder(batch.X_obs, batch.T_obs, batch.M_obs, batch.context,
                         getattr(batch, "delta_t", None))
        self._hidden = self.graph(z)
        return self._hidden

    def loss(self, batch):
        pred = self.head(self.encode_variables(batch))          # [B, pred_len*N]
        return (pred - batch.y_flat).abs().mean()               # MAE

    def predict(self, batch):
        return self.head(self.encode_variables(batch))          # [B, pred_len*N]
```
> `config` 需带 `pred_len`、`num_sensors`；复用 `KSTProbFlowConfig` 即可（它已有这些字段）。
> 消融标志：把任务 4 的 `AblationConfig` 也传给 `MultiScaleKAFEncoder`，使轻量模型同样支持逐项消融。

### 3.2 不要走 run_experiment.py —— 新增 `code/evaluate_ch3.py`
独立脚本，复用数据管线、自算 Ch3 指标：
```python
# 伪代码骨架（关键步骤）
from kaf_profiti.experiments.datasets import create_protocol_datasets
from kaf_profiti.experiments.masks import MaskedWindowDataset, generate_or_load_split_masks
from kaf_profiti.industrial.batch import IndustrialCollator
from kaf_profiti.models.kst_probflow import KSTProbFlowLight, KSTProbFlowConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 1) 数据（与 run_experiment 同协议）
bundle = create_protocol_datasets(dataset, data_root, seed, history_len, pred_len, stride, async_mode="none")
split_masks = generate_or_load_split_masks(...)           # 同 run_experiment
train_set = MaskedWindowDataset(bundle.train, split_masks["train"])
...
loader = DataLoader(train_set, batch_size=..., collate_fn=IndustrialCollator())

# 2) 模型
cfg = KSTProbFlowConfig(num_sensors=bundle.num_sensors, context_dim=bundle.context_dim,
                        hidden_dim=..., pred_len=pred_len, patch_lens="12,24,48", ...)
# 若做消融：cfg 加 ablation 字段并传给 encoder（见任务4）
model = KSTProbFlowLight(cfg).to(device)

# 3) 训练
opt = AdamW(model.parameters(), lr=..., weight_decay=...)
for epoch in range(epochs):
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad(); loss = model.loss(batch); loss.backward(); opt.step()

# 4) 评估：MAE / RMSE / 相对下降 / 参数量 / 训练&推理时间
def point_metrics(y, pred, mask):
    valid = (mask > 0) & torch.isfinite(y) & torch.isfinite(pred)
    mae = ((y - pred).abs() * mask).sum() / mask.sum().clamp_min(1)
    rmse = torch.sqrt((((y - pred) ** 2) * mask).sum() / mask.sum().clamp_min(1))
    return mae, rmse
# 在不同 missing_rate {0,0.1,0.3,0.5,0.7} 与不同 mode {random,low_rate,block_offline,mixed}
# 各跑一遍，记录 MAE/RMSE，并算「相对性能下降」= (高缺失MAE - 0.1缺失MAE)/0.1缺失MAE
# 计时：time.perf_counter() 包住训练循环与单次推理
# 参数量：sum(p.numel() for p in model.parameters())
```
> 输出建议直接存 json（`result/ch3/.../metrics_...json`），后续 `code/build_tables.py` 取数填 Table 2/4/5/6。

### 3.3 注册（可选，谨慎）
**不要**把 `kst_probflow_light` 标成 `enabled` 并塞进 `run_experiment.py`（会崩）。
若要登记，只在 `registry.py` 加一类新 status 如 `"ch3_only"` 并让 `run_experiment.py` 开头对 `"ch3_only"` 拒绝（提示用 `evaluate_ch3.py`）。最简做法：**不登记，脚本里直接 `KSTProbFlowLight(cfg)` 构造**。

---

## 任务 4：消融开关（逐项消融，孤立 MSGA 等创新）

### 目标
把 Ch3 设计里的「去时间戳 / 去 mask / 去通道身份 / 去工况 / 去多尺度 / 去跨传感器交互」变成可一键关闭的开关。
其中 **去多尺度分支 = MSGA 的 w/o 消融**（用基类行为，跳过 `z_local` 与 `merge_gate`）。

### 4.1 新增配置（放 `models/kafnet_encoder.py` 或 `kst_probflow.py` 顶部）
```python
from dataclasses import dataclass
@dataclass
class AblationConfig:
    ablate_multiscale: bool = False     # 去 MSGA 多尺度分支（孤立创新）
    ablate_cross_sensor: bool = False   # 去 FreqBlock 跨传感器交互
    ablate_timestamp: bool = False       # 去时间嵌入（T 置 0）
    ablate_mask: bool = False            # 去观测掩码（M 置 1）
    ablate_channel_id: bool = False      # 去通道身份（PositionalEncoding）
    ablate_context: bool = False         # 去工况 FiLM
```

### 4.2 接到编码器
`MultiScaleKAFEncoder.__init__` 增加 `ablation: AblationConfig = None`，存为 `self.ablation`。
改写 `encode()`（保持原顺序：pos → blocks → var_agg）：
```python
def encode(self, X, T_obs, M_obs, context=None, delta_t=None):
    batch_size, _, num_sensors = X.shape
    if num_sensors != self.num_sensors:
        raise ValueError(f"Expected {self.num_sensors} sensors, got {num_sensors}")
    T = self._expand_times(T_obs, num_sensors)
    abl = self.ablation
    if abl is not None and abl.ablate_timestamp:
        T = torch.zeros_like(T)
    if abl is not None and abl.ablate_mask:
        M_obs = torch.ones_like(M_obs)

    z_global = self._kaf_pool(X, T, M_obs, self.intra, self.feat_proj)
    if abl is not None and abl.ablate_multiscale:
        z = z_global                                     # 跳过 z_local / merge_gate = 基类行为
    else:
        z_local = self._local_representation(X, T, M_obs)
        gate = self.merge_gate(torch.cat([z_global, z_local], dim=-1))
        z = gate * z_global + (1.0 - gate) * z_local

    if delta_t is not None and self.delta_proj is not None:
        d_feat = delta_t.mean(dim=1, keepdim=True)
        z = z + self.delta_proj(d_feat)

    if not (abl is not None and abl.ablate_channel_id):
        z = self.pos(z)
    if not (abl is not None and abl.ablate_cross_sensor):
        for block in self.blocks:
            z = block(z)
    z = self.var_agg(z)

    if not (abl is not None and abl.ablate_context) and self.context_proj is not None and context is not None:
        gamma, beta = self.context_proj(context).chunk(2, dim=-1)
        z = z * (1.0 + gamma[:, None, :]) + beta[:, None, :]
    return z
```
> KAFNetEncoder（基类）的 `encode` 不受影响；`ablate_multiscale=True` 让子类退化为基类路径，正好对应「w/o 多尺度」。

### 4.3 轻量模型透传
`KSTProbFlowLight.__init__` 收 `ablation` 并传给 `MultiScaleKAFEncoder`；`evaluate_ch3.py` 用 `--ablation {multiscale,cross_sensor,timestamp,mask,channel_id,context}` 控制。

---

## 5. 验证与运行顺序

1. **先建字段链路（任务 2）并冒烟**：
   ```bash
   python -c "
   from kaf_profiti.experiments.datasets import create_protocol_datasets
   from kaf_profiti.experiments.masks import MaskedWindowDataset, generate_or_load_split_masks
   from kaf_profiti.industrial.batch import IndustrialCollator
   b = create_protocol_datasets('metropt3_chrono_502030','dataset',2026,168,24,60)
   m = generate_or_load_split_masks('/tmp/m.npz',{'t':(len(b.train),168,b.num_sensors)},0.3,2026,'mixed')
   dl = __import__('torch').utils.data.DataLoader(MaskedWindowDataset(b.train,m['t']),2,collate_fn=IndustrialCollator())
   s = next(iter(dl))
   print('delta_t' in dir(s), s.delta_t.shape)
   "
   ```
2. **轻量头单测（任务 3）**：小批量跑 `KSTProbFlowLight.loss` 反传，确认能出 MAE、梯度不炸。
3. **消融（任务 4）**：逐一 `AblationConfig(ablate_xxx=True)` 跑 `evaluate_ch3.py`，确认性能相对全模型下降且方向合理。
4. **对比基线（Ch3 设计 ⑥）**：LI+TCN/FF+GRU/Masked TCN/GRU-D/mTAN 当前仍是 `not_implemented`——属另一批任务，不在本四项内，需另行实现后才会有 Table 2 的对比行。

---

## 6. 四项 → 表格映射（与 3.6 节一致）
| 任务 | 落点 | 对应 Ch3 节 / 表 |
|---|---|---|
| 1 真·异步时间戳 | metropt.py:194（档B可选） | 3.1 动机 / 3.5 编码器（叙事） |
| 2 时间间隔 delta_t | masks.py / batch.py / encoder | 3.6.5 消融（时间间隔项） |
| 3 轻量头 | 新 KSTProbFlowLight + evaluate_ch3.py | 3.6.1/3.6.3 → Table 2；3.6.4 → Table 4；效率 → Table 6 |
| 4 消融开关 | AblationConfig + encoder | 3.6.5 → Table 5（MSGA 孤立项） |

> 注意：对比基线（LI+TCN 等）与缺失机制复现（random/low_rate/block_offline 已实现于 missing.py）不在本四项中；缺失机制已就绪，基线实现是独立缺口。
