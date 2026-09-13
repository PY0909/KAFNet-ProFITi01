# Ch3 真异步时间戳处理方案

> 边界说明：本文只给出代码改造边界与决策，不改动 `code/` 下任何文件。
> 改动需由用户（或显式授权）执行。

## 0. 现实澄清（必须先读）

核查 `code/kaf_profiti/industrial/metropt.py` 与 `experiments/datasets.py` 后的硬事实：

1. **CSV 有真实时间戳列**。`load_metropt_frame`（L54-78）已用
   `parse_dates=["timestamp"]` 读取 `datetime`，并算好
   `relative_time = (timestamp - timestamp.iloc[0]).dt.total_seconds()`（L74-76）。
   所以真实秒数时间轴**已经存在**，只是被 `__getitem__` 丢掉了。
2. **MetroPT-3 是同步采样数据**。单台空压机、15 个传感器共享同一个
   `timestamp`，即所有通道在同一时刻被采样。**物理上不存在"每个传感器
   独立真实时刻"**——这是数据集本身的属性，不是代码问题。
3. **当前 T_obs 是假的**。`MetroPTWindowDataset.__getitem__` 在 L194 用
   `torch.arange(history_len)`，把真实时间轴换成了整数索引。
4. **编码器已兼容连续时间戳**。`kafnet_encoder.py` 的 `encode()` 在
   L157-159 对 `T_obs` 做样本内 min-max 归一化到 [0,1]，且 `_time_embedding`
   （L140）用可学习的线性层 `te_scale/te_per_sin/te_per_cos`，能自适应任意
   尺度。因此换成真实秒数**不会数值爆炸、无需改编码器**。

结论：**"真异步时间戳"在 MetroPT 上 = 用真实秒数替换 `arange`（层次1）；
而"逐传感器独立时间戳"级真异步需要换数据集（层次2）。**

---

## 1. 层次 1：真实时间轴（推荐，最小改动，立刻满足"时间戳是真的"）

### 1.1 改动点
文件：`code/kaf_profiti/industrial/metropt.py`
位置：`MetroPTWindowDataset.__getitem__`，原 L194（及 T_q 的 L196-200）。

原代码：
```python
        return MetroPTWindowSample(
            X_obs=X_obs,
            T_obs=torch.arange(self.history_len, dtype=torch.float32),
            M_obs=M_obs,
            T_q=torch.arange(
                self.history_len,
                self.history_len + self.pred_len,
                dtype=torch.float32,
            ),
            Y_q=sensors_future,
            M_q=torch.ones_like(sensors_future),
            context=context,
            rul=future_fault,
            unit_id=0,
        )
```

改为（真实相对秒数，窗口内归零以保证跨窗口一致）：
```python
        # 真实观测时间戳：用 CSV 的 relative_time（秒），窗口内归零跨窗口一致
        rel_time = hist["relative_time"].to_numpy(dtype="float32")
        t_obs = torch.tensor(rel_time, dtype=torch.float32) - float(rel_time[0])
        # 预测目标时间轴也用真实相对时间（从窗口起点起算）
        fut_rel = fut["relative_time"].to_numpy(dtype="float32")
        t_q = torch.tensor(fut_rel, dtype=torch.float32) - float(rel_time[0])

        return MetroPTWindowSample(
            X_obs=X_obs,
            T_obs=t_obs,
            M_obs=M_obs,
            T_q=t_q,
            Y_q=sensors_future,
            M_q=torch.ones_like(sensors_future),
            context=context,
            rul=future_fault,
            unit_id=0,
        )
```

### 1.2 为什么"减窗口起点"是必要的
- 不减去窗口起点时，不同时间窗口的 `relative_time` 绝对值相差巨大
  （2月 vs 7月，差值可达千万秒），`te_scale` 线性层会学到"绝对日历时间
  偏移"而非"窗口内相对时间模式"，引入不想要的季节/日历偏置。
- 减窗口起点后，每个窗口的 `T_obs` 都从 0 开始，跨窗口一致，且最接近
  原 `arange` 的语义（只是坐标从整数变成真实秒数）。

### 1.3 编码器兼容性（无需改动）
- `encode()` L157-159 样本内 min-max 归一化 → 形状行为与原 `arange` 一致。
- `_time_embedding` L140 为可学习线性层 → 自动吸收"秒数 vs 索引"的尺度差
  （约 ×60，若原始采样严格 60s）。
- 因此**只改 `metropt.py` 一行来源，编码器零改动**。

### 1.4 与缺失/降采样机制的协同 → 形成"逐通道异步观测"
MetroPT 全通道共享同一真实时刻网格，但 `MissingMechanismSimulator`
（`random` / `low_rate` 通道降采样 / `block_offline` 连续离线）会让**不同
通道在不同真实时刻有/无观测**。于是：
- 时间坐标 = 真实秒数（真）；
- 各通道观测 = 在真实时刻上稀疏/不规则分布（异步由机制驱动）。

这就是在 MetroPT 上可辩护的"以真实观测时间戳为坐标的逐通道异步建模"，
且与你之前的设计口径（"异步由缺失模拟造成"）**自洽升级**。

---

## 2. 协同升级：delta_t 用真实秒间隔（对应"四个待补项"之②）

既然有了真实 `relative_time`，`delta_t`（时间间隔）不必再从网格 mask 派
生"连续缺失步数"，可改为**相邻真观测的真实秒间隔**：

- `metropt.py`：样本加 `delta_t` 占位字段（或复用 `T_obs` 差值）。
- `masks.py` `__getitem__`（L102-105）：从最终 `mask` 与真实时间算
  "距上次观测的真实秒数" 并 `replace` 进 sample。
- `batch.py`：`IndustrialBatch` 加字段 + collator 堆叠 + `.to()` 透传。
- `kafnet_encoder.py`：加可选 `use_delta_t`（默认关，行为不变），
  `encode` 末尾把 `delta_t` 时间维均值嵌入后加到 `z`。

注意：`delta_t` 的派生**必须在 `masks.py`**（活路径里
`MaskedWindowDataset` 才持有最终 `M_obs`，在 `metropt.py` 里算会被覆盖）。

---

## 3. 层次 2：逐传感器真异步事件流（架构级，主数据集不支持）

若你的"真异步"指**每个传感器有独立真实观测时刻的事件序列**，则：

### 3.1 数据集现状不支持
MetroPT-3 所有通道同步采样，**没有 per-channel 真实时间戳**。要做到
per-channel 事件流，事件时刻只能来自全局 `relative_time`，且只有"有观测"
的时刻才生成事件——异步性仍由缺失/降采样机制驱动，并非数据自带。

### 3.2 所需架构改动（代价大）
- 输入从规则网格 `[L, N]` + mask，改为 N 个通道各自变长事件列表
  `events_n = [(value_k, t_k, present_k)]`，长度随缺失变化。
- 编码器需改为事件序列编码器（per-channel set/sequence encoder +
  跨通道交互），**动摇 MSGA 的 patch 切块设计**（patch 依赖规则网格）。
- 这会推翻现有 KAFNet + MSGA 架构，论文需重述编码器章节。

### 3.3 建议
- **不阻塞当前论文**。把层次 2 列为"未来工作 / 跨数据集验证"。
- 若导师坚持"真·逐传感器异步"，需引入一个**天然异步数据集**
  （如多设备物联网、医疗 ICU 多源异步监测）作为附加验证，而非改写
  MetroPT 主实验。

---

## 4. 论文口径建议（诚实且自洽）

在 3.1/3.5 节采用如下表述：

> 本文以传感器原始采集的真实观测时间戳（秒）为时间坐标，不对异步观测
> 做规则重采样或插值；缺失与采样率差异（通道级降采样、连续离线）由
> 受控模拟机制引入，使各通道在真实时间轴上呈现稀疏、不规则的观测模式。
> 由此，观测过程本身（何时、以何种方式被观测）被显式建模为设备状态
> 的一部分。

这既满足"时间戳真实"的诉求，又诚实说明 MetroPT 的同步采样属性，避免
过度宣称"真·逐传感器异步"。

---

## 5. 验证清单（改动后冒烟）
1. `T_obs` 形状仍为 `[history_len]`，值域为窗口内真实秒数（从 0 起）。
2. 打印一个样本 `T_obs`，确认非等差/含真实间隔（若原始采集 jitter 则
   非严格等差，正说明时间轴真实不规则）。
3. 跑一次 KST + 轻量头（见 `Ch3_四个待补项实现指南.md` 的 `evaluate_ch3.py`），
   确认 MAE 量级与 `arange` 版可比（编码器自适应，不应崩）。
4. 消融"去时间戳"：将 `T_obs` 置常数，性能应下降，证明时间戳贡献。

---

## 6. 与"四个待补项"的关系
- 本方案 = 待补项①的"真异步"升级版（替换原 0 代码叙事对齐选项）。
- ② delta_t 可借真实时间协同升级（见 §2）。
- ③ 轻量头、④ 消融开关 不变，仍见 `Ch3_四个待补项实现指南.md`。
