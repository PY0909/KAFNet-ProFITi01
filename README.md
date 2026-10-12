# KST ProbFlow 工业多传感器概率预测

当前仓库正在从零实现论文主线的统一模型包。新的主线代码位于相对路径
`code/kaf_profiti/models/unified/`，MetroPT-3、C-MAPSS FD004 和 TEP
分别实例化并训练独立参数；旧模型身份和旧结果不纳入新模型 artifact。

阶段性实现目前覆盖：通道级多尺度历史编码器、数据集特定状态/RUL/TEP 任务头、
任务配置与 registry、train-only 特征/状态目标归一化、split/manifest 合同以及
validation-only checkpoint 选择。状态任务在标准化空间训练，artifact 中的预测和
MAE/RMSE 恢复为原始传感器量纲；单 seed 运行仅用于诊断，不构成正式多 seed 论文结论。

本地验证环境使用 conda `torch23`：

```bash
PYTHONPATH=code conda run -n torch23 python -m pytest code/tests/ -q
```

本项目保留 legacy 模型文档，同时以 `code/` 下的 unified 实现作为当前论文主线。正式多 seed 结果尚未运行。

统一训练入口支持 CUDA、AMP、train-only 特征/目标归一化、学习率调度和 validation 早停：

```bash
PYTHONPATH=code python code/run_unified_experiment.py \
  --dataset metropt3 --data-root /path/to/metropt-root \
  --metropt-protocol official_first_month \
  --result-root /path/to/result --device cuda \
  --epochs 50 --batch-size 256 --patience 8 --scheduler plateau \
  --evaluate-test
```

只有显式提供 `--evaluate-test` 才会在 validation-best checkpoint 上评估 test 一次；每次默认生成新的 run 目录，也可用 `--run-id` 指定唯一名称。结果目录包含 `test_metrics.json`、`test_predictions.pt` 和 `efficiency.json`：MetroPT 指标及预测/目标均恢复为 raw sensor units；FD004 保存 raw-cycle 总体与逐 engine 误差及官方 test prediction/target；TEP 保存分类指标、逐类/逐 composite-run 汇总、类别映射、逐窗口标签和诊断概率。特征和 state target 的归一化统计量只从 train split 拟合，并记录 `target_normalizer_identity` 与 `target_normalizer_source_split`。

统一编码器支持历史 V2、V3-A/B/C 和 V4。V4 在共享历史编码器中加入观测掩码安全的一阶/二阶差分、dilation 为 1/2/4 的 depthwise temporal convolution，并通过逐传感器可学习门控把 recent/trend 表示融合进任务表示；跨传感器注意力采用输入残差连接，避免注意力分支直接覆盖传感器表示。三个数据集仍通过各自任务头独立训练。V4 试验入口使用 `--encoder-variant v4`，例如：

```bash
PYTHONPATH=code conda run -n torch23 python -u code/run_unified_experiment.py \
  --dataset cmapss_fd004 --data-root dataset/CMAPSSData \
  --result-root result --run-id cmapss_fd004_v4_seed2026 \
  --seed 2026 --device cuda --epochs 80 --batch-size 128 \
  --history-len 50 --training-cap 125 --encoder-variant v4 --evaluate-test
```

MetroPT 默认采用官方首月基准：起始所在日历月构成训练期，该月末尾 20% 的唯一时间戳组作为 validation，后续所有月份只作为 test。validation 子段仅用于 checkpoint/调度选择，不进入最终参数更新；这样可保留 validation-only 选择且不触碰测试期。所有窗口在各自 split 内生成，不跨边界。若需复现旧 50/20/30 结果，显式设置 `--metropt-protocol chrono_502030`；两种协议的结果不可混为同一比较。

V4 recent/trend 门控融合与跨传感器 residual 修正已在 batch size 128、seed 2026 下完成 MetroPT 和 FD004 单次重训并生成测试产物。MetroPT 测试总体 MAE/RMSE 为 0.4765/1.6087；FD004 测试 MAE 为 29.2260，NASA score 为 29863.41。上述数字仅用于本次架构修正的诊断对照，不替代多 seed 正式论文结论；TEP 本轮按实验范围未重训。

TEP repeated-run 数据使用 `dataset/dataverse_files` 下的四个 RData 文件。当前内存型训练器默认每个故障读取 20 个仿真运行，先按完整运行划分 train/validation，再以 history 96、stride 12 生成窗口；训练模式不读取约 9.6M 行的测试 RData，以避免在 10 GiB 内存环境中被系统终止。该设置用于可复现实验开发和训练入口验证，不等价于论文全量 formal 结果。

```bash
PYTHONPATH=code conda run -n torch23 python -u code/run_unified_experiment.py \
  --dataset tep \
  --data-root dataset/dataverse_files \
  --result-root result \
  --run-id tep_dataverse_seed2026 \
  --seed 2026 --device cuda --epochs 50 --batch-size 128 \
  --history-len 96 --tep-window-stride 12 \
  --tep-max-runs-per-fault 20 --scheduler plateau
```

`--tep-max-runs-per-fault 0` 表示不限制运行数，但当前实现会把窗口保存在内存中，不建议在现有内存配置上直接使用。`--evaluate-test` 会尝试读取 testing RData 并执行一次测试评估；该文件较大，建议在内存充足环境运行。TEP 第三章按故障类别点分类报告 Macro-F1、Balanced Accuracy、Accuracy、逐类指标、混淆矩阵和 composite-run 汇总；本代码输出不代表正式多 seed 论文结果。

第三章比较模型位于独立的 `compare_code/`，支持 MetroPT、C-MAPSS FD004 和 TEP，并复用统一的 train-only 归一化、validation-best checkpoint 和 test evaluator。当前已实现 TCN、GRU、PatchTST、TimesNet、iTransformer；五个模型均提供 TEP 故障类别分类适配：

```bash
PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python -m compare_code.run_experiment \
  --model tcn --dataset metropt3 \
  --metropt-protocol official_first_month \
  --data-root /path/to/metropt-root --result-root /path/to/result \
  --device cuda --epochs 50 --batch-size 256 --history-len 168 --pred-len 24 \
  --stride 60 --patience 8 --scheduler plateau --evaluate-test
```

将 `--model` 替换为 `gru`、`patchtst`、`timesnet` 或 `itransformer` 即可切换结构。比较模型的结构超参数由 CLI 显式记录在 checkpoint config；没有真实同协议多模型运行前，不报告模型排名或性能优越性。

FD004 训练使用 train engines 内部的 train/validation split，官方 test 每台 engine 只保留最后一个 history window；不足 50 cycle 的轨迹在左侧重复首个观测补齐。训练目标按 cap 125 截断，官方 test target 与模型输出均为 raw RUL cycles。运行单个比较模型的命令如下：

正式比较前可运行 protocol checker，逐个核对 Unified、TCN、GRU、PatchTST、TimesNet 和 iTransformer 的 FD004 manifest：

```bash
PYTHONPATH=code python -m kaf_profiti.experiments.protocol_checker \
  --result-root result
```

检查项包括数据集和任务、engine split、train-only normalization、输入 schema、padding、RUL cap 语义、raw test target、seed 和 evaluator。发现 manifest 缺失或字段冲突时返回非零退出码；报告不会修改 `result/`。

```bash
PYTHONPATH=code:. conda run -n torch23 python -m compare_code.run_experiment \
  --model tcn --dataset cmapss_fd004 \
  --data-root dataset/CMAPSSData --result-root result \
  --seed 2026 --epochs 80 --batch-size 128 --history-len 50 --pred-len 10 \
  --training-cap 125 --device cuda --scheduler plateau --evaluate-test \
  --run-id cmapss_fd004_tcn_seed2026
```

2026-10-06 已完成 Unified、TCN、GRU、PatchTST、TimesNet、iTransformer 在 FD004 上的 seed 2026 单次运行。表内为官方 248 台 test engine 上的 raw-cycle 指标，结果仅作单 seed 诊断，不代表正式多 seed 论文结论；运行 artifact 保存在各自的 `result/cmapss_fd004_<model>_seed2026_20261006/` 目录。

| 模型 | Epoch | Test MAE (cycles) | Test RMSE (cycles) | NASA score | 训练时间 (s) |
|---|---:|---:|---:|---:|---:|
| Unified | 80 | 25.919 | 34.095 | 19106.973 | 334.4 |
| TCN | 80 | 23.968 | 31.346 | 9098.729 | 142.5 |
| GRU | 80 | 23.649 | 31.707 | 13709.411 | 124.4 |
| PatchTST | 80 | 31.321 | 38.808 | 39149.543 | 461.2 |
| TimesNet | 80 | 28.885 | 37.879 | 37981.141 | 250.3 |
| iTransformer | 80 | 29.424 | 38.249 | 373368.469 | 189.7 |

训练设备为本地 RTX 4060 Ti 8 GiB；训练耗时不与其他硬件上的结果直接比较。正式模型结论仍需按固定协议完成 seed 2027、2028 及其统计汇总。

主入口：

```text
code/run_experiment.py
```

## 1. 任务定义

主任务是 **未来多传感器值的概率预测**：

```text
输入：规整历史窗口 X_obs, T_obs, context
输出：未来窗口 Y_q 的预测均值、预测样本、预测区间和概率指标
```

辅助任务是 **设备风险评分**：

```text
MetroPT-3：由故障报告区间构造风险标签
C-MAPSS：由 RUL <= risk_threshold 构造风险标签
TEP：由故障 run / fault label 构造风险标签
```

风险任务用于评估预测结果是否能支持预警分析，不替代主预测任务。

## 2. 当前模型

已启用模型：

| 模型名 | 状态 | 说明 |
|---|---|---|
| `kst_probflow` | enabled | 当前主模型 |
| `kst_light_v2` | enabled | 方案 B 多尺度层次 KAF + residual point head |
| `kst_flow_v2` | enabled | 方案 B KST 表示 + ProFITi triangular conditional flow |
| `kaf_profiti_joint` | enabled | 早期 KAFNet + ProFITi joint flow 实现 |

`kst_probflow` 结构：

```text
X_obs, T_obs, context
  -> MultiScaleKAFEncoder
  -> DynamicSensorGraphBlock
  -> QueryConditionAdapter
  -> LowRankCopulaFlowHead
  -> QuantileHead
  -> RiskHead
```

当前 registry 中还包含 `TCN`、`GRU-D`、`mTAN`、`GraFITi`、`ProFITi original` 等对比模型名，但这些模型目前是 `not_implemented`，不能作为正式实验结果。

方案 B 的两个 v2 模型共享多尺度历史表示；跨变量模块在 FLA、sensor graph 和 identity 中单选。`kst_flow_v2` 的 NLL、样本、区间和风险输入来自同一个 ProFITi 条件流，旧 `kst_probflow` artifact 不与 v2 结果混合。

## 3. 目录结构

```text
/home/work/new_work/
  code/
    run_experiment.py
    evaluate_risk_calibration.py
    build_tables.py
    kaf_profiti/
      experiments/
      industrial/
      models/
    tests/
  dataset/
    CMAPSSData/
    metropt+3+dataset/
  result/
  requirement.txt
  统一对比实验方案.md
```

## 4. 环境

推荐使用当前已验证的 conda 环境：

```bash
TMPDIR=/tmp /root/anaconda3/bin/conda run -n torch23 python -m pytest /home/work/new_work/code/tests -q
```

`requirement.txt` 中列出的基础依赖：

```text
torch
numpy
pandas
scikit-learn
pytest
```

如果在 Featurize 服务器上运行，通常直接使用 base 环境即可；以服务器实际 `python` 和 `torch.cuda.is_available()` 为准。

## 5. 数据集协议

当前支持的数据集名：

| 数据集名 | 说明 |
|---|---|
| `metropt3` | 默认官方基准：数据起始所在首个日历月作为训练期（该月尾段留作 validation），其余时间只作 test；可用 `--metropt-protocol chrono_502030` 选择旧协议 |
| `metropt3_chrono_602020` | chronological 60/20/20，无跨边界窗口 |
| `metropt3_chrono_502030` | 可选旧协议：chronological 50/20/30，不跨 split 生成窗口 |
| `cmapss_fd001` 到 `cmapss_fd004` | C-MAPSS FD001-FD004 |
| `tep` | TEP 预留适配 |

划分约束：

```text
1. 先划分 train / valid / test，再生成滑动窗口
2. 归一化统计只来自 train
3. train/valid/test 复用同一份划分与窗口登记文件
4. C-MAPSS 按 engine id 划分 train/valid，不能按窗口随机划分
5. MetroPT chronological 协议不跨 train/valid/test 边界生成窗口
```

### 5.1 MetroPT-3 统一比较口径

本小节及其 `code/run_experiment.py` 命令是旧版 `kst_probflow` 实验入口的历史说明，保持 `metropt3_chrono_502030` 以便复现旧 artifact；它们不是当前 Unified/compare_code 主线的官方首月协议命令。新实验请使用本文开头的 `run_unified_experiment.py` 或 `compare_code.run_experiment` 并采用 `--metropt-protocol official_first_month`。

在比较 `kst_probflow`、`tcn_gaussian` 以及后续其他模型的 MetroPT-3 风险预警结果时，统一采用以下口径：

```text
同一数据协议：所有对比 run 固定同一个 `--metropt-protocol`；默认是 `official_first_month`
同一 dataset split
同一 seed
同一数据划分与窗口集合
同一 risk label rule
同一 q = 0.95
同一使用 Platt calibration
同一 checkpoint selection 规则
```

具体含义：

| 项目 | 固定规则 |
|---|---|
| 数据协议 | `metropt3_chrono_502030`，chronological 50/20/30 划分 |
| dataset split | 每个模型使用同一份 split JSON，不重新随机划分 |
| seed | 默认 `2026`；多 seed 实验时所有模型使用同一组 seed |
| availability map | 每个 seed 和可用率共用同一份 `.npz` availability_map |
| risk label rule | 使用相同的 MetroPT 故障/预警窗口标签构造规则 |
| q | `q=0.95`，阈值取 validation normal risk scores 的 95% 分位数 |
| Platt calibration | 所有模型的风险分数均在 validation split 上拟合 Platt calibration，再用于 test |
| checkpoint selection | 默认使用 validation CRPS 最优的 `checkpoint_seed{seed}_best.pt` |

正式风险指标中的 F1 使用 Platt 后风险分数和 `validation_normal_q95` 阈值计算。`AUROC/AUPRC` 只依赖排序；`ECE` 使用 Platt 后风险概率计算。阈值、方向和校准参数均来自 validation split，不使用 test label。

## 6. 本地训练命令

### 6.0 方案 B 首次接线 smoke（限制 batch，不作为正式指标）

```bash
cd /home/new_work/new_work
/root/anaconda3/bin/conda run -n torch23 python -u code/run_scheme_b_matrix.py \
  --matrix point --mode smoke --group ours \
  --data-root /home/new_work/new_work/dataset \
  --result-root /home/new_work/new_work/result \
  --device cuda --num-workers 0
```

该命令用于接线 smoke，会执行受限的 test 评估但不产生正式论文指标；正式训练前需完成方案 B 实施计划 P4/P5 的 validation-only 门禁。

### 6.1 MetroPT-3 风险可评估协议

```bash
cd /home/work/new_work

TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 /root/anaconda3/bin/conda run -n torch23 python -u /home/work/new_work/code/run_experiment.py \
  --dataset metropt3_chrono_502030 \
  --model kst_probflow \
  --seed 2026 \
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

### 6.2 C-MAPSS FD004

```bash
cd /home/work/new_work

TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 /root/anaconda3/bin/conda run -n torch23 python -u /home/work/new_work/code/run_experiment.py \
  --dataset cmapss_fd004 \
  --model kst_probflow \
  --seed 2026 \
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

## 7. Featurize 服务器路径

如果代码位于 `/home/featurize/work/code`，数据位于 `/home/featurize/data`，结果保存到 `/home/featurize/result`，将命令中的路径替换为：

```text
/home/work/new_work/code/run_experiment.py  -> /home/featurize/work/code/run_experiment.py
/home/work/new_work/dataset                 -> /home/featurize/data
/home/work/new_work/result                  -> /home/featurize/result
```

示例：

```bash
cd /home/featurize
mkdir -p /home/featurize/result

TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 python -u /home/featurize/work/code/run_experiment.py \
  --dataset metropt3_chrono_502030 \
  --model kst_probflow \
  --seed 2026 \
  --history-len 168 \
  --pred-len 24 \
  --stride 60 \
  --data-root /home/featurize/data \
  --output-dir /home/featurize/result \
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

## 8. 使用 checkpoint 重新评估

```bash
TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 /root/anaconda3/bin/conda run -n torch23 python -u /home/work/new_work/code/run_experiment.py \
  --dataset metropt3_chrono_502030 \
  --model kst_probflow \
  --seed 2026 \
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
  --checkpoint /path/to/checkpoint_seed2026_best.pt
```

## 9. 风险阈值和校准重评估

只比较风险阈值策略，不重训、不改变预测结果：

```bash
TMPDIR=/tmp /root/anaconda3/bin/conda run -n torch23 python -u /home/work/new_work/code/evaluate_risk_calibration.py \
  --run-dir /home/work/new_work/result/metropt3_chrono502030_mskaf_20260609_084658 \
  --data-root /home/work/new_work/dataset \
  --device cuda \
  --batch-size 128
```

该脚本会生成：

```text
calibration/{dataset}/{model}/risk_calibration_e1_seed{seed}.json
```

会比较以下策略：

```text
raw_validation_best_f1
raw_validation_normal_q90
raw_validation_normal_q95
platt_validation_best_f1
platt_validation_normal_q95
```

正式评估默认使用按“报警预算”定阈值：

```text
risk_threshold = quantile(validation_normal_risk_scores_after_platt, 0.95)
risk_threshold_source = validation_normal_quantile
risk_calibration_method = platt
risk_calibration_quantile = 0.95
calibration_uses_test_labels = false
```

`test best_f1` 只作为诊断上限，不能作为正式结果。

## 10. 结果目录

每次实验保存在：

```text
{output_dir}/{run_id}/
```

主要文件：

| 文件 | 说明 |
|---|---|
| `metrics/{dataset}/{model}/metrics_seed{seed}.json` | 最终 test 指标 |
| `training_history/{dataset}/{model}/history_seed{seed}.json` | 每个 epoch 的训练和验证过程 |
| `checkpoints/{dataset}/{model}/checkpoint_seed{seed}.pt` | final checkpoint |
| `checkpoints/{dataset}/{model}/checkpoint_seed{seed}_best.pt` | validation CRPS 最优 checkpoint |
| `splits/{dataset}/{model}/{dataset}_split_seed{seed}.json` | 划分统计和协议说明 |
| `availability_maps/{dataset}/{dataset}_availability_{rate}_seed{seed}.npz` | 固定多源可用性映射 |
| `predictions/{dataset}/{model}/mean_seed{seed}.npy` | 预测均值 |
| `predictions/{dataset}/{model}/samples_seed{seed}.npy` | 预测样本 |
| `predictions/{dataset}/{model}/risk_seed{seed}.npy` | 风险分数 |
| `configs/{dataset}/{model}/config_seed{seed}.yaml` | 本次实验配置 |
| `calibration/{dataset}/{model}/calibration_seed{seed}.json` | validation 校准信息 |

## 11. 指标解释

主预测指标：

| 指标 | 优劣方向 | 说明 |
|---|---|---|
| MAE | 越低越好 | 预测均值的平均绝对误差 |
| RMSE | 越低越好 | 对少量大误差窗口敏感 |
| NLL | 越低越好 | 真实值在预测分布下的负对数似然 |
| CRPS | 越低越好 | 分布预测综合质量 |
| PICP | 接近目标覆盖率更好 | 95% 区间实验中目标接近 0.95 |
| MPIW | PICP 达标时越低越好 | 预测区间平均宽度 |

风险指标：

| 指标 | 优劣方向 | 说明 |
|---|---|---|
| AUROC | 越高越好 | 风险分数排序能力 |
| AUPRC | 越高越好 | 正类稀少时比 AUROC 更关键 |
| F1 | 越高越好 | 固定阈值下 precision 和 recall 的折中 |
| ECE | 越低越好 | 风险概率校准误差 |

判断顺序：

```text
1. 先检查 nonfinite_sample_rows / nonfinite_mean_rows / nonfinite_risk_rows
2. 再看 MAE、RMSE、CRPS 是否下降
3. NLL 必须 finite，且不能出现长尾爆炸
4. PICP 应接近 0.95，MPIW 在 PICP 达标后再比较
5. 风险指标只在 test 同时有正负类时解释
6. F1 必须说明阈值来源，默认是 validation_normal_quantile
```

## 12. 构建结果表

```bash
TMPDIR=/tmp /root/anaconda3/bin/conda run -n torch23 python /home/work/new_work/code/build_tables.py \
  --results-dir /home/work/new_work/result
```

输出：

```text
/home/work/new_work/result/tables/
  table1_dataset_split.csv
  table2_main_forecasting.csv
  table3_risk_prediction.csv
  table4_availability_robustness.csv
  table5_ablation.csv
  table6_efficiency.csv
  table7_statistical_test.csv
```

未实现模型会在表格中保留空指标和 `status=not_implemented`，不能填入虚构结果。

## 13. 测试

```bash
TMPDIR=/tmp /root/anaconda3/bin/conda run -n torch23 python -m pytest /home/work/new_work/code/tests -q
```

如果只检查实验框架：

```bash
TMPDIR=/tmp /root/anaconda3/bin/conda run -n torch23 python -m pytest /home/work/new_work/code/tests/test_experiment_framework.py -q
```

## 14. 相关方案文档
