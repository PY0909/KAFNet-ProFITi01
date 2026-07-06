# KST ProbFlow 工业异步传感器概率预测

本项目实现工业多变量传感器的未来状态概率预测实验框架。当前主模型为 `kst_probflow`，用于在异步缺失观测下预测未来多传感器联合分布，并输出辅助风险分数。

当前代码目录：

```text
/home/work/new_work/code
```

主入口：

```text
/home/work/new_work/code/run_experiment.py
```

## 1. 任务定义

主任务是 **未来多传感器值的概率预测**：

```text
输入：历史窗口 X_obs, T_obs, M_obs, context
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
| `kaf_profiti_joint` | enabled | 早期 KAFNet + ProFITi joint flow 实现 |

`kst_probflow` 结构：

```text
X_obs, T_obs, M_obs, context
  -> MultiScaleKAFEncoder
  -> DynamicSensorGraphBlock
  -> QueryConditionAdapter
  -> LowRankCopulaFlowHead
  -> QuantileHead
  -> RiskHead
```

当前 registry 中还包含 `TCN`、`GRU-D`、`mTAN`、`GraFITi`、`ProFITi original` 等对比模型名，但这些模型目前是 `not_implemented`，不能作为正式实验结果。

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
  KAFNet-ProFITi工业异步传感器状态预测缝合方案.md
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
| `metropt3` | UCI 推荐扩展原则：first-month train，remaining-months test |
| `metropt3_chrono_602020` | chronological 60/20/20，无跨边界窗口 |
| `metropt3_chrono_502030（当前使用）` | chronological 50/20/30，test 中保留故障正类，适合风险指标 |
| `cmapss_fd001` 到 `cmapss_fd004` | C-MAPSS FD001-FD004 |
| `tep` | TEP 预留适配 |

划分约束：

```text
1. 先划分 train / valid / test，再生成滑动窗口
2. 归一化统计只来自 train
3. 缺失 mask 固化为 .npz，train/valid/test 复用同一份 mask 文件
4. C-MAPSS 按 engine id 划分 train/valid，不能按窗口随机划分
5. MetroPT chronological 协议不跨 train/valid/test 边界生成窗口
```

### 5.1 MetroPT-3 统一比较口径

在比较 `kst_probflow`、`tcn_gaussian` 以及后续其他模型的 MetroPT-3 风险预警结果时，统一采用以下口径：

```text
同一数据协议：metropt3_chrono_502030
同一 dataset split
同一 seed
同一 missing mask
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
| missing mask | 每个 seed 和缺失率共用同一份 `.npz` mask |
| risk label rule | 使用相同的 MetroPT 故障/预警窗口标签构造规则 |
| q | `q=0.95`，阈值取 validation normal risk scores 的 95% 分位数 |
| Platt calibration | 所有模型的风险分数均在 validation split 上拟合 Platt calibration，再用于 test |
| checkpoint selection | 默认使用 validation CRPS 最优的 `checkpoint_seed{seed}_best.pt` |

正式风险指标中的 F1 使用 Platt 后风险分数和 `validation_normal_q95` 阈值计算。`AUROC/AUPRC` 只依赖排序；`ECE` 使用 Platt 后风险概率计算。阈值、方向和校准参数均来自 validation split，不使用 test label。

## 6. 本地训练命令

### 6.1 MetroPT-3 风险可评估协议

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

### 6.2 C-MAPSS FD004

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
  --missing-rate 0.3 \
  --missing-mode mixed \
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
| `masks/{dataset}/{dataset}_missing_{rate}_seed{seed}.npz` | 固定异步缺失 mask |
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
  table4_missing_robustness.csv
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

- [统一对比实验方案.md](/home/work/new_work/统一对比实验方案.md)
- [KAFNet-ProFITi工业异步传感器状态预测缝合方案.md](/home/work/new_work/KAFNet-ProFITi工业异步传感器状态预测缝合方案.md)
