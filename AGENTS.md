# AGENTS.md

本文件用于约束本仓库的默认协作流程，目标是让 Claude Code 在论文写作、实验管理、文献查阅等任务中按明确上下文执行，减少重复沟通和返工。

如果本文件与仓库中的代码、文档或配置现状不一致，以实际可执行内容为准，并在相关改动中同步修正文档，避免规则漂移。

## 1. 硬规则

- `AGENTS.md` 是仓库内 AI 协作规则的唯一真源。
- `CLAUDE.md` 应作为指向 `AGENTS.md` 的兼容入口；若因平台限制不能使用软链接，内容也必须只指向本文件。
- 论文正文、实验数据、文献引用等任何内容修改后，必须同步检查受影响的相关文件（引用编号、表格数据、章节交叉引用）。
- 不写死密钥、账号、AutoDL 实例 IP、绝对路径（跨机器不兼容的路径）。
- 论文写作：修改正文后必须验证引用编号连续性和对应关系。
- 实验管理：每次实验必须记录 `run_id`、seed、数据集、模型名、关键指标，结果存入 `result/` 目录。
- 文献管理：引用增删后必须同步更新 `文献汇总表.md` 核验状态和覆盖性检查。
- 注释、docstring、日志文案以清晰准确为准，不强制要求英文，但应与文件语境保持一致。

## 2. 仓库速览

- 项目定位：计算机硕士毕业论文课题——面向工业设备异步多传感器的概率状态预测与风险评估方法研究（KST ProbFlow）。
- 主流程：
  1. 阅读文献 → 更新文献汇总表 → 写文献综述
  2. 设计实验协议 → 在 AutoDL 租 GPU → 训练模型 → 收集指标
  3. 校准风险阈值 → 生成结果表和图 → 写入论文章节
  4. 全文统稿 → 导师审阅 → 修改 → 查重 → 送审
- 第一版边界：
  - 主模型为 `kst_probflow`（MultiScaleKAFEncoder + DynamicSensorGraphBlock + LowRankCopulaFlowHead + QuantileHead + RiskHead）。
  - 主数据集为 MetroPT-3（chronological 50/20/30协议），C-MAPSS 和 TEP 为辅助验证。
  - 风险阈值采用 validation-only Platt calibration，`q=0.95`。
  - 对比模型包括 TCN-Gaussian、PatchTST-Gaussian、GRU-D、ODE-RNN、mTAN、GraFITi、ProFITi、KAFNet 等。
  - 论文使用 Markdown 起草，后续转为 Word .docx。
  - GPU 资源通过 AutoDL 按需租用（RTX 3090 24GB），实验使用 `nohup` 或 `tmux` 后台运行。SSH: `ssh -p 43676 root@connect.nmb2.seetacloud.com`。

## 3. 目录边界

- `code/`：模型训练主工程。
  - `code/run_experiment.py`：统一训练与评估入口。
  - `code/evaluate_risk_calibration.py`：风险阈值与 Platt 校准重评估。
  - `code/build_tables.py`：从 result/ 生成结果表（Table 1-7）。
  - `code/kaf_profiti/experiments/`：实验配置、registry、数据协议。
  - `code/kaf_profiti/models/`：模型定义（kst_probflow.py、kafnet_encoder.py 等）。
  - `code/kaf_profiti/industrial/`：数据集读取、collate、缺失模拟、风险构造。
  - `code/tests/`：框架测试。
- `compare_code/`：对比模型独立工程。
  - `compare_code/TCN-Gaussian/`：TCN-Gaussian 对比模型。
  - `compare_code/probabilistic_baselines/`：PatchTST、ODE-RNN、GraFITi、ProFITi、KAFNet 等基线入口。
- `dataset/`：数据集原始文件（CMAPSSData、metropt+3+dataset）。
- `result/`：每次实验的完整保存（metrics、checkpoints、predictions、calibration、history）。
- `plan/`：进度跟踪、结构蓝图、证据映射、任务分包、30天推进计划。
- `refs/`：外部参考资料。
- `图/`：论文插图。
- `文献阅读.md`：文献综述正文（65条引用，已重写为正式综述结构）。
- `文献汇总表.md`：66条参考文献的详细条目表（含DOI、核验状态、阅读优先级）。
- `README.md`：项目总览、快速开始、数据集协议、指标解释。
- `补充建议.md`：开题审核补充建议（模型命名一致性、风险校准口径、对比模型状态等）。
- `统一对比实验方案.md`：实验矩阵、统一划分原则、评价指标、结果保存格式。
- `KAFNet-ProFITi工业异步传感器状态预测缝合方案.md`：初始技术方案（已被当前 kst_probflow 实现取代，部分章节仍有参考价值）。
- `3.2_结构框架.md`：论文章节结构（六章递进逻辑）。
- `开题报告修改建议_基于3.2结构框架.md`：开题报告修改建议。
- `开题报告_2_3_6对应修改建议.md`：第2/3/6章对应修改建议。

## 4. 默认工作流

1. 先判断任务类型：`writing（论文写作）/ experiment（实验）/ review（审查）/ lit（文献）/ plan（规划）/ fix（修复）`。
2. 先读 `AGENTS.md`、`plan/30天推进计划.md` 和相关文档，再动手。
3. 识别改动边界：
   - 论文正文（文献阅读.md 或章节日后创建的文件）→ 引用一致性 + 术语统一
   - 实验代码（code/ 或 compare_code/）→ 不影响已有模型注册 + 测试不退化
   - 文档（plan/、*.md）→ 数据一致性
   - 结果（result/）→ 只读，不手动修改 metrics JSON
4. 论文写作改动：
   - 新增引用 → 同步更新 文献汇总表.md 和 文献阅读.md 参考文献列表
   - 新增实验数据 → 验证来源（run_id、seed、checkpoint）
   - 修改术语 → 全文件统一替换
5. 实验改动：
   - 新增模型 → 注册到 registry + 补充训练命令 + 说明与现有模型的关系
   - 修改训练目标/损失权重 → 记录到 config 并更新 README 训练命令
6. 如果发现文档和代码不一致，优先信任可执行代码，并同步修正文档。
7. 最终交付默认说明：改了什么、为什么、验证情况、未验证项、风险点。

## 5. 常用命令

### 服务器连接

```bash
# SSH 连接
ssh -p 43676 root@connect.nmb2.seetacloud.com

# 工作目录（数据盘，关机不丢失）
cd /root/autodl-tmp
```

**当前实例配置**：

| 项 | 值 |
|---|---|
| 平台 | AutoDL（容器 ID: `autodl-container-511a4fa322-e1e797dc`） |
| GPU | RTX 3090 (24GB) × 1 |
| CPU | 14 vCPU Intel Xeon Gold 6330 @ 2.00GHz |
| 内存 | 90 GB |
| 系统盘 | 30 GB |
| 镜像 | PyTorch 2.5.1 + Python 3.12 + CUDA 12.4 (Ubuntu 22.04) |
| 数据盘路径 | `/root/autodl-tmp` |
| SSH 端口 | 43676 |

### 文件传输

```bash
# ===== 从本地上传代码到 AutoDL =====

# 方式一：打包上传（推荐，速度快）
cd /Users/ppy/研/00提交资料汇总/new_work
tar --exclude='__pycache__' --exclude='.DS_Store' --exclude='*.pyc' \
    --exclude='.pytest_cache' --exclude='code/code.zip' \
    -czf /tmp/new_work_code.tar.gz code/ compare_code/ dataset/ requirement.txt
scp -P 43676 /tmp/new_work_code.tar.gz root@connect.nmb2.seetacloud.com:/root/autodl-tmp/

# SSH 登录后解压
ssh -p 43676 root@connect.nmb2.seetacloud.com
cd /root/autodl-tmp
tar -xzf new_work_code.tar.gz && rm new_work_code.tar.gz

# ===== 从 AutoDL 下载结果到本地 =====

# 方式一：拉取整个 result 目录
scp -P 43676 -r root@connect.nmb2.seetacloud.com:/root/autodl-tmp/result/ /Users/ppy/研/00提交资料汇总/new_work/result/

# 方式二：打包后下载（推荐用于大批量结果）
ssh -p 43676 root@connect.nmb2.seetacloud.com "cd /root/autodl-tmp && tar -czf /tmp/results.tar.gz result/"
scp -P 43676 root@connect.nmb2.seetacloud.com:/tmp/results.tar.gz /tmp/
tar -xzf /tmp/results.tar.gz -C /Users/ppy/研/00提交资料汇总/new_work/
```

### 实验后台运行

```bash
# ===== 方式一：nohup（简单，推荐单次实验） =====
nohup python -u code/run_experiment.py \
  --dataset metropt3_chrono_502030 \
  --model kst_probflow \
  --seed 2027 \
  --missing-rate 0.3 --missing-mode mixed \
  --history-len 168 --pred-len 24 --stride 60 \
  --data-root /root/autodl-tmp/dataset \
  --output-dir /root/autodl-tmp/result \
  --run-id metropt3_kst_seed2027_$(date +%Y%m%d_%H%M%S) \
  --epochs 50 --batch-size 128 --nsamples 100 --device cuda \
  > logs/run.log 2>&1 &

tail -f logs/run.log   # 查看日志

# ===== 方式二：tmux（可断线重连，推荐长实验） =====
tmux new -s train_kst
python -u code/run_experiment.py ...
# Ctrl+B D 断开，tmux attach -t train_kst 重连

# ===== 查看运行状态 =====
jobs -l          # nohup 任务
tmux ls          # tmux 会话
nvidia-smi       # GPU 占用

# ===== 批量提交多 seed 实验 =====
mkdir -p logs
for seed in 2026 2027 2028; do
  nohup python -u code/run_experiment.py \
    --dataset metropt3_chrono_502030 \
    --model kst_probflow \
    --seed $seed \
    --missing-rate 0.3 --missing-mode mixed \
    --history-len 168 --pred-len 24 --stride 60 \
    --data-root /root/autodl-tmp/dataset \
    --output-dir /root/autodl-tmp/result \
    --run-id metropt3_kst_seed${seed}_$(date +%Y%m%d_%H%M%S) \
    --epochs 50 --batch-size 128 --nsamples 100 --device cuda \
    --hidden-dim 64 --te-dim 10 --kernel-count 4 \
    --n-layers 2 --n-heads 2 --preconv-dim 16 \
    --patch-lens 12,24,48 --graph-layers 1 --copula-rank 32 \
    --lambda-point 0.5 --lambda-quantile 0.2 --lambda-risk 0.05 \
    --lr 2e-4 --weight-decay 1e-4 \
    --sample-clip 30.0 --attention-diag-floor 0.05 \
    > logs/kst_seed${seed}.log 2>&1 &
done
wait && echo "All seeds done"
```

### 环境准备

### MetroPT-3 主实验（主模型 kst_probflow）

```bash
cd /root/autodl-tmp

TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 python -u code/run_experiment.py \
  --dataset metropt3_chrono_502030 \
  --model kst_probflow \
  --seed 2026 \
  --missing-rate 0.3 \
  --missing-mode mixed \
  --history-len 168 \
  --pred-len 24 \
  --stride 60 \
  --data-root /root/autodl-tmp/dataset \
  --output-dir /root/autodl-tmp/result \
  --run-id metropt3_kst_$(date +%Y%m%d_%H%M%S) \
  --epochs 50 \
  --batch-size 128 \
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

### MetroPT-3 多 seed 训练

```bash
# seeds: 2026, 2027, 2028
for seed in 2026 2027 2028; do
  nohup python -u code/run_experiment.py \
    --dataset metropt3_chrono_502030 \
    --model kst_probflow \
    --seed $seed \
    ... \
    --run-id metropt3_kst_seed${seed}_$(date +%Y%m%d_%H%M%S) \
    > logs/kst_seed${seed}.log 2>&1 &
done
```

### C-MAPSS FD004 实验

```bash
cd /root/autodl-tmp

TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 python -u code/run_experiment.py \
  --dataset cmapss_fd004 \
  --model kst_probflow \
  --seed 2026 \
  --missing-rate 0.3 \
  --missing-mode mixed \
  --history-len 50 \
  --pred-len 10 \
  --stride 1 \
  --data-root /root/autodl-tmp/dataset \
  --output-dir /root/autodl-tmp/result \
  --run-id cmapss_fd004_kst_$(date +%Y%m%d_%H%M%S) \
  --epochs 80 \
  --batch-size 128 \
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

### 使用已有 checkpoint 只做评估

```bash
python -u code/run_experiment.py \
  --dataset metropt3_chrono_502030 \
  --model kst_probflow \
  --seed 2026 \
  --missing-rate 0.3 \
  --missing-mode mixed \
  --history-len 168 \
  --pred-len 24 \
  --stride 60 \
  --data-root /root/autodl-tmp/dataset \
  --output-dir /root/autodl-tmp/result \
  --run-id reeval_$(date +%Y%m%d_%H%M%S) \
  --epochs 0 \
  --batch-size 128 \
  --nsamples 100 \
  --device cuda \
  --checkpoint /root/autodl-tmp/result/某次实验/checkpoints/.../checkpoint_seed2026_best.pt
```

### 风险校准重评估（不改模型，只改阈值策略）

```bash
python -u code/evaluate_risk_calibration.py \
  --run-dir /root/autodl-tmp/result/metropt3_chrono502030_kst_20260609_084658 \
  --data-root /root/autodl-tmp/dataset \
  --device cuda \
  --batch-size 128
```

### 构建结果表

```bash
python code/build_tables.py --results-dir /root/autodl-tmp/result
# 输出：result/tables/table1_*.csv ~ table7_*.csv
```

### 测试

```bash
python -m pytest code/tests/ -q
python -m pytest code/tests/test_experiment_framework.py -q
```

## 6. 验证矩阵

- 论文写作改动：
  - 引用增删 → 检查 文献阅读.md 正文引用编号连续无跳跃，参考文献编号与正文一一对应
  - 章节新增 → 检查 3.2_结构框架.md 与正文章节标题一致
  - 术语变更 → 全仓库 grep 旧术语确认无残留
- 实验代码改动：
  - 优先执行 `python -m pytest code/tests/ -q`
  - 若影响模型注册、数据协议、缺失构造、风险标签或校准流程，必须覆盖对应测试或说明未覆盖风险
  - 新增模型必须注册到 registry，指定 status（enabled / not_implemented）
- 文档改动：
  - 确认命令、配置项、文件名、模块职责与当前仓库一致
  - README 只更新核心入口和快速开始，细节写入专题文档
- 结果改动：
  - `result/` 下的 metrics JSON 不允许手动修改
  - 任何指标数字的变动必须有对应的 run_id 和 checkpoint 溯源

## 7. 稳定性护栏

- 单个实验失败不能中断整批实验（`nohup` 或 `tmux` 独立运行）。
- 单次 GPU 实例异常关机不能丢失已完成的结果（定期 `scp -P 43676 -r` 下载 result/ 到本地）。
- 缺失 mask 文件（`.npz`）在训练前固化，同一 seed+rate 固定复用，不重新生成。
- 归一化统计量只能来自 train split，不能从 valid/test 泄露。
- 风险阈值只能用 validation split 确定，不能用 test label 搜索。
- `AUROC/AUPRC/F1` 只在 test split 同时有正负类时解释；无正类时报告为 `null`。
- 新增对比模型必须使用与主模型相同的数据划分、缺失 mask、seed、风险标签规则和阈值来源。
- 论文正文引用编号按首次出现顺序排列，不使用合并引用（如 [1-3] 或 [1,2,3]）。

## 8. 文档治理

- 规划入口：`plan/30天推进计划.md`。
- 进度跟踪：`plan/progress.md`。
- 结构蓝图：`plan/chapter-blueprints/`。
- 文献总表：`文献汇总表.md`（66条，含DOI、核验状态、阅读优先级）。
- 文献正文：`文献阅读.md`（65条引用，正式综述结构）。
- 实验协议：`统一对比实验方案.md`。
- 项目总览：`README.md`。
- 不新增平行说明文档；如果需要新专题，先在对应目录下创建，并在 `plan/progress.md` 中记录。
