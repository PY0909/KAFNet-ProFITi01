# Round 1：Unified V3C 单 seed 复现与第三章三 seed 执行计划

> 日期：2026-10-12  
> 状态：规划文档，尚未执行数据拷贝、Git 推送或训练。  
> 主工程：`/Users/ppy/研/kaf_profiti_unified_v3`。  
> 事实依据：本仓库当前代码、`plan/experiment-protocol.md`、`plan/stage6_run_matrix_20261011.sh`、`plan/2026-10-09-current-baseline-handoff.md`。  
> 文档结构参考：按 Phase、Task、门禁和证据记录组织；不继承其他工程的实验内容、模型 ID 或结果。

## 0. 目标与边界

第三章最终矩阵为：

```text
MetroPT-3 / C-MAPSS FD004 / TEP
× {unified v3c, tcn, gru, patchtst, timesnet, itransformer}
× seeds {2026, 2027, 2028}
= 54 个正式 run
```

C-MAPSS 只跑 FD004。`v3c` 是第三章 point-only 主模型；`v4` 属于第四章，本计划完全不运行、不迁移、不解释 v4。

本计划分两轮：

- **Round 1：** FD004 unified v3c 和 MetroPT unified v3c 先跑；随后 TEP 五个 comparison baseline；最后 TEP unified v3c。共 8 个 seed 2026 诊断 run。
- **Round 2：** Round 1 通过门禁后，运行第三章完整 54-run、3-seed 矩阵，并生成 mean±std 汇总。Round 1 单 seed 结果不能直接进入最终性能表。

## 1. 事实与参数决策

### 1.1 已锁定参数

| 数据集 | Round 1 参数 | Round 2 正式参数 |
|---|---|---|
| FD004 | history 80、pred 1、120 epochs、training cap 150、`all_windows`、condition normalization、standard 14-sensor filter | 以冻结的第三章配置为准，不复用 Round 1 结果 |
| MetroPT | `official_first_month`、history 168、pred 24、stride 60、120 epochs | 以冻结的第三章配置为准 |
| TEP | history 96、pred 1、window stride 12、50 epochs、`tep-max-runs-per-fault=20` | 先通过全量数据显存/耗时 preflight，再冻结为 `0`；若只能使用 20，必须降级声明为预注册子集协议 |
| 所有 run | batch 128、seed 2026、Adam lr 1e-3、plateau scheduler、显式 `--evaluate-test` | seeds 2026/2027/2028，其他科学参数冻结后执行 |
| early stopping | 禁用：不传 `--patience`，完整执行预算 epoch；plateau scheduler 只调学习率 | 由冻结的第三章协议统一决定，同一数据集六模型一致 |

TEP 的 50 epochs 和 `20` 均采用 `plan/stage6_run_matrix_20261011.sh` 的显式 stage6 口径，用于 Round 1 复现；不能依赖 CLI 默认值。Round 2 开始前必须对 `0` 做单模型 1-epoch GPU preflight。若全量数据可运行，54 个正式 run 中的 TEP 全部显式使用 `0`；若当前 resident-batch 实现无法承载，必须先优化 loader 或把正式口径降级为明确的 20-run 子集，不能静默沿用默认值。

### 1.2 TEP head 语义

统一 CLI 的 `--head-mode` 目前只有 `point` 和 `quantile`。TEP 是 `fault_detection_diagnosis` 任务，传入 `--head-mode point` 后，训练入口（`experiments/entrypoint.py:338`）会将实际配置规范为 `probability`。因此 TEP unified 命令仍传 `point` 以满足 CLI，但放行依据必须是 manifest 和 test artifact：

```text
task = fault_detection_diagnosis
head_mode = probability
diagnosis_probability 存在
threshold_selection.source_split = validation
```

### 1.3 现有 protocol checker 的边界

当前 `code/kaf_profiti/experiments/protocol_checker.py` 只检查固定命名的 FD004 六模型，并不能验证本轮 8 个 run，也不覆盖 MetroPT/TEP。因此本轮不能只执行现有 checker 就宣称通过；需要新增或扩展一个显式接收 run 目录的 Round 1 manifest audit。

## 2. 复现判定标准

### 2.1 FD004 unified v3c

冻结靶值为 MAE 20.16、RMSE 27.49、NASA score 8375。对三项分别计算 `abs(observed-target)/target`：

| 判定 | 条件 |
|---|---|
| 复现成功 | MAE ≤5%、RMSE ≤5%、NASA score ≤10% |
| 待裁决 | 三项均未超过 15%，但未满足上一行；逐字段核对 manifest、split、normalizer、sensor filter、training cap、epoch/early-stop 和 checkpoint |
| 失败/停止 | 任一项超过 15%，或协议、数据、代码或环境身份不一致 |

不能只看 MAE；RMSE 和 NASA score 是辅助判定，但必须同时记录。

### 2.2 MetroPT unified v3c

当前没有已登记的 v3c 旧靶值；Round 1 产出首个 v3c 基线。该结果只按自身 manifest、split、数据、训练历史和 finite artifact 判定，不与第四章 v4 做优劣或复现比较。唯一附加护栏是**非正式量级检查**：历史 V4 单 seed overall MAE 为 0.485，若本次 v3c raw MAE 超过 2.0（约 4 倍量级），必须先排查数据、归一化与评估链路，再决定是否登记为基线；该检查只防数量级异常，不用于优劣判定，也不把 v4 引入为比较对象。

### 2.3 TEP

Round 1 仅建立 seed 2026 诊断基线。必须检查 classification accuracy、Balanced Accuracy、Macro-F1、逐类 recall、event-level 指标均为 finite，且阈值只来自 validation。Round 1 的 20-run 截断必须写入 manifest 和报告，资格标记为 diagnostic，不得进入第三章全量表。

## 3. Phase A：数据拷贝与本地验证

- [x] **Phase A 完成：数据 SHA 一致，本地测试/smoke 通过，训练 provenance 与 TEP 公平性缺口已补齐。**

> 2026-10-12 执行记录：A1–A4 全部完成并通过断言式验收。A1–A3：拷贝 22 文件、TEP 官方清单 4/4 OK、源/目标全量 SHA-256 22/22 一致、pytest 250 passed、FD004/MetroPT unified v3c 1-epoch CPU smoke 全产物齐全且指标 finite（MetroPT `split_identity=metropt3_official_first_month_v1_val20pct-units1516948` 与既有冻结 run 一致；TEP 本地 smoke 按 libiconv 约束跳过，留待 Phase C）。A4：TDD 完成（红 15 失败 → 绿），全量 **265 passed**（新增 15 个测试），真实数据 FD004/MetroPT smoke 复跑确认 manifest 新字段端到端齐全。详见 [progress.md](progress.md) 当日条目。

### Task A1：只拷贝原始数据

- [x] **Task A1 完成：三个原始数据目录已迁移，未覆盖任何 v3 代码或结果。**

不能用 `kst_light_v2` 的代码、results、archive 或 checkpoint 覆盖 v3。只拷贝以下目录：

```bash
mkdir -p dataset
cp -Rc "/Users/ppy/研/kst_light_v2/dataset/CMAPSSData" dataset/
cp -Rc "/Users/ppy/研/kst_light_v2/dataset/metropt+3+dataset" dataset/
cp -Rc "/Users/ppy/研/kst_light_v2/dataset/dataverse_files" dataset/
```

`dataset/` 已在 `.gitignore` 中，原始数据不进入 GitHub；历史 `result/` 不复制、不手工修改。

### Task A2：SHA-256 校验

- [x] **Task A2 完成：TEP 官方清单与源/目标全量 SHA-256 均一致。**

TEP 官方清单：

```bash
(cd dataset/dataverse_files && shasum -a 256 -c SHA256SUMS.txt)
```

完整源/目标目录比较必须统一 SHA-256 并排序，不能使用默认 SHA-1 或未排序的 `find` 输出：

```bash
make_sha256_manifest() {
  local root="$1"
  (cd "$root" && find . -type f -print | LC_ALL=C sort | while IFS= read -r path; do
    shasum -a 256 "$path"
  done)
}

make_sha256_manifest "/Users/ppy/研/kst_light_v2/dataset" > /tmp/kst_light_v2_dataset.sha256
make_sha256_manifest "dataset" > /tmp/kaf_v3_dataset.sha256
diff -u /tmp/kst_light_v2_dataset.sha256 /tmp/kaf_v3_dataset.sha256
```

预期完整 diff 无输出；TEP 清单全部 `OK`。任何差异均停止后续训练。

### Task A3：本地 smoke 与环境边界

- [x] **Task A3 完成：250 个测试通过，MetroPT/FD004 CPU smoke 产物 finite。**

本地测试必须使用已验证的 `/opt/anaconda3/envs/kaf_profiti/bin/python`，不能使用没有 torch/pytest 的系统 Python：

```bash
PYTHONPATH=code /opt/anaconda3/envs/kaf_profiti/bin/python -m pytest code/tests/ -q
```

预期 `250 passed`。本地只执行 MetroPT/FD004 unified v3c 各 1 epoch 的 CPU loader/artifact smoke。当前 TEP loader 显式加载 Linux `libiconv.so`；Mac 缺少该库时不执行 TEP smoke，TEP smoke 改在 AutoDL 环境门禁执行，不把环境缺库误报为模型失败。

### Task A4：补齐实验 provenance 后再跑训练

- [x] **Task A4 完成：两个 runner 记录同一组科学身份字段，TEP loss/概率语义可公平比较。**

当前 manifest 尚未完整记录 `epochs`、`batch_size`、`scheduler`、`patience`、history/pred/stride 和请求的 `tep_max_runs_per_fault`；严格 audit 不能从命令文本反推这些字段。Round 1 前必须完成以下最小代码改动：

- 修改 `code/kaf_profiti/industrial/datasets.py`：各 bundle metadata 显式记录实际 history/pred/stride、协议参数和 TEP 请求/实际选择的 run 数。
- 当前 `split_identity` 只包含 unit 数量、`normalizer_identity` 只包含 feature 数量，不能证明内容相同；新增 canonical `split_members_sha256`、`window_catalog_sha256`、`normalizer_stats_sha256`，并写入 manifest。
- 修改 `code/kaf_profiti/experiments/entrypoint.py`：manifest 写入 epochs、batch size、learning rate、scheduler、patience、device/AMP 和 resolved dataset metadata。
- 修改 `compare_code/cli.py`：新增 `--lr` 并显式传给 `run_bundle(learning_rate=...)`；禁止依赖隐藏的 1e-3 默认值，确保 Round 2 可以做 validation-only tuning。
- 修改 `compare_code/models/base.py` 及 factory 接线：TEP baseline 使用与 unified 相同的 train-frequency class weights，并从 diagnosis probability 派生 `detection_probability = 1 - p(class 0)`，使五个 baseline 与 unified 使用同一训练权重、validation-only threshold 和 event evaluator。
- TEP head 模式从 checkpoint config 或独立 manifest 字段核验；不得检查当前并不存在的 `manifest.head_mode`。
- 修改 `code/tests/test_experiment_entrypoint.py`、`code/tests/test_compare_models.py`、`code/tests/test_cli.py` 和 `code/tests/test_compare_cli.py`，覆盖内容 SHA、TEP detection probability 以及 unified/comparison 两个 runner 的 provenance 一致性。
- 运行全量 pytest；只有测试通过后，才冻结 Git commit 并启动 AutoDL smoke。

## 4. Phase B：Git 身份冻结与 GitHub

- [ ] **Phase B 完成：v3c 代码身份已冻结到可追溯分支并推送，数据/结果/指针备份未入库。**

> 2026-10-12 执行记录：指针备份（仓库外 `/private/tmp/`）→ `git init -b unified-v3-ch3-seeds` → import 审计版 `requirement.txt`（scikit-learn 无导入、不列入）→ 暂存审查（111 文件，无 dataset/result/备份入索引）→ commit 冻结完成。**push 停在门禁**：github.com 443 直连不通且本机无代理，远端侦察不可达，按规划走独立快照分支兜底（不连通远端历史属预期，已在 progress.md 登记）。待代理可用后：先 `git ls-remote` 复核远端是否已有同名分支，再 `git push -u origin unified-v3-ch3-seeds`，不静默强推。

当前 v3 的 `.git` 是失效的服务器 worktree 指针，其分支元数据只存在于 `/home/new_work/kaf_profiti`（RTX 4060 Ti 机器）的 `.git/worktrees` 下；AutoDL 上的 `new_work` 是独立 clone，只包含**已推送到 GitHub 的分支**。因此主路径不是“在 AutoDL 恢复 worktree 分支”，而是：先用 `git ls-remote --heads origin`（可加 `--tags`）检查远端是否已存在与本地代码对应的分支或等价树——存在则比较树 SHA 后从该历史创建 `unified-v3-ch3-seeds`；不存在（含 4060 机器不可达的情形）直接使用下列“独立快照分支”兜底。兜底分支与远端历史不连通属预期行为，须在 Phase F 登记中注明分支来源。备份放在仓库外，避免被 `git add -A` 提交：

```bash
git_pointer_backup="/private/tmp/kaf_profiti_unified_v3.git.server-pointer.$(date +%Y%m%d%H%M%S).bak"
mv .git "$git_pointer_backup"
git init -b unified-v3-ch3-seeds
```

创建经过实际 import 审计的依赖清单；至少包含当前代码需要的 `torch`、`numpy`、`pandas`、`scikit-learn`、`pytest`、`tqdm`、`pyreadr`。不能无审计地复制旧工程依赖文件。

提交前确认数据、结果和备份均不在索引：

```bash
git check-ignore -q dataset result
git status --short
git ls-files | rg '(^|/)(dataset|result|results|archive|\\.git\\.server-pointer)' || true
git add -A
git diff --cached --check
git commit -m "ch3: freeze unified v3c reproduction snapshot"
git remote add origin https://github.com/PY0909/KAFNet-ProFITi01.git
git ls-remote --heads origin unified-v3-ch3-seeds
git push -u origin unified-v3-ch3-seeds
```

若远端分支已存在，先比较 commit 和工作区，不得静默强推；无 GitHub 凭据时停在推送门禁。不得为了省事把独立 root snapshot 强推到 `main`。

## 5. Phase C：AutoDL 准备

- [ ] **Phase C 完成：远端 commit、环境、数据 SHA、250 tests 和三数据集 GPU smoke 全部通过。**

在 `/root/autodl-tmp/kaf_v3` 独立 clone `unified-v3-ch3-seeds`，不修改旧 `/root/autodl-tmp/new_work`。

- 记录 commit SHA、`git status --porcelain`、Python/Torch/CUDA 版本。
- 优先服务器内复制旧 `new_work/dataset`；缺失或 SHA 不一致才从 Mac 上传。
- 确认 `pyreadr` 可导入且 `libiconv.so` 可加载；否则先修环境。
- 服务器解释器执行全量 pytest，`250 passed` 才放行。
- GPU 上执行 unified × FD004、unified × MetroPT、tcn × TEP 的 1-epoch smoke；TEP smoke 显式 `--tep-max-runs-per-fault 2`，只验证 loader、finite 和 artifact。

## 6. Phase D：Round 1 八个 run

- [ ] **Phase D 完成：8/8 seed-2026 诊断 run 均生成完整 artifact。**

所有命令从仓库根执行。Unified 和 comparison 命令均显式提供 `--result-root`；comparison runner 显式设置 `PYTHONPATH`。批处理脚本为每个 run 单独日志，单 run 失败后继续并写入 `summary.log`。

```bash
ROOT=/root/autodl-tmp/kaf_v3
PYTHON=/root/anaconda3/envs/torch23/bin/python
DATA="$ROOT/dataset"
OUT="$ROOT/result"
LOG="$ROOT/logs/round1_v3c_repro_20261012"
mkdir -p "$OUT" "$LOG"
```

### Task D1：FD004 unified v3c（顺序 1）

- [ ] **Task D1 完成：FD004 v3c artifact 已生成。**

```bash
PYTHONPATH="$ROOT/code" "$PYTHON" -u "$ROOT/code/run_unified_experiment.py" \
  --dataset cmapss_fd004 --data-root "$DATA/CMAPSSData" --result-root "$OUT" \
  --run-id round1_fd004_unified_v3c_seed2026 \
  --encoder-variant v3c --head-mode point --cross-channel-mode attention \
  --history-len 80 --pred-len 1 --epochs 120 --training-cap 150 \
  --fd004-validation all_windows --fd004-normalization condition \
  --fd004-sensor-filter standard --batch-size 128 --lr 1e-3 \
  --scheduler plateau --seed 2026 --device cuda --evaluate-test
```

### Task D2：MetroPT unified v3c（顺序 2）

- [ ] **Task D2 完成：MetroPT v3c artifact 已生成。**

```bash
PYTHONPATH="$ROOT/code" "$PYTHON" -u "$ROOT/code/run_unified_experiment.py" \
  --dataset metropt3 --data-root "$DATA" --result-root "$OUT" \
  --run-id round1_metropt3_unified_v3c_seed2026 \
  --metropt-protocol official_first_month --encoder-variant v3c \
  --head-mode point --cross-channel-mode attention \
  --history-len 168 --pred-len 24 --stride 60 --epochs 120 \
  --batch-size 128 --lr 1e-3 --scheduler plateau \
  --seed 2026 --device cuda --evaluate-test
```

### Task D3：TEP 五个 comparison baseline（顺序 3–7）

- [ ] **Task D3 完成：五个 TEP baseline artifact 均已生成。**

依赖声明：本 Task 命令中的 `--lr 1e-3` 依赖 Task A4 已为 `compare_code/cli.py` 完成 `--lr` 接线；A4 未完成时本命令会因未知参数直接失败，不得跳过 A4 执行本 Task。

```bash
for model in tcn gru patchtst timesnet itransformer; do
  PYTHONPATH="$ROOT/code" "$PYTHON" -u -m compare_code.run_experiment \
    --model "$model" --dataset tep --data-root "$DATA/dataverse_files" --result-root "$OUT" \
    --run-id "round1_tep_${model}_seed2026" \
    --history-len 96 --pred-len 1 --tep-window-stride 12 \
    --tep-max-runs-per-fault 20 --tep-valid-fraction 0.2 \
    --epochs 50 --batch-size 128 --lr 1e-3 --scheduler plateau \
    --seed 2026 --device cuda --evaluate-test \
    >"$LOG/round1_tep_${model}_seed2026.log" 2>&1 \
    || echo "FAILED round1_tep_${model}_seed2026" >>"$LOG/summary.log"
done
```

### Task D4：TEP unified v3c（顺序 8）

- [ ] **Task D4 完成：TEP unified v3c artifact 已生成且实际 head 为 probability。**

```bash
PYTHONPATH="$ROOT/code" "$PYTHON" -u "$ROOT/code/run_unified_experiment.py" \
  --dataset tep --data-root "$DATA/dataverse_files" --result-root "$OUT" \
  --run-id round1_tep_unified_v3c_seed2026 \
  --encoder-variant v3c --head-mode point --cross-channel-mode attention \
  --history-len 96 --pred-len 1 --tep-window-stride 12 \
  --tep-max-runs-per-fault 20 --tep-valid-fraction 0.2 \
  --epochs 50 --batch-size 128 --lr 1e-3 --scheduler plateau \
  --seed 2026 --device cuda --evaluate-test
```

## 7. Phase E：八 run 审计门禁

- [ ] **Phase E 完成：8-run artifact、内容 SHA、跨 runner 公平性和 FD004 复现判定全部通过。**

新增或扩展 `code/kaf_profiti/experiments/round1_protocol_audit.py`，显式接收 run 目录。它必须检查：

```text
8 个 manifest、checkpoint、metrics.json、test_metrics.json、efficiency.json 均存在；
run_id/model_name/dataset/task/seed 对应；
split_identity、split_members_sha256、window_catalog_sha256、target_schema、dataset_schema、normalizer_stats_sha256、evaluator_version 一致；
FD004/MetroPT/TEP 的窗口、预算、training_cap、sensor filter 或 tep-max-runs 明确记录；
TEP comparison 与 unified 的 split、normalizer、target、evaluator 跨 runner 一致；
test_evaluation_count == 1 且全部指标 finite；
TEP unified checkpoint config（或新增的 manifest 字段）head_mode == probability；
TEP threshold_selection.source_split == validation；
Round 1 TEP 的 max-runs=20 明确标记 diagnostic。
```

执行：

```bash
PYTHONPATH="$ROOT/code" "$PYTHON" -m kaf_profiti.experiments.round1_protocol_audit \
  --result-root "$OUT" \
  --run-id round1_fd004_unified_v3c_seed2026 \
  --run-id round1_metropt3_unified_v3c_seed2026 \
  --run-id round1_tep_tcn_seed2026 \
  --run-id round1_tep_gru_seed2026 \
  --run-id round1_tep_patchtst_seed2026 \
  --run-id round1_tep_timesnet_seed2026 \
  --run-id round1_tep_itransformer_seed2026 \
  --run-id round1_tep_unified_v3c_seed2026
```

8/8 artifact、跨 runner audit 和 FD004 三指标判定全部通过后，才允许进入 Round 2。

## 8. Phase F：结果归档与 Round 2 放行

- [ ] **Phase F 完成：Round 1 证据已登记，Round 2 的 tuning、TEP 数据规模和 54-run 配置已冻结。**

- 下载 manifest、metrics、test_metrics、efficiency、history、checkpoint SHA 和 audit report；不手工修改任何 JSON 指标。
- 把 run_id、seed、commit SHA、数据 SHA、manifest SHA、判定和未决风险写入 `plan/progress.md`。
- FD004 按 MAE/RMSE/NASA 三指标判定；MetroPT 登记 v3c 首个基线；TEP 登记 20-run diagnostic baseline。
- 只有 Round 1 门禁通过、AutoDL 环境可复现、Git 分支已推送，才建立 Round 2 执行计划。
- Round 1 的 test 结果只用于发现实现/数值异常，不得用于选择正式学习率或训练预算。Round 2 前按 `plan/experiment-protocol.md` 使用 train/validation-only tuning 冻结各数据集×模型配置；54 个正式 run 不包含 tuning run。
- Round 2 先对 TEP `--tep-max-runs-per-fault 0` 做 1-epoch GPU preflight。通过后 54-run 中的 TEP 全部显式使用 `0`；若不通过，先优化数据加载，或经书面范围变更统一使用 20 并降级结论。不得复用 Round 1 截断 artifact。

## 9. 八个 Round 1 run 清单

| 顺序 | run-id | 模型 | 数据集 | 角色 |
|---:|---|---|---|---|
| 1 | `round1_fd004_unified_v3c_seed2026` | unified v3c | FD004 | 复现靶值 |
| 2 | `round1_metropt3_unified_v3c_seed2026` | unified v3c | MetroPT | 建立 v3c 基线 |
| 3 | `round1_tep_tcn_seed2026` | TCN | TEP | 对比基线 |
| 4 | `round1_tep_gru_seed2026` | GRU | TEP | 对比基线 |
| 5 | `round1_tep_patchtst_seed2026` | PatchTST | TEP | 对比基线 |
| 6 | `round1_tep_timesnet_seed2026` | TimesNet | TEP | 对比基线 |
| 7 | `round1_tep_itransformer_seed2026` | iTransformer | TEP | 对比基线 |
| 8 | `round1_tep_unified_v3c_seed2026` | unified v3c | TEP | 对比完成后建立基线 |

## 10. 本轮禁止事项

- 不跑 Unified V4、第四章概率模型或第五章风险实验。
- 不把 `kst_light_v2` 旧代码、旧结果或旧 checkpoint 当作 v3 正式结果。
- 不把 smoke、单 seed 或 TEP 20-run 截断结果写成第三章最终性能结论。
- 不使用 test 标签选择学习率、epoch、checkpoint、TEP 阈值或正式数据规模。

## 11. 时间预算（用于安排 AutoDL 开机窗口）

| 阶段 | 预计耗时 | 位置 |
|---|---|---|
| Task A1–A3（拷贝、SHA 校验、本地测试/smoke） | 约 0.5 小时 | Mac |
| Task A4（provenance/公平性代码改动 + 测试） | 约 0.5–1 天，本轮最大前置工作量 | Mac |
| Phase B（git 冻结与推送） | 约 0.5 小时 | Mac |
| Phase C（AutoDL 环境 + GPU smoke） | 约 0.5–1 小时；TEP 数据若需从 Mac 上传另加 0.5–1 小时（1.3G） | AutoDL |
| Phase D（8 个 run） | 约 2.5–5.5 GPU 小时：FD004 5–15 分钟、MetroPT 30–60 分钟、TEP 每个 15–40 分钟（3090 量级估计） | AutoDL |
| Phase E–F（审计与归档） | 约 1 小时 | AutoDL + Mac |

GPU 租用窗口建议：A4 完成并推送后再开机，一次开机覆盖 Phase C–F（约半天）。
