#!/usr/bin/env bash
# Round 1 v3c reproduction matrix: 8 runs, seed 2026, per plan/2026-10-12-round1-v3c-single-seed-reproduction-plan.md Phase D.
# Single-run failures do not stop the batch; every run logs separately into logs/round1_v3c_repro_20261012/.
set -u

ROOT=/root/autodl-tmp/kaf_v3
PYTHON=/root/miniconda3/envs/kaf_profiti/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/kaf_profiti/lib:${LD_LIBRARY_PATH:-}
DATA="$ROOT/dataset"
OUT="$ROOT/result"
LOG="$ROOT/logs/round1_v3c_repro_20261012"
mkdir -p "$OUT" "$LOG"

run_one() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name" | tee -a "$LOG/summary.log"
  if (cd "$ROOT" && PYTHONPATH="$ROOT/code" "$PYTHON" -u "$@") >"$LOG/$name.log" 2>&1; then
    echo "[$(date -Is)] DONE $name" | tee -a "$LOG/summary.log"
  else
    echo "[$(date -Is)] FAILED $name" | tee -a "$LOG/summary.log"
  fi
}

# Order 1: FD004 unified v3c (reproduction target MAE 20.16 / RMSE 27.49 / NASA 8375)
run_one round1_fd004_unified_v3c_seed2026 \
  code/run_unified_experiment.py --dataset cmapss_fd004 --data-root "$DATA/CMAPSSData" --result-root "$OUT" \
  --run-id round1_fd004_unified_v3c_seed2026 --encoder-variant v3c --head-mode point --cross-channel-mode attention \
  --history-len 80 --pred-len 1 --epochs 120 --training-cap 150 --fd004-validation all_windows \
  --fd004-normalization condition --fd004-sensor-filter standard --batch-size 128 --lr 1e-3 \
  --scheduler plateau --seed 2026 --device cuda --no-progress --evaluate-test

# Order 2: MetroPT unified v3c (establish the first registered v3c baseline)
run_one round1_metropt3_unified_v3c_seed2026 \
  code/run_unified_experiment.py --dataset metropt3 --data-root "$DATA" --result-root "$OUT" \
  --run-id round1_metropt3_unified_v3c_seed2026 --metropt-protocol official_first_month \
  --encoder-variant v3c --head-mode point --cross-channel-mode attention \
  --history-len 168 --pred-len 24 --stride 60 --epochs 120 --batch-size 128 --lr 1e-3 \
  --scheduler plateau --seed 2026 --device cuda --no-progress --evaluate-test

# Orders 3-7: TEP five comparison baselines (task D3; --lr depends on Task A4 wiring, already merged)
for model in tcn gru patchtst timesnet itransformer; do
  run_one "round1_tep_${model}_seed2026" \
    -m compare_code.run_experiment --model "$model" --dataset tep --data-root "$DATA/dataverse_files" --result-root "$OUT" \
    --run-id "round1_tep_${model}_seed2026" --history-len 96 --pred-len 1 --tep-window-stride 12 \
    --tep-max-runs-per-fault 20 --tep-valid-fraction 0.2 --epochs 50 --batch-size 128 --lr 1e-3 \
    --scheduler plateau --seed 2026 --device cuda --no-progress --evaluate-test
done

# Order 8: TEP unified v3c (head resolves to probability inside the runner)
run_one round1_tep_unified_v3c_seed2026 \
  code/run_unified_experiment.py --dataset tep --data-root "$DATA/dataverse_files" --result-root "$OUT" \
  --run-id round1_tep_unified_v3c_seed2026 --encoder-variant v3c --head-mode point --cross-channel-mode attention \
  --history-len 96 --pred-len 1 --tep-window-stride 12 --tep-max-runs-per-fault 20 --tep-valid-fraction 0.2 \
  --epochs 50 --batch-size 128 --lr 1e-3 --scheduler plateau --seed 2026 --device cuda --no-progress --evaluate-test

echo "[$(date -Is)] ROUND1 MATRIX FINISHED" | tee -a "$LOG/summary.log"
