#!/usr/bin/env bash
set -u

PYTHON=/root/anaconda3/envs/torch23/bin/python
ROOT=/home/new_work/kaf_profiti
DATA=$ROOT/dataset
OUT=$ROOT/result
LOG=$ROOT/logs/stage6_20261011
mkdir -p "$LOG"

run_one() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name" | tee -a "$LOG/summary.log"
  if PYTHONPATH="$ROOT/code" "$PYTHON" -u "$ROOT/code/run_unified_experiment.py" "$@" \
      --seed 2026 --encoder-variant v4 --batch-size 128 --device cuda \
      --scheduler plateau --patience 8 --evaluate-test \
      --run-id "$name" >"$LOG/$name.log" 2>&1; then
    echo "[$(date -Is)] DONE $name" | tee -a "$LOG/summary.log"
  else
    echo "[$(date -Is)] FAILED $name" | tee -a "$LOG/summary.log"
  fi
}

# MetroPT: official first-month protocol, 168/24 windows.
for arm in attention cudtr_only cudtr_fallback; do
  case "$arm" in
    attention)
      extra=(--cross-channel-mode attention);;
    cudtr_only)
      extra=(--cross-channel-mode cudtr_only --cudtr --cudtr-top-k 2);;
    cudtr_fallback)
      extra=(--cross-channel-mode attention --cudtr --cudtr-top-k 2 --cudtr-fallback-init 0.5);;
  esac
  run_one "stage6_20261011_metropt3_${arm}_v4" \
    --dataset metropt3 --data-root "$DATA" --result-root "$OUT" \
    --metropt-protocol official_first_month --history-len 168 --pred-len 24 \
    --stride 60 --epochs 50 --head-mode point "${extra[@]}"
done

# FD004: V3C/V4-compatible formal protocol, 14 sensors, condition normalization.
for arm in attention cudtr_only cudtr_fallback; do
  case "$arm" in
    attention) extra=(--cross-channel-mode attention);;
    cudtr_only) extra=(--cross-channel-mode cudtr_only --cudtr --cudtr-top-k 2);;
    cudtr_fallback) extra=(--cross-channel-mode attention --cudtr --cudtr-top-k 2 --cudtr-fallback-init 0.5);;
  esac
  run_one "stage6_20261011_fd004_${arm}_v4" \
    --dataset cmapss_fd004 --data-root "$DATA/CMAPSSData" --result-root "$OUT" \
    --history-len 80 --pred-len 1 --epochs 120 --training-cap 150 \
    --fd004-validation all_windows --fd004-normalization condition \
    --fd004-sensor-filter standard --head-mode point "${extra[@]}"
done

# TEP: validation-only event threshold and balanced train-frequency weights.
for arm in attention cudtr_only cudtr_fallback; do
  case "$arm" in
    attention) extra=(--cross-channel-mode attention);;
    cudtr_only) extra=(--cross-channel-mode cudtr_only --cudtr --cudtr-top-k 2);;
    cudtr_fallback) extra=(--cross-channel-mode attention --cudtr --cudtr-top-k 2 --cudtr-fallback-init 0.5);;
  esac
  run_one "stage6_20261011_tep_${arm}_v4" \
    --dataset tep --data-root "$DATA/dataverse_files" --result-root "$OUT" \
    --history-len 96 --pred-len 1 --epochs 50 --tep-window-stride 12 \
    --tep-max-runs-per-fault 20 --head-mode point "${extra[@]}"
done

echo "[$(date -Is)] MATRIX FINISHED" | tee -a "$LOG/summary.log"
