# CUDTR Task Progress

## 2026-10-11

- Created goal to execute controlled quantile baseline, calibration comparison, validation separation, and sensor/time-block diagnostics.
- Confirmed prior WIS/validation-history edits are uncommitted and in scope; no unrelated worktree changes detected.
- Read repository instructions and relevant TDD/planning guidance.
- Verified torch23 is PyTorch 2.5.1 with CUDA available; conda command is not on PATH, invoke environment interpreter directly.
- Added two regression assertions; red run failed as expected for missing WIS and missing validation quantile history fields.
- Implemented validation raw-unit Pinball/WIS/coverage/width/median MAE/RMSE history; target tests 34 passed and full suite 230 passed.
- Ran CUDTR validation-diagnostics training with original configuration. Run: `metropt3_cudtr_v4_quantile_route_teacher_seed2026_valmetrics`; best epoch 50, validation Pinball 0.0726027, WIS 0.1942, coverage 0.8718, width 0.1601. Test raw coverage/WIS 0.6968/0.5068; calibrated 0.7701/0.5329.
- Phase 1 still needs the matched no-CUDTR quantile baseline and a phase commit.
- Matched no-CUDTR quantile baseline completed as `metropt3_v4_quantile_nocudtr_seed2026_matched`; raw/calibrated metrics were compared with CUDTR and recorded in `findings.md`.
- Phase 1 code and artifact checks are ready for commit; phases 2-4 remain pending.
- Implemented CQR as an explicit validation-only calibration option and passed the full 231-test suite.
- Offline CQR comparison completed for both frozen Phase 1 checkpoints; results are recorded in `findings.md`.
- Implemented chronological validation selection/calibration isolation with a configurable 50% selection fraction; full suite passes (232 tests). Controlled retraining is in progress.
- Retrained both models with isolated validation: `metropt3_v4_quantile_nocudtr_seed2026_valsplit50` and `metropt3_cudtr_v4_quantile_seed2026_valsplit50`.
- Added `state_quantile_diagnostics` and generated `interval_diagnostics.json` for both runs; full suite passes (233 tests).
