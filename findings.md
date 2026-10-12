# CUDTR 验证发现

## Current artifacts

- CUDTR: `result/metropt3_cudtr_v4_quantile_route_teacher_seed2026_valmetrics/`.
- Frozen V4 teacher: `final_result/ch3/metropt3/metropt3_unified_v4_recenttrend_residual_seed2026_bs128_retrain_20261009/`.
- MetroPT protocol: `metropt3_official_first_month_v1_val20pct-units1516948`, 15 sensors, history 168, horizon 24, stride 60, seed 2026.
- CUDTR training config: V4, quantile head, mean pooling, batch 128, 50 epochs, learning rate 2e-4, patience 8, plateau scheduler, 5 router-only warmup epochs, top-k 2; utility weights 0.2/0.1/0.01/0.05.
- Latest CUDTR metrics: test raw coverage 0.6968, width 0.3605, WIS 0.5068; calibrated coverage 0.7701, width 0.5579, WIS 0.5329. These are single-seed diagnostics.
- Prior turn added WIS and raw-unit per-epoch validation metrics while retaining Pinball checkpoint selection. Targeted tests and full suite pass (230 tests) before this goal's remaining phases.

## Method decisions

- Use central 80% interval from P10/P90; report empirical coverage, mean width, WIS, Pinball, P50 MAE/RMSE.
- Compare current median residual conformal radius against CQR nonconformity `max(q10-y, y-q90, 0)` using validation calibration data only.
- Selection/calibration separation must be chronological; use the earlier part of validation for checkpoint selection and later part for calibration, with no window/sample overlap.
- Time-block diagnostics use ordered test samples, reporting each block's own metrics and sample count; avoid relabeling highly overlapping windows as independent observations.
- Verify WIS convention for one central interval: `(0.5*|y-median| + alpha/2 * interval_score) / 1.5`, where `alpha=0.2`.

## Phase 3 isolated validation and Phase 4 diagnostics

Both models were retrained with the same chronological 50/50 validation split:
356 selection windows for Pinball checkpoint selection and 357 later windows for
calibration. The split identities are recorded in each manifest as
`chronological_tail_disjoint_from_selection`.

| Run | Raw coverage | Raw width | Raw WIS | CQR coverage | CQR width | CQR WIS |
|---|---:|---:|---:|---:|---:|---:|
| No CUDTR | 0.7138 | 0.4834 | 0.5380 | 0.8897 | 28.3925 | 7.8227 |
| CUDTR | 0.7619 | 0.3446 | 0.5022 | 0.9125 | 28.8832 | 7.9360 |

The CUDTR raw interval is narrower and has higher coverage in this controlled
rerun. CQR calibration becomes unusably wide because the later validation tail
has very large channel-wise nonconformity; this is a distribution-drift signal.

`interval_diagnostics.json` in each run reports raw and calibrated metrics for
all sensors and four ordered contiguous test blocks.

## Phase 2 calibration comparison

Using frozen checkpoints and identical validation/test predictions, CQR was
compared with the existing median-residual calibration. Both use validation
targets only.

| Run | Calibration | Coverage | Width | WIS | Pinball |
|---|---|---:|---:|---:|---:|
| CUDTR | median residual | 0.7701 | 0.5579 | 0.5328 | 0.2106 |
| CUDTR | CQR | 0.7199 | 0.4308 | 0.5136 | 0.2137 |
| No CUDTR | median residual | 0.8394 | 0.7247 | 0.5773 | 0.2162 |
| No CUDTR | CQR | 0.7674 | 0.5702 | 0.5508 | 0.2184 |

CQR produces narrower intervals and lower WIS in this diagnostic, but lowers
test coverage for both models. It is recorded as an alternative calibration
method, not selected using test labels.

## Phase 1 matched comparison

The matched no-CUDTR quantile control completed with the same official first-month
MetroPT protocol, seed, V4 encoder, quantile head, normalization, batch size,
epoch budget, optimizer, and learning rate. Only CUDTR-specific routing and
auxiliary losses were disabled.

| Run | Raw Pinball | Raw WIS | Raw coverage | Raw width | P50 MAE | P50 RMSE | Calibrated coverage | Calibrated WIS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CUDTR | 0.2174 | 0.5068 | 0.6968 | 0.3605 | 0.4833 | 1.5847 | 0.7701 | 0.5329 |
| No CUDTR | 0.2214 | 0.5415 | 0.7480 | 0.4940 | 0.5147 | 1.6858 | 0.8394 | 0.5773 |

This is a single-seed diagnostic. CUDTR improves point and raw interval loss,
but its interval is narrower and under-covers more before calibration. The
calibrated control reaches the nominal 0.80 target while CUDTR remains slightly
below it, motivating the calibration-method and validation-isolation phases.
