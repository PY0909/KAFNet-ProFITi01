# CH3-S05-T01 Baseline Evidence Audit

> Audit scope: 30 baseline point runs downloaded from AutoDL under `~/Downloads/run/runs` on 2026-09-18. This catalog records validation results; runtime artifacts remain outside Git and are not edited.

## Coverage

| Model | mixed_030 reused | random_000 | random_030 | random_070 | low_rate_030 | block_offline_030 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| LI-TCN | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| FF+GRU | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| Masked-TCN | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| GRU-D | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| ODE-RNN | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| **Total** | **5** | **5** | **5** | **5** | **5** | **5** | **30** |

## Audit result

All 30 baseline run directories passed the downloaded-artifact audit:

- `status=completed`, `run_level=pilot`, model identity, and `test_evaluation_count=1` are valid;
- history, metrics, checkpoint, and predictions exist and every listed SHA matches the local downloaded file;
- every history has 50 entries and finite validation scores;
- within each condition, all five baselines share one non-mask protocol identity;
- each model has six distinct condition mask identities and one shared target schema;
- realized rates match the registered conditions: random 0/30/70, low-rate 30, block-offline 30, and mixed 30;
- no duplicate metrics or prediction artifact SHA exists across distinct model/condition keys.

`random_000` LI-TCN and Masked-TCN share history/checkpoint SHA because at zero missingness their input transforms are mathematically equivalent under the same seed; their metrics and predictions remain distinct, and all nonzero-missingness conditions differ.

## Provenance caveat

The reused CH3-S04 mixed runs carry the historical code fingerprint `f44952ed…`; the 25 new runs carry `fdd06602…` from commit `2b47c07`. This is expected and documented in `plan/pilot-evidence-index.md`: the code delta is limited to sanity-gate and manifest provenance paths and does not alter formal train/validation/test execution. Some historical manifests lack embedded Git provenance; new runs record it. This is an auditability caveat, not a demonstrated metric defect.

## T01 disposition

CH3-S05-T01 is complete and its 30 baseline results are sufficient to proceed to T02. T02 must still enforce baseline-first gating per condition and must not overwrite any of these verified source keys.
