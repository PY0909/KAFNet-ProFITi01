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

## Provenance caveat

The reused CH3-S04 mixed runs carry the historical code fingerprint `f44952ed…`; the 25 new runs carry `fdd06602…` from commit `2b47c07`. The code delta is limited to sanity-gate and manifest provenance paths and does not alter formal train/validation/test execution. Some historical manifests lack embedded Git provenance; new runs record it. This is an auditability caveat, not a demonstrated metric defect.

Future manifests additionally record `model_class`, `model_config`, `optimizer_config`, `training_command_hash`, and `environment_preflight_sha` so a run's model identity and recipe no longer depend on checkpoint binaries alone. The existing 30 runs predate those fields and must not be re-signed; checkpoint payloads stay `state_dict`-only and the scientific results are unchanged.

## LI-TCN vs Masked-TCN at `point_random_000`

The two runs share identical `history.json` and `checkpoint.pt` bytes. Investigation found no evidence of a shared/misassigned checkpoint:

- both checkpoints are `state_dict`-only with identical tensor key sets and zero differing tensors;
- model classes are distinct (`LITCNPoint` vs `MaskedTCNPoint`), tested in `code/tests/pilot/test_pilot_model_identity.py`;
- at 0% history missingness both encoders receive the same effective input (interpolation fill and `X_obs * M_obs` both reduce to `X_obs`), with identical architecture, parameter count, and seed;
- all five nonzero-missingness conditions differ between the two models.

This is an expected degenerate-equivalence sanity check, not a defect.

## Mask semantics

The `mask` field in point `predictions.json` is the query/evaluation-validity mask, not the history input mask. For this MetroPT point protocol it is all ones because every query target is valid; `mask_allones=true` does **not** mean a random, low-rate, block-offline, or mixed condition had no history missingness.

To compare error on observed versus missing history points, use `manifest.json` → `protocol_sha.mask_sha.{train,valid,test}` to locate the protocol mask bundle and reconstruct the input history mask. Never use `predictions.json["mask"]` for that analysis.

## Point metadata semantics

For `track=point`, `manifest.nsamples=20` is the matrix's shared configuration field, while `predictions.json.nsamples=null` is intentional because no sample ensemble is emitted. For probabilistic runs, `predictions.json.nsamples` must equal the manifest value. Validators must branch on `track` rather than require literal equality for point runs.

## ODE-RNN `point_mixed_030` boundary

The ODE-RNN `point_mixed_030` run is artifact-valid and finite, but its validation history contains large spikes (max ≈13.96 vs best ≈0.355) and late degradation, and residual analysis shows heavy-tail/extreme-prediction pathology (extreme predictions tens of σ, high kurtosis). It is retained as a diagnostic baseline, **not** as evidence of numerically stable training.

- T02 may proceed without rerunning it;
- T03 must expose `instability_flag`, `best_epoch`, `max_residual_or_prediction`, and `valid_spike`/late-degradation status, and label the run `numerically finite but stability-risk / heavy-tail pathology`;
- before multi-seed formal evaluation, decide whether to add gradient diagnostics, compare clip/no-clip, verify input scaling, test a registered stabilization policy, or reclassify ODE-RNN as diagnostic-only;
- no post-hoc clipping or recipe change is permitted in the current single-seed result.

## T01 disposition

CH3-S05-T01 is complete and its 30 baseline results are sufficient to proceed to T02. T02 must still enforce baseline-first gating per condition and must not overwrite any of these verified source keys.
