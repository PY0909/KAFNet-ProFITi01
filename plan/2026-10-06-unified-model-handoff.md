# Unified Model Reimplementation Handoff

## Workspace

- Repository: `/home/new_work/new_work`
- Branch: `main`
- Current code commit: `8a324c3 fix: complete dataset loaders and task semantics`
- Date: 2026-10-06

## Completed work

The shared encoder and dataset-specific task-head pipeline has been implemented and wired into the task-aware runner.

- `UnifiedTemporalEncoder` and `SharedEncoderTaskModel` exist.
- MetroPT, C-MAPSS FD004 and TEP task heads are registered.
- C-MAPSS RUL-only history windows and raw official test RUL are implemented.
- TEP `.dat` loading supports `d00..d21`, `_te` files, normal `N x 52` data and the checked-out transposed `52 x N` layout.
- TEP train/validation chronological separation and official test loading are implemented.
- TEP detection/diagnosis metrics include LogLoss, Brier and ECE paths.
- Dataset-task configs exist under `configs/ch3`, `configs/ch4` and `configs/ch5`.

## Verification evidence

- Targeted regression tests: `23 passed`.
- Full test suite: `435 passed, 6 skipped`.
- The six skipped tests require TEP RData files absent from the current checkout.
- Python compilation and `git diff --check` passed.
- Real TEP `.dat` smoke produced train/valid/test window counts `419/419/2002`, sample shape `[50, 52]`, and finite statistics.

These are implementation and smoke checks only. No formal multi-seed paper result has been established.

## Current uncommitted files

The following are pre-existing user document changes and must not be reverted or mixed into an implementation commit:

- `3.2_结构框架.md`
- `README.md`
- `plan/2026-10-05-dataset-specific-ch3-ch5-experiment-plan.md`
- `plan/experiment-protocol.md`
- `plan/implementation-plan.md`
- `plan/project-overview.md`
- `plan/2026-10-05-shared-encoder-task-head-future-direction.md`

## Architecture assessment

The current implementation uses a common encoder structure but trains dataset-local parameters. This matches the future-direction document; do not describe it as cross-dataset parameter sharing.

The current main risks are:

1. `HierarchicalMissingnessKAFEncoder.encode_with_patches()` passes raw `X` into the local KAF path. Values where `M_obs == 0` can affect neighboring convolution outputs. A minimal behavior check observed a non-zero prediction change after perturbing masked values.
2. `UnifiedTemporalEncoder.VERSION` is named `hierarchical_kaf_v1` while it wraps a missingness-aware hierarchical encoder. Version naming needs to be made explicit.
3. `kafnet_encoder.py` contains unreachable legacy code after an unconditional return.
4. `RULHead._apply_cap()` is defined but unused. Raw/capped target semantics need an explicit decision and test.
5. `TEPFaultHead` returns binary detection probability as the formal output and stores diagnosis probability in metadata. Detection latency is not yet a first-class task/evaluation interface.

## Missing-data code status

Missing-data code has not been deleted and should not be deleted wholesale.

- `industrial/missing.py` remains used by all dataset loaders.
- `models/missingness_features.py` remains used by the hierarchical encoder.
- `experiments/masks.py` remains used by pilot timeline-mask protocols.
- `baselines/point.py` and `baselines/grud.py` contain comparison-model adapters.
- `kst_probflow.py`, `kst_light.py` and `kst_flow.py` remain needed for legacy identity and existing tests.

The new implementation should isolate the main model in `code/kaf_profiti/models/unified/` and migrate through the registry. Delete old code only after an import/reference audit and compatibility verification.

## Required next action

Start with Stage 1 from `plan/2026-10-06-unified-model-reimplementation-prompt.md`:

1. Create the `models/unified/` package and its public contracts.
2. Add tests before implementation details.
3. Implement mask-safe encoder behavior in the new package.
4. Run targeted tests, then the full suite.
5. Commit the stage separately and record the commit in `plan/progress.md`.

## Canonical documents

- `AGENTS.md`
- `plan/2026-10-05-shared-encoder-task-head-code-refactor-plan.md`
- `plan/2026-10-05-shared-encoder-task-head-future-direction.md`
- `plan/2026-10-05-dataset-specific-ch3-ch5-experiment-plan.md`
- `plan/experiment-protocol.md`
- `plan/implementation-plan.md`
- `plan/stage-gates.md`
- `plan/progress.md`
- `3.2_结构框架.md`

## Guardrails

- Do not modify existing `result/` metrics, predictions or checkpoints.
- Do not start formal long training before the new encoder passes masked-input invariance tests.
- Do not use smoke metrics as paper claims.
- Do not claim real TEP training or latency validation without the required data and artifacts.
- Keep one commit per completed implementation stage.
