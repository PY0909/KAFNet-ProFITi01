# CH3-S04 Pilot Evidence Index

> Scope: the single-seed CH3-S04 center-condition pilot and preceding CH34-S03 sanity gate. Runtime files under `result/` remain machine-local evidence and are not edited by hand.

## Evidence levels

- `pilot`: complete 50-epoch run with `test_evaluation_count=1`, prediction artifact, metrics, history, checkpoint, manifest, and SHA validation. Seven runs support point-estimate validation/test, per-channel, and timing observations only.
- `sanity_train`: validation-only short run with `test_evaluation_count=0`, no prediction/checkpoint, and a separate standardized masked-query persistence contract. It is not a formal result and must not enter model ranking.
- `data_gate`: history-only persistence reference. Its raw/all-query and standardized `std_micro` values are not interchangeable with the sanity loader contract.
- All claims are `seed=2026` single-seed observations. No significance, confidence interval, or general model-superiority claim is permitted.

## Formal pilot catalog

The seven source keys are the seven `point_mixed_030` entries under `result/pilot/metropt3/runs/`:

| Evidence ID | Scientific key | Level | Required artifact chain | Allowed claim |
|---|---|---|---|---|
| S04-01 | `...|point|ff_gru|linear|point_mixed_030|2026` | pilot | manifest → history/metrics/predictions/checkpoint | point-estimate validation/test and timing |
| S04-02 | `...|point|gru_d|linear|point_mixed_030|2026` | pilot | manifest → history/metrics/predictions/checkpoint | point-estimate validation/test and timing |
| S04-03 | `...|point|li_tcn|linear|point_mixed_030|2026` | pilot | manifest → history/metrics/predictions/checkpoint | point-estimate validation/test and timing |
| S04-04 | `...|point|masked_tcn|linear|point_mixed_030|2026` | pilot | manifest → history/metrics/predictions/checkpoint | point-estimate validation/test and timing |
| S04-05 | `...|point|ode_rnn|linear|point_mixed_030|2026` | pilot | manifest → history/metrics/predictions/checkpoint | point-estimate validation/test and timing |
| S04-06 | `...|point|kst_light|linear|point_mixed_030|2026` | pilot | manifest → history/metrics/predictions/checkpoint | point-estimate validation/test and timing |
| S04-07 | `...|point|kst_light|mlp|point_mixed_030|2026` | pilot | manifest → history/metrics/predictions/checkpoint | point-estimate validation/test and timing |

The `...` prefix is a display abbreviation only. Exact keys and complete hashes are in each runtime manifest; a report claiming a numeric row must cite its Evidence ID and corresponding runtime manifest/artifact SHA set. Do not treat directory names or filesystem mtime as evidence identity.

## Sanity and diagnostic catalog

| Evidence ID | Runtime source | Level | Contract | Allowed claim |
|---|---|---|---|---|
| S03-SAN-LI | `result/pilot/metropt3/sanity/<li_tcn-key>/` | sanity_train | standardized masked-query LOCF persistence micro, validation-only, no test | finite/updated/validation-improved and gate status after aligned rerun |
| S03-SAN-ODE | `result/pilot/metropt3/sanity/<ode_rnn-key>/` | sanity_train | same as above | same; never formal ranking |
| S01-GATE | `result/pilot/metropt3/diagnostics/data_gate.json` | data_gate | raw and standardized all-query floors | formal all-query reference only |

Historical sanity manifests generated before the aligned baseline implementation are superseded for `beat_naive`; they must not be presented as current gate evidence. After the code fix, rerun only the two validation-only sanity jobs and regenerate their manifests.

## Provenance contract

A complete pilot evidence chain is:

`claim/table -> aggregate command -> scientific key/seed -> prediction/metrics -> checkpoint -> resolved config -> manifest -> protocol/shared-artifact/code hash -> preflight identity`.

New manifests record full commit SHA, clean/dirty state, and UTC generation time. The existing formal run artifacts retain their original code/protocol/artifact hashes; this index does not rewrite or re-sign them. `result/` is ignored runtime evidence, so a future formal report must attach exact manifest/artifact SHA sets and validator command output (or an evidence-bundle SHA) rather than relying on this catalog's abbreviated display keys.

The CH3-S04 pilot `go` is outside D3-D5 formal stage-gate closure: `plan/stage-gates.md` remains authoritative for those gates.
