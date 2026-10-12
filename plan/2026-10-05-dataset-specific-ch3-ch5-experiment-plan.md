# Dataset-Specific Experiments for Chapters 3--5

## Scope and decision

This plan defines the experiment logic around the native task of each benchmark. The three datasets are not forced into one common point-forecast protocol:

| Chapter | Dataset | Native task | Primary unit of evaluation |
|---|---|---|---|
| 3 | MetroPT-3 | online compressor/APU failure prediction | failure event / time interval |
| 4 | C-MAPSS FD004 | run-to-failure degradation and final RUL prediction | engine |
| 5 | Tennessee Eastman Process (TEP) | process fault detection and early detection | simulation run and fault type |

The common methodological thread is historical multivariate observations -> model output -> operational decision. The output, labels, splits, and metrics remain task-specific.

## Evidence from public protocols

- MetroPT-3 is a 1 Hz multivariate APU stream with maintenance-report failure intervals. The dataset paper states that failure prediction is evaluated by overlap between predicted and observed intervals, with false positives and false negatives, and that the operational objective is at least two hours of warning before the train becomes non-operational. The raw release is a complete, regularly sampled benchmark.
- NASA C-MAPSS provides complete run-to-failure training trajectories, truncated test trajectories, and a final-cycle RUL vector. FD004 contains 248 training engines, 249 test engines, six operating conditions, and two fault modes. The official test file and RUL file are retained; no artificial availability is required for the main RUL claim.
- Standard TEP benchmark files use regularly sampled process simulations with fault-number and simulation-run identifiers. The classical files contain 500 training samples and 960 test samples per run at three-minute sampling. Extended TEP data provide repeated runs and multiple operating/fault settings; run-level separation is required to avoid treating overlapping windows from one simulation as independent examples. Detection rate, false-alarm rate, and average detection delay are native process-monitoring outcomes.

Sources used for protocol design: MetroPT dataset paper, DOI `10.1038/s41597-022-01877-3`; NASA PCoE C-MAPSS data description and PHM08 protocol; Reinartz, Kulahci and Ravn, *An extended Tennessee Eastman simulation dataset for fault-detection and decision support systems*, Computers & Chemical Engineering 149 (2021), 107281; and recent TEP fault-detection benchmark implementations that report detection TPR/FPR and average detection delay.

## Chapter 3: MetroPT-3 failure prediction

### 3.1 Task definition

Use the original MetroPT-3 stream and maintenance report timestamps. For each inference time `t`, use a history window ending at `t` and predict whether a failure event will begin within a registered warning horizon. The main operational horizon is two hours before the LPS/non-operational point; an auxiliary horizon may be 30/60/120 minutes, but it must be declared before tuning. A prediction is an event interval or an alarm episode, not a future 24-step sensor reconstruction target.

The maintenance intervals are weak labels: the release is sensor data plus post-hoc reports, not a dense fault-state annotation. Report the number of events and their durations, and do not claim population-level generalization from only a few events.

### 3.2 Data protocol

1. Sort by timestamp and retain all native samples and native sensor channels.
2. Build event labels from the published maintenance intervals and retain the LPS signal as an observed feature only if the prediction-time information policy allows it. Run a sensitivity check with LPS excluded because it is itself an alarm channel.
3. Use a chronological operational split. The split must be checked for event support before training; if a strict chronological test contains no event, report the split as infeasible for event metrics rather than moving labels across the boundary.
4. Fit normalization on the training time range only. Generate overlapping inference windows inside each split; do not randomly split adjacent windows.
5. Use the native complete-observation protocol for the main result.

### 3.3 Main comparison

Baselines: persistence/rule threshold, logistic regression or gradient-boosted trees on window statistics, GRU/TCN classifier, reconstruction-based autoencoder, and the proposed model. All models receive the same history, feature policy, chronological split, and warning horizon.

### 3.4 Experiments

| ID | Experiment | Purpose | Primary metrics |
|---|---|---|---|
| M3-1 | Main event prediction | Compare models under native protocol | event recall, event precision, event F1, PR-AUC |
| M3-2 | Horizon sensitivity | Test 30/60/120-minute warning policy | recall, precision, median lead time |
| M3-3 | LPS-policy ablation | Quantify dependence on the built-in alarm channel | event F1, false alarms/hour |
| M3-4 | Representation ablation | Remove time/availability_map/context modules only if those inputs are actually used | event F1, PR-AUC |
| M3-5 | Operational timeline | Show alarms against reported intervals | lead time, overlap/IoU, false alarms/hour |
| M3-6 | Optional observation damage | One pre-registered random or block outage stress test | relative degradation, not a native score |

The event evaluator must count an event as detected only when the predicted alarm episode overlaps the ground-truth event and must separately report warning lead time. Window-level accuracy is secondary and must not be the headline metric.

## Chapter 4: C-MAPSS degradation and RUL prediction

### 4.1 Task definition

Use FD004 as the main subset because it contains six operating conditions and two fault modes. Given the observed prefix of an engine trajectory, predict RUL at the final observed cycle. A secondary trajectory experiment may estimate RUL at intermediate cycles for training diagnostics, but the final table is engine-level RUL on the official test trajectories.

### 4.2 Data protocol

1. Read official `train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt`.
2. Keep engine IDs intact. Split the official training engines into train/validation by engine, never by windows; retain the official test engines for one final evaluation.
3. Fit sensor normalization on training engines only. For FD004, record whether normalization is global or operating-condition-aware; the choice is a registered preprocessing factor, not a post-hoc optimization.
4. Register one RUL target convention before training: raw RUL and, optionally, a capped-125 target as a separate ablation. Report both if the cap is used because the cap changes the estimand.
5. Use the official complete trajectory with a truncated test prefix.

### 4.3 Main comparison

Baselines: last-value/linear degradation extrapolation, MLP or CNN window regressor, GRU/LSTM, TCN, Transformer/PatchTST-style regressor, and the proposed probabilistic model. Use the same engine split, sensor selection, target convention, and final test evaluator.

### 4.4 Experiments

| ID | Experiment | Purpose | Primary metrics |
|---|---|---|---|
| C4-1 | FD004 main RUL | Compare degradation predictors | RMSE, MAE, PHM08/NASA asymmetric score |
| C4-2 | Distributional RUL | Evaluate uncertainty if the model outputs it | NLL/CRPS, interval coverage, interval width |
| C4-3 | Operating-condition normalization | Test multi-regime handling | RMSE and asymmetric score by condition |
| C4-4 | RUL cap sensitivity | Separate target-construction effect from architecture | raw/capped RMSE and score |
| C4-5 | Sensor/representation ablation | Test contribution of sensor selection and temporal encoder | engine-level metrics |
| C4-6 | Cross-subset transfer (optional) | Train on FD004, evaluate only if a prespecified transfer question is desired | subset-wise RMSE, no pooled headline |

The official test RUL vector is used only after the checkpoint and preprocessing policy are frozen. Report per-engine residual plots and early/late error asymmetry because a lower RMSE alone does not guarantee safer maintenance decisions.

## Chapter 5: TEP process fault detection

### 5.1 Task definition

Treat TEP as process monitoring/fault detection. The main output is a binary normal/fault alarm and, in a secondary analysis, fault-number diagnosis. The task is not future sensor reconstruction unless reconstruction is explicitly used as an unsupervised detector.

### 5.2 Data protocol

1. Choose and name the exact TEP release before experiments: classic Braatz-style files for a direct historical comparison, or the extended repeated-run release for statistically independent run-level evaluation. Do not mix releases in one table.
2. Use the provided fault activation point and `faultNumber`/`simulationRun` metadata only to form labels and splits; fault identity must not be passed as a feature.
3. Split by complete simulation run. If repeated-run data are available, use disjoint random seeds/runs for train, validation, and test. For the classic single-run files, state the limited replication and use case-level rather than window-count confidence intervals.
4. Fit scaling on normal/training runs only. Keep the regular three-minute sampling and the original run length.
5. Use the original process samples and run metadata for the main TEP result.

### 5.3 Main comparison

Baselines: PCA with Hotelling `T²` and `Q` charts, DPCA/PLS where implemented, one-class/reconstruction baseline, GRU/TCN classifier, and the proposed model. For supervised diagnosis, include a multiclass classifier using identical run splits.

### 5.4 Experiments

| ID | Experiment | Purpose | Primary metrics |
|---|---|---|---|
| T5-1 | Binary fault detection | Detect any active fault | TPR/recall, FPR or false alarms/run, macro-F1, AUPRC |
| T5-2 | Detection delay | Measure operational timeliness | average/median detection delay, missed-run rate |
| T5-3 | Per-fault analysis | Expose fault-specific weaknesses | per-fault recall/F1/delay, macro averages |
| T5-4 | Multiclass diagnosis | Identify fault type after detection | macro-F1, balanced accuracy, confusion matrix |
| T5-5 | Operating-mode robustness | Test nominal mode/transition variation if release supports it | mode-wise metrics |

The headline TEP table must contain run-level detection metrics and delay. Window-level accuracy may be included for reproducibility but is not sufficient evidence because many windows from one run are correlated.

## Shared fairness and reporting contract

- Each dataset gets its own task-specific label, split, baseline family, and primary metric. Do not rank MetroPT event F1, C-MAPSS RMSE, and TEP F1 in one aggregate score.
- Shared model components may be reused, but input heads and losses are task-specific: event classification/alarm head for MetroPT, RUL/probabilistic regression head for C-MAPSS, and detection/diagnosis head for TEP.
- Every run records `run_id`, dataset release, task, split identity, seed, preprocessing version, model, checkpoint, and metric aggregation unit.
- Main results should use at least three seeds when repeated runs are feasible. If only one seed is run, label it as a pilot and report no formal multi-seed claim.
- Report confidence intervals or run-level bootstrap where the number of independent events/runs permits; never treat overlapping windows as independent replicates.
- The native protocol of each dataset is the sole basis of the main comparison.

## Chapter boundary after replanning

Chapter 3 establishes failure-event prediction on a real compressor stream. Chapter 4 establishes degradation/RUL prediction on run-to-failure engine trajectories. Chapter 5 establishes online process fault detection and diagnosis on simulated process runs. This progression is defensible because it spans three distinct PHM decision tasks while preserving a common probabilistic risk perspective; it is not a claim that one numerical metric or one data protocol applies to all three.
