# Cross-comparison arms: ACORec vs AutoDP over both operator spaces

Companion to `docs/OPERATOR_SPACE_COMPARISON.md`, which explains *why* the operators differ. This
file is the run book: what each arm is, what is implemented, what is not, and the exact commands.

## The grid

Three dimensions — **data** (our 23 eval datasets / their ~10), **operators** (ours = 19 / theirs =
22 reimplemented), **method** (ACORec = ACO + AutoGluon CV gate / AutoDP = MCTS).

| arm | data | operators | method | status |
|---|---|---|---|---|
| *(published)* | ours | ours | ACORec | ✅ already have it |
| `1-adp-ourops` | ours | ours | AutoDP | run me |
| `1-aco-theirops` | ours | theirs | ACORec | run me — the clean control |
| `2-aco-ourops` | theirs | ours | ACORec | run me |
| `2-adp-ourops` | theirs | ours | AutoDP | run me |
| `3-aco-theirops` | theirs | theirs | ACORec | run me |
| *(their paper)* | theirs | theirs | AutoDP | ✅ published |

Filling these makes every pairwise comparison hold two of the three dimensions fixed.

## What "same operator space" now means

Previously the equalisation was **family-level** (`--exclude-steps dimensionality_reduction`), which
is not the same thing at all — their `ZS` and our `standard` sit in the same family and behave
differently. Both directions are now real:

- **their operators → our discipline**: `src/automl_aco/preprocessing/autodp_ops.py` reimplements 22
  of their 24 operators as leak-free fit/transform steps. Selected with `--operator-space theirs`.
  The module docstring carries the **complete deviation table** — read it before reporting anything.
- **our operators → their framework**: `scripts/autodp_our_space.py` (pre-existing) makes their MCTS
  search our space. Still uses code aliasing rather than a retrain; see §5 of
  `OPERATOR_SPACE_COMPARISON.md` for why that debt is real for their family-ordering meta-learner
  and moot for their value model.

### Disclosures that must accompany any arm using `--operator-space theirs`

1. **22 of 24 operators.** `EM` (impyute, no fit/transform equivalent) and `AD` (py-stringsimjoin,
   no wheel for python ≥ 3.10) are dropped.
2. **Their bugs are fixed, their parameters are kept.** Integer-truncated means, the broken chi²
   negativity guard, `LOF`'s fixed 30-row deletion, unseeded `ExtraTrees`. Every fix makes their
   operator stronger, so it biases against our own result.
3. **`DROP` cannot delete test rows.** Their version does; ours drops incomplete rows from train and
   fills the test split from surviving-train statistics, because refusing to predict a row has no
   leak-free counterpart.
4. **`WR`/`TB` keep categoricals.** Theirs delete every categorical column. Kept because ACORec
   searches step order, so those operators may legally run before encoding.
5. **The transfer arm is off.** The reference matrix is in *our* operator codes; retrieved pipelines
   would be silently coerced into their vocabulary. Override with `--transfer-arm-anyway` only if
   you also rebuild the matrix.

## Validation

`scripts/test_operator_spaces.py` checks six invariants across every operator and random full
pipelines. I6 (row independence) is the direct proof the leak is gone: feeding test rows in reverse
must yield the reversed outputs, which fails for any concat-then-fit operator.

```
$ python scripts/test_operator_spaces.py --space theirs --pipelines 60
  -- single operators: 66/66 passed
  -- full pipelines:   60/60 passed
ALL INVARIANTS HOLD
```

`ours` does **not** pass — see the open issues below. Those failures are pre-existing and unrelated
to this work.

## Protocol: `--protocol leakfree` (the default, and the ONLY reported protocol)

Every reported number now comes from `--protocol leakfree`. Both sides fit on train only:
`--prepare-mode leakfree` for ACORec, and for AutoDP its search scores candidates on our seed-42
0.6 train / 0.2 val, its evaluation layer moved onto ours by `scripts/autodp_protocol.py` — see
that module for the exact list of what moved (split, CBE's label channel, scorer seeding, scorer
objective) and what stayed theirs (MCTS itself, the value estimate, operator semantics).

**Consequence, and it is a real improvement to the control arm:** this lifts
`_ROW_DROPPING_OPERATORS`, so ACORec's search now covers all **22 of their 22** reimplemented
operators, not 17.

`--protocol native` still exists but is **not a reported path for either method**. It is AutoDP's
literal published API — the MCTS searches the full dataset and its internal scorer holds out its
own *unseeded* random 20%, so our test rows are visible to the search — kept, unpatched, only so
the deviation from `leakfree` is inspectable. `native`'s one remaining structural difference from
`leakfree` even after a hypothetical split-fix is the **apply** step: it fits on `concat(train,
test)` via `merge_datasets`, which `leakfree` never does. Verified to actually take effect on a
deliberately shifted split (`_fit_pipeline`, `ZS` scaling):

```
leakfree   train column mean = +0.0000     (fitted on train)
native     train column mean = -0.5768     (fitted on the union)
```

### Historical note: the asymmetry `native` could never remove

Under `native`, `transform()` never deletes rows, so row-dropping operators were excluded from
ACORec's search there (`_ROW_DROPPING_OPERATORS`: `ZSB`, `IQR`, `LOF`, `ED`, `DROP` in their space;
`iqr`, `zscore`, `lof`, `isolation_forest` in ours) — a silent no-op otherwise, measured before the
fix (`ZSB` cut 90 train rows to 73 under `leakfree`, to 90 under `native`). A hard guard in
`_fit_pipeline` still raises rather than letting this recur, but it is moot now that `native` is
not reported.

## Running

Print one command per dataset — fastest first result, and a dying notebook costs you one dataset:

```bash
python scripts/run_arms.py --print-commands --arm 3-aco-theirops
```

Or coarser shards:

```bash
python scripts/run_arms.py --print-commands --arm 3-aco-theirops --shards 4
```

A single run (safe to rerun; completed rows are skipped, keyed on arm + dataset + protocol):

```bash
python scripts/run_arms.py --arm 3-aco-theirops --datasets 1461 \
    --out arms_3.jsonl --data-dir /kaggle/input/openml --time-limit 300
```

See everything collected so far, across any number of files:

```bash
python scripts/run_arms.py --summarize "arms_*.jsonl"
```

which prints per-run rows, a per-arm mean, and a warning listing any rows whose `eval_method` is not
`autogluon` (those are proxy scores and must be excluded before averaging).

Output is one JSONL row per (arm, dataset) with `score`, `eval_method`, `pipeline`, `step_order`,
`ag_candidate_scores`, `total_seconds`. **Audit `eval_method` before averaging** — `proxy` or
`autogluon_failed` means the number is not an AutoGluon score.

On Kaggle keep `--autogluon-profile best_quality` (the default). `local_rf_xt` is for local smoke
tests only; `best_quality` segfaults on macOS via AutoGluon's OpenMP layer.

### Second downstream evaluator: `--evaluator h2o` (AutoDP arms only)

`--evaluator h2o` re-scores an AutoDP arm with **H2O AutoML at its defaults** instead of AutoGluon,
to get the ACORec-vs-AutoDP comparison under a second downstream AutoML. Only stage-3 scoring
changes — the AutoDP search, the `leakfree` protocol, the seed-42 split, the fit-on-train+val /
predict-20%-test convention, the original-target re-attachment and the `score_full`/`score_kept`
accounting are all identical to the AutoGluon path (`scripts/eval_autodatapre.py`,
`score_prepared(evaluator=...)`). H2O runs with `preprocessing=None` (downstream preprocessing
OFF), `max_runtime_secs` = `--time-limit`, and everything else (nfolds, StackedEnsemble,
sort_metric) at H2O defaults; residual categoricals go to H2O as native `enum` factors rather than
being one-hot encoded. `--time-limit` sets `max_runtime_secs` to the same *number* the AutoGluon
path uses, but it is not the same *quantity* (whole-AutoML budget vs AutoGluon's per-model budget),
and H2O overruns it by ~20% in practice. Rows are tagged `evaluator: "h2o"` and `--summarize`
groups the mean by
`(arm, evaluator)`, so an H2O run never averages into the same arm's AutoGluon column. Keep H2O
runs in their own JSONL files (`arms_<arm>_h2o_*.jsonl`). Run book:
`docs/kaggle_adp_arm1_h2o_cells.md`. ACORec's downstream evaluator is set inside `run_recommend.py`,
so `--evaluator h2o` is rejected on ACORec arms.

## Open issues, in priority order

1. **`THEIR_DATASETS` in `run_arms.py` is still UNVERIFIED against the OpenML API** — the ids were
   transcribed from their paper and several are OCR-risky (`8335`, `43723`, `42493`). Confirm each
   before reporting any arm-2 or arm-3 number.

   The **overlap-with-the-reference-library** half of this issue is FIXED. Checked directly: 5 of
   the 10 (`1461`, `1590`, `184`, `31`, `40701`) are performance-matrix columns and a 6th (`40945`)
   is a metafeature row. None are in `EVAL_IDS` (they moved from the legacy 23 to today's 30), so
   the default eval-ID holdout never touched them — `184` and `31` used to be protected under the
   legacy set and silently stopped being protected when it changed. `run_recommend.py --holdout-ids`
   (`src/automl_aco/eval_ids.py::holdout_reference(extra_ids=...)`) closes this; `run_arms.py`'s
   `acorec_cmd` now passes all ten `THEIR_DATASETS` automatically whenever `data == "theirs"`, and
   `scripts/build_adp_meta_corpus.py` holds the same ten out of the retrained corpus. Regression
   tests: `tests/test_leakage_holdout.py::test_acorec_cmd_holds_out_their_datasets_on_their_data`
   and `::test_holdout_reference_extra_ids_*`.
2. **`ag_candidate_scores` under-reports the gate by one candidate.** Instrumented on dataset 1054,
   the gate receives the floor in both spaces:

   ```
   theirs  extra=[('no_preprocessing','OE')]                                 ag_candidates=2  ag_candidate_scores=1
   ours    extra=[('no_search_retrieval','onehot'),('no_preprocessing','onehot')]  ag_candidates=3  ag_candidate_scores=2
   ```

   The floor is present and is also passed as `select_default_name`, so the three-way gate is
   intact — this is a **reporting** gap, not a method defect. (An earlier draft of this file claimed
   the floor was missing; that was wrong, and the local 20-dataset run contradicts it directly, with
   `no_preprocessing` winning the gate 7 times.) Consequence: do not infer the winning arm from
   `ag_candidate_scores`, because the floor can win without appearing there.
3. **`ours` crashes on ~11% of pipelines.** 13/120 random full pipelines raise, all one root cause:
   NaN reaching a NaN-intolerant operator (`SelectKBest`, `PCA`, `TruncatedSVD`, `LocalOutlierFactor`)
   when `imputation="none"`, plus outlier removal occasionally emptying the frame. In search these
   are swallowed and scored as failures, which silently biases the ACO away from `imputation=none`
   combinations. Left unchanged deliberately — it is your method and changing it needs a
   test-set-independent rationale — but it should be a conscious choice, not an accident.
4. **DECIDED: `score_full` is the headline column for the AutoDP arms; `score_kept` is reported
   alongside, never in its place.** Their `ZSB`/`LOF` delete test rows, so the two diverge whenever
   a pipeline is non-empty. `score_full` charges AutoDP for the rows it declined to predict
   (classification: wrong; regression: training-target-mean fallback) and is therefore the column
   directly comparable to ACORec, which never deletes test rows under any protocol. Fixed here,
   before any reported number exists — `scripts/adp_bench.py`'s `run()` already writes both fields
   per row (`eval_autodatapre.py::score_prepared`); the summary/report layer must read `score`
   (== `score_full`), not `score_kept`.
5. **Dataset 1047 has a mixed-type target** (`float` and `str`), which crashes `np.unique` in
   `evaluate_candidates_simple`. Pre-existing, reproduces in both spaces.
6. **AutoDP's search can crash-loop into an empty pipeline on small datasets, and it is now
   VISIBLE, not fixed.** `MCTS.CLA_Without_TimeBudget`'s search loop wraps every iteration in a
   bare `except:`, so a search that fails on every single iteration still "converges" and reports a
   pipeline that was never actually evaluated. Root cause (found validating the exception counter
   on dataset 862, 87 rows): a classifier's internal `k_folds` guard returns `None` on a batch
   shrunk by `Is_BatchTraining`'s progressive subsampling, and `drop_unpromising`'s `profit.sort()`
   then raises comparing `None` to `float` — on every iteration for the rest of the run. Measured:
   1.19M swallowed exceptions, pipeline `['NB']`. This is search machinery (`drop_unpromising`,
   `best_child`, `Is_BatchTraining`), so per Part 3 it is DISCLOSED, not patched. What changed:
   `scripts/autodp_protocol.py::ExceptionCounter` counts it, and `search_iteration_exceptions` /
   `search_iteration_exception_kinds` are now threaded through `eval_autodatapre.py` →
   `adp_bench.py` → `run_arms.py` into every reported row, so a row with a large count can be
   excluded or flagged rather than silently averaged in as a genuine AutoDP preference. Both
   `adp_bench.py --summarize` and `run_arms.py --summarize` print a warning listing affected rows.
   Full writeup: `docs/AUTODP_BASELINE.md` "Known behaviours of their code". Regression test:
   `tests/test_autodp_protocol.py::test_exception_counter_counts_and_reraises`.

   **What the spin produces now (updated).** `ExceptionCounter` aborts the spin after 500
   consecutive raises with no node evaluated (`os._exit(3)` — their bare `except:` swallows
   anything softer). Two outcomes:
   - **≥1 node scored a real profit before the spin** (e.g. 378 on `theirs` ops: 7 nodes, best
     `['RF','CBE']`): `SearchCheckpoint` salvages that pipeline apply-only. Row carries
     `salvaged_from_checkpoint: true`.
   - **0 nodes ever scored** (e.g. 862 and 27 on `ours` ops under `leakfree`: 3 nodes, all
     `profit=None`): there is no pipeline, so the salvage applies the **empty pipeline = raw
     frame**. Row carries `dead_search: true` + `empty_pipeline: true` + the spin counts
     (`dead_search_none_profit_evals`, `search_iteration_exceptions`). This is an **AutoDP
     failure reported as the raw-frame number** — the same class as 378/722 timing out on arm 0,
     not a hole in the table and not a genuine "AutoDP chose no preprocessing". Both
     `--summarize` layers call these rows out separately from ordinary empty pipelines.
     `scripts/autodp_protocol.py` writes `dead_search.json` next to the checkpoint before the
     abort so the counts survive `os._exit`; `run_autodatapre.py::_dead_search_worker` does the
     apply. Regression test:
     `tests/test_autodp_protocol.py::test_dead_search_with_no_scored_node_writes_a_marker_for_the_raw_frame_salvage`.
