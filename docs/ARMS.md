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

## Protocol: `--protocol native` (the default)

AutoDP is scored under its **published protocol** — its search and its pipeline application both see
the whole table, because that is what the released API does. To keep the comparison fair, **ACORec
is prepared the same way** in these arms: `--prepare-mode native` fits the chosen pipeline on the
full dataset (train and test together) before the AutoGluon split evaluation, mirroring their
operators fitting on `concat(train, test)`.

`--protocol native` sets both sides at once and is the default, so the two can never drift apart by
accident. `--protocol leakfree` gives the fit-on-train discipline to both.

Verified to actually take effect, on a deliberately shifted split (`_fit_pipeline`, `ZS` scaling):

```
leakfree   train column mean = +0.0000     (fitted on train)
native     train column mean = -0.5768     (fitted on the union)
```

**Anything scored under `native` is not a measurement of generalisation, for either method.** It is
a like-for-like comparison under their protocol. Label the column that way, and if you also want a
generalisation number, run `--protocol leakfree` for both sides and report it separately.

### The one asymmetry `native` cannot remove — disclose it

Under `native` the pipeline is fitted on the full frame and applied with `transform()`, which never
deletes rows. **Row-dropping operators are therefore excluded from ACORec's search under `native`**
(`_ROW_DROPPING_OPERATORS`): `ZSB`, `IQR`, `LOF`, `ED`, `DROP` in their space; `iqr`, `zscore`,
`lof`, `isolation_forest` in ours. Selecting one would otherwise be a **silent no-op** — measured
before the fix: `ZSB` cut 90 train rows to 73 under `leakfree` and to 90 (i.e. nothing) under
`native`. A hard guard in `_fit_pipeline` now raises rather than letting that recur.

Consequences to state in the paper:

- ACORec's **native** search covers **17 of their 22** operators, not 22. The full 22 are available
  under `--protocol leakfree`.
- AutoDP under native **does** delete rows, which is why its results carry `score_full` /
  `score_kept`. ACORec never deletes test rows under either protocol.

The alternative — tracking surviving row indices through the full-frame fit and slicing the splits
from them, as `run_autodatapre.py` does with `__adp_row__` — would restore the missing 5 operators.
It is not implemented; this is the honest, verifiable version, and it makes ACORec weaker under
native rather than stronger.

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

## Open issues, in priority order

1. **`THEIR_DATASETS` in `run_arms.py` is UNVERIFIED.** The ids were transcribed from their paper
   and several are OCR-risky (`8335`, `43723`, `42493`). Confirm each against OpenML *before* any
   arm-2 or arm-3 run, and check overlap with the 901-dataset reference library — `184` and `31` are
   known to appear in both, and any overlap must be held out for those arms to be leak-free.
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
4. **`score_full` vs `score_kept` for the AutoDP arms.** Their `ZSB`/`LOF` delete test rows, so the
   two diverge now that pipelines are non-empty. Fix the choice before running, not after seeing
   both numbers. (ACORec never deletes test rows in either protocol, so this asymmetry is theirs
   alone and should be stated wherever the AutoDP column appears.)
5. **Dataset 1047 has a mixed-type target** (`float` and `str`), which crashes `np.unique` in
   `evaluate_candidates_simple`. Pre-existing, reproduces in both spaces.
