# AutoDP baseline (`autodatapre==0.1.12`)

An external comparison point: Auto-DP's MCTS data-preparation search, scored with **our** AutoGluon
protocol on **our** split, so its numbers sit in the same table as ours.

## What is held constant, and what is not

Constant across AutoDP and our method — the same rows, the same row order, the same seed-42
0.6/0.2/0.2 split, fit on train+val (80%), predicting the same 20% test rows, scored by the same
`TabularPredictor` settings (`best_quality`, DyStack off, `IdentityFeatureGenerator`, XGB-retry
fallback), accuracy for classification and R² for regression. Stages 1 and 3 reuse
`automl_aco.data.splits.split_train_val_test`, `_detect_problem_type` and
`_fit_predict_with_autogluon` directly rather than reimplementing them.

AutoDP's METHOD runs as published: MCTS's tree policy, UCB, backup and pruning, its pretrained
meta-learner picking the search-space ordering from its own `Metafeature.csv` neighbours, its
value estimate, and operator semantics (per-split statistics, union-fitted encoders) are all
untouched. What changed is its EVALUATION LAYER — moved onto our setting, because a split, a
label-leakage guard, and scorer determinism/objective are not pipeline-selection logic:
`scripts/autodp_protocol.py` seeds the internal RF/ExtraTrees scorer, gives LDA the same
holdout-accuracy objective NB and RF already use, and closes a leak where `CBE` consumed the
labels of the rows it was encoding. The final scorer is swapped to AutoGluon, as before.

## `fair` is the only reported protocol

`AutoDP.Classifier(df, ...)` takes one dataframe and returns a prepared dataframe, so a fit/transform
split has to be built on top of it:

| mode | what the search sees | reading |
|---|---|---|
| `fair` (default, reported) | our **seed-42 0.6 train / 0.2 val** — `scripts/autodp_protocol.py::build_search_dataset` replaces their own `read_dataset`, whose `train_test_split` had no `random_state`; the winning chain is then applied to our 80%/20% using AutoDP's operator classes, `CBE` now leak-free | leak-free, the protocol our own method is held to |
| `native` (NOT reported) | the **full** dataset, including our test rows — literally their published API, deliberately left unpatched; its internal scorer holds out its own unseeded random 20% | transductive; kept only so the deviation from `fair` is inspectable |

Only `fair` numbers go in any table. `native` exists as a disclosure artifact, not a second column
to average or compare against.

## Known behaviours of their code, and how each is handled

- **Row deletion on the test split.** `DROP` imputation, `ZSB`/`IQR`/`LOF` outlier removal and
  `ED`/`AD` dedup delete rows from the *test* frame. Our own evaluator forbids that outright. Rather
  than let AutoDP score on a smaller, easier test set, stage 3 reports `score_full` (all our test
  rows; deleted ones count as wrong, or the training mean for regression — **this is the comparable
  number and the report default**) alongside `score_kept` (only the rows it kept — the convention
  its own internal evaluation uses) plus `test_coverage`. At coverage 1.0 the two coincide.
- **Train and test are transformed independently** (e.g. `ZS` z-scores test with test's own
  mean/std). Theirs, preserved as-is.
- **`read_dataset` LabelEncodes the target**, which destroys continuous targets on regression
  datasets. Kept inside their search (it is their signal); stage 3 re-attaches the **original** y by
  row position, so the reported R² means what it says.
- **Bare `except:` in their entry points** silently returns the RAW frame when the winning pipeline
  crashes on apply. Stage 2 detects it and records `status`; the report flags those rows with ⚠️
  instead of passing off raw data as prepared.
- **Bare `except:` inside the SEARCH loop too** (`CLA_Without_TimeBudget`), which can swallow every
  MCTS iteration and still report a "converged" pipeline that was never actually evaluated even
  once. `scripts/autodp_protocol.py::ExceptionCounter` counts these; a non-zero
  `search_iteration_exceptions` on a reported row means treat the pipeline with caution.
  DISCLOSED, NOT FIXED (it is search machinery -- `drop_unpromising`, `best_child`,
  `Is_BatchTraining`, all theirs). One concrete trigger, found while validating this counter:
  `classifier.py`'s internal `k_folds`-based `None` guard combined with `Is_BatchTraining`'s
  progressive subsampling can shrink a small dataset's usable batch below the guard threshold at
  shallow tree depth, and `drop_unpromising`'s `profit.sort()` then raises `TypeError: '<' not
  supported between instances of 'NoneType' and 'float'` -- on EVERY iteration, for the rest of the
  search. Measured on dataset 862 (87 rows, one of the smallest of the 30): 1.19M swallowed
  exceptions, pipeline `['NB']` (no preprocessing, because nothing was ever actually evaluated).
  Smaller datasets are more exposed in general -- our seed-42 0.6 train (vs their published 80%)
  gives `Is_BatchTraining` less headroom before the guard fires -- but which specific datasets
  trip it is a property of the running search, not a static property of the split, so this is not
  precomputable the way the (separate, fixed) index-labelling bug below was. Watch
  `search_iteration_exceptions` per row instead of trying to predict it. Regression test (a
  different, already-fixed trigger on the same dataset):
  `tests/test_autodp_protocol.py::test_search_dataset_reproduces_the_862_failure_mode_directly`.
  **Outcome for a fully-dead search (0 nodes scored):** the spin is aborted and the row becomes
  the raw frame, tagged `dead_search: true` + `empty_pipeline: true` + the spin counts. Under
  `leakfree` this is what 862 and 27 do on `ours` ops (3 nodes, all `profit=None`). Treat it as
  an AutoDP failure whose reported value is the untouched-frame accuracy — same standing as a
  timeout on arm 0. See `docs/ARMS.md` item 6.
- **A pipeline of length 1** means AutoDP picked a classifier and *no* preprocessing operator, so its
  output is the raw data. The report calls those out — on a 30s smoke run of dataset 31 this is
  exactly what happened, so watch for it at full budget.
- **Residual categoricals.** When AutoDP selects `enc_null`, object columns survive and AutoGluon
  under `IdentityFeatureGenerator` cannot consume them. Stage 3 one-hot encodes the leftovers with
  the same `OneHotEncoder(handle_unknown="ignore")` our `no_preprocessing` baseline uses, fit on
  training rows only, and records `residual_encoding_applied`.

## Why a second environment

AutoDP hard-pins `numpy==1.21.6` / `pandas==1.3.5` / `scikit-learn==1.0.2`, and its code uses
`np.bool`, `DataFrame.append` and positional `drop(..., 1)` — all removed in the versions AutoGluon
needs. It cannot share an environment with AutoGluon. `.venv-autodp` holds the oldest pins that both
build on a modern toolchain and keep every AutoDP path working; see `requirements-autodp.txt`.

## Running it

Local:

```bash
bash scripts/setup_autodp_env.sh                  # once: builds .venv-autodp
bash scripts/run_autodatapre_all.sh outputs/autodp 1800 300
```

Args: `<output_dir> [cap_seconds] [ag_time_limit] [modes] [ids]`. AutoDP runs to **its own
convergence rule** (its default, strongest setting); `cap_seconds` is only a watchdog — if
convergence never fires the dataset is killed and retried with an explicit budget, and the report
flags it. Set `ag_time_limit` to the same AutoGluon `--time-limit` your own runs used, or the
comparison is not compute-matched.

Kaggle. The repo arrives as a read-only dataset, so copy it to `/kaggle/working` first, and install
AutoGluon the way the ACORec runs do — `--target` plus `PYTHONPATH`:

```python
!mkdir -p /kaggle/working/repo && cp -r /kaggle/input/datasets/<user>/<slug>/. /kaggle/working/repo/
%cd /kaggle/working/repo
!bash scripts/setup_autodp_env.sh
!pip install -q --target=/kaggle/working/acorec_deps "numpy<2" "pandas<3" autogluon.tabular==1.5.0 openml

!PYTHONPATH=/kaggle/working/acorec_deps MAIN_PY=python \
  bash scripts/run_autodatapre_all.sh outputs/autodp 600 300
```

`PYTHONPATH` is applied **per stage**, never exported globally: main-env stages get
`src:$PYTHONPATH`, while the AutoDP stage gets `src` alone. That separation is load-bearing —
`PYTHONPATH` outranks a venv's own site-packages, so exporting `acorec_deps` globally would pull
numpy>=1.26 and pandas 2.x into `.venv-autodp` and break AutoDP.

Without internet, stage 1 falls back to any `<id>.csv` it can find: it walks `/kaggle/input`,
`data/openml` and `test_data_local` automatically (`--openml-local-folder` / `--local-root` to add
your own), and prints the file and label column it picked per dataset. Note that a third-party CSV
mirror is not guaranteed to match the OpenML API's rows or column order — if your own scores came
from the API, enable Internet so both sides use the same source.

The batch is **resumable** — any dataset with an `autodp_eval.json` is skipped — so if a Kaggle
session times out, rerun the same command and it continues. Per-dataset logs land in
`outputs/autodp/<mode>_<id>.log`.

## The table

```bash
python scripts/report_autodatapre.py --input-dir outputs/autodp \
    --ours "ACORec=outputs/<your_run_dir>" \
    --ours "no-prep=outputs/<your_baseline_dir>"
```

Writes `outputs/autodp/AUTODP_REPORT.md`: accuracy and runtime per dataset per mode, a mean row,
head-to-head W/T/L against each of our runs, and the caveat notes above. Runtime is AutoDP's own
search time (`--runtime-column total` adds AutoGluon scoring, which is our harness and identical
work for every method).
