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

AutoDP's method runs as published: its MCTS, its pretrained meta-learner picking the search-space
ordering from its own `Metafeature.csv` neighbours, and its internal NB/LDA/RF scoring signal are
all untouched. Only the **final** scorer is swapped to AutoGluon — exactly what we do to our own
pipelines.

## Two protocols, because their API has no fit/transform split

`AutoDP.Classifier(df, ...)` takes one dataframe and returns a prepared dataframe. So:

| mode | what the search sees | reading |
|---|---|---|
| `native` | the **full** dataset, including our test rows (its published API; its internal scorer holds out its own unseeded random 20%) | transductive, **generous to AutoDP** |
| `fair` | only our **80% train+val**; the winning operator chain is then applied to our 80%/20% using AutoDP's own operator classes | leak-free, the protocol our method is held to |

Both are run. If we win under `native`, we win under their best case.

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

Kaggle (internet on, so stage 1 can reach OpenML):

```python
!cd /kaggle/working/SolutionRecommendation && bash scripts/setup_autodp_env.sh

# MAIN_PY=python -> Kaggle's own env, which already has AutoGluon
!cd /kaggle/working/SolutionRecommendation && \
  MAIN_PY=python bash scripts/run_autodatapre_all.sh outputs/autodp 1800 300
```

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
