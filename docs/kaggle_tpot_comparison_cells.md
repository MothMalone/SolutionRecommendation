# Kaggle cells — AutoDP arm (`1-adp-ourops`) scored by **estimator-only TPOT**

The TPOT analogue of `docs/kaggle_adp_arm1_h2o_cells.md`. Produces the **AutoDP** number for the
ACORec-vs-AutoDP comparison under a *third* downstream evaluator, to sit next to the AutoGluon and
H2O columns.

**Identical to the H2O arm-1 run in every way except the downstream model:** the AutoDP search
(MCTS over ACORec's operator space, retrained 1-NN meta-learner from `data/adp_ourops_corpus`), the
`leakfree` protocol (`--mode fair`), the seed-42 0.6/0.2/0.2 positional split
(`automl_aco.data.splits.split_train_val_test`), fit-on-train+val / predict-the-20%-test,
original-target re-attachment, `score_full` / `score_kept` row-coverage accounting. The only fork is
`tpot.TPOTClassifier` / `TPOTRegressor` in place of `_fit_predict_with_h2o`.

**TPOT settings** (`scripts/_tpot_eval.py`, shared with the ACORec arm so the two cannot drift):
estimator-only search space (`get_search_space("classifiers" | "regressors")`), `preprocessing=False`,
`validation_strategy="none"`, `max_time_mins=5`, `max_eval_time_mins=1`, `population_size=20`,
`cv=min(5, smallest post-AutoDP class count)`, `early_stop=5`, `random_state=1` (a **separate** knob
from the split seed 42). Residual categoricals AutoDP left in place are one-hot encoded, fit on
train only (`residual_encoding_applied`), same as the AutoGluon arm. A frame with NaN in it
(`imp_null`) is a hard failure — `status: "failed"`, excluded from the mean — because imputing here
would be downstream preprocessing.

---

## Why this needs a different shape from the H2O run

The H2O arm ran search + score in one `run_arms.py` command because AutoGluon and H2O share the
main env. TPOT 1.1.0 pins `numpy>=1.25` / `scikit-learn<1.8` and **cannot** share that env, nor
AutoDP's (`numpy<1.24`). So it is a three-environment pipeline, driven by
`scripts/run_autodp_tpot_arm.py` (the TPOT analogue of `adp_bench.py`):

| step | env | what |
|---|---|---|
| AutoDP search | `.venv-autodp` (`/tmp/adpenv`) | `run_autodatapre.py`, **persisted** under `--prepared-root` — done once, reused |
| TPOT score | TPOT (`/tmp/tpotlibs`) | `evaluate_autodp_tpot.py` reads the persisted `prepared.csv` |

The driver runs in the **base env** and shells out to both. It is resumable and shardable exactly
like `adp_bench.py`, and writes `adp_bench`-compatible JSONL rows tagged `evaluator: "tpot"`.

**The AutoDP search is deterministic** at `--seed 42` with the same corpus and operator space, so a
fresh search here re-derives the identical frame the H2O column scored — the H2O run's scratch dirs
were deleted, so there is nothing saved to reuse, but the re-derivation is exact.

---

## Sharding — 5 sessions, 6 datasets each

Round-robin over the 30 `EVAL_IDS`, identical assignment to the H2O run:

| shard | datasets |
|---|---|
| `1/5` | 1066, 876, 1485, 42932, 41001, 802 |
| `2/5` | 1047, 18, 14, 40668, 41671, **722** |
| `3/5` | 862, 1520, 27, 1471, 1046, 40922 |
| `4/5` | 40663, 1548, 44956, 100000, 46597, 1119 |
| `5/5` | 1054, **378**, 1037, 42165, 30, 1497 |

Per dataset: AutoDP search (862/27 dead-search in ~10 s; 378/722 use the checkpoint salvage —
"likely ~30 s", measured on arm 0 not arm 1) + 5 min TPOT. A 6-dataset shard ≈ 1–2 h. Restart the
identical command to resume — rows already in the `--out` file are skipped.

---

## Cell 1 — code-freshness check

```bash
%%bash
REPO=$(ls -d /kaggle/input/*/*/acorec /kaggle/input/*/acorec /kaggle/input/acorec 2>/dev/null | head -1)
if [ -z "$REPO" ]; then
  echo "ERROR: acorec not found under /kaggle/input"; ls /kaggle/input
else
  echo "$REPO" > /tmp/repo_path
  echo "REPO=$REPO"; echo
  fail=0
  check() {
    if grep -q -- "$2" "$REPO/$1" 2>/dev/null; then printf '  ok    %-28s %s\n' "$2" "$1"
    else printf '  STALE %-28s %s\n' "$2" "$1"; fail=1; fi
  }
  check scripts/run_autodp_tpot_arm.py      _run_tpot_score
  check scripts/evaluate_autodp_tpot.py     score_prepared_tpot
  check scripts/evaluate_autodp_tpot.py     'from eval_autodatapre import'
  check scripts/_tpot_eval.py               'def build_model'
  check scripts/_tpot_eval.py               TPOT_RANDOM_STATE
  check scripts/autodp_our_space.py         _retrained_order_fn
  check scripts/autodp_protocol.py          SearchCheckpoint
  check scripts/run_autodatapre.py          salvaged_from_checkpoint
  check requirements-tpot-kaggle.txt        'TPOT==1.1.0'
  for f in data/adp_ourops_corpus/label.csv data/adp_ourops_corpus/Metafeature.csv; do
    [ -f "$REPO/$f" ] && printf '  ok    %s\n' "$f" || { printf '  STALE %s\n' "$f"; fail=1; }
  done
  echo
  [ $fail -eq 1 ] && echo "  >>> STALE - refresh the acorec dataset (Add Data), do NOT continue" \
                   || { echo "  >>> code is current"; echo -n "  >>> corpus fingerprint: "; \
                        cat "$REPO/data/adp_ourops_corpus/corpus_fingerprint.txt"; }
fi
echo
[ -d /kaggle/input/datasets/mathurinache/openml ] \
  && echo "  openml mount: attached" || echo "  openml mount: NOT attached (Cell 4 needs it)"
```

## Cell 2 — the two shell-out environments  (Internet ON)

```bash
%%bash
REPO=$(cat /tmp/repo_path)

# AutoDP's pinned env (python 3.10 / numpy 1.23) -> /tmp/adpenv
ADP_VENV=/tmp/adpenv bash "$REPO/scripts/setup_autodp_env.sh"

# TPOT env, installed to a target dir so it never shadows the base stack -> /tmp/tpotlibs
python -c "import tpot" 2>/dev/null || \
  pip install -q --target=/tmp/tpotlibs -r "$REPO/requirements-tpot-kaggle.txt"

# sanity: TPOT imports against the base interpreter + that dir
PYTHONPATH=/tmp/tpotlibs python -c "import tpot, sklearn, numpy; print('tpot', tpot.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)"
```

The driver runs in the base env (only stdlib + `automl_aco.eval_ids`); it does **not** need
AutoGluon or H2O.

## Cell 3 — export all 30 eval CSVs into one dir  (Internet ON)

The same files the ACORec arm reads. If the `eval_all` dir from the H2O run is still attached, reuse it.

```bash
%%bash
set +e
REPO=$(cat /tmp/repo_path)
rm -rf /kaggle/working/eval_all
python "$REPO/scripts/export_diffprep_datasets.py" --download \
  --copy-openml-from /kaggle/input/datasets/mathurinache/openml \
  --out-dir /kaggle/working/eval_all
python "$REPO/scripts/export_eval_datasets.py" \
  --ids "1066 1047 862 40663 1054 876 18 1520 1548 378 1485 14 27" \
  --openml-local-folder /kaggle/input/datasets/mathurinache/openml \
  --out-dir /kaggle/working/eval_all
python "$REPO/scripts/export_diffprep_datasets.py" --fingerprint-only --out-dir /kaggle/working/eval_all
echo "CSV count: $(ls /kaggle/working/eval_all/*.csv | wc -l)  (expect 30)"
```

## Cell 4 — run this shard  (change `1/5` and the `--out` name per session)

```bash
%%bash
set +e
REPO=$(cat /tmp/repo_path)

python -u "$REPO/scripts/run_autodp_tpot_arm.py" \
  --shard 1/5 \
  --protocol leakfree \
  --operator-space ours \
  --adp-meta-corpus "$REPO/data/adp_ourops_corpus" \
  --data-dir /kaggle/working/eval_all \
  --prepared-root /kaggle/working/adp_prepared \
  --adp-python /tmp/adpenv/bin/python \
  --tpot-libs /tmp/tpotlibs \
  --cap-seconds 5400 \
  --out /kaggle/working/arms_1-adp-ourops_tpot_1of5.jsonl \
  2>&1 | tee /kaggle/working/log_arm1_tpot_1of5.txt
```

- `--operator-space ours` + `--adp-meta-corpus data/adp_ourops_corpus` = arm `1-adp-ourops`
  (retrained meta-learner; value estimator NOT retrained — disclose per `docs/ARMS.md`).
- `--prepared-root /kaggle/working/adp_prepared` persists each search under
  `fair_ourops/dataset_<id>/`; add it as a Kaggle dataset output so later evaluators can `--skip-search`.
- `--tpot-libs /tmp/tpotlibs` is prepended to `PYTHONPATH` for the TPOT step only.
- resumable: rerun the identical command and it skips `(dataset, mode)` rows already in the file.
- if you would rather not risk shard 5 on 378: `--ids "1054 1037 42165 30 1497"` first, then
  `--ids 378` in its own session.

## Cell 5 — read before trusting

```bash
%%bash
REPO=$(cat /tmp/repo_path)
python "$REPO/scripts/adp_bench.py" --summarize '/kaggle/working/arms_1-adp-ourops_tpot_*.jsonl'
```

`adp_bench --summarize` accepts the `tpot` rows and gives a `fair/tpot` column. Check the callouts:

| flag | meaning | action |
|---|---|---|
| **No score** / `status: failed` mentioning **NaN** | AutoDP's frame had NaN (`imp_null`); TPOT with `preprocessing=False` cannot consume it | **exclude from the mean**, footnote as an AutoDP failure — do NOT count as 0 |
| `!! DEAD SEARCH -> raw frame` | AutoDP scored zero nodes; the row is the raw frame, not a searched pipeline | exclude from the "AutoDP-search-succeeded" mean; report flagged |
| `EMPTY pipeline` | AutoDP chose no preprocessing; score is the raw frame | footnote it, don't average as a search result |
| `AutoDP deleted test rows` (`test_coverage < 1`) | `score` (= `score_full`) counts dropped rows wrong | report `score_full`, `score_kept` alongside |
| `search-iteration exceptions` | their MCTS swallowed exceptions; pipeline may be unevaluated | treat with caution |

To join the TPOT column to the ACORec+TPOT numbers, check the `dataset_csv.target_sha1_16`
fingerprint in each row matches the ACORec arm's row for that dataset.

## Cell 6 — pull the file

```python
from IPython.display import FileLink
FileLink('/kaggle/working/arms_1-adp-ourops_tpot_1of5.jsonl')
```

Commit the five `arms_1-adp-ourops_tpot_*of5.jsonl`; they concatenate for the table.

---

## The ACORec side of this comparison

`scripts/run_acorec_tpot_eval.py` (batch runner) / `scripts/evaluate_acorec_tpot.py` score frozen
ACORec `recommendation.json`s with the same TPOT. **Reuse the `recommendation.json` files from the
run that produced the AutoGluon/H2O columns** — do not re-run ACORec search, or a flag mismatch
silently changes which pipeline TPOT scores. It runs in the TPOT env only (no AutoDP stage):

```bash
%%bash
REPO=$(cat /tmp/repo_path); export PYTHONPATH="/tmp/tpotlibs:$REPO/src"
ID=1066
python "$REPO/scripts/evaluate_acorec_tpot.py" \
  --recommendation-json /kaggle/input/<your-acorec-run>/dataset_$ID/recommendation.json \
  --dataset-csv /kaggle/working/eval_all/$ID.csv --dataset-id $ID \
  --output-json /kaggle/working/acorec_tpot/dataset_$ID/tpot_evaluation.json
```
