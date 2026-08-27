# Kaggle cells — arm 1 (`1-adp-ourops`) re-scored with **H2O AutoML**

Produces the **AutoDP** number for the ACORec-vs-AutoDP comparison under a *second* downstream
evaluator, to sit next to ACORec's H2O column.

**Identical to the AutoGluon arm-1 run in every way except the downstream model:** the AutoDP
search (MCTS over ACORec's operator space, retrained 1-NN meta-learner from
`data/adp_ourops_corpus`), the `leakfree` protocol, the seed-42 0.6/0.2/0.2 positional split
(`automl_aco.data.splits.split_train_val_test`), fit-on-train+val / predict-the-20%-test,
original-target re-attachment, `score_full` / `score_kept` row-coverage accounting — all of it the
same `score_prepared` code path. The only fork is `_fit_predict_with_h2o` in place of
`_fit_predict_with_autogluon`.

**H2O settings:** `preprocessing=None` (downstream preprocessing off), `max_runtime_secs` =
`--time-limit` (300), `max_runtime_secs_per_model=60`, `nfolds=5`, `seed=42`; StackedEnsemble and
sort_metric at H2OAutoML defaults. Residual categoricals go to H2O as native `enum` factors.
Recorded per row under `protocol.h2o` / `evaluator_meta`.

---

## Sharding — 5 sessions, 6 datasets each

`--shard i/5` is round-robin over the 30 `EVAL_IDS`; the 5 output files concatenate with no overlap:

| shard | datasets | note |
|---|---|---|
| `1/5` | 1066, 876, 1485, 42932, 41001, 802 | — |
| `2/5` | 1047, 18, 14, 40668, 41671, **722** | 722 runs last |
| `3/5` | 862, 1520, 27, 1471, 1046, 40922 | — |
| `4/5` | 40663, 1548, 44956, 100000, 46597, 1119 | — |
| `5/5` | 1054, **378**, 1037, 42165, 30, 1497 | 378 runs 2nd |

**378 and 722** previously burned ~10,800 s each for nothing. The dead-loop-detection + node-level
`SearchCheckpoint` salvage (commit `ba76dcc`) got 378 down to ~30 s with
`salvaged_from_checkpoint: true` — **but that was measured on arm 0 (`theirs` ops)**. Arm 1 is
`ours` ops via `autodp_our_space.py`; the same `get_profit` hook applies but has not been measured
there. Treat ~30 s as "likely", not "confirmed". So:

**862 and 27** collapse a different way under `leakfree` on `ours` ops: the search scores
*zero* nodes (all `profit=None` on the tiny 0.6 train split), so there is no checkpoint to
salvage. As of the dead-search change, the row is now the **raw frame** — `dead_search: true`,
`empty_pipeline: true`, `score` = H2O on the untouched frame, aborted in ~10 s. Report these as
AutoDP failures (raw-frame number), the same standing as 378/722 timing out on arm 0 — `--summarize`
lists them under a "DEAD SEARCH" callout. Local end-to-end check: 862 → H2O `score_full` 0.706.
(876 spun on **arm 0** only; under arm-1 `ours` ops its smoke run produced `['NB']` — an ordinary
`empty_pipeline`, not `dead_search`. Watch the rerun but don't assume it.)

- `--cap-seconds 5400` on every shard (only costs wall time when a search needs it; salvage makes a
  kill non-fatal);
- runs are **resumable** — restart the identical command and it skips `(arm, dataset, protocol,
  evaluator)` rows already in the file;
- if you would rather not risk shard 5 on 378, split it: run `--datasets 1054,1037,42165,30,1497`
  first, then `--datasets 378` in its own session.

---

## Cell 1 — code-freshness check  (your existing check, plus H2O)

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
    if grep -q -- "$2" "$REPO/$1" 2>/dev/null; then printf '  ok    %-26s %s\n' "$2" "$1"
    else printf '  STALE %-26s %s\n' "$2" "$1"; fail=1; fi
  }
  check scripts/eval_autodatapre.py         _fit_predict_with_h2o
  check scripts/eval_autodatapre.py         H2O_RUNTIME_PER_MODEL
  check scripts/adp_bench.py                'evaluator=args.evaluator'
  check scripts/run_arms.py                 'evaluator mismatch'
  check scripts/autodp_our_space.py         _retrained_order_fn
  check scripts/autodp_protocol.py          SearchCheckpoint
  check scripts/run_autodatapre.py          salvaged_from_checkpoint
  check requirements-kaggle.txt             h2o==
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

## Cell 2 — main env: AutoGluon deps **+ H2O**  (Internet ON)

```bash
%%bash
python -c "import autogluon" 2>/dev/null || \
  pip install -q --target=/tmp/aglibs "autogluon.tabular[all]==1.5.0" "numpy<2"
python -c "import h2o" 2>/dev/null || \
  pip install -q --target=/tmp/aglibs "h2o==3.46.0.12"
java -version 2>&1 | head -1 || apt-get -qq install -y openjdk-11-jre-headless
```

`autogluon.tabular` is not used for scoring in this run, but `eval_autodatapre.py` imports it at
module load and `setup_autodp_env.sh` / other paths still reference it — keep it installed.

## Cell 3 — AutoDP's pinned env

```bash
%%bash
REPO=$(cat /tmp/repo_path)
ADP_VENV=/tmp/adpenv bash "$REPO/scripts/setup_autodp_env.sh"
```

## Cell 4 — export all 30 eval CSVs into one dir  (Internet ON)

```bash
%%bash
set +e
REPO=$(cat /tmp/repo_path)
export PYTHONPATH=/tmp/aglibs
rm -rf /kaggle/working/eval_all

# 17 DiffPrep datasets, and copy the 13 OpenML-native ones from the mount into the same dir
python "$REPO/scripts/export_diffprep_datasets.py" --download \
  --copy-openml-from /kaggle/input/datasets/mathurinache/openml \
  --out-dir /kaggle/working/eval_all

# 13 OpenML-native (fetches any the copy above didn't find; Internet ON covers it)
python "$REPO/scripts/export_eval_datasets.py" \
  --ids "1066 1047 862 40663 1054 876 18 1520 1548 378 1485 14 27" \
  --openml-local-folder /kaggle/input/datasets/mathurinache/openml \
  --out-dir /kaggle/working/eval_all

python "$REPO/scripts/export_diffprep_datasets.py" --fingerprint-only --out-dir /kaggle/working/eval_all
echo "CSV count: $(ls /kaggle/working/eval_all/*.csv | wc -l)  (expect 30)"
```

## Cell 5 — run this shard  (change `1/5` and the `--out` name per session)

```bash
%%bash
set +e
REPO=$(cat /tmp/repo_path)
export PYTHONPATH=/tmp/aglibs

python -u "$REPO/scripts/run_arms.py" \
  --arm 1-adp-ourops \
  --shard 1/5 \
  --evaluator h2o \
  --data-dir /kaggle/working/eval_all \
  --out /kaggle/working/arms_1-adp-ourops_h2o_1of5.jsonl \
  --protocol leakfree --time-limit 300 \
  --cap-seconds 5400 \
  --adp-extra "--adp-python /tmp/adpenv/bin/python" \
  2>&1 | tee /kaggle/working/log_arm1_h2o_1of5.txt
```

- `--evaluator h2o` is threaded to `adp_bench.py` → `eval_autodatapre.py`; `run_arms.py` aborts if
  the scored row comes back tagged with a different evaluator.
- arm `1-adp-ourops` auto-uses `data/adp_ourops_corpus` (retrained meta-learner; value estimator
  NOT retrained — disclose per `docs/ARMS.md`).
- `--time-limit 300` sets H2O `max_runtime_secs`. Note this is the *whole-AutoML* budget, whereas on
  the AutoGluon path 300 is the *per-model* budget — same number, different quantity — and H2O
  overran a 60 s budget by ~20 % in local testing. Per dataset budget ~6–12 min H2O + AutoDP
  search + H2OFrame parse; a 6-dataset shard is ~1–2 h.

Print all five commands:

```bash
%%bash
REPO=$(cat /tmp/repo_path); export PYTHONPATH=/tmp/aglibs
python "$REPO/scripts/run_arms.py" --print-commands --arm 1-adp-ourops --shards 5 \
  --evaluator h2o --time-limit 300
```

## Cell 6 — read before trusting

```bash
%%bash
REPO=$(cat /tmp/repo_path); export PYTHONPATH=/tmp/aglibs
python "$REPO/scripts/run_arms.py" --summarize '/kaggle/working/arms_1-adp-ourops_h2o_*.jsonl'
```

Per-arm line reads `1-adp-ourops  h2o  n=..  mean=..`. Check the warnings:

| flag | meaning | action |
|---|---|---|
| `EMPTY pipeline` | AutoDP chose no preprocessing; score is the **raw frame** | footnote it, don't average as a search result |
| `SALVAGED from a cap-killed search` | pipeline is the best completed iteration, not converged | reportable, disclose |
| `search-iteration exceptions` | their MCTS swallowed exceptions; pipeline may be unevaluated | treat with caution |
| `AutoDP deleted test rows` (`test_coverage < 1`) | `score` (=`score_full`) counts dropped rows wrong | report `score_full`, `score_kept` alongside |

## Cell 7 — pull the file

```python
from IPython.display import FileLink
FileLink('/kaggle/working/arms_1-adp-ourops_h2o_1of5.jsonl')
```

Commit the five `arms_1-adp-ourops_h2o_*of5.jsonl`; they concatenate for the table.
