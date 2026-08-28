# Kaggle cells — arm 1 (`1-adp-ourops`) scored with **AutoGluon**

Produces the **AutoDP** number for the ACORec-vs-AutoDP main comparison (RQ2.4): AutoDP's MCTS
searching ACORec's 19-operator space, on our 30 evaluation datasets, scored with AutoGluon
`best_quality` under the `leakfree` protocol.

This is the AutoGluon twin of `docs/kaggle_adp_arm1_h2o_cells.md` — **same AutoDP search, same
protocol, same seed-42 0.6/0.2/0.2 split, same `score_full`/`score_kept` accounting**; only the
downstream model differs (`_fit_predict_with_autogluon` instead of `_fit_predict_with_h2o`).

**Corpus:** arm 1 reads the retrained 1-NN family-order meta-learner from `data/adp_ourops_corpus`.
The parity rebuild covers **645 reference datasets** drawn from the same 901-column library ACORec's
Siamese trains on (901 → 775 after the classification + ≤1000-feature filters the LogReg proxy needs
and the eval/`THEIR_DATASETS` holdout → 645 after attrition), scored with the LogReg proxy, 10
shuffled-order pipelines each — replacing the earlier 108-dataset build
(`data/adp_ourops_corpus_108`, fingerprint `d5b76a950e749ead`). Closes the order-of-magnitude scale
gap; not exact parity (ACORec's 879 applies neither filter). Fingerprint
of the parity corpus: **`80c470059a49543c`**. The value estimator `model_CLA.pickle` is **not**
retrained — see `docs/DATASET_CHANGE_AND_RQ3.md` §C.1–C.3.

> **Before running:** re-upload the `acorec` Kaggle dataset so it carries the rebuilt
> `data/adp_ourops_corpus/{Metafeature.csv,label.csv,corpus_fingerprint.txt}`. Cell 1 prints the
> fingerprint; it must read `80c470059a49543c` (NOT `d5b76a950e749ead` — that is the old 108 corpus).

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

- **378 and 722** previously burned ~10,800 s each. `--cap-seconds 5400` bounds them; the
  node-level `SearchCheckpoint` salvage makes a cap-kill non-fatal (`salvaged_from_checkpoint:
  true`). If you would rather not risk shard 5 on 378, split it: run
  `--datasets 1054,1037,42165,30,1497` first, then `--datasets 378` in its own session.
- **862 and 27** collapse under `leakfree` on `ours` ops with *zero* scored nodes → the row becomes
  the **raw frame** (`dead_search: true`, `empty_pipeline: true`), aborted in ~10 s. Report as
  AutoDP failures (raw-frame number); `--summarize` lists them under a "DEAD SEARCH" callout.
- Runs are **resumable** — restart the identical command; it skips `(arm, dataset, protocol,
  evaluator)` rows already in the file.

---

## Cell 1 — code-freshness + corpus check

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
  check scripts/eval_autodatapre.py         _fit_predict_with_autogluon
  check scripts/adp_bench.py                'evaluator=args.evaluator'
  check scripts/run_arms.py                 'evaluator mismatch'
  check scripts/autodp_our_space.py         _retrained_order_fn
  check scripts/autodp_protocol.py          SearchCheckpoint
  check scripts/run_autodatapre.py          salvaged_from_checkpoint
  for f in data/adp_ourops_corpus/label.csv data/adp_ourops_corpus/Metafeature.csv; do
    [ -f "$REPO/$f" ] && printf '  ok    %s\n' "$f" || { printf '  STALE %s\n' "$f"; fail=1; }
  done
  echo
  if [ $fail -eq 1 ]; then
    echo "  >>> STALE - refresh the acorec dataset (Add Data), do NOT continue"
  else
    echo "  >>> code is current"
    n=$(($(wc -l < "$REPO/data/adp_ourops_corpus/Metafeature.csv") - 1))
    echo "  >>> corpus datasets: $n   (parity rebuild = 645; 108 = OLD, stop and refresh)"
    echo -n "  >>> corpus fingerprint: "; cat "$REPO/data/adp_ourops_corpus/corpus_fingerprint.txt"
  fi
fi
echo
[ -d /kaggle/input/datasets/mathurinache/openml ] \
  && echo "  openml mount: attached" || echo "  openml mount: NOT attached (Cell 4 needs it)"
```

## Cell 2 — main env: AutoGluon deps  (Internet ON)

```bash
%%bash
python -c "import autogluon" 2>/dev/null || \
  pip install -q --target=/tmp/aglibs "autogluon.tabular[all]==1.5.0" "numpy<2"
```

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

## Cell 5 — run this shard  (change `1/5` and the `--out` name per session)

```bash
%%bash
set +e
REPO=$(cat /tmp/repo_path)
export PYTHONPATH=/tmp/aglibs

python -u "$REPO/scripts/run_arms.py" \
  --arm 1-adp-ourops \
  --shard 1/5 \
  --data-dir /kaggle/working/eval_all \
  --out /kaggle/working/arms_1-adp-ourops_1of5.jsonl \
  --protocol leakfree --time-limit 300 \
  --cap-seconds 5400 \
  --adp-extra "--adp-python /tmp/adpenv/bin/python" \
  2>&1 | tee /kaggle/working/log_arm1_ag_1of5.txt
```

- No `--evaluator` flag → defaults to `autogluon`; rows are tagged `evaluator:"autogluon"`.
- arm `1-adp-ourops` auto-uses `data/adp_ourops_corpus` (retrained meta-learner; value estimator
  NOT retrained — disclose per `docs/ARMS.md` / `docs/DATASET_CHANGE_AND_RQ3.md` §C.2).
- `--time-limit 300` is AutoGluon's **per-model** budget (contrast the H2O doc, where it is the
  whole-AutoML budget). Per-dataset ~ search + up to ~8 AG fits × 300 s; a 6-dataset shard is
  ~2–4 h, comfortably under Kaggle's 12 h.
- `--autogluon-profile` is not exposed by `run_arms.py`; AutoDP arms inherit `adp_bench.py`'s
  default `best_quality` — the reported profile. (To smoke locally where `best_quality` segfaults,
  add `--adp-extra "--adp-python … --autogluon-profile local_rf_xt"`; that is a plumbing check, not
  a reportable number.)

Print all five commands:

```bash
%%bash
REPO=$(cat /tmp/repo_path); export PYTHONPATH=/tmp/aglibs
python "$REPO/scripts/run_arms.py" --print-commands --arm 1-adp-ourops --shards 5 --time-limit 300
```

## Cell 6 — read before trusting

```bash
%%bash
REPO=$(cat /tmp/repo_path); export PYTHONPATH=/tmp/aglibs
python "$REPO/scripts/run_arms.py" --summarize '/kaggle/working/arms_1-adp-ourops_*of5.jsonl'
```

Per-arm line reads `1-adp-ourops  autogluon  n=..  mean=..`. Check the warnings:

| flag | meaning | action |
|---|---|---|
| `EMPTY pipeline` | AutoDP chose no preprocessing; score is the **raw frame** | footnote it, don't average as a search result |
| `DEAD SEARCH` | search scored zero nodes; score is the raw frame | report as an AutoDP failure |
| `SALVAGED from a cap-killed search` | pipeline is the best completed iteration, not converged | reportable, disclose |
| `search-iteration exceptions` | their MCTS swallowed exceptions; pipeline may be unevaluated | treat with caution |
| `AutoDP deleted test rows` (`test_coverage < 1`) | `score` (=`score_full`) counts dropped rows wrong | report `score_full`, `score_kept` alongside |

## Cell 7 — pull the file

```python
from IPython.display import FileLink
FileLink('/kaggle/working/arms_1-adp-ourops_1of5.jsonl')
```

Commit the five `arms_1-adp-ourops_<i>of5.jsonl`; they concatenate for the table.
```
