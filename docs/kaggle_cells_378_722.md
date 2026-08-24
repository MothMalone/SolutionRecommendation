# Kaggle cells — arm 0 rerun for 378 and 722

Run each dataset in its **own notebook session** (two notebooks, or two runs of the same one back
to back) — Kaggle's per-session time limit is the whole reason these two datasets need a rerun in
the first place, and the checkpoint/salvage fix only helps if the cap you give it is generous.

Turn Internet **ON** for the notebook (Settings → Internet) — cell 3 needs it to build the AutoDP
env, and if the CSVs aren't already on Kaggle, `adp_bench.py` falls back to a live OpenML fetch.

---

## Cell 1 — clone the repo

```bash
!git clone https://github.com/MothMalone/SolutionRecommendation.git
%cd SolutionRecommendation
!git log --oneline -3
```

Confirm the top commit is `2531b97` (or later) — that's the one with the checkpoint/salvage fix.
If it isn't, the rerun will just reproduce the same failures.

## Cell 2 — main environment (AutoGluon side)

```bash
!pip -q install -r requirements-kaggle.txt
```

## Cell 3 — AutoDP's pinned environment

```bash
!bash scripts/setup_autodp_env.sh
```

This builds `.venv-autodp` (python 3.10, numpy 1.23, pandas 1.5 — incompatible with AutoGluon on
purpose, which is why it's a separate env). Takes a few minutes; it also builds
`py-stringmatching`/`py-stringsimjoin` from source.

## Cell 4 — dataset 378

```bash
!python scripts/run_arms.py --arm 0-adp-baseline --datasets 378 \
    --out /kaggle/working/arms_0_378.jsonl \
    --protocol leakfree \
    --time-limit 300 \
    --cap-seconds 5400 \
    --data-dir data/eval_datasets
```

## Cell 4′ (separate session) — dataset 722

```bash
!python scripts/run_arms.py --arm 0-adp-baseline --datasets 722 \
    --out /kaggle/working/arms_0_722.jsonl \
    --protocol leakfree \
    --time-limit 300 \
    --cap-seconds 5400 \
    --data-dir data/eval_datasets
```

`--cap-seconds 5400` (90 min) is generous on purpose — 722 previously burned 6549s across two
kills and got nothing; the checkpoint fix means a big cap now only costs time when the search
actually needs it, not when it's wasted. `data/eval_datasets` doesn't need to exist ahead of time:
`adp_bench.py` exports the CSV itself (from a local copy if `--openml-local-folder` finds one via
`--data-dir`, otherwise straight from OpenML with Internet on).

## Cell 5 — read the result before trusting it

```bash
!python scripts/run_arms.py --summarize "/kaggle/working/arms_0_*.jsonl"
```

Check the printed warnings, not just the score column:

| flag | what it means | what to do |
|---|---|---|
| `EMPTY pipeline` | AutoDP chose no preprocessing; the score is the **raw frame**, not a search result. Reproduced on 378 during testing — the explicit-runTime retry can return this on large frames. | Do not report as a normal AutoDP number; note it as a degenerate run in the table footnote. |
| `SALVAGED from a cap-killed search` | The pipeline is the best of the iterations that completed before the kill, not a converged search. | Reportable, but disclose it — it's a weaker guarantee than the other 28 rows in the column. |
| `search-iteration exceptions` | Their MCTS loop swallowed exceptions; the reported pipeline may never have been evaluated. | Pre-existing check; treat with the same caution as before. |

If a row comes back `eval_crashed` instead of `ok`, the search itself succeeded — only AutoGluon
scoring crashed. The row's `prepared_dir_kept` path holds the prepared frame; rerun scoring alone
against it rather than repeating the multi-hour search.

## Cell 6 — pull the results back

```bash
from IPython.display import FileLink
FileLink('/kaggle/working/arms_0_378.jsonl')
```
```bash
from IPython.display import FileLink
FileLink('/kaggle/working/arms_0_722.jsonl')
```
