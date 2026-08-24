# The selection ladder: measured, null — and the diagnosis it forced

Reproduce:

    python scripts/validate_ladder.py --summarize outputs/ladder_validation.jsonl
    python scripts/measure_gate_fidelity.py --datasets 746 --summarize --out outputs/gate_fidelity.jsonl
    cat outputs/selector_compare.jsonl

24 reference-holdout datasets (`data/adp_ourops_corpus`, verified disjoint from EVAL_IDS and
THEIR_DATASETS), every one held out of the reference on every run, both arms sharing one
pre-trained Siamese, `local_rf_xt` at `time_limit=30`. **No evaluation dataset was touched.**

---

## 1. The ladder does nothing

| | simulation predicted | measured (n=24) |
|---|---|---|
| mean delta vs REF | **+0.06 to +0.07** | **−0.0004** |
| paired t | — | **−0.18** |
| win / loss / tie | — | **4 / 4 / 16** |
| runtime | "cheap" | **3.80× REF** (127s vs 33s) |

Stable throughout: −0.0013 at n=8, −0.0011 at n=19, −0.0004 at n=24. A real null, not an
underpowered one. Four wins and four losses is a coin flip.

`docs/SIGNAL_DIAGNOSIS.md` §6 argued the gate was a high-fidelity signal starved of candidates.
That argument is wrong, and the three things I checked next each killed a different explanation
for why.

## 2. It is not "the sample was too easy"

The obvious defence — small local datasets have no room — does not hold. Against the reference
library, my 24 datasets had median headroom **0.0393** vs the library's **0.0323**: *more* room than
typical. And headroom *falls* with size (median 0.050 under 500 rows, 0.020–0.030 above 5000), so
larger datasets would not have helped. The ladder failed on above-average-headroom data.

## 3. It is not "there is nothing to gain"

I nearly published this one. Measured over 9 datasets with a 12-candidate grid, the gate's top-1
pick sits **0.0202** below the best candidate available. Against headroom of ~0.039, that means
**roughly half the achievable gain is genuinely lost to imperfect selection.** Selection is a real
bottleneck. The ladder simply failed to move it.

## 4. It is not "the gate is too noisy for K=5"

Gate fidelity measured **ρ=+0.567** (median 0.645), squarely the regime where the K sweep put the
optimum at ~5 — the value the preset uses. Small validation blocks are not the culprit either:
ρ averages 0.550 under 120 val rows and 0.603 above, a gap swamped by per-dataset variance.

## 5. What it actually is: ranking cannot be made reliable at this scale

I predicted the screening rung was the flaw — it filters 20→5 with a single validation split, then
hands survivors to a 3-fold CV gate whose own docstring calls CV "a far less noisy estimate than a
single validation split." Filtering with the weaker judge would actively discard good candidates.

**That prediction was wrong too.** Running both selectors over identical candidates, n=7:

| selector | ρ (range) | top-1 regret | cost |
|---|---|---|---|
| single split | **+0.561** [−0.23, +0.86] | 0.0207 | 20s |
| 3-fold CV (the gate) | **+0.327** [−0.41, +0.71] | 0.0198 | 75s (**3.7×**) |

paired regret difference **+0.0010, t=+0.06**

Three things fall out:

* **The CV gate ranks *worse* than a single split** (ρ 0.327 vs 0.561) despite costing 3.7× — the
  opposite of what `cv_select_folds`' docstring claims. Plausible mechanism: CV selects using models
  fit on ~53% of the data while the reported score comes from a model fit on 80%, so the selection
  signal and the scored object are different training regimes. The single-split selector scores and
  reports from the *same* fitted model.
* **On the metric that decides anything — regret — they are identical** (t=+0.06). So
  `--cv-select-folds 3`, a REF default, costs 3.7× and buys nothing measurable.
* **The selector is anti-correlated with test on 1/7 (single) and 2/7 (CV) datasets**, and ρ ranges
  from −0.41 to +0.86. Per-dataset, selection is not merely imprecise; it is sometimes inverted.

## 6. The real conclusion

Every intervention I tried is the same operation in different clothes: **rank the candidates
better.** More candidates into the gate (K=5), a cheap pre-filter (screening), a costlier selector
(3-fold CV). All three return the same nothing, and the measurements say why — at this data scale
the ranking signal is unreliable enough to be inverted on a third of datasets, so no ranking
machinery recovers the 0.02 that is genuinely available.

**The lever that follows is different in kind: stop trying to pick the best pipeline; reduce the
cost of picking wrong.** When the selector is anti-correlated on 2/7 datasets, committing to its
single choice *is* the failure mode. Averaging the predictions of the top few pipelines converts
selection error into variance reduction instead of a committed mistake. AutoGluon already ensembles
over *models*; nothing here ensembles over *preprocessing pipelines*.

This is a hypothesis, not a result — it has not been measured. But it is the first proposal in this
document that is not another attempt to rank better, and it is cheap to test: the candidates are
already being fit.

## What survives in the code

* **`--screen-topk` stays, defaulting to 0.** Nothing changed for existing runs. The contract test
  proves the rung really reorders candidates; it just does not help. Worth retrying only where
  candidates have real spread.
* **The `ladder` preset stays, relabelled measured-null**, so nobody rediscovers this.
* **`--hybrid-no-search-neighbor-k` stays** — it fixed a genuinely mis-set default. But note what it
  bought: the ladder never selected `no_preprocessing` where REF did so 9 times, substituting
  `no_search_retrieval` **at the same score**. It changes which candidate wins, not what winning is
  worth. Reporting that as "ACORec no longer recommends doing nothing" would be cosmetic.

## Next, in order

1. **Test pipeline ensembling** (§6). The one idea here that is not "rank better."
2. **Question `--cv-select-folds 3`.** 3.7× cost, t=+0.06 on regret, worse ρ. Re-measure at n≈20
   and on larger frames before changing REF — but if it holds, that is a free 3.7× on the gate.
3. **Re-derive §4's proxy-fidelity claim.** Its +0.0148 rests on the same inflated simulation as
   §6's +0.07. Measured top-1 regret is 0.0202 *total*, so the proxy lever cannot be worth more
   than that no matter how good the proxy gets.
4. **Question the downstream model.** AutoGluon absorbs the preprocessing these operators perform,
   which is *why* candidates are near-interchangeable — `outputs/proxy_fidelity.log` shows
   preprocessing-sensitive logreg at ρ=0.42 against AutoGluon while hist_gbdt reaches 0.06.
   DiffPrep and AutoDP evaluate against fixed simple models, where preprocessing genuinely moves the
   number. Reporting both targets would make the contribution visible rather than compressed.

## Why the simulation was wrong

Structural, not a tuning error. `diagnose_budget_allocation.py` builds each candidate pool by
resampling a dataset's 19 measured scores **with kernel smoothing**, then drawing N. Drawing 200
smoothed samples from a 19-point distribution produces maxima well above anything measured, so the
simulated oracle — and the room for selection to matter — is manufactured. Its regret scale (~0.08)
is four times the measured one (0.0202). Two assumptions flagged in review before the run also
mattered: rungs modelled as independent observers when screening and the gate share a split, and
gate fidelity held constant in K.

The narrow lesson: **a simulation calibrated on a resampled candidate pool cannot answer a question
about selection, because the answer is dominated by the pool's spread — which is exactly what the
resampling fabricates.**
