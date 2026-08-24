# Where the signal actually is — and where it is not

Reproduce every number here with:

    python scripts/diagnose_signal_contribution.py

All of it comes from the reference library (`data/openml/*`) with `EVAL_IDS` dropped and the drop
asserted. **No number below touched the 30 evaluation datasets.** That is what makes these
legitimate tuning targets rather than test-set fitting.

---

## The question the arms table cannot answer

The four columns of the results table land within 0.03 of each other, and the per-dataset winner
flips almost at random. That pattern has several possible causes — a weak retrieval signal, a weak
proxy, an under-powered search, or simply no headroom — and the table cannot distinguish them,
because it only ever shows the *end* of the chain. These diagnostics take the chain apart.

## 1. There is less headroom than the table implies

Over 779 reference datasets, comparing the best of 12 pipelines against no preprocessing at all:

| | |
|---|---|
| best pipeline beats no-preprocessing by ≤ 0.005 | **23.2%** of datasets |
| best pipeline beats no-preprocessing by ≥ 0.05 | 38.8% of datasets |
| median headroom | 0.0323 |

On roughly a quarter of datasets **the contest is unwinnable by construction** — no pipeline in the
space beats doing nothing by a meaningful margin. A CV gate that returns the floor on those rows is
behaving correctly, not failing. The averages in the results table are therefore dominated by the
~39% of datasets where preprocessing genuinely matters, diluted by a quarter where nothing can move.

This alone explains part of why all four columns compress together.

## 2. Metafeature retrieval contributes almost nothing query-specific

Leave-one-out over the same 779 datasets. Each strategy recommends one pipeline per dataset; the
score is what that pipeline actually achieved.

| strategy | mean score | regret vs oracle |
|---|---|---|
| oracle (best pipeline per dataset) | 0.8096 | 0.0000 |
| **cosine top-1 neighbour's best pipeline** | 0.7708 | **0.0371** |
| cosine top-5 weighted vote | 0.7699 | 0.0391 |
| **global default** (metafeatures never consulted) | 0.7691 | **0.0405** |
| random neighbour | 0.7517 | 0.0563 |
| no preprocessing | 0.7441 | 0.0588 |

**Metafeature retrieval buys +0.0034 over a constant recommendation** that ignores the query
entirely, in this cosine-top-1 configuration. A random neighbour costs 0.0192, so the metafeatures
are *not* pure noise — but nearly all of their value is "pick a generally-good pipeline," not "pick
the right one for *this* dataset."

§3 revisits this with the trained metric and top-5 aggregation, which raises the figure to +0.0062
— still under a percentage point over hardcoding one pipeline.

### It is not the metric model's fault

An RF regressor trained directly on (metafeatures → per-pipeline gain), 5-fold CV, upper-bounds
what *any* model over these metafeatures can extract. It reaches regret **0.0362** — statistically
indistinguishable from the 1-NN's 0.0371. Per-pipeline Spearman between predicted and true gain
runs **0.34–0.48**.

So a better retrieval model cannot rescue this. The metafeature→pipeline-gain signal itself is
capped at roughly ρ ≈ 0.45. **This is a ceiling, not a bug.**

## 3. The learned metric is FINE — an earlier draft of this document was wrong

The `outputs/rq3_metric_neighbor_diagnostics_*/` files record a collapsed metric (score spread
0.0055, top1–top2 margin 0.0013, one dataset winning 80–90% of queries), and an earlier version of
this document called the metric degenerate on that basis. **That does not reproduce.** Those files
are `..._smoke_...` runs with `n_query_datasets` of 10.

Retraining the metric exactly as `metric.py` does — full pair set, 100 epochs, Pearson loss — on
the 779-dataset reference library:

| | plain cosine | trained Siamese |
|---|---|---|
| per-query similarity spread | 0.0833 | **0.1113** |
| top1–top2 margin | 0.0043 | **0.0140** |
| most dominant neighbour wins | 0.9% of queries | **0.6% of queries** |

The trained metric is *more* discriminative than cosine on every one of those, not less. The
architecture is also fine: dropping the embedder's final `ReLU` (a plausible cause of cosine
collapse, since it confines embeddings to the positive orthant) makes similarity-matrix
reconstruction slightly **worse**, 0.393 vs 0.421 Spearman.

And it earns its place on the actual task, provided the aggregation matches:

| retrieval variant | regret vs oracle |
|---|---|
| **siamese top-5 aggregated** | **0.0343** |
| cosine top-1 | 0.0371 |
| siamese top-1 | 0.0373 |
| cosine top-5 | 0.0391 |
| global default | 0.0405 |

So retrieval is worth **+0.0062** over a constant recommendation, not the +0.0034 in §2 — but only
in the `siamese + top-5` cell. The production no-search arm used `neighbor_k=1, top_l=1`, hardcoded
with no CLI flag, which is the *worst* Siamese cell in the table. `--hybrid-no-search-neighbor-k`
and `--hybrid-no-search-top-l` now expose it.

Retire nothing. Retrieval is still small next to §4 and §6, but it is real and it was mis-configured.

## 4. The proxy is where the accuracy actually is

Measured proxy↔AutoGluon rank agreement (`outputs/proxy_fidelity.log`, 12 reference datasets):

| proxy model | mean Spearman |
|---|---|
| **logreg (current)** | **+0.417** |
| random_forest | +0.113 |
| hist_gbdt | +0.062 |
| extra_trees | +0.060 |

Simulating selection among all 12 pipelines with a proxy of a given rank fidelity:

| proxy Spearman | mean score | regret vs oracle |
|---|---|---|
| **0.43 (current)** | 0.7796 | **0.0300** |
| 0.60 | 0.7890 | 0.0206 |
| 0.70 | 0.7944 | 0.0152 |
| 0.90 | 0.8055 | 0.0041 |

**Lifting the proxy from ρ=0.43 to ρ=0.70 is worth +0.0148 accuracy — about 4× everything
metafeature retrieval contributes (+0.0034), and it compounds with a larger search budget rather
than competing with it.**

## 5. Is the optimizer an overfitting machine?

No — if anything the opposite. The REF configuration is `--n-ants 4 --n-iterations 3` = **12 proxy
evaluations** over a space of roughly 15,000 configurations. That is not enough to overfit; it is
barely enough to explore. And the final CV gate (`--hybrid-select --cv-select-folds 3`, selection
on a held-out split, reporting test) is a genuine guard against the winner's curse.

The real failure mode is the reverse: **a 12-sample search steered by a ρ=0.43 proxy rarely finds
anything the gate prefers over the floor, so the gate correctly keeps the floor** — which is why so
many rows tie the no-preprocessing baseline. The system is under-searching with a noisy compass,
not overfitting.

## 6. Saturation is real, but it is on the *search* axis, not the budget axis

Reproduce with `python scripts/diagnose_budget_allocation.py`. Selection is modelled as a ladder of
noisy observers of one true score; candidate scores are resampled from each dataset's own measured
spread, so per-dataset difficulty stays empirical. Candidates are drawn IID, which a real ACO run
does not do — it concentrates — so the search axis below is an **optimistic** bound: a real search
saturates sooner than these curves, never later.

**Widening the search while only its #1 survives (REF's shape) saturates:**

| proxy evals | ρ=0.43 | ρ=0.70 | ρ=0.90 |
|---|---|---|---|
| 12 (REF) | 0.7866 | 0.8114 | 0.8295 |
| 50 | 0.7990 | 0.8317 | 0.8524 |
| 200 | 0.8114 | 0.8456 | 0.8665 |
| 800 | 0.8215 | 0.8572 | 0.8770 |

A 66× budget increase buys +0.035 at ρ=0.43. That is the saturation to be afraid of, and the
mechanism is precise: **the argmax of a noisy surrogate over a bigger pool converges on whichever
candidate drew the luckiest noise**, so the marginal candidate is increasingly a noise artifact.

**But the binding constraint is how many candidates reach the real evaluator:**

| N proxy | K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| 12 | 0.7861 | 0.8153 | 0.8229 | 0.8296 | — |
| 200 | 0.8113 | 0.8392 | 0.8473 | 0.8549 | 0.8597 |

At N=200, moving K from 1→5 gains **+0.036** — more than the entire 12→200 search widening
(+0.028) that it sits next to. REF pins K to 1.

**And the ladder beats either end:**

| configuration | score | vs REF |
|---|---|---|
| REF today — N=12, K=1 | 0.7859 | — |
| wider search only — N=200, K=1 | 0.8135 | +0.028 |
| wider gate — N=200, K=5 | 0.8473 | +0.061 |
| **3-rung — N=200, 20 → 5 → 1** | **0.8602** | **+0.074** |
| 3-rung wider — N=400, 40 → 8 → 1 | 0.8683 | +0.082 |

Read these as *relative* comparisons. The absolute levels are inflated by the resampling, which
gives a richer candidate pool than 19 real pipelines; the ordering and the ratios are the result.

**The cost objection does not survive measurement.** The gate was budgeted as if AutoGluon fits
cost `time_limit` (300 s). Measured fits from the arm runs: **4–20 s**, including 19.4 s on a
48,842-row dataset. A K of 8 with 3-fold CV is roughly 8 × 4 × 10 s ≈ 320 s. AutoDP spends
3,300–6,500 s per large dataset. There is room for the whole ladder inside a comfortable win on time.

---

## What to change, ranked by accuracy per unit of compute

**A. Give the real evaluator more than one candidate.** ✅ *Implemented.* `--screen-topk` adds the
middle rung: N proxy-ranked candidates → one short AutoGluon fit each, ranked on validation → the
best `--final-autogluon-topk` to the full CV gate. This is the +0.06 to +0.07 row of §6 and it is
cheap because fits are 4–20 s. Suggested starting point, to be validated on reference-holdout:

    --n-ants 8 --n-iterations 6 --screen-topk 20 --screen-profile local_rf_xt \
    --screen-time-limit 30 --final-autogluon-topk 5

**B. Fix the no-search arm's aggregation.** ✅ *Exposed.* It ran at `neighbor_k=1, top_l=1`, the
worst Siamese cell in §3. `--hybrid-no-search-neighbor-k 5 --hybrid-no-search-top-l 1` moves it to
the best one (+0.003). Nearly free.

**C. Then proxy fidelity.** §4 says ρ=0.43→0.70 is worth +0.0148 at K=1, and it *multiplies* with
the ladder rather than competing with it: a better shortlist makes every rung above it better.
Measurable on reference-holdout before any eval run, using the harness that produced
`outputs/proxy_fidelity.log`. Note from that file that logreg is anti-correlated with AutoGluon on
some datasets (dataset 43: every proxy family negative), so a rank-averaged ensemble across
families is the first thing to try — it costs little and truncates the catastrophic cases.

**D. Only then raise search width.** §6's first table is the argument for doing this last: more
ants are worth little while the compass reads ρ=0.43 and the gate takes one candidate. After A and
C, width multiplies instead of saturating.

**E. Reframe the contribution.** §2 stands: retrieval is worth +0.0062 over a constant
recommendation, and a reviewer running that ablation will find it. The defensible and more
interesting story: *a warm-started search with a leak-free multi-fidelity gate matches a far
heavier MCTS baseline at a fraction of the search cost — and we quantify how weak the
metafeature-transfer signal actually is.* Presented as a finding it is a contribution; discovered
by a reviewer it is a hole.
