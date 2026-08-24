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
entirely. A random neighbour costs 0.0192, so the metafeatures are *not* pure noise — but nearly
all of their value is "pick a generally-good pipeline," not "pick the right one for *this* dataset."

The whole retrieval apparatus — metafeatures, learned metric, top-K neighbours, Eq-7 aggregation —
is worth about a third of a percentage point over hardcoding one pipeline.

### It is not the metric model's fault

An RF regressor trained directly on (metafeatures → per-pipeline gain), 5-fold CV, upper-bounds
what *any* model over these metafeatures can extract. It reaches regret **0.0362** — statistically
indistinguishable from the 1-NN's 0.0371. Per-pipeline Spearman between predicted and true gain
runs **0.34–0.48**.

So a better retrieval model cannot rescue this. The metafeature→pipeline-gain signal itself is
capped at roughly ρ ≈ 0.45. **This is a ceiling, not a bug.**

## 3. The learned metric is currently degenerate

From the existing diagnostics in `outputs/rq3_metric_neighbor_diagnostics_*/`:

| | plain cosine | trained Siamese |
|---|---|---|
| mean score spread across candidates | 0.0731 | **0.0055** |
| mean top1–top2 margin | 0.0062 | **0.0013** |
| single dataset that is top-1 for … | 10% of queries | **80–90% of queries** |

The trained metric returns **essentially the same neighbour regardless of query**, and is *less*
discriminative than the untrained cosine baseline it replaced. Combined with §2, this is why the
"our operator space" and "their operator space" columns sit so close: in both, the transfer
heuristic is close to query-independent, so both arms warm-start from nearly the same prior.

This is a correctness problem, not a tuning knob. A metric whose top-1 is constant is not a metric.

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

---

## What to change, ranked by accuracy per unit of compute

**A. Fix or retire the Siamese metric (correctness, not tuning).** Its top-1 is constant across
80–90% of queries and it is less discriminative than plain cosine. Either train it against a
held-out ranking objective with early stopping on reference-holdout NDCG, or fall back to cosine
and say so. Shipping a "learned metric" that its own diagnostics show to be degenerate is the one
thing in this list that is a reviewer risk rather than a score risk.

**B. Spend the entire remaining budget on proxy fidelity.** This is the only lever with a
multi-point payoff. Concretely, and each is measurable on reference-holdout *before* any eval run:
  - a short AutoGluon fit (`medium_quality`, 15–30 s) as the proxy for the surviving top-N
    candidates, instead of one logreg;
  - a rank-averaged ensemble (logreg + HistGBDT, 3 seeds) rather than a single model;
  - re-measure with the existing harness that produced `outputs/proxy_fidelity.log` — that file is
    already a legitimate, test-set-independent tuning loop. Only adopt a change that moves ρ on it.

**C. Turn on the global prior.** `--global-prior-weight` already exists and its own docstring says
0.3–0.5 stabilizes the weak neighbour signal. §2 is direct evidence for exactly that: the global
default is within 0.0034 of retrieval, so blending them should dominate either alone. Validate the
weight on reference-holdout, then lock it.

**D. Reframe the contribution.** Given §2, "metafeature transfer picks the right pipeline" is not a
claim the data supports, and a reviewer running the obvious ablation will find what §2 found. The
defensible and more interesting story: *a warm-started search with a leak-free CV gate matches a
far heavier MCTS baseline at a fraction of the search cost — and we quantify how weak the
metafeature-transfer signal actually is.* An ablation table showing global-default ≈ retrieval is a
strong honest result. Presented as a finding it is a contribution; discovered by a reviewer it is a
hole.

**E. Only then, raise the search budget.** More ants are worth little while the compass reads
ρ=0.43 (§4 caps what any amount of search can deliver through a noisy proxy). Do B first, then
scale ants/iterations — they multiply.
