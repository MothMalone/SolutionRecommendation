> # ⚠️ INVALIDATED — DO NOT CITE
>
> Every quantitative claim below was computed from an AutoDP environment missing
> `category_encoders`, an **undeclared runtime dependency** of `autodatapre==0.1.12`
> (imported lazily inside `Search_Space/encoding.py` for its `BE`/`FE`/`CBE` encoders).
> Without it those encoders raise, every MCTS candidate scores `pre_profit = 0.0`,
> `gap > 0` is never satisfied, `best_node` never leaves the root, and the search returns an
> empty pipeline while reporting itself converged. Three layers of bare `except:` hide it.
>
> With the dependency installed, AutoDP selects real preprocessing on **8 of 18** completed
> searches (not 2), overwhelmingly via the previously-broken encoders — see
> `docs/autodp_pipeline_census_local.tsv`.
>
> Consequently: the "AutoDP returns raw data on 16-19 of 21" finding is an artifact; the
> `Mingap`-below-the-noise-floor mechanism is wrong; the three-group decomposition is void
> (its groups were defined by the empty-pipeline test); and the AutoDP score column itself
> must be re-run, since it was produced under the same broken environment.
>
> Rewrite this document only after re-running AutoDP with `category_encoders` present.

# Why ACORec outperforms AutoDP, and how the two pipelines differ

Evidence base: the paired AutoGluon evaluation over 20 datasets (AutoDP 0.7675 / ACORec 0.7875,
11W-4T-5L, Wilcoxon p = 0.0131), an AutoDP **pipeline census** (its search run alone, recording what
it selects), and a full local ACORec run capturing the final gate's decision and every candidate's
score. Sources: `docs/autodp_pipeline_census_local.tsv`, `scripts/analyze_autodp_gap.py`.

---

## 1. The short answer

**AutoDP almost never preprocesses.** Its MCTS terminates at depth ≤ 1 — it picks an internal
classifier and stops before selecting any preprocessing operator — on **19 of 21** datasets whose
search completed, converging in under 5 seconds on 12 of them. Its "prepared" dataset is therefore
the raw input, and its score is a no-preprocessing baseline.

ACORec's advantage is **not** that it searches the same space better. It is that ACORec sometimes
returns a pipeline worth applying, and AutoDP essentially never does.

| group | n | AutoDP | ACORec | Δ |
|---|---|---|---|---|
| both returned raw data | 3 | 0.8170 | 0.8170 | +0.0000 |
| AutoDP returned raw data, ACORec preprocessed | 13 | 0.7548 | 0.7805 | **+0.0257** |
| both preprocessed (contested) | 2 | 0.7850 | 0.7890 | +0.0040 |
| AutoDP search never finished | 2 | 0.7590 | 0.7875 | +0.0285 |

The headline +0.0200 lives almost entirely in row 2. Row 3 — the only genuine
pipeline-versus-pipeline comparison — is 1 win each over 2 datasets and carries no signal.

---

## 2. Why AutoDP stops before preprocessing anything

Three design choices compound:

1. **Its selection signal is one fit on one split.** Each candidate is scored by a single
   NB / LDA / RandomForest fit on an unseeded 20% split (`MCTS_DATA.read_dataset`,
   `Search_Space/classifier.py`). On the 150–200-row datasets here that estimate carries roughly
   ±0.05 of noise.
2. **Its stopping threshold is 0.001.** The search halts after 20 consecutive iterations that fail
   to improve by more than `Mingap = 0.001` (`MCTS.py`). A 0.001 threshold is two orders of
   magnitude below the noise in the signal judging it, so improvement is indistinguishable from
   chance and the counter runs out almost immediately.
3. **Nothing biases it toward acting.** With no improvement detectable, the incumbent — the empty
   pipeline at the root — is returned.

Measured consequence: pipelines of `[]`, `['NB']`, `['RF']`, `['LDA']` (element 0 is the classifier;
everything after it would be preprocessing) on 19 of 21 datasets, with search times of 0.8–5 s. It
is **not** a budget problem: it converges by its own rule, long before any cap.

## 3. Why ACORec returns a usable pipeline more often

- **Lower-variance candidate scoring.** 3-fold CV × 3 seeds (`--cv-select-folds 3 --proxy-seeds
  42,52,62`) instead of one split, so a real improvement is separable from noise.
- **A transfer arm.** The final gate always includes `no_search_retrieval`: the best pipeline of the
  nearest reference dataset under the learned Siamese metric. **AutoDP has no analogue** — its
  pretrained meta-learner only chooses the *order in which operator families are searched*, never a
  concrete pipeline. In the local run this arm won the gate on **6 of 17** datasets, making it the
  single largest contributor among the non-trivial candidates.
- **Verification on the real evaluator.** The winner is confirmed by AutoGluon CV, not by the weak
  proxy that ranked it.

Gate decisions across the 20-dataset local run:

| final choice | n |
|---|---|
| `no_preprocessing` (the floor) | 7 |
| `no_search_retrieval` (transfer from nearest dataset) | 6 |
| ACO search pipeline | 4 |
| evaluation failed | 3 |

---

## 4. Concrete differences in the operator space

| stage | AutoDP | ACORec |
|---|---|---|
| imputation | RAND, MF, MICE, KNN, EM, MEDIAN, MEAN, DROP | none, mean, median, most_frequent, constant, knn |
| encoding | OE, BE, FE, CBE | onehot (fixed in the RQ3 protocol) |
| scaling / normalization | ZS, DS, MM | none, standard, minmax, robust, maxabs |
| feature selection | MR (missing-ratio), WR (chi²-KBest), LC (collinearity), TB (tree-based) | none, variance_threshold, k_best, mutual_info |
| outlier removal | ZSB, IQR, LOF | none, iqr, zscore, lof, isolation_forest |
| duplicate removal | ED (exact), AD (approximate string) | **absent** |
| dimensionality reduction | **absent** | none, pca, svd |
| model choice | **part of the searched pipeline** (NB / LDA / RF) | not searched — AutoGluon is fixed |
| step order | fixed by the meta-learner per run | **searched**, under precedence constraints |

Two structural differences matter most:

- **AutoDP searches the classifier as part of the pipeline.** It optimises the preprocessing *for
  one of three weak models*, then the pipeline is handed to AutoGluon. Preprocessing that helps
  Naive Bayes (aggressive scaling, discretisation, feature pruning) is frequently worthless to a
  bagged gradient-boosting ensemble. ACORec never searches the model; the pipeline is always judged
  for the estimator that will actually be used.
- **ACORec searches the order of operations, AutoDP does not.** AutoDP's meta-learner fixes the
  family order up front; ACORec treats order as a search dimension under precedence constraints.

---

## 5. What this evidence does NOT support

State these before a reviewer does.

- **Not "our search finds better pipelines than theirs."** Only 2 datasets had both methods
  preprocessing, and they split 1–1. There is no evidence of superior operator selection
  head-to-head.
- **ACORec often declines too.** Its gate chose `no_preprocessing` on 7 of 17 completed local runs —
  the same output AutoDP produces. The four exact ties in the paired table are precisely these
  datasets. On them the two methods are indistinguishable by construction.
- **The gate is not always right.** On 5 of 17 local datasets the selected candidate scored *worse*
  on the held-out test set than `no_preprocessing` (1066, 1485, 27, 862, 876 — by up to 0.10 on
  876). The CV gate exists to prevent this and does not fully succeed.
- **Some reported cells may not be AutoGluon scores.** Three datasets (1047, 378, 381) reported
  `final_evaluation.method = "autogluon_failed"`, in which case the LogisticRegression proxy score
  is written into the AutoGluon column. Audit `final_evaluation.method` across all results before
  publishing any mean.

## 6. The defensible claim

> AutoDP's MCTS converges to the empty pipeline on the large majority of these datasets — its
> stopping threshold (0.001) sits far below the noise of its single-split NB/LDA/RF selection signal
> — so in practice it reduces to the no-preprocessing baseline. ACORec, using lower-variance
> candidate scoring and a meta-learned transfer arm that proposes a concrete pipeline from the
> nearest reference dataset, returns a preprocessing pipeline worth applying on a substantial subset
> of datasets, gaining 2.6 accuracy points there and 2.0 points overall (Wilcoxon p = 0.013).

That claim is measured, falsifiable — a reviewer can reproduce the census by printing what
`autodatapre` returns — and it does not overstate the operator-selection result.
