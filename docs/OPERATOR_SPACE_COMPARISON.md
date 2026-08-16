# AutoDP's operator space vs ACORec's, operator by operator

Every claim below is read from `autodatapre==0.1.12` as installed (verified byte-identical to the
PyPI wheel) and from `src/automl_aco/preprocessing/`. The behavioural claims in §2 were executed,
not inferred; reproduce with `scripts/demo_autodp_cbe_leak.py` and the checks quoted inline.

Source map:

| family | AutoDP | ACORec |
|---|---|---|
| imputation | `Search_Space/imputer.py` | `preprocessing/imputation.py` |
| encoding | `Search_Space/encoding.py` | `preprocessing/encoding.py` |
| scaling | `Search_Space/normalizer.py` | `preprocessing/scaling.py` |
| feature selection | `Search_Space/feature_selector.py` | `preprocessing/feature_selection.py` |
| outlier removal | `Search_Space/outlier_detector.py` | `preprocessing/outliers.py` |
| dim. reduction | *absent* | `preprocessing/dimred.py` |
| duplicate removal | `Search_Space/duplicate_detector.py` | *absent* |
| orchestration | none — each operator owns the whole dict | `preprocessing/preprocessor.py` |

---

## 1. The architectural difference, which drives everything else

**ACORec has a fit/transform boundary. AutoDP does not.**

Ours is a single `Preprocessor` object. `fit_transform(X_train, y_train)` learns state and stores it
(`self.num_imputer`, `self.encoder`, `self.selector`, `self.scaler`, `self.reducer`);
`transform(X_test)` replays that stored state and raises `AssertionError` if you never fitted. One
function is learned from training data and then applied everywhere.

Theirs has no such object. Each operator is a class constructed around the whole
`{train, test, target, target_test}` dict, exposing a single method `transform()` that mutates the
dict and returns it. There is no `fit`. There is nowhere to *put* fitted state, so each operator
improvises — and they do not agree with each other. Three different conventions coexist inside one
search space:

| convention | operators | what happens to the test split |
|---|---|---|
| **recompute per split** | imputer, normalizer, outlier detector — `for key in ['train','test']:` | test is transformed by statistics computed **from test itself** |
| **fit on train, project columns** | feature selector — `to_keep` from train, then `fsd['test'] = df_test[to_keep]` | the right idea: train decides, test complies (but see §2.4) |
| **concatenate** | encoder — `pd.concat([train, test])` then fit on the union | train and test transformed as one table |

This is the single most important thing to understand about their operator space. It is not that
their operators are weaker than ours. It is that **an AutoDP pipeline is not a function you can
apply to unseen data.** It is a procedure for rewriting a table you already hold in full. Every
difference below is a consequence of that.

Our `transform()` even contains an explicit `elif step == "outlier_removal": pass`
([preprocessor.py:126-127](../src/automl_aco/preprocessing/preprocessor.py#L126-L127)) — rows are
dropped when fitting on train and never from test, because deleting test rows would change what the
score is measuring. Their `Outlier_detector.transform()` loops over `['train','test']` and deletes
from both. That asymmetry is exactly why our harness has to carry `score_full` and `score_kept`.

---

## 2. Family by family

### 2.1 Imputation

| | AutoDP | ACORec |
|---|---|---|
| operators | `RAND` `MF` `MICE` `KNN` `EM` `MEDIAN` `MEAN` `DROP` (8) | `mean` `median` `most_frequent` `constant` `knn` `none` (6) |
| implementation | hand-written pandas + `impyute` | `sklearn` `SimpleImputer` / `KNNImputer` |
| fitted on | each split separately | train only |
| categorical columns | untouched by MEAN/MEDIAN/KNN | always imputed (`most_frequent`) |

Theirs is the **larger** space — `MICE` and `EM` are genuinely more sophisticated than anything we
offer, and we should say so rather than pretend our six beat their nine.

Three implementation details matter:

**(a) Their mean is truncated to an integer.**

```python
X[i] = X[i].fillna(int(X[i].mean()))          # imputer.py:25, and :39 for median
```

Measured: a column `[1.0, 2.0, 8.0, NaN]` has mean `3.667`; AutoDP fills `3.0`. On a feature scaled
to `[0, 1]`, *every* mean-imputed value becomes `0`. This is a bug, not a design choice, and it
silently degrades their most commonly selected imputer.

**(b) Their imputation is transductive.** `transform()` loops `for key in ['train','test']`, so test
NaNs are filled with the **test split's own** column mean. Ours fits `SimpleImputer` on train and
calls `.transform()` on test.

**(c) Their `DROP` deletes rows from the test split.** Ours has no row-deleting imputer.

### 2.2 Encoding

| | AutoDP | ACORec |
|---|---|---|
| operators | `OE` ordinal, `BE` binary, `FE` frequency, `CBE` CatBoost target | `onehot` (fixed in the RQ3 protocol) |
| fitted on | `pd.concat([train, test])` | train only |
| unseen test category | **cannot occur** | `handle_unknown="ignore"` → all-zero row |

Theirs is again the larger space, and target encoding is a real technique we don't offer.

But this family is where the concatenation convention does the most damage, and there are two
distinct severities:

- `OE`, `BE`, `FE` fit on concatenated **features**. Transductive, no labels involved. `FE`'s
  frequencies (`X.groupby(x).size() / len(X)`) are computed over the union.
- `CBE` additionally concatenates the **targets** and passes them to a *supervised* encoder, so test
  labels become an input to the feature values (`encoding.py:79-94`):
  ```python
  X = pd.concat([self.dataset['train'], self.dataset['test']], axis=0)
  ...
  elif (self.strategy == "CBE"):
      target = pd.concat([self.dataset['target'], self.dataset['target_test']], axis=0)
      dn = self.CatBoost_encoding(X, target)     # enc.fit_transform(X, target)
  normd['train'] = dn.head(trainlen)
  normd['test']  = dn.tail(totallen - trainlen)
  ```
  Full worked example with hand-checkable arithmetic in
  [scripts/demo_autodp_cbe_leak.py](../scripts/demo_autodp_cbe_leak.py).

Note the second row of that table, which is a quiet advantage independent of the leak: because they
fit on the union, **a category present only in test is never unknown to them.** Our one-hot encoder
maps it to all zeros and loses the information, honestly. Theirs gives it a real code. On
high-cardinality categoricals with a small test split, that alone can move a score.

### 2.3 Scaling / normalization

| | AutoDP | ACORec |
|---|---|---|
| operators | `ZS` z-score, `MM` min-max, `DS` quantile (`n_quantiles=10`) | `standard` `minmax` `robust` `maxabs` |
| implementation | `ZS` hand-written; `MM`/`DS` sklearn, re-instantiated per split | sklearn scalers, fitted once |
| fitted on | each split separately | train only |
| constant column | set to **`1`** (`normalizer.py:29-30`) | `StandardScaler` → `0` |

`DS` (quantile transform) has no counterpart on our side; `robust` and `maxabs` have none on theirs.

The per-split fitting is worth seeing concretely, because it does more than introduce a small bias —
it **erases distribution shift**. Train `[0,1,2,3]` (mean 1.5) against a test split of
`[100,101,102,103]` (mean 101.5), a shift a deployed model would certainly suffer from:

```
AutoDP  train -> [-1.162 -0.387  0.387  1.162]
AutoDP  test  -> [-1.162 -0.387  0.387  1.162]     identical
train-fitted  -> [79.1   79.9   80.7   81.5]      the shift survives, as it must
```

Their test split is standardised into the same coordinate system as train no matter how far it has
drifted. This is not label leakage — no target is touched — but it is a **free covariate-shift
correction that no deployed model receives**, and unlike the `CBE` leak it plausibly *helps* their
reported numbers on shifted splits.

### 2.4 Feature selection

| | AutoDP | ACORec |
|---|---|---|
| operators | `MR` missing-ratio, `LC` collinearity, `WR` chi²-KBest, `TB` ExtraTrees | `variance_threshold` `k_best` (f_classif) `mutual_info` |
| supervised? | `MR`/`LC` no, `WR`/`TB` yes | `variance_threshold` no, other two yes |
| k | hardcoded `10` (`WR`) | `min(20, n_features)` |
| boundary | right in direction — fit on train, `df_test[to_keep]` | correct |

This is the one family where their train/test *direction* is right, and the credit belongs to them.
It is not clean, though:

**It desynchronises train from its own labels.** `transform()` does `dn = dn.dropna()`
(`feature_selector.py:116`) and writes the shortened frame to `fsd['train']`, but only a local copy
`dt` is realigned via `.loc[dn.index]` — `dataset['target']` is never written back. Measured on a
60-row frame with 5 NaN rows and `WR`:

```
before: train=60 rows, target=60 rows
after : train=55 rows, target=60 rows      <- out of sync
```

Downstream code survives only because their classifiers index with `dataset['target'].loc[X.index]`.
Any consumer that pairs them by *position* — ours does, when re-attaching the original `y` — would
silently mislabel every row.

Three further implementation notes:

- **`WR` silently deletes every categorical column.** `FS_WR_identify_best_subset` and
  `FS_Tree_based` both start with `df_train.select_dtypes(['number'])` and return only those
  columns; `transform()` then propagates that column list to test. Selecting `WR` or `TB` therefore
  drops all non-numeric features regardless of their usefulness.
- **`WR`'s non-negativity guard is buggy.** chi² requires non-negative inputs, so they strip columns
  containing negatives:
  ```python
  for i in range(0, len(lsv) - 1):      # skips the last column
      if lsv[i] > 0:
          del lis[i]                    # deletes while indexing -> positions shift
  ```
  Both defects push it toward dropping the wrong columns. Our `k_best` uses `f_classif`, which has
  no sign constraint and needs no such guard.
- **`TB` is nondeterministic** — `ExtraTreesClassifier(n_estimators=10)` with no `random_state`. Ours
  is deterministic except `mutual_info`, which is also stochastic; that one is a fair criticism of
  our space too.

### 2.5 Outlier removal

| | AutoDP | ACORec |
|---|---|---|
| operators | `ZSB` modified z-score, `IQR`, `LOF` | `iqr` `zscore` `lof` `isolation_forest` |
| threshold | `ZSB` at **1.6** modified-z; `IQR` at 1.5 | `zscore` at 3.0; `iqr` at 1.5 |
| applied to | **train and test** | train only (`transform` is a deliberate no-op) |
| on removal | rows deleted from both splits | rows deleted from train only |

(`preprocessing/outliers.py` also implements an `outlier_cleaning` step that winsorises and
re-imputes rather than deleting. It is **not** in `config.OPERATORS` and is never searched, so it is
not part of this comparison.)

Two measured behaviours:

**`ZSB` deletes ~8% of clean noise.** On 200 rows of pure `N(0,1)` with zero planted outliers, it
deleted 16. The threshold is `1.6` on a modified z-score (`outlier_detector.py:52`) where the
conventional cutoff is 3.5. Their MAD constant is `1.4296`, an apparent typo for the standard
`1.4826`.

**`LOF` deletes a fixed row count, not a fixed fraction.** `k = int(threshold * 100)`, so with their
default `threshold=0.3` it removes exactly 30 rows — on a 100-row dataset that is 30% of the data;
on a 10,000-row dataset, 0.3%. Measured on 100 rows with 5 planted outliers:

```
rows in 100 -> kept 70          (exactly 30 deleted, as k dictates)
of 5 planted OUTLIERS, kept 2   it caught 3
of 95 planted INLIERS,  kept 68 it destroyed 27 good rows to do it
```

It does rank outliers correctly — I initially suspected the comparison was sign-inverted and the
test disproved that. The defect is the fixed count: having decided to delete 30 rows it deletes 30,
and once the real outliers run out it takes inliers.

**On a split of fewer than ~30 rows it deletes everything.** `np.argsort(scores)[-30:]` then returns
every index, so the cutoff becomes the minimum score and `scores < min` is all-False. Measured:

```
test split of 25 rows -> kept 0
test split of 40 rows -> kept 10
```

Not triggered by our census — the two datasets that chose `LOF` are 184 (28,056 rows, test ≈ 5,611)
and 31 (1,000 rows, test = 200) — but the harness should record an empty prepared frame as a failure
rather than scoring it.

**`IQR` can also reorder rows.** `to_keep = set(X.index) - set(to_drop)` followed by
`X.loc[list(to_keep)]` with no `sort_index()` (`outlier_detector.py:33-38`); `ZSB` uses an
order-preserving list comprehension and is safe. No dataset in this census selected `IQR`, so it
never fired — but combined with positional `y` re-attachment it would misalign labels, so guard it
before any run where `IQR` can be chosen.

Ours uses `n_neighbors=20` and sklearn's own contamination logic, and — decisively — **never touches
the test split**.

### 2.6 The two unmatched families

- **AutoDP has duplicate removal, we don't.** `ED` (exact) and `AD` (approximate string matching via
  `py-stringsimjoin`). A real capability we lack. Note that our `--exclude-steps` equalisation was
  described as removing this asymmetry and does so only at the *family* level.
- **We have dimensionality reduction, they don't.** `pca`, `svd`, `n_components=min(10, ...)`.

Any "same operator space" claim has to state how these two were handled. They cannot be aligned;
one side simply lacks each.

### 2.7 The model is inside their pipeline

Their search selects the classifier (`NB` / `LDA` / `RF`) as element 0 of the pipeline and scores
candidates with it. Ours never searches the model. This means their preprocessing is optimised *for
Naive Bayes or LDA*, and aggressive scaling and feature pruning that suits those models is often
worthless to the gradient-boosting ensemble AutoGluon actually fits. It is a genuine confound in any
comparison where AutoGluon does the final scoring, and it cuts against them through no fault of
their operator implementations.

---

## 3. Summary table

| | AutoDP | ACORec |
|---|---|---|
| operators searched | 24 across 6 families (+3 classifiers) | 19 across 6 families |
| fit/transform boundary | **absent** | enforced (`transform` raises if unfitted) |
| test split may be modified | yes — rewritten, and rows deleted | no |
| test labels reachable | yes, via `CBE` | no |
| model in the pipeline | yes | no |
| step order | fixed per run by their meta-learner | searched, under precedence constraints |
| unique to them | `MICE`, `EM`, `DS` quantile, target encoding, duplicate removal | `robust`/`maxabs`, `isolation_forest`, PCA/SVD, outlier *cleaning* |

---

## 4. What is fair to claim

**Fair.** Their operator space is broader than ours in imputation and encoding, and narrower in
scaling and dimensionality reduction. The decisive difference is not the inventory, it is that their
operators have no fit/transform boundary: they rewrite the test split using the test split's own
statistics, delete test rows, and in one case consume test labels. An AutoDP pipeline is a
table-rewriting procedure, not a transformation deployable to unseen data.

**Also fair, and it cuts against us.** Several of their conventions plausibly *help* their scores —
per-split normalisation removes covariate shift for free, and union-fitted encoders never face an
unseen category. Our honest handling of both costs us accuracy. If we win under our protocol, we win
despite giving them these advantages, not because we removed them.

**Not fair.** Do not claim their published results are inflated by leakage. The `CBE` demo shows the
mechanism exactly but no measurable accuracy gain (paired difference `-0.013 ± 0.063` over 20 seeds),
`CBE` was selected on 1 of the 8 datasets where AutoDP preprocessed at all, and their internal "test"
split is a search-time validation scratchpad rather than a held-out set. The defensible statement is
about **protocol validity** — scores computed this way do not measure generalisation, in either
direction — not about the size or sign of an effect.

---

## 5. "If they use our operators, must their model be retrained?"

Yes in principle — and the answer turns out to matter less than it should, for reasons that are
worth stating carefully because they cut against them.

### 5.1 The objection is legitimate

AutoDP has two learned components, both trained over **their** operator vocabulary:

| component | what it is | trained on |
|---|---|---|
| `get_CLA_meta_task_order` | picks *which operator families* to search and in what order | `datasets/dataset/label.csv` — 2,000 curated (dataset, pipeline, score) rows harvested from Kaggle notebooks, pipelines written in their codes |
| `Estimate_after_profit.get_Estimate` | the MCTS rollout value estimate | `Estimation_Model/model_CLA.pickle`, an `MLPRegressor(n_features_in_=14)` |

`scripts/autodp_our_space.py` does **not** retrain either. It aliases each ACORec operator onto
their nearest class id (`ALIAS`), which is defensible for near-equivalents (`mean`→`MEAN`) and
indefensible for `pca`/`svd`, which have no counterpart at all and are aliased to `TB`. It also
remaps their family 5 (duplicate removal) onto our dimensionality reduction — so when their
meta-learner says "duplicate removal pays off on this dataset," we hear "do PCA." That is not a
translation, and the arm must disclose it.

The obligation splits unevenly across the two components, and §5.2 is why:

- **The family-ordering meta-learner does work**, is deterministic, and is genuinely trained on their
  operator vocabulary. A retrain here would be meaningful — and the family-5 remap
  (duplicate removal → dimensionality reduction) is exactly the kind of nonsense it would fix. This
  is a real, unpaid debt in arm 2.
- **The value model's retrain is moot**, because its input featuriser is randomised per call
  (§5.2). Retraining it would change nothing that matters.

No training script ships with the package — only the two pickles, `label.csv`, and
`Metafeature.csv` — so either retrain means reconstructing their training procedure from the paper.

### 5.2 But their value model is not doing what it appears to do

Before costing a retrain, check what would be recovered. `get_Estimate` builds its 14 features by
passing the meta-features through a `MultiHeadAttention` module — **constructed fresh on every
call**, with default `nn.Linear` initialisation:

```python
attention = MultiHeadAttention(metalen, choices.size(1), 7, 1)   # Estimate_after_profit.py:60
out1, scores1 = attention(matrix, choices)
result = rfc1.predict(df1)                                        # the pickled MLP
```

There is no `load_state_dict`, no `torch.load`, and no `manual_seed` anywhere in the package
(`grep` over `Pipeline_Generation/` returns nothing). The featuriser feeding the pretrained model is
re-initialised randomly on every call.

**It is not pure noise, and it is important not to overstate this** — half of the 14 features are
`cat(out1, keys)`, and `keys` is a projection of the pipeline itself, so a real dependence on the
pipeline survives. Both quantities were measured on the bundled `42493.csv`:

| | spread |
|---|---|
| seed fixed, **pipeline** varied across 6 candidates | 0.686 |
| pipeline fixed, **seed** varied across 12 draws | **1.019** |

So the pipeline signal is present but **dominated by initialisation noise, at about 1.5×**. The
estimates are also unbounded — draws of `1.68` and `1.52` appear, on what is nominally an accuracy.

The decisive ratio is against the quantity it is *added to*. Node selection uses:

```python
def get_profit_value(self):
    return self.pre_profit + self.after_profit      # MCTS.py:210
```

`pre_profit` is their real measurement (one NB/LDA/RF fit). Across those same 6 candidate pipelines
it spans `0.5625 – 0.6045`, a spread of **0.042**. The random term added on top of it has a spread
of **1.019** — roughly **24× larger than the signal it is meant to rank**. `default_policy` returns
that value as the MCTS reward and `backup` propagates it as a running maximum, so it reaches every
ancestor.

**Consequence for the arm.** Retraining `model_CLA.pickle` on our operator space would not repair
this, because the corruption is upstream of the model. Any honest arm-2 result must report that in
the released artifact the value estimate is dominated by initialisation noise, and that aliased and
retrained variants inherit that equally. Fixing it would mean changing their method, which puts it
outside a baseline comparison.

### 5.3 A third defect in the same selection path

`best_child`'s exploration term is:

```python
right = 15 * math.floor(math.log(0.001, 10)) * math.sqrt(temp)     # MCTS.py:305
```

`floor(log10(0.001))` is `-3`, so the coefficient is **−45**. Since
`temp = log(parent_visits + 1) / (child_visits + 1)`, `sqrt(temp)` is *larger* for less-visited
children — multiplied by −45 they receive a larger penalty. The exploration bonus is inverted.

Magnitude matters as much as sign: `sqrt(temp)` is order 1, so the penalty is order **45** against a
`profit` of order 1. This is not a mild exploitation bias — the first child to accumulate visits wins
essentially unconditionally, and the tree cannot recover. It independently explains the depth-≤1
trees measured in the earlier census.

### 5.4 What this means for the comparison

The meta-learner's *family ordering* (`get_CLA_meta_task_order`, driven by `label.csv` and
meta-feature similarity) is deterministic and is a genuine contribution — it is the part worth
preserving, and the adapter does preserve it. The *value estimate* is not functioning as a learned
component in the released version.

Say this descriptively, not as an accusation, and keep it separate from the accuracy claim: it
explains why AutoDP's search behaves erratically, and it is why "we should retrain their model on
our operators" — a correct instinct — would not have produced a materially fairer arm.

---

## 6. Can their meta-learning be reimplemented over our operator space? Yes.

The instinct that a model trained on *their* operators has no business scoring *our* operators is
correct, and `ALIAS` is not a defence. This section is the feasibility answer: a faithful retrain is
possible for both components, requires **no new expensive compute**, and would be *more* faithful to
their paper than the shipped artifact. It is worth doing.

### 6.1 Component A — family ordering. No training required at all.

`get_CLA_meta_task_order` (MCTS.py:75-120) has no learned weights. It is 1-NN retrieval:

1. Compute their 7 meta-features for the query dataset and average over columns → a 7-vector.
   (`MetaFeature.py`: instances, distinct values, is-string, is-unique, missing rate, skew, kurtosis.)
2. Find the nearest of the 200 rows in `Metafeature.csv` under a std-normalised Euclidean distance.
3. Take that dataset's 10 rows in `label.csv`, pick the highest `EvaluationMetric`.
4. Read the family sequence off that pipeline.

To port it, we need one thing: **a reference table of (meta-features, best pipeline) in our operator
codes.** We already have it — `data/openml/training_performance_matrix_autogluon.csv` is 12 pipelines
× 901 datasets with 10,233 observed scores, and their 7 meta-features can be computed for those
datasets by calling their own `MetaFeature.getfeature`. `argmax` down each column gives the best
pipeline per dataset.

This is a faithful port, not an approximation, and it removes the worst wart in the current adapter —
the family-5 remap that currently makes their meta-learner say "duplicate removal" and us hear "PCA."
Our families would be *our* families throughout.

### 6.2 Component B — the value model. Reimplementable, with two disclosed decisions.

Everything needed is recoverable:

| ingredient | status |
|---|---|
| architecture | `MultiHeadAttention(7, len, 7, 1)` → 14 features |
| regressor | `MLPRegressor(hidden_layer_sizes=(64,16), activation='relu', solver='lbfgs', alpha=1e-4, max_iter=5000)`, read from the pickle |
| training-set format | `label.csv`: (dataset → meta-features, pipeline, achieved score) |
| our training set | the same performance matrix — **10,233 (dataset, pipeline, score) triples vs their 2,000** |

Two things are underdetermined by the artifact and must be decided and stated:

**(i) The attention must be seeded and persisted.** This is unavoidable, because the current
behaviour is untrainable, not merely noisy — you cannot fit a regressor to features drawn from a new
random projection each call. A random projection is a legitimate technique (random features / ELM)
*provided the same projection is used at fit and predict time*. Note the real reason they
re-instantiate: `W_key`'s `in_features` is `choices.size(1)`, the **pipeline length**, so a single
module cannot serve variable-length pipelines. The fix is a cache of one seeded module per length
(1…7), persisted alongside the model. Doing this is best described as *implementing the architecture
their paper describes*, and should be disclosed exactly that way.

**(ii) How to represent absent steps.** Every one of their 2,000 training pipelines is length-7 with
`*_null` placeholders — and `class_mapping` contains **no null codes**, so a verbatim row from their
own training file cannot pass through their own estimator:

```
mapped 'MR,MEDIAN,BE,ZS,dup_null,out_null,CART' -> [15.0, 5.0, 9.0, 12.0, nan, nan, 26.0]
get_Estimate(...)  ->  ValueError: Input X contains NaN
```

So their published training table is not consumable by their published inference path, and we must
choose: drop nulls (variable-length sequences) or add an explicit null class. Either is defensible;
the choice must be recorded because their paper does not determine it.

### 6.3 What this costs, and the honest weaknesses

**Cost is low** — this is the key point. Unlike arm 3, which needs a *new* performance matrix over
their operator space, arm 2's retraining reuses a matrix we already own. No AutoGluon re-runs. The
work is a meta-feature pass over the reference datasets plus fitting a small MLP.

Three weaknesses to disclose rather than hide:

- **Pipeline diversity.** Our matrix has only 12 distinct pipelines (repeated across 901 datasets);
  their `label.csv` has more varied ones. The retrained model sees 12 points in pipeline-space, so it
  will learn "which dataset" far better than "which operator." This is the strongest objection to the
  arm and it should be stated up front.
- **Score provenance.** Our matrix was built with AutoGluon `medium_quality` and has a silent
  `RandomForest` fallback on ~5.4% of cells; their scores are single-model accuracies scraped from
  Kaggle notebooks. The two targets are not the same quantity.
- **Leakage.** The 901 reference datasets overlap our 23 evaluation IDs (184 and 31 are known). Those
  must be held out of the retrained tables, exactly as planned for arm 2's reference library.

### 6.4 The recommendation

Do it — but understand what it buys. It removes a real and fair objection (aliased operator codes,
and a meta-learner whose family 5 means something else entirely), at low compute cost. It does **not**
rescue the value estimate from being dominated by noise in the *shipped* baseline, which is a
separate fact about the released artifact and must still be reported for the un-retrained runs.

If it is not done, the fallback is honest disclosure: report arm 2 as *"AutoDP's search procedure over
ACORec's operator space, with its meta-models applied out-of-domain via code aliasing"* and state
plainly that its meta-learning is therefore not operating as designed. That is a weaker but not
dishonest arm. The retrained version is the better paper.

---

## 7. Two decisions the re-run forces

**(a) `score_full` or `score_kept` — decide before running.** Every AutoDP result so far had
`test_coverage == 1.0`, but only because the searches returned empty pipelines. That is over.
`ZSB` and `LOF` delete test rows and were selected on 4 of the 8 datasets that preprocessed (ZSB on
1164 and 14; LOF on 184 and 31), and §2.5 measures ZSB deleting ~8% of even clean data. The two
scores will diverge and they answer different questions — `score_full` charges AutoDP for rows it
declined to predict, `score_kept` does not. Choosing after seeing both numbers is precisely what a
reviewer will challenge, so fix the choice now and record it here.

**(b) The `fair` arm is not yet leak-free.** We hand their operators our real `target_test`. Until
that is replaced with a dummy target, `CBE` can consume genuine held-out labels inside the arm whose
whole purpose is to exclude them.
