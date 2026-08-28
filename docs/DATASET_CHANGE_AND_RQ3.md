# Response to the supervisor's six items: what the dataset change costs, and how to fix RQ3

Two questions were asked directly: **(1)** does the code need to change for the new dataset
requirement, and **(2)** what metric would separate the RQ3 hyperparameter settings, whose accuracies
are nearly identical. Both answers are below, followed by what each of the six items actually
requires.

Every claim here was executed against the repo, not inferred. Reproduce with
`scripts/` + `/private/tmp/.../audit.py` (arithmetic audit, quoted inline).

---

## Answer to (1): yes — and the cascade is the point

Changing the evaluation set is not a table edit. `src/automl_aco/eval_ids.py` encodes the project's
leakage policy: *one frozen Siamese, trained on a reference library from which every evaluation ID is
removed.* So:

> **Grow `EVAL_IDS` → the reference library shrinks → the frozen Siamese was trained on datasets that
> are now test data → it must be retrained → every RQ1/RQ2/RQ3 number changes.**

There is no version of item 1 that does not require re-running the suite. Plan for that.

### 1.1 The leak that already exists (this is the headline)

The paper states, §4.2 line 528: *"The 30 evaluation datasets are excluded from this collection to
prevent information leakage."* **The code that produced the numbers does the opposite.**

Table 3's numbers come from `solrec-aco-our-valid-data.ipynb` (the Kaggle notebook), not from the
leak-free repo path `scripts/run_recommend.py`. In that notebook:

| check | result |
|---|---|
| eval IDs inside `train_dataset_ids` (the 912-dataset reference library) | **22 of 23** (only 1485 absent) |
| DiffPrep eval datasets inside `train_dataset_ids` | **7** — ada_prior, eeg, mozilla4, page-blocks, pbcseq, pol, shuttle |
| query-dataset exclusion at inference (`query_dataset_id`) | **absent** — the string does not occur in the notebook's recommender |

So when the notebook recommends a pipeline for e.g. dataset 248, it retrieves neighbours from a
library that contains 248 itself, and aggregates Eq. 9 heuristics over *248's own observed best
pipelines*. The nearest neighbour is the query. That is direct self-transfer, and it inflates exactly
the quantity RQ1 and RQ2 report.

The repo path (`run_recommend.py`) is clean — it calls `holdout_reference()` and passes
`query_dataset_id` at inference. **The fix is to regenerate every number through the repo path.** That
is required work regardless of the dataset change, so it should be folded into the same re-run.

**Measured size of the gap.** `docs/local_full_run/` holds 20 runs from the clean repo path. For the
13 datasets that also appear in Table 3, **not one value matches — and Table 3 is higher on 13 of
13:**

| | clean repo path | Table 3 (BETA) | gap |
|---|---|---|---|
| mean over the 13 | **0.7605** | **0.8032** | **+0.043** |
| direction | — | — | Table 3 higher 13/13, max +0.103 |

Per-dataset examples: fri-c1 0.600 → 0.700, kc1-binary 0.759 → 0.862, mfeat-morphological 0.748 →
0.813. A one-directional gap on every dataset is what a self-retrieval leak produces; it is not run
variance.

**Consequence for item 2: no row of Table 3 has verified provenance.** Rows 1–13 do *not* come from
the clean runs on disk, and the notebook state that produced rows 14–30 is not in the file (its
`diffprep_test_dataset_ids` is fully commented out). The supervisor's suspicion — "I'm not sure I
took the current version's results" — is correct, and it applies to the whole table. This makes the
conclusion firmer, not weaker: **regenerate everything under one protocol.** Expect the honest BETA
average to land near 0.76 on this subset, not 0.80, before the dataset change is even applied.

(Also worth noting for item 2: 3 of the 20 clean runs record `final_evaluation.method =
"autogluon_failed"` and fell back to the LogReg proxy — 1047, 378, 381. A proxy fallback on a subset
is a candidate explanation for the Tables 9–12 scale mismatch discussed in §2, and the re-run should
treat an AutoGluon failure as a hard error rather than silently scoring with the proxy.)

### 1.2 A second leak that the new datasets would introduce silently

`holdout_reference()` drops a column when `normalize_id(col) in EVAL_ID_SET`. `normalize_id` strips a
`.0` suffix (`"248.0"` → `"248"`) but **not** other decimals. The reference matrix contains exactly
six such columns:

```
D_1037.1  D_1471.1  D_1046.1  D_802.1  D_722.1  D_40685.1
ada_prior    eeg     mozilla4   pbcseq   pol      shuttle
```

All six are DiffPrep evaluation datasets, appended to the matrix a second time. `normalize_id` leaves
them untouched, so they survive the holdout — **and `assert_disjoint` uses the same function, so the
post-condition does not fire either.** Adding these IDs to `EVAL_IDS` would therefore *not* remove
them. The leak would be silent and the guard would report success.

Fix `normalize_id` to strip any `.<digits>` suffix, and add a duplicate-column check to
`holdout_reference`.

### 1.3 Eight of the new datasets get an all-zero metafeature vector

`extract_enhanced_metafeatures()` (`src/automl_aco/data/metafeatures.py:15`) is a **pure lookup** by
dataset id against `dataset_feats.csv`. On a miss it returns `{}`. Downstream,
`recommender.py:1767` does:

```python
new_mf_df = pd.DataFrame([new_mf]).reindex(columns=self.metafeatures_df.columns, fill_value=0)
```

`{}` → a row of **zeros**, which is then imputed, scaled, and used for retrieval. No exception, no
warning. The dataset is embedded at the origin and its "behaviourally similar" neighbours are
meaningless.

These DiffPrep datasets have no row in `dataset_feats.csv`:

**avila, google, house, jungle_chess, micro, uscensus, abalone, obesity**

The notebook works around this with `extract_69_metafeatures` (cell 9), which computes meta-features
from raw data, plus synthetic IDs 100000–100003. **That function must be ported into the repo path**,
and `extract_enhanced_metafeatures` must raise on a miss rather than return `{}`.

### 1.4 The rest of the change list

| # | change | file | status |
|---|---|---|---|
| 1 | replace `EVAL_IDS` with the 30-dataset set, keyed by name in `EVAL_DATASETS`; fix the docstring (said 24, held 23) | `eval_ids.py` | **done** |
| 2 | `normalize_id`: strip any `.<digits>` suffix, incl. behind a `D_` prefix (§1.2) | `eval_ids.py` | **done** |
| 3 | make `run_arms.OUR_DATASETS` import `EVAL_IDS` instead of re-declaring it (it had drifted by 3 datasets) | `run_arms.py` | **done** |
| 4 | gate AutoDP-over-our-operators arms behind `--ack-autodp-not-retrained` | `run_arms.py` | **done** |
| 4b | delete the recommender's private `_normalize_id` copy, which had the same `.N` bug and is what `query_dataset_id` exclusion runs through — a duplicated dataset read as two, so the query could still retrieve itself | `metalearning/recommender.py` | **done** |
| 5 | port the metafeature computation; a lookup miss now computes (or raises) instead of returning `{}` (§1.3) | `data/metafeatures.py` | **done** |
| 6 | translate the 17 DiffPrep `<name>/data.csv` folders into the `<id>.csv` layout the runners expect | `scripts/export_diffprep_datasets.py` (new) | **done** |
| 7 | ACORec arms now run the deployed REF config; `--extra` no longer breaks ACORec arms | `run_arms.py` | **done** |
| 8 | add wall-clock timers to the search loop — see §2, and do it **before** the re-run | `run_recommend.py` | *open* |
| 9 | row cap removed: `--max-rows` now defaults to **no cap** and is recorded in the manifest | `export_eval_datasets.py` | **done** |
| 10 | local `<id>.csv` is now authoritative (`prefer_local`), not a fallback-on-exception | `data/loaders.py` | **done** |

### Item 10 — the loader was silently scoring the wrong table

`load_openml_dataset` consulted the OpenML API first and only fell back to the local CSV when the
API path *raised*. With internet on (Kaggle: `OpenML reachable: True`) the API therefore won for
every real OpenML id — including the 17 DiffPrep datasets, whose ids in several cases point at a
**different table of the same name**. Observed: `722 [native] 5000 rows x 48 features`, where
DiffPrep's `pol` has 15,000 rows. Ids the API could reach but not parse (44956, 46597, 40668, 41671,
100000) failed outright rather than reading the file sitting next to them.

`prefer_local=True` (the default) makes a supplied local CSV authoritative. Side effect: the two
long-standing `test_loaders` failures were caused by this same ordering and now pass, so
`tests/test_loaders.py` is 4/4.

### Item 9 — the 5000-row cap, and who it applied to

`export_eval_datasets.py` never declared the eval ids as "test", so the loader's
`max_samples_default=5000` truncated **every evaluation dataset**. 13 of the 30 were affected:

`40922` 88,588 · `40668` 67,557 · `41001` 44,819 · `1119` 32,561 · `42932` 20,867 · `41671` 20,000 ·
`1046` 15,545 · `722` 15,000 · `1471` 14,980 · `100000` 9,367 · `378` 8,844 · `30` 5,473 ·
`1497` 5,456

The other 17 are ≤5,000 rows and were never touched.

> **Deferred, and it needs deciding before any table is final.** Every method's numbers on those 13
> datasets were produced on truncated data, not just AutoDP's — so a consistent table requires
> re-running all methods on them.
>
> One mitigating fact makes this smaller than it looks: **the notebook did not cap.**
> `load_kaggle_diffprep_dataset` uses `max_samples = 100000 if dataset_id in test_dataset_ids else
> 5000` (cell 21), so Table 3 rows 14–30 were computed on full-size data while every repo-path run
> was capped at 5,000. Removing the cap therefore moves the repo path *toward* both the notebook and
> Table 2 rather than away from them — it is a divergence being closed, not a new one opened. It is
> also one more respect in which the two halves of Table 3 were never comparable (§1.1).

### Item 5 — three scale bugs inherited from the notebook

The notebook's `extract_69_metafeatures` produced values on a **different scale from the reference
table it would be compared against**, which would have misplaced the 8 computed datasets in
retrieval space regardless of how good the Siamese was:

| quantity | notebook | reference table | fix |
|---|---|---|---|
| `*ErrRate` | stored **accuracy** (`1 - err` where `err = 1 - acc`) | true error rate (DecisionStumpErrRate mean 0.321) | store the error rate |
| `ClassEntropy` | nats (scipy default) | bits — dataset 27: 0.659 nats vs table's 0.950, and 0.659/ln2 = 0.951 | `base=2` |
| all 7 `Percentage*` | 0–1 fractions | 0–100 (all seven max ≈100, means 57.0 / 83.3) | ×100 |

Also: OpenML counts the **target among the features** (dataset 27 reads 23 features / 16 symbolic
where `X` alone has 22 / 15), so the counts and every percentage denominator now include it.

Validated against the table on 7 datasets that appear in both. After the fixes `NumberOfFeatures`
matches exactly on all 7, and `ClassEntropy`, `MajorityClassPercentage`, `DecisionStumpErrRate` and
`DecisionStumpAUC` agree to rounding (e.g. dataset 27: entropy 0.95/0.95, majority 63.0/63.0,
stump-err 0.185/0.185). The residual `NumberOfSymbolicFeatures` gap on those 7 is an artifact of our
*exported* eval CSVs being pre-label-encoded; the raw DiffPrep CSVs keep their string columns and
detect correctly (obesity → 9 symbolic, google → 4).

### Item 7 — the arms were running the RQ2.1 ablation, not ACORec

`acorec_cmd` passed no metric flag, and `run_recommend.py` defaults `train_metric_inline=False`, so
every ACORec arm **silently used cosine similarity on raw meta-features** — which is precisely the
RQ2.1 ablation. It also omitted `--require-autogluon`, so an AutoGluon failure fell back to the
LogReg proxy and still wrote a score (the `autogluon_failed` rows in `docs/local_full_run`). Both are
now in `ACOREC_REF_FLAGS`, along with the rest of `EXPERIMENTS.md`'s REF, and `--n-ants`/
`--n-iterations` default to REF's 4/3 rather than 10/10. `--acorec-config minimal` restores the old
behaviour and warns.

Separately, `--extra` is appended to **both** command builders, so an AutoDP flag string
(`--adp-python`, `--cap-seconds`) made `run_recommend.py` exit 2 on *every* dataset of an ACORec arm.
There are now `--acorec-extra` / `--adp-extra`, and any passthrough flag is validated against the
receiving tool's `--help` **before** the run starts.

Regression tests for 1–4b are in `tests/test_leakage_holdout.py` (**11/11 pass**), including one that
fails if `run_arms.OUR_DATASETS` ever drifts from `EVAL_IDS` again, and one asserting the `.N` rule
touches nothing that is not an `<int>.<int>` id.

Rest of the suite: every other file passes. Three pre-existing problems are **not** caused by these
changes — each was confirmed by re-running with the changes stashed:

| pre-existing issue | evidence |
|---|---|
| `test_loaders::test_openml_local_fallback_csv[_zip]` fail | fail identically on the unmodified tree (the loader picks up real repo CSVs ahead of the tmp fixture) |
| `test_optimizers` hangs after its 1st test | hangs identically on the unmodified tree, even at a 900 s timeout |
| `test_recommender_retrieval_local` fails **in isolation** | fails on the unmodified tree too; passes inside a full-suite run — a test-ordering dependency, and its error (`fake_autogluon() got an unexpected keyword argument 'select_on_val'`) is a stale test double |

**Effect of the `normalize_id` fix on the shipped matrix:** the holdout now drops **29 columns
(901 → 872) covering 24 evaluation datasets**, where the old code dropped 22 columns and left five
`.1` duplicates behind. The one surviving `.1` column is `D_40685.1` (shuttle), which is correct —
shuttle is not an evaluation dataset. The remaining 6 of the 30 (madelon, avila, house, jungle_chess,
run_or_walk, google) are simply absent from the matrix.

Item 6 matters for honesty as much as for runtime: capping a 88k-row dataset to 5,000 is now a
**disclosed methodological choice** affecting a headline table, not an incidental default. The paper
must state it.

### 1.5 Which 10 datasets were dropped — please confirm

Table 2 is 13 + 17: our 13, plus DiffPrep's 18 minus `shuttle`. Matching Table 2's `#S/#F/#C` against
the local CSVs gives the mapping; the **10 dropped** from our original 23 are:

| dropped ID | why it plausibly looked "abnormal" |
|---|---|
| 1164 | 10,935 features on 185 rows |
| 1387 | 24 classes |
| 184 | 18 classes on 6 features |
| two of {378, 381, 382} | near-duplicate ipums variants — see caveat below |
| 993 | same 60-feature family |
| 248, 2, 29, 31 | — |

**Open point:** 378, 381 and 382 all have shape 5000 × 60 with 7–8 classes after capping, so Table 2's
`#S/#F/#C` cannot say which one is "ipums-la-99". No file in the repo carries dataset *names* —
`dataset_feats.csv` is keyed by numeric id only. Please confirm which of the three was kept; the other
two are dropped.

Two Table 2 rows to check while confirming: **usp05** is listed as 203 rows / 11 classes but the
exported CSV holds 191 rows / 4 classes (rare-class filtering), and **google** carries a row identical
to connect-4's (67,557 / 43 / 3) although its target is binary (`Rating>4.2`) — it looks copy-pasted.

---

## Answer to (2): RQ3 — the tables are not measuring what RQ1 measures

Before choosing a better metric, there is a blocking problem: **Tables 9–12 were produced by a
different scoring path than Table 3.** For the same datasets and the default configuration:

| dataset | Table 3 (BETA) | Table 9 (K=5) | Table 11/12 |
|---|---|---|---|
| madelon | 0.887 | 0.989 | ~0.61 |
| ipums-la-99 | 0.814 | 0.978 | 0.79–0.82 |
| mc2 | 0.813 | 0.598 | — |
| fri-c1 | 0.700 | 0.425 | — |

The default configuration should reproduce Table 3's column in *one* of K=1/3/5. It matches none.
Tables 3–8 agree with each other perfectly (BETA's 13 rows are byte-identical across Tables 3, 4, 5,
6-ACO and 8), but Tables 9–10, Tables 11–12, and Tables 3–8 form **three mutually inconsistent
groups** — most likely the LogReg proxy versus AutoGluon, and possibly different splits.

So RQ3 is not "add a column." It is **re-run RQ3 under the RQ1 protocol, with cost instrumentation.**

### 2.1 The two metrics to lead with

`aco_history` in every `recommendation.json` already records what is needed — no new experiment type,
just reporting:

```json
{"iteration": 1, "global_best_score": 0.886, "sampled_unique_count": 4, "cache_size": 4,
 "step_entropy": {...}, "mean_entropy": 0.241, "pheromone_min": 0.8, "pheromone_max": 2.80,
 "pheromone_saturation": 0.0}
```

**(a) Exploration dynamics — lead with this.** `mean_entropy` and per-step `step_entropy` over
iterations (exploration versus premature collapse), plus `pheromone_saturation` and
`pheromone_min`/`max` against the MMAS bounds. These are per-iteration *distributions*, so they stay
informative at any budget, and they show a mechanism where accuracy shows nothing: α, β and ρ visibly
change *how* the search moves through the space even when it lands in the same place. This is what
"space" most likely meant in the request.

**(b) Cost to reach the accuracy.** Unique evaluations until `global_best_score` last improved, plus
wall-clock. The finding to state is *K buys cost, not accuracy* when K=1 and K=5 both land on 0.798 —
a result, not a null result.

> **Resolution warning — settle before the re-run.** REF is `--n-ants 4 --n-iterations 3`, and the
> K/H sweeps inherit it. "Iterations to plateau" therefore has only **three** possible values, and
> `sampled_unique_count` tops out at 12 against a 3,600-pipeline space (0.3% coverage). Both are
> near-degenerate at REF budget. Either **raise the budget for the RQ3 sweeps specifically** (and say
> in the paper that RQ3 uses a larger budget so the cost axis is measurable), or report cost as
> *unique evaluations to last improvement* — 12 levels instead of 3, usable as-is. Treat raw coverage
> counts as secondary to the entropy signals either way.

Add wall-clock timers now — the re-run is happening anyway, so instrumenting first is free and
instrumenting later costs a second full re-run.

### 2.2 Free improvement to the accuracy tables themselves

Replace mean accuracy with **mean rank + win/tie/loss counts** across datasets. Pure post-processing,
no re-run, and it separates settings whose means agree to three decimals — 0.798 vs 0.798 vs 0.798 in
Table 10 hides a real per-dataset ordering (H=3 wins robot-failures 0.887 vs 0.842).

---

## Arithmetic audit of the current PDF (item 2, the part checkable now)

All 12 tables recomputed. **The column averages are correct everywhere except Table 4:**

| table | column | reported | computed |
|---|---|---|---|
| Table 4 | Meta-feature Similarity | 0.770 | **0.7725** |
| Table 4 | Behavioral Similarity | 0.800 | **0.8032** |

Table 4's Behavioral column is byte-identical to Table 5's "w/ Transfer" column, which correctly
reports 0.803. Table 4's averages are simply wrong. Everything else (Tables 3, 5, 6, 7, 8, 9, 10, 11,
12) reconciles to ±0.0005, including the columns with missing entries.

### Textual inconsistencies to fix in the same pass

| location | problem |
|---|---|
| Abstract & line 86 | "20 real-world datasets" vs §4.1 "30" vs §4.2 line 480 "**all 35** evaluation datasets" |
| §4.2 line 527 | "knowledge base of **900** datasets" — the matrix has 901 columns, 6 of them duplicates (§1.2) |
| §4.2 line 528 | "The 30 evaluation datasets are excluded … to prevent information leakage" — **false for the code that produced Table 3** (§1.1) |
| §2.1 line 172 | search space "**4,320** candidate pipelines". Table 1 lists 6 outlier operators including `mad`; `config.OPERATORS` has **5** (no `mad`) → repo path searches **3,600**. The notebook is smaller still: its `OPTIONS_CURRENT` gives encoding a single option (`onehot`) → **1,800**, and that is the space actually searched for Table 3 rows 14–30. Three different numbers for one claim. Either add `mad` and a second encoding option, or correct the text — the coverage denominator in §2.1(b) depends on which |
| §5.1 line 602 | "with TPOT and AUTOGLUON … respectively" — no TPOT table exists (this is item 4) |
| §4.2 line 594 vs Table 12 | *M* is defined as "elite ants contributing to the pheromone update" in the text but "number of ants" in the caption; Eq. 10–11 use *M* for the ant count |

---

## Appendix A — Table 2 name → dataset ID

Rows 1–13 mapped by matching Table 2's `#S/#F/#C` against the exported CSVs in `data/eval_datasets/`.
Rows 14–30 read from the notebook's own `DATASET_NAME_TO_ID` (cell 22) — authoritative for what the
notebook loaded, but not independently verified against the OpenML API.

| # | Table 2 name | ID | source | match |
|---|---|---|---|---|
| 1 | kc1-binary | 1066 | OpenML | exact (145/94/2) |
| 2 | usp05 | 1047 | OpenML | **partial** — Table 2 says 203 rows/11 classes, CSV has 191/4 (rare-class filtering) |
| 3 | sleuth-ex2016 | 862 | OpenML | exact (87/10/2) |
| 4 | calendarDOW | 40663 | OpenML | exact (399/32/5) |
| 5 | mc2 | 1054 | OpenML | exact (161/39/2) |
| 6 | fri-c1 | 876 | OpenML | exact (100/50/2) |
| 7 | mfeat-morphological | 18 | OpenML | exact (2000/6/10) |
| 8 | robot-failures-lp5 | 1520 | OpenML | exact (164/90/5) |
| 9 | autoUniv-au4 | 1548 | OpenML | exact (2500/100/3) |
| 10 | ipums-la-99 | 378 | OpenML | **confirmed by the supervisor** — 381 and 382 are therefore dropped |
| 11 | madelon | 1485 | OpenML | exact (2600/500/2) |
| 12 | mfeat-fourier | 14 | OpenML | exact (2000/76/10) |
| 13 | colic | 27 | OpenML | exact (368/22/2) |
| 14 | abalone | 44956 | DiffPrep CSV | not in `dataset_feats.csv` |
| 15 | ada_prior | 1037 | DiffPrep CSV | duplicate column `D_1037.1` (§1.2) |
| 16 | avila | 42932 | DiffPrep CSV | not in `dataset_feats.csv` |
| 17 | connect-4 | 40668 | DiffPrep CSV | 67,557 rows — breaks the 5,000-row cap |
| 18 | eeg | 1471 | DiffPrep CSV | duplicate column `D_1471.1` (§1.2) |
| 19 | google | **100000** (synthetic) | DiffPrep CSV | **no OpenML entry** — see below; Table 2 row duplicates connect-4's |
| 20 | house | 42165 | DiffPrep CSV | not in `dataset_feats.csv` |
| 21 | jungle_chess | 41001 | DiffPrep CSV | not in `dataset_feats.csv` |
| 22 | micro | 41671 | DiffPrep CSV | `microaggregation2`; **in the reference library — must be held out** |
| 23 | mozilla4 | 1046 | DiffPrep CSV | duplicate column `D_1046.1` (§1.2) |
| 24 | obesity | 46597 | DiffPrep CSV | not in `dataset_feats.csv` |
| 25 | page-blocks | 30 | DiffPrep CSV | in the reference library |
| 26 | pbcseq | 802 | DiffPrep CSV | duplicate column `D_802.1` (§1.2) |
| 27 | pol | 722 | DiffPrep CSV | duplicate column `D_722.1` (§1.2) |
| 28 | run_or_walk | 40922 | DiffPrep CSV | 88,588 rows — breaks the 5,000-row cap |
| 29 | uscensus | **1119** | DiffPrep CSV | **in the reference library — must be held out**; 32,561 rows breaks the cap |
| 30 | wall-robot-nav | 1497 | DiffPrep CSV | in the reference library |

`shuttle` (40685) appears in the notebook's mapping and carries a duplicate column `D_40685.1`, but is
**not** in Table 2 — it is the DiffPrep dataset that was dropped. Its duplicate column correctly
stays in the reference library.

### The three missing IDs, resolved

All three were traced by reading the raw CSVs out of `github.com/chu-data-lab/DiffPrep` (`gh api
repos/chu-data-lab/DiffPrep/contents/data/<name>/data.csv`), which is the same source the Kaggle
`diffprep-dataset` input came from.

**uscensus = OpenML 1119 (`adult-census`). Confirmed.** Its header is
`Age, Workclass, Fnlwgt, Education, Education-num, Marital-status, Occupation, Relationship, Race,
Sex, Capital-gain, Capital-loss, Hours-per-week, Native-country, Income` and row 1 is
`39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174,
0, 40, United-States, <=50K` — the canonical first row of UCI Adult. OpenML 1119 is the Adult
**training split only**, 32,561 instances, which matches Table 2 exactly (the other Adult copies,
179 and 1590, hold all 48,842). **1119 is in the reference library and must now be held out.**

**micro = `microaggregation2`, OpenML 41671. Confirmed locally.** openml.org was unreachable from
this machine (504 on every endpoint), so instead the id was corroborated against
`dataset_feats.csv`, which already carries row 41671:

| quantity | `dataset_feats.csv` row 41671 | DiffPrep `microaggregation2/data.csv` |
|---|---|---|
| instances | 20,000 | 20,000 (Table 2) |
| features | 21 | `a1..a20 + class` = 21 |
| numeric / symbolic | 20 / 1 | 20 numeric + 1 class |
| classes | 5 | 5 (Table 2) |
| missing values | 0 | 0 |

Four independent quantities agree and nothing else in the reference library has that signature.
Controls behave as expected (1119 → 32,561×16, 2 classes, 9 symbolic = Adult; 722 → 15,000×49 = pol).
**41671 is in the reference library and must now be held out.**

**google has no OpenML entry.** Its header is
`Category, Reviews, Size, Type, Price, Content Rating, Genres, Install, Rating>4.2` — the Kaggle
Google-Play-Store apps table with a *derived* binary target. DiffPrep's claim that all 18 datasets
come from OpenML is loose. It keeps the synthetic id **100000** (the notebook's convention), frozen in
`DIFFPREP_SYNTHETIC_IDS`. Being synthetic it cannot collide with the reference library, so no holdout
is needed — but it does need computed meta-features (§1.3).

Neither 1119 nor 41671 was in the old eval set, so **both sat inside the reference library for every
run to date** — a leak channel independent of the notebook leak in §1.1, and one that also applies to
the clean repo-path runs. Any earlier `uscensus` or `micro` result is contaminated.

**Blast radius of the `normalize_id` fix**, measured over the shipped tables: it changes exactly six
performance-matrix columns (`D_1037.1 D_1046.1 D_1471.1 D_40685.1 D_722.1 D_802.1`) and nothing else
— the metafeature index is unaffected (2,560 keys, no collapses), and no key that is not
`<int>.<int>` changes. Five of the six are now held out; the sixth, shuttle, is now correctly
de-duplicated in the neighbour pool rather than counted twice.

---

## Appendix B — is the arm-0 AutoDP run still valid?

**Yes, the numbers are sound; no, the run is not complete for the new table.**

**What arm 0 actually is.** `run_arms.py:71` defines it as `data="ours", ops="theirs"` and
`run_arms.py:141` passes `--operator-space theirs`. So it is **our datasets, their operator space,
their search** — AutoDP exactly as published. That is the right configuration for a Table 3 baseline
column, but note what it is *not*: it is **not** an operator-space-controlled comparison. The arms
that isolate the operator space (`1-adp-ourops`, `1-aco-theirops`) are unrun, and they are what RQ2.4
("Contribution of Operator Space") needs.

**Why they survive.** Arm 0 is AutoDP driven by *its own* meta-learner (`label.csv` / `Metafeature.csv`,
200 Kaggle-scraped datasets). It never touches our reference library or our Siamese, so the leak in
§1.1 does not reach it. It was scored through `adp_bench.py` with AutoGluon on our splits, and the
printed `score` column is **`score_full`** (`adp_bench.py:233` — "deleted test rows count as wrong").
That settles decision (a) in `OPERATOR_SPACE_COMPARISON.md` §7: the strict scoring was used.

**Why the run is incomplete.** Three separate gaps:

1. **It ran the wrong 23.** `run_arms.py:52` defines `OUR_DATASETS`, which differs from `eval_ids.py`'s
   `EVAL_IDS` by three substitutions: **1049, 1050, 1063 are in; 2, 382, 993 are out.** Two "canonical"
   dataset lists in one repo that disagree — worth reconciling regardless.
2. **1049, 1050, 1063 are not in the supervisor's new set** — those three runs are discarded, and must
   not be averaged in.
3. **Zero of the 17 DiffPrep datasets were run.** Arm 0 covers 12 of the new 30 (40%).

**Failures inside the covered set:** `862` errored after 7.7 s (a genuine hole — 862 is kept), and
`248` timed out at 3619 s (harmless — 248 is dropped anyway).

### The number you asked for, and the reason not to publish it yet

On the **same 12 datasets**, all clean, all AutoGluon-scored:

| method | mean | protocol |
|---|---|---|
| **AutoDP (arm 0, clean)** | **0.7756** | native/transductive, `score_full` |
| **ACORec (`docs/local_full_run`, clean)** | **0.7601** | leak-free repo path |
| BETA as printed in Table 3 | 0.8015 | **leaky — do not use** |

**Clean AutoDP currently outscores the clean ACORec evidence on disk by +0.016.** Only the leaky
column wins. This is the single most important thing the re-run has to resolve, and it should not be
discovered late.

Read it as *whole method vs whole method*, not as an operator-space result — the two sides search
different spaces. Two things are working for AutoDP in that 0.7756, both documented in
`OPERATOR_SPACE_COMPARISON.md` §2 and both live under the `native` protocol:

- **A partly richer operator space** — `MICE`, `EM`, target encoding and duplicate removal have no
  counterpart on our side (we have PCA/SVD and `robust`/`maxabs`, which they lack).
- **Transductive advantages.** Their operators have no fit/transform boundary: per-split normalisation
  silently erases covariate shift, and union-fitted encoders never meet an unseen category. Our
  honest handling costs us accuracy here.

So the comparison as it stands **gives AutoDP those advantages and it still only wins by 0.016.**
Re-running arm 0 under `--protocol leakfree` (adp_bench `fair` mode) would strip the transductive
ones and is the more informative second data point. Decide which of the two is the reported baseline
before the re-run, not after seeing both.

It is not yet a defeat. `docs/local_full_run` cannot be confirmed as the REF config — its JSONs do not
serialise `hybrid_select`, `cv_select_folds` or `autogluon_profile`, and 3 of 20 fell back to the
proxy. `EXPERIMENTS.md` records the CV-selection config as worth **+0.033** over baseline off-test;
0.760 + 0.033 = 0.793, which would clear AutoDP's 0.776. So the honest position is: **the margin is
real but unproven, and it is thinner than Table 3 implies.**

### Can the comparison table be filled now?

**Fill the AutoDP column — it is final for these 12. Do not put it next to the current BETA column,**
which is leak-inflated by ~0.043; that comparison flatters ACORec by exactly the amount that is not
real.

| # | dataset | ID | AutoDP arm-0 | note |
|---|---|---|---|---|
| 1 | kc1-binary | 1066 | 0.8621 | |
| 2 | usp05 | 1047 | 0.9444 | |
| 3 | sleuth-ex2016 | 862 | — | **error @ 7.7 s — must re-run** |
| 4 | calendarDOW | 40663 | 0.6203 | |
| 5 | mc2 | 1054 | 0.7188 | |
| 6 | fri-c1 | 876 | 0.6500 | |
| 7 | mfeat-morphological | 18 | 0.8050 | |
| 8 | robot-failures-lp5 | 1520 | 0.6250 | |
| 9 | autoUniv-au4 | 1548 | 0.6520 | |
| 10 | ipums-la-99 | 381 | 0.8170 | 378 also ran → 0.8030, pending §1.5 |
| 11 | madelon | 1485 | 0.8846 | |
| 12 | mfeat-fourier | 14 | 0.8650 | |
| 13 | colic | 27 | 0.8630 | |
| | **mean (n=12)** | | **0.7756** | |
| 14–30 | all 17 DiffPrep datasets | | — | **not run** |

### Arm-0 status against the new 30-dataset set

Now that `run_arms.OUR_DATASETS` tracks `EVAL_IDS`, arm 0 stands at **12 of 30 usable**:

- **12 usable and still in the set** — the table above minus 862.
- **9 runs discarded**: `29 31 184 381 1049 1050 1063 1164 1387`. Six were dropped by the supervisor;
  three (1049, 1050, 1063) were never in any declared eval set and came from the drifted list.
- **18 still to run**: `862` (re-run after its error) plus all 17 DiffPrep datasets.

`run_arms.py` keys resumption on `(arm, dataset, protocol, status=="ok")`, so pointing it at the same
JSONL will run exactly those 18 and skip the 12.

Also still open: decide whether the `native` protocol (AutoDP's search sees our test rows) is the
reported baseline — if so it must be disclosed in the paper, since it favours AutoDP.

---

## What each of the six items now costs

| item | status | blocking dependency |
|---|---|---|
| 1. DiffPrep + our datasets | **code changes required** — §1.4, six items | confirm the 10 dropped IDs (§1.5) |
| 2. Verify all numbers | arithmetic done above (one error, Table 4). Per-dataset values are **not reproducible from anything on disk**: 0 of 13 match the clean runs in `docs/local_full_run/`, which sit 0.043 lower on average, and the notebook state behind rows 14–30 is commented out (§1.1) | item 1 |
| 3. Sync to Overleaf | **do not start** — downstream of 1 and 2 | items 1, 2 |
| 4. AutoDP results + TPOT | AutoDP arms are analysed in `OPERATOR_SPACE_COMPARISON.md`; §7 there flags two unresolved decisions (`score_full` vs `score_kept`; the `fair` arm is not yet leak-free). TPOT is a new downstream framework — new harness path | item 1 |
| 5. RQ3 another view | **§2** — cost-to-accuracy + space-explored, plus timers added before the re-run | item 1 + timers |
| 6. Good/bad examples | best drawn from the same re-run: pick from per-dataset win/loss once one protocol produces all rows | items 1, 2 |

**Recommended order:** confirm the dataset list → land the six code changes → add timers → one
re-run of everything through `run_recommend.py` → then items 2, 3, 5, 6 all fall out of that single
consistent output set.

---

## Appendix C — Retraining AutoDP's meta-learner over our operator space

Supervisor item 4 requires AutoDP results on our operators. AutoDP has two learned components.
One can be retrained and now is; the other cannot be, for a reason worth reporting in its own right.

### C.1 The meta-learner — retrained

`get_CLA_meta_task_order` (`Pipeline_Generation/MCTS.py`) is a deterministic 1-NN: compute 7
metafeatures for the query dataset, find the nearest row of `Metafeature.csv`, take that
neighbour's best-scoring pipeline from `label.csv`, and read the operator-family order off it.
Both files ship describing AutoDP's own operators, so applying it to our space previously required
aliasing every operator onto their nearest class id — and `pca`/`svd`, which have no counterpart in
their space at all, both collapsed onto `TB` (tree-based feature selection).

`scripts/build_adp_meta_corpus.py` regenerates both CSVs over our 19 operators: sample library
datasets, sample pipelines over our six families in a shuffled order, score each with the LogReg
proxy, and write the corpus in their exact format. Their 7-slot pipeline shape is preserved (six
preprocessing families plus a model slot), so `n_features_in_` stays 14 — the arm varies the
vocabulary, not the architecture.

Verified: with a retrained corpus, `dimensionality_reduction` is returned as a first-class family.
Under aliasing that was unreachable.

**Scale (parity rebuild, 2026-08-28).** The first build under-delivered against its own
`--n-datasets 200` target and produced only **108** datasets (`data/adp_ourops_corpus_108`,
fingerprint `d5b76a950e749ead`). ACORec's Siamese metric is fit on the 901-column AutoGluon
performance matrix (879 after the eval holdout), so a 108-row 1-NN table put the two methods an
order of magnitude apart on reference-set size. `data/adp_ourops_corpus` was rebuilt over the
**same reference datasets ACORec uses**: the performance-matrix `D_<id>` columns, restricted to
classification and ≤1000 features and with the 30 `EVAL_IDS` + 10 `THEIR_DATASETS` removed
(`scripts/adp_parity_ref_ids.py` → 775 ids), 4 round-robin shards of
`build_adp_meta_corpus.py --ids … --shard i/4`. After attrition (regression targets, all-tie
"no signal", proxy failures, and 7 datasets whose AutoDP metafeatures came back NaN) the corpus is
**645 datasets × 10 pipelines**, fingerprint `80c470059a49543c` — 6× the old build.

The 645 are drawn from the **same 901-column reference library ACORec's Siamese trains on**, so this
closes the order-of-magnitude scale gap; it is not exact parity. The chain is 901 → 775 after
restricting to classification and ≤1000 features (both filters inherent to the LogReg proxy scorer —
it cannot fit continuous targets and its cost scales with column count) and removing the eval /
`THEIR_DATASETS` holdout → 645 after attrition. ACORec's 879 applies neither the classification nor
the width filter, so the two reference sets differ in composition (≈77 wide + 4 regression datasets
ACORec's metric sees and this corpus does not), not only in count.

**Two asymmetries the parity rebuild does *not* remove, by design:** the corpus is scored with the
**LogReg proxy** (ACORec's matrix is AutoGluon; ~6,500 AutoGluon fits was not affordable) and uses
**10 randomly-sampled shuffled-order pipelines** per dataset (ACORec's matrix is 12 fixed canonical
pipelines with full coverage). The proxy only has to rank pipelines well enough to read a family
order off the best one; disclose both alongside the arm-1 result.

Two correctness details:

- **Metafeatures are computed by AutoDP itself**, not by a port. A pure-python port was written and
  abandoned: pandas 1.x (their pin) returns True from `is_string_dtype` for *any* object dtype
  while 2.x does not, and `unique()`/`LabelEncoder` differ on NaN. The residual was small — max
  relative 4e-3 — but a nearest-neighbour lookup can flip on that silently. `adp_metafeatures.py`
  shells into `.venv-autodp`, the same two-environment split `adp_bench.py` already uses.
  `tests/test_adp_metafeature_port.py` diffs the two implementations.
- **Exactly k rows per dataset.** Their reader slices `df.iloc[k*minid : k*minid+k]`, which assumes
  a fixed block. Their own shipped `label.csv` violates this (group sizes 10/6/4/11/9), so their
  neighbour lookup misaligns for datasets after the first irregular one. Writing a fixed k makes
  the indexing correct here. Disclose as a side effect of retraining, not as an operator-space
  effect.

### C.2 The value estimator — not retrained, and why

`model_CLA.pickle` is deliberately left alone. `Estimate_after_profit.get_Estimate` constructs a
`MultiHeadAttention` with fresh random weights on **every call** and feeds its output to the MLP.
The weights are never loaded — there is no `torch.save`, `torch.load`, or `state_dict` anywhere in
the package.

Measured, one dataset, 4 distinct pipelines x 40 calls each:

| quantity | value |
|---|---|
| between-pipeline sd of means | 0.0235 |
| within-pipeline sd (re-instantiation noise) | 0.3114 |
| signal / noise | **0.076** |

Pipeline identity moves the output ~13x less than re-instantiating the module does. Refitting the
MLP would fit one random projection and be queried through a different one on every call; making it
coherent requires seeding and persisting the attention, which changes their architecture and
forfeits the arm's claim to be "their search".

### C.3 Why AutoDP still works, and what to claim

The estimator is not load-bearing. `get_profit` calls `mctsdata.getAcc` — a real accuracy
measurement — for every expanded node, and `drop_unpromising` prunes on those real `pre_profit`
values alone. The estimate enters only `best_child`'s score and `backup`. So noise there costs
search efficiency, not correctness: the returned pipeline was always genuinely evaluated.

What remains once the learned components are discounted is guided random search over preprocessing
pipelines with real evaluation and real pruning — a strong baseline that will beat no-preprocessing
comfortably. AutoDP's published gains are consistent with the search budget rather than with
transfer.

Two further signals consistent with the estimator not mattering: `best_child`'s exploration term is
`15 * floor(log10(0.001)) * sqrt(temp)` = **-45*sqrt(temp)**, a negative coefficient that penalises
under-visited nodes (inverted UCB); and `backup` propagates a running max of the *estimates*, so the
noise accumulates upward.

**Claim to make:** AutoDP's advantage comes from search budget and real evaluation, not from its
learned transfer. Supported by the 0.076 ratio, by pruning being estimator-independent, and
testably by the constant-estimator ablation (replace `get_Estimate` with a constant; if scores hold,
the learned components are decorative).

**Caveat to state:** the 0.076 figure is one dataset and four pipelines. The mechanism is structural
— random init per call, visible in the code — so it does not depend on the dataset or the operator
space, but the exact ratio would vary.

### C.4 Why AutoDP does not terminate on large datasets

AutoDP has two search modes, selected by whether a `runTime` budget is supplied.

**With a budget** (`CLA_With_TimeBudget`) the loop is `while True: if elapsed > time_budget: break`.
It always consumes the entire budget and never stops early, so a reported runtime is a
configuration choice, not a property of the method.

**Without one** (`CLA_Without_TimeBudget`) it stops after 20 *consecutive* iterations whose
improvement falls below `Mingap = 0.001`, with no time bound whatsoever. Two independent sources
of randomness keep that counter from reaching 20:

| source | measured effect |
|---|---|
| `RandomForestClassifier()` constructed with no `random_state` (`Search_Space/classifier.py:90`) | re-scoring the SAME node on the same data with the same seeded CV split varies by **0.008**, i.e. **8x** `Mingap` |
| `MultiHeadAttention` re-initialised per call inside `get_Estimate` | `best_child` descends a different path each iteration, so a *different* node is evaluated (signal/noise **0.076**, §C.2) |

The convergence threshold is thus an order of magnitude finer than the method's own evaluation
noise. On small data each iteration is milliseconds and the streak completes by luck; on
15,000+ rows each iteration costs real model fits and the unbounded mode has no reliable stopping
point. Observed: `378` (8,844 rows) and `722` (15,000 rows) were killed at a 3,600s cap and again
at the retry, ~7,200s each.

**Reporting:** always pass an explicit budget. Record datasets that exceed it as timeouts at a
stated cap rather than omitting them -- non-termination at this data scale is a property of the
method, not a harness failure.

### C.5 The cost of transferring their prior (arm 1 without retraining)

Running arm 1 without a retrained corpus reuses AutoDP's shipped `label.csv` for the task order
and reaches our operators by aliasing. The search space really is ours -- verified, dataset 184
selected `AR_encoding:onehot` and `AR_outlier_removal:zscore` -- but the *prior* was learned over
a different vocabulary, and one family does not survive the mapping:

- their family 5 is **deduplication** (`ED`/`AD`); our family 5 is **dimensionality reduction**
  (`pca`/`svd`)
- their own training pipelines skip deduplication in **97.8%** of cases (1955/2000 use `dup_null`)
- so the transferred prior selects our `dimensionality_reduction` family in only **~10.2%** of
  searches, while the other five families are selected normally

This is a *wrong* prior rather than a missing one, and it biases the arm **toward ACORec**, whose
ACO searches all six families on every run. Disclose it, or remove it by retraining the corpus
(`scripts/build_adp_meta_corpus.py`) or by passing `--adp-family-order all`, which supplies all six
families in ACORec's canonical order with no prior transferred. Note that the neutral order is
>3x slower per dataset, since the deeper tree costs more real evaluations per iteration.

### C.6 Reading rows with low `test_coverage`

AutoDP's outlier operators delete rows from the TEST split, so some rows have no prediction. Every
row records both accountings:

| field | meaning |
|---|---|
| `score` (`score_full`) | deleted test rows counted as wrong -- the comparable number |
| `score_kept` | accuracy over only the rows it agreed to predict |
| `test_coverage` | fraction of the test set surviving |

Example, dataset 184 (18 classes, majority-class baseline 0.1676):

```
score 0.1121 | score_kept 0.1630 | test_coverage 0.688
```

31% of the test set was deleted, and on the rows it did predict the pipeline matched the
majority-class baseline. Report `score_full` for comparison against methods that predict every
row, and state `test_coverage` wherever it is below 1.0.
