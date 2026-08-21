#!/usr/bin/env python3
"""Make AutoDP's MCTS search ACORec's operator space, without modifying its source.

Arm "their method, our operators" of the cross-comparison. Everything AutoDP contributes is
preserved -- the MCTS itself, its UCB/backup/prune logic, its pretrained profit estimator, its
internal NB/LDA/RF scoring signal and its convergence rule. Only the *set of operators the search
may choose from* is replaced, and each chosen operator is executed by ACORec's own implementation,
so a pipeline means the same thing in both systems.

Installed by monkeypatching module attributes at runtime; `autodatapre`'s files on disk are never
touched (verified byte-identical to the PyPI wheel).

## The three things that must be patched, and why

1. **The family lists.** `MCTS_DATA.list1..list6` and `MCTS.list1..list6` enumerate the operators of
   each family. The dispatch in `getAcc`/`getdataset` is a chain of `if op in listN`, so the lists
   are both the search space AND the routing table.

2. **`Estimate_after_profit.class_mapping`.** A hard-coded 31-entry dict of THEIR operator codes,
   consumed by their pretrained MLP (`model_CLA.pickle`) via `choices.map(class_mapping)`. None of
   ACORec's 19 operators appear in it, so without aliasing every choice becomes NaN and their value
   estimate -- which drives `backup` and therefore `best_child`'s UCB -- degrades to noise silently.
   Each ACORec operator is aliased to their nearest equivalent class id (see ALIAS below).
   `pca`/`svd` have NO counterpart: their space has no dimensionality-reduction family. They are
   aliased to the closest structural operator (`TB`, tree-based feature selection) purely so the
   estimator receives a valid index, and this is a genuine limitation of the arm -- their value
   model cannot represent those operators. It must be disclosed when reporting the result.

3. **`MCTS.get_CLA_meta_task_order`.** It reads their `label.csv` (best pipelines over their
   reference library, written in THEIR codes) and builds the search space by testing
   `if code in listN`. Once the lists hold our codes, every test fails and the space collapses to
   `[list7]` -- classifier only, no preprocessing at all, silently. The patch keeps their original
   codes for the membership test and substitutes our family lists as the content, so their
   meta-learner still chooses WHICH families to search and in WHAT order (its actual contribution),
   while the operators inside each family are ours.

## Deliberate semantic difference

AutoDP's operators transform train and test independently (e.g. `ZS` z-scores test with test's own
statistics) and delete rows from the test split. ACORec's operators fit on train and transform test,
and never drop test rows. This adapter uses ACORec's semantics, because the point of the arm is to
hold the operator space constant, and mixing in their leakier transform convention would confound
the comparison. Row drops on the TRAIN split are applied to the targets too, keeping their
`dataset['target'].loc[X.index]` lookups valid.

Usage:
    import autodp_our_space; autodp_our_space.install()
    # ... then call AutoDP's CLA_With_TimeBudget / CLA_Without_TimeBudget as usual
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

from automl_aco.preprocessing.preprocessor import Preprocessor  # noqa: E402

# ACORec's operator space, one list per family. Prefixed so a code can never collide with one of
# AutoDP's own (which the meta-learner still needs to recognise in its label.csv).
STEP_OPERATORS = {
    "imputation": ["mean", "median", "most_frequent", "constant", "knn"],
    "encoding": ["onehot"],
    "scaling": ["standard", "minmax", "robust", "maxabs"],
    "feature_selection": ["variance_threshold", "k_best", "mutual_info"],
    "outlier_removal": ["iqr", "zscore", "lof", "isolation_forest"],
    "dimensionality_reduction": ["pca", "svd"],
}
PREFIX = "AR_"


def _code(step: str, op: str) -> str:
    return f"{PREFIX}{step}:{op}"


CODE_TO_STEP_OP = {_code(s, o): (s, o) for s, ops in STEP_OPERATORS.items() for o in ops}

# ACORec operator -> nearest AutoDP class id, so their pretrained estimator gets a valid index.
# pca/svd have no counterpart in their space; TB is the closest structural operator.
ALIAS = {
    "mean": "MEAN", "median": "MEDIAN", "most_frequent": "MF", "constant": "MF", "knn": "KNN",
    "onehot": "OE",
    "standard": "ZS", "minmax": "MM", "robust": "DS", "maxabs": "MM",
    "variance_threshold": "MR", "k_best": "WR", "mutual_info": "TB",
    "iqr": "IQR", "zscore": "ZSB", "lof": "LOF", "isolation_forest": "LOF",
    "pca": "TB", "svd": "TB",
}

# Which of OUR families each of THEIR family lists maps onto, so their meta-learner's family
# ordering (read from label.csv in their codes) still selects a real family.
THEIR_FAMILY_TO_OUR_STEP = {
    1: "imputation",           # list1
    2: "encoding",             # list2
    3: "scaling",              # list3 (their "normalization")
    4: "feature_selection",    # list4
    5: "dimensionality_reduction",  # list5 (their "duplicate removal" -> our only unmatched family)
    6: "outlier_removal",      # list6
}

_installed = False


def _apply_step(dataset: dict, step: str, op: str) -> None:
    """Run one ACORec operator on AutoDP's {train,test,target,target_test} dict, in place."""
    cfg = {s: "none" for s in STEP_OPERATORS}
    cfg[step] = op
    pre = Preprocessor(cfg, step_order=[step])

    X_tr = dataset["train"]
    y_tr = dataset["target"]
    y_series = y_tr.iloc[:, 0] if isinstance(y_tr, pd.DataFrame) else y_tr

    # ORIGINAL row positions. read_dataset splits with train_test_split, which keeps the source
    # index, so these ARE positions in the input file. run_autodatapre writes them out as
    # __adp_row__ and eval_autodatapre uses them to attach each prepared row to its true label.
    # Renumbering here (the old reset_index(drop=True)) silently broke that link the moment any
    # operator dropped a row: prepared row k stopped being source row k, labels were matched to
    # the wrong features, and the classifier collapsed to chance -- 0.4944 on binary run_or_walk
    # with 97.8% of rows and all 6 features intact. Arm 0 was unaffected because AutoDP's own
    # operators keep their index and never reach this adapter.
    orig_index = np.asarray(X_tr.index)

    res = pre.fit_transform(X_tr.reset_index(drop=True), y_series.reset_index(drop=True))
    if isinstance(res, tuple):
        X_tr_p, y_tr_p = res
    else:
        X_tr_p, y_tr_p = res, y_series.reset_index(drop=True)

    # Ask the Preprocessor which rows survived. Its returned INDEX cannot be trusted for this:
    # some operators hand back a subset of the input labels, others a fresh 0..m-1 range (lof did
    # exactly that -- 2885 rows with max index 2884 out of 3200 input rows), so reading identity
    # off the index silently mismatched every label.
    survivors = getattr(pre, "kept_positions_", None)
    survivors = (np.arange(len(X_tr_p)) if survivors is None
                 else np.asarray(survivors, dtype=int))
    if len(survivors) != len(X_tr_p):        # defensive: never guess at a mapping
        raise RuntimeError(
            f"row-identity tracking failed for {step}:{op} -- Preprocessor reported "
            f"{len(survivors)} survivors but returned {len(X_tr_p)} rows")
    kept_index = orig_index[survivors]

    # When fit_transform returns only X, y is still the FULL column while X has lost rows --
    # subset it by the same survivors or every label shifts up by the number dropped.
    y_arr = np.asarray(y_tr_p)
    if len(y_arr) != len(survivors):
        y_arr = y_arr[survivors]
    X_tr_p = X_tr_p.reset_index(drop=True)
    X_tr_p.index = kept_index
    y_tr_p = pd.Series(y_arr, index=kept_index)   # target.loc[X.index] must work
    dataset["train"] = X_tr_p
    dataset["target"] = y_tr_p.to_frame(name=(y_tr.columns[0] if isinstance(y_tr, pd.DataFrame) else "target"))

    if not isinstance(dataset["test"], dict) and dataset["test"] is not None:
        X_te = dataset["test"]
        te_index = np.asarray(X_te.index)
        X_te_p = pre.transform(X_te.reset_index(drop=True))
        te_survivors = np.asarray(X_te_p.index, dtype=int)
        te_kept = te_index[te_survivors]
        X_te_p = X_te_p.reset_index(drop=True)
        X_te_p.index = te_kept
        dataset["test"] = X_te_p
        y_te = dataset["target_test"]
        y_te_s = (y_te.iloc[:, 0] if isinstance(y_te, pd.DataFrame) else y_te)
        y_te_s = pd.Series(np.asarray(y_te_s))
        y_te_s = y_te_s.iloc[te_survivors] if len(y_te_s) == len(te_index) else y_te_s
        y_te_s.index = te_kept
        dataset["target_test"] = y_te_s.to_frame(
            name=(y_te.columns[0] if isinstance(y_te, pd.DataFrame) else "target"))


def _apply_pipeline(dataset: dict, order) -> None:
    """Dispatch AutoDP's chosen operator chain to ACORec operators (order[0] is its classifier)."""
    for code in list(order)[1:]:
        mapped = CODE_TO_STEP_OP.get(code)
        if mapped is None:
            continue  # not one of ours (e.g. a stray original code) -> no-op
        _apply_step(dataset, *mapped)


def _retrained_order_fn(corpus_dir: Path, our_lists: dict, MCTS):
    """Their 1-NN meta-learner, reading a corpus built over OUR operators.

    Same algorithm as `get_CLA_meta_task_order` -- 7 metafeatures, nearest row of
    Metafeature.csv by their std-normalised euclidean distance, best-scoring pipeline of that
    neighbour, family order read off it. The only difference is that the corpus describes our
    operator space, so no aliasing is involved and pca/svd are selectable in their own right.
    """
    import numpy as _np
    import pandas as _pd
    import torch as _torch
    from autodatapre.Pipeline_Generation import MetaFeature

    meta = _pd.read_csv(corpus_dir / "Metafeature.csv")
    label = _pd.read_csv(corpus_dir / "label.csv")
    k = len(label) // len(meta)
    if k * len(meta) != len(label):
        raise ValueError(
            f"{corpus_dir}: label.csv has {len(label)} rows for {len(meta)} datasets; their "
            "reader slices a fixed block per dataset, so it must divide exactly."
        )
    code_to_family = {_code(step, op): step for step, ops in STEP_OPERATORS.items() for op in ops}

    def fn(df):
        matrix = MetaFeature.getfeature(df)
        query = _torch.Tensor(_np.transpose(matrix.numpy()).mean(axis=1))

        best, best_id = None, 0
        for idx, row in meta.iterrows():
            row_t = _torch.Tensor(row.values)
            d = _np.linalg.norm(query - row_t)
            d = d / _np.sqrt(_torch.std(query) ** 2 + _torch.std(row_t) ** 2)
            if best is None or d < best:
                best, best_id = d, idx

        block = label.iloc[k * best_id: k * best_id + k]
        pipeline = block.loc[block["EvaluationMetric"].idxmax()]["Pipeline"].split(",")

        order, seen = [], set()
        for token in pipeline:
            fam = code_to_family.get(token)
            if fam is None or fam in seen:
                continue          # the model slot, or a family already placed
            seen.add(fam)
            order.append(fam)
        if not order:
            raise ValueError(f"{corpus_dir}: neighbour pipeline had no known operator: {pipeline}")
        step_to_index = {v: i for i, v in THEIR_FAMILY_TO_OUR_STEP.items()}
        return [MCTS.list7] + [our_lists[step_to_index[f]] for f in order]

    return fn


def _neutral_order_fn(our_lists: dict, MCTS):
    """Give the search all six families in ACORec's canonical order, with no transferred prior.

    The alternative -- reusing their shipped corpus -- hands AutoDP a prior learned over a
    DIFFERENT operator vocabulary. Their family 5 is deduplication, which their own training
    pipelines skip 97.8% of the time (1955/2000 use dup_null), and it maps onto our
    dimensionality_reduction. So the transferred prior selects pca/svd in roughly one search in
    ten while every other family is selected normally.

    That is not a missing prior, it is a WRONG one, and it biases the arm in ACORec's favour:
    ACORec's ACO searches all six families on every run, so AutoDP seeing five of them most of the
    time is an asymmetry in our own favour -- the direction that most damages the comparison.
    """
    from automl_aco.preprocessing.preprocessor import DEFAULT_PREPROCESSOR_ORDER

    step_to_index = {v: i for i, v in THEIR_FAMILY_TO_OUR_STEP.items()}
    ordered = [our_lists[step_to_index[step]] for step in DEFAULT_PREPROCESSOR_ORDER
               if step in step_to_index]

    def fn(df):
        return [MCTS.list7] + ordered

    return fn


def install(verbose: bool = False, retrained_dir=None, family_order: str = "prior") -> None:
    """Patch autodatapre in memory so its MCTS searches ACORec's operator space.

    ``retrained_dir`` points at a corpus from scripts/build_adp_meta_corpus.py. With it, the
    meta-learner is retrained over our operators. Without it, their shipped corpus is reused and
    the family order is transferred through aliasing -- the original behaviour, kept so the two
    can be compared.
    """
    global _installed
    if _installed:
        return

    from autodatapre.Pipeline_Generation import MCTS, MCTS_DATA as D
    from autodatapre.Pipeline_Generation import Estimate_after_profit as EAP

    # Capture MCTS's own family lists (NOT MCTS_DATA's -- those carry extra '*_null' entries).
    # FamList subclasses list so their `if code in listN` membership test still works, while
    # carrying the family index so we can tell which families their meta-learner selected without
    # relying on list equality.
    original_lists = {i: _FamList(getattr(MCTS, f"list{i}"), i) for i in range(1, 7)}
    our_lists = {i: [_code(THEIR_FAMILY_TO_OUR_STEP[i], o)
                     for o in STEP_OPERATORS[THEIR_FAMILY_TO_OUR_STEP[i]]]
                 for i in range(1, 7)}

    # 1. family lists (both modules keep their own copies)
    for i in range(1, 7):
        setattr(D, f"list{i}", our_lists[i])
        setattr(MCTS, f"list{i}", our_lists[i])

    # 2. estimator class mapping: alias our codes onto their nearest class id
    for code, (step, op) in CODE_TO_STEP_OP.items():
        EAP.class_mapping[code] = EAP.class_mapping[ALIAS[op]]

    # 3. operator dispatch -> ACORec implementations
    def getAcc(dataset, order, target):
        _apply_pipeline(dataset, order)
        return D.choose_classifier(dataset, order[0], target).get("quality_metric")

    def getMse(dataset, order, target):
        try:
            _apply_pipeline(dataset, order)
            mse = D.choose_regressor(dataset, order[0], target).get("quality_metric")
        except Exception:
            mse = -1
        return 1 / mse if mse else -1

    def getdataset(dataset, order, target):
        _apply_pipeline(dataset, order)
        D.choose_classifier(dataset, order[0], target)
        return dataset

    D.getAcc, D.getMse, D.getdataset = getAcc, getMse, getdataset
    MCTS.mctsdata.getAcc, MCTS.mctsdata.getMse = getAcc, getMse

    # 4. meta-learner: keep THEIR codes for the label.csv membership test, return OUR families
    orig_order_fn = MCTS.get_CLA_meta_task_order

    def get_CLA_meta_task_order(df):
        their_List = _their_family_indices(df, orig_order_fn, original_lists, MCTS)
        return [MCTS.list7] + [our_lists[i] for i in their_List]

    if retrained_dir is not None:
        MCTS.get_CLA_meta_task_order = _retrained_order_fn(Path(retrained_dir), our_lists, MCTS)
    elif family_order == "all":
        MCTS.get_CLA_meta_task_order = _neutral_order_fn(our_lists, MCTS)
    else:
        MCTS.get_CLA_meta_task_order = get_CLA_meta_task_order

    _installed = True
    if verbose:
        if retrained_dir:
            mode = f"meta-learner RETRAINED from {retrained_dir}"
        elif family_order == "all":
            mode = "NEUTRAL family order (all 6, ACORec canonical); no prior transferred"
        else:
            mode = "meta-learner ALIASED from their shipped corpus"
        print(f"[adapter] AutoDP now searches ACORec's space: "
              f"{sum(len(v) for v in our_lists.values())} operators across {len(our_lists)} "
              f"families; {mode}")
        print("[adapter] value estimator model_CLA.pickle is NOT retrained: its input is a "
              "per-call random projection (signal/noise 0.076). See docs.")


class _FamList(list):
    """A family list that remembers which family index it is."""

    def __init__(self, items, idx):
        super().__init__(items)
        self.idx = idx


def _their_family_indices(df, orig_order_fn, original_lists, MCTS) -> list:
    """Which family indices their meta-learner selected, and in what order.

    Runs their own `get_CLA_meta_task_order` against the ORIGINAL lists (so its label.csv lookup
    still resolves), then reads back the family index each appended list carries. Their prior
    therefore still decides WHICH families to search and in what order -- only the operators inside
    each family are swapped for ours.
    """
    saved = {i: getattr(MCTS, f"list{i}") for i in range(1, 7)}
    try:
        for i in range(1, 7):
            setattr(MCTS, f"list{i}", original_lists[i])
        their_List = orig_order_fn(df)
    finally:
        for i in range(1, 7):
            setattr(MCTS, f"list{i}", saved[i])

    indices = [fam.idx for fam in their_List[1:] if isinstance(fam, _FamList)]
    # Include 5: it maps onto our dimensionality_reduction. Omitting it made pca/svd unreachable
    # whenever their meta-learner returned nothing usable, on top of the 10% selection rate the
    # transferred prior already imposes.
    return indices or [1, 2, 3, 4, 5, 6]


if __name__ == "__main__":
    install(verbose=True)
    print("operator codes:")
    for step, ops in STEP_OPERATORS.items():
        print(f"  {step:26} {[_code(step, o) for o in ops]}")
