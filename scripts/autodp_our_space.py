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

    res = pre.fit_transform(X_tr.reset_index(drop=True), y_series.reset_index(drop=True))
    if isinstance(res, tuple):
        X_tr_p, y_tr_p = res
    else:
        X_tr_p, y_tr_p = res, y_series.reset_index(drop=True)

    # Keep index and target aligned: their classifiers do dataset['target'].loc[X.index].
    X_tr_p = X_tr_p.reset_index(drop=True)
    y_tr_p = pd.Series(y_tr_p).reset_index(drop=True)
    dataset["train"] = X_tr_p
    dataset["target"] = y_tr_p.to_frame(name=(y_tr.columns[0] if isinstance(y_tr, pd.DataFrame) else "target"))

    if not isinstance(dataset["test"], dict) and dataset["test"] is not None:
        X_te_p = pre.transform(dataset["test"].reset_index(drop=True)).reset_index(drop=True)
        dataset["test"] = X_te_p
        y_te = dataset["target_test"]
        y_te_s = (y_te.iloc[:, 0] if isinstance(y_te, pd.DataFrame) else y_te).reset_index(drop=True)
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


def install(verbose: bool = False, retrained_dir=None) -> None:
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
    else:
        MCTS.get_CLA_meta_task_order = get_CLA_meta_task_order

    _installed = True
    if verbose:
        mode = f"meta-learner RETRAINED from {retrained_dir}" if retrained_dir else \
               "meta-learner ALIASED from their shipped corpus"
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
    return indices or [1, 2, 3, 4, 6]  # fall back to the five shared families


if __name__ == "__main__":
    install(verbose=True)
    print("operator codes:")
    for step, ops in STEP_OPERATORS.items():
        print(f"  {step:26} {[_code(step, o) for o in ops]}")
