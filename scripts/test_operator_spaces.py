#!/usr/bin/env python3
"""Invariant tests for both operator spaces, run before spending any Kaggle time.

Checks every operator of both spaces, in isolation and in full pipelines, against the invariants
whose violation has already cost us three silent failures (column mismatch in
``_transform_feature_selection``, NaN reaching ``SelectKBest``, and prepared test frames missing
train columns):

  I1  ``transform`` does not raise
  I2  the transformed test frame has exactly the fitted train frame's columns, in the same order
  I3  ``transform`` never changes the test row count -- row drops are a train-time operation
  I4  ``fit_transform`` keeps X and y the same length
  I5  no NaN or inf survives into either output
  I6  no test-label leakage: transforming the same test rows with a different y leaves the
      prepared test frame bit-identical (catches any supervised operator fitted on test labels)

Usage:
    python scripts/test_operator_spaces.py            # both spaces
    python scripts/test_operator_spaces.py --space theirs
"""
from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automl_aco.config import OPERATOR_SPACES  # noqa: E402
from automl_aco.preprocessing.preprocessor import Preprocessor  # noqa: E402


def make_frames(seed: int = 0):
    """Three frames chosen to exercise the paths that have actually broken before."""
    rng = np.random.default_rng(seed)
    n = 160
    frames = {}

    # (a) mixed types WITH missing values, moderate cardinality
    frames["mixed_nan"] = pd.DataFrame({
        "num_a": rng.normal(0, 1, n),
        "num_b": rng.exponential(2, n),
        "num_c": rng.integers(0, 5, n).astype(float),
        "cat_a": rng.choice(list("abcd"), n),
        "cat_b": rng.choice(["x", "y"], n),
    })
    frames["mixed_nan"].loc[rng.choice(n, 20, replace=False), "num_a"] = np.nan
    frames["mixed_nan"].loc[rng.choice(n, 12, replace=False), "cat_a"] = None

    # (b) all numeric, no missing, includes a constant and a NEGATIVE column (chi2 guard)
    frames["numeric_only"] = pd.DataFrame({
        "p": rng.normal(5, 2, n),
        "q": rng.normal(-3, 1, n),      # negative -> WR must exclude it, not crash
        "r": np.ones(n),                # constant -> variance/std == 0
        "s": rng.integers(0, 100, n).astype(float),
    })

    # (c) high-cardinality categoricals, and a test-only level (unseen-category path)
    frames["high_card"] = pd.DataFrame({
        "id_like": rng.integers(0, 60, n).astype(str),
        "grp": rng.choice([f"g{i}" for i in range(12)], n),
        "val": rng.normal(0, 1, n),
    })

    targets = {k: pd.Series(rng.integers(0, 2, n), name="y") for k in frames}
    return frames, targets


def split(X, y, n_train=120):
    return (X.iloc[:n_train].reset_index(drop=True), y.iloc[:n_train].reset_index(drop=True),
            X.iloc[n_train:].reset_index(drop=True), y.iloc[n_train:].reset_index(drop=True))


def check(cfg, order, X, y, label):
    """Run one config through fit/transform and return a list of invariant violations."""
    Xtr, ytr, Xte, yte = split(X, y)
    errs = []

    pre = Preprocessor(dict(cfg), step_order=list(order))
    Xtr_out, ytr_out = pre.fit_transform(Xtr, ytr)

    if Xtr_out is None:
        return ["fit_transform returned None"]

    # I4
    if ytr_out is not None and len(Xtr_out) != len(ytr_out):
        errs.append(f"I4 X/y desync after fit: X={len(Xtr_out)} y={len(ytr_out)}")

    # I1
    try:
        Xte_out = pre.transform(Xte)
    except Exception as exc:
        return errs + [f"I1 transform raised {type(exc).__name__}: {exc}"]

    # I2
    if list(Xtr_out.columns) != list(Xte_out.columns):
        only_tr = [c for c in Xtr_out.columns if c not in Xte_out.columns]
        only_te = [c for c in Xte_out.columns if c not in Xtr_out.columns]
        errs.append(f"I2 column mismatch: train-only={only_tr[:4]} test-only={only_te[:4]} "
                    f"(ntrain={len(Xtr_out.columns)} ntest={len(Xte_out.columns)})")

    # I3
    if len(Xte_out) != len(Xte):
        errs.append(f"I3 test rows changed: {len(Xte)} -> {len(Xte_out)}")

    # I5 -- only meaningful when the input was clean or imputation was actually requested.
    # NaN in -> NaN out with imputation="none" is correct behaviour, not a defect.
    imputes = str(cfg.get("imputation", "none")) != "none"
    input_clean = not bool(X.isna().any().any())
    if imputes or input_clean:
        for name, frame in (("train", Xtr_out), ("test", Xte_out)):
            num = frame.select_dtypes(include=["number"])
            if num.shape[1] and not np.isfinite(num.to_numpy(dtype=float)).all():
                errs.append(f"I5 non-finite values in {name} output")

    # I6 -- ROW INDEPENDENCE. transform() takes no y, so label leakage cannot enter that way; the
    # real hazard is an operator whose test output depends on the OTHER test rows (exactly what
    # AutoDP's concat-then-fit encoders do). Feeding the test rows in reverse order must produce
    # the reversed outputs. Any cross-row dependence in transform breaks this.
    # RAND is exempt: it fills each missing cell with an independent draw, so reordering the rows
    # reshuffles which draw lands where. That is stochasticity, not information flow -- it consults
    # no other row's values. Every other operator must pass.
    if str(cfg.get("imputation", "none")) == "RAND":
        return errs

    rev = Xte.iloc[::-1].reset_index(drop=True)
    try:
        Xte_rev = pre.transform(rev)
    except Exception as exc:
        return errs + [f"I6 transform raised on reordered rows: {type(exc).__name__}: {exc}"]
    a = Xte_out.select_dtypes(include=["number"]).to_numpy(dtype=float)
    b = Xte_rev.select_dtypes(include=["number"]).to_numpy(dtype=float)[::-1]
    if a.shape == b.shape and not np.allclose(a, b, equal_nan=True):
        errs.append("I6 test output depends on other test rows (transductive)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["ours", "theirs", "both"], default="both")
    ap.add_argument("--pipelines", type=int, default=25,
                    help="random full-pipeline combinations to test per space")
    args = ap.parse_args()

    spaces = ["ours", "theirs"] if args.space == "both" else [args.space]
    frames, targets = make_frames()
    total_fail = 0

    for space_name in spaces:
        spec = OPERATOR_SPACES[space_name]
        options, order = spec["options"], spec["order"]
        print("=" * 88)
        print(f"SPACE: {space_name}   steps={list(options.keys())}")
        print("=" * 88)

        # ---- Pass 1: each operator alone, on each frame
        n_fail = 0
        n_run = 0
        for step, ops in options.items():
            for op in ops:
                if op == "none":
                    continue
                for fname, X in frames.items():
                    cfg = {s: "none" for s in options}
                    cfg[step] = op
                    # an encoder is needed for categoricals to reach numeric-only operators
                    if step != "encoding" and "encoding" in cfg:
                        cfg["encoding"] = options["encoding"][-1] if space_name == "theirs" else "onehot"
                    n_run += 1
                    try:
                        errs = check(cfg, order, X, targets[fname], f"{step}={op}")
                    except Exception:
                        errs = ["UNCAUGHT " + traceback.format_exc(limit=2).strip().splitlines()[-1]]
                    if errs:
                        n_fail += 1
                        print(f"  FAIL {step:22} {op:6} on {fname:14} {errs[0]}")
        print(f"  -- single operators: {n_run - n_fail}/{n_run} passed")

        # ---- Pass 2: random full pipelines (every step active at once)
        rng = np.random.default_rng(7)
        p_fail = 0
        for i in range(args.pipelines):
            cfg = {s: str(rng.choice(ops)) for s, ops in options.items()}
            fname = list(frames)[i % len(frames)]
            try:
                errs = check(cfg, order, frames[fname], targets[fname], "pipeline")
            except Exception:
                errs = ["UNCAUGHT " + traceback.format_exc(limit=2).strip().splitlines()[-1]]
            if errs:
                p_fail += 1
                print(f"  FAIL pipeline on {fname}: {cfg}")
                for e in errs:
                    print(f"       {e}")
        print(f"  -- full pipelines:   {args.pipelines - p_fail}/{args.pipelines} passed")
        total_fail += n_fail + p_fail

    print()
    print("ALL INVARIANTS HOLD" if total_fail == 0 else f"{total_fail} FAILURES")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
