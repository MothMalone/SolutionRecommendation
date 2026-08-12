#!/usr/bin/env python3
"""Export the evaluation datasets to CSV exactly as ``run_recommend.py`` sees them.

External baselines (AutoDP / autodatapre) run in their own pinned Python environment, which
cannot import our loaders. To keep the comparison apples-to-apples, this script materialises the
*same* frame ``run_recommend.py`` builds in memory:

    dataset = load_openml_dataset(id)      # drop all-NaN columns, drop NaN targets, coerce target,
    df = dataset["X"]; df["target"] = y    # drop classes with <5 rows, 5000-row cap @ random_state=42

and writes it to ``<out-dir>/<id>.csv``. Row ORDER is the contract: every downstream stage
re-derives the seed-42 0.6/0.2/0.2 split positionally from this row order, so the same physical
rows land in train/val/test for our method and for every external baseline.

The manifest records a SHA-256 per file so a Kaggle run can prove it used the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd

from automl_aco.data.loaders import load_openml_dataset
from automl_aco.eval_ids import EVAL_IDS


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _task_type(target: pd.Series) -> str:
    return "regression" if (target.nunique() > 50 and target.dtype.kind in "iufc") else "classification"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default=",".join(EVAL_IDS), help="comma/space separated OpenML ids (default: the eval set)")
    ap.add_argument("--out-dir", default="data/eval_datasets")
    ap.add_argument("--openml-local-folder", default=None, help="fallback folder of <id>.csv files")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ids = [t for t in args.ids.replace(",", " ").split() if t]
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = []
    failed = []
    for did in ids:
        out_path = os.path.join(args.out_dir, f"{did}.csv")
        if os.path.exists(out_path) and not args.overwrite:
            existing = pd.read_csv(out_path)
            manifest.append({
                "dataset_id": did, "path": out_path, "n_rows": int(existing.shape[0]),
                "n_features": int(existing.shape[1] - 1), "sha256": _sha256(out_path),
                "task_type": _task_type(existing["target"]),
                "n_classes": int(existing["target"].nunique()),
                "reused_existing": True,
            })
            print(f"[skip] {did} (exists: {existing.shape[0]} rows)")
            continue

        print(f"[load] {did} ...", flush=True)
        ds = load_openml_dataset(did, verbose=args.verbose, local_data_folder=args.openml_local_folder)
        if ds is None or "X" not in ds or "y" not in ds:
            print(f"[FAIL] {did}: loader returned nothing")
            failed.append(did)
            continue

        X, y = ds["X"], ds["y"]
        if "target" in X.columns:
            raise ValueError(f"dataset {did} already has a column named 'target'; cannot export unambiguously")
        df = X.copy()
        df["target"] = y.to_numpy()
        df.to_csv(out_path, index=False)

        manifest.append({
            "dataset_id": did, "path": out_path, "n_rows": int(df.shape[0]),
            "n_features": int(df.shape[1] - 1), "sha256": _sha256(out_path),
            "task_type": ds.get("task_type", _task_type(df["target"])),
            "n_classes": int(pd.Series(y).nunique()),
            "n_object_cols": int(X.select_dtypes(include=["object", "category"]).shape[1]),
            "reused_existing": False,
        })
        print(f"[ok  ] {did}: {df.shape[0]} rows x {df.shape[1] - 1} features ({manifest[-1]['task_type']})")

    man_path = os.path.join(args.out_dir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump({"datasets": manifest, "failed": failed}, f, indent=2)
    n_reg = sum(1 for m in manifest if m["task_type"] == "regression")
    print(f"\nWrote {len(manifest)} datasets to {args.out_dir} "
          f"({n_reg} regression, {len(manifest) - n_reg} classification); manifest -> {man_path}")
    if failed:
        print(f"FAILED ids: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
