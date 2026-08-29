#!/usr/bin/env python3
"""Export the evaluation datasets to CSV exactly as ``run_recommend.py`` sees them.

External baselines (AutoDP / autodatapre) run in their own pinned Python environment, which
cannot import our loaders. To keep the comparison apples-to-apples, this script materialises the
*same* frame ``run_recommend.py`` builds in memory:

    dataset = load_openml_dataset(id)      # drop all-NaN columns, drop NaN targets, coerce target,
    df = dataset["X"]; df["target"] = y    # drop classes with <5 rows, optional --max-rows cap
                                           # (default: NO cap) @ random_state=42

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
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd

from automl_aco.data.loaders import (
    _detect_target_column,
    load_gitlab_openml_dataset,
    load_openml_dataset,
)
from automl_aco.eval_ids import EVAL_IDS

# Searched for <id>.csv when OpenML is unreachable (Kaggle sessions with internet off).
DEFAULT_LOCAL_ROOTS = ("/kaggle/input", "data/openml", "test_data_local")

# --max-rows 0 means "keep every row"; the loader wants a concrete bound.
_NO_CAP = 10 ** 9


def _internet_available(host: str = "api.openml.org", port: int = 443, timeout: float = 5.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def _index_local_csvs(ids, roots) -> dict:
    """id -> directory holding <id>.csv, by walking each root once."""
    wanted = {f"{i}.csv": str(i) for i in ids}
    wanted.update({f"{i}.csv.zip": str(i) for i in ids})
    found = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                did = wanted.get(fn)
                if did is not None and did not in found:
                    found[did] = dirpath
    return found


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
    ap.add_argument("--openml-local-folder", default=None,
                    help="folder of <id>.csv files to fall back on. Without it, /kaggle/input, "
                         "data/openml and test_data_local are searched automatically.")
    ap.add_argument("--openml-backend", choices=["openml", "gitlab"], default="openml",
                    help="dataset backend: OpenML API (default) or the GitLab/DataGit Parquet mirror")
    ap.add_argument("--gitlab-cache-dir", default=None,
                    help="writable cache root for the GitLab backend; defaults to <out-dir>/.gitlab_cache")
    ap.add_argument("--local-root", action="append", default=[],
                    help="repeatable: extra root to search for <id>.csv")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    # The loader caps at 5000 rows unless the id is declared "test". This script never declared
    # them, so every EVALUATION dataset was silently downsampled to 5000 -- which is why paper
    # Table 2 lists ipums-la-99 at 8,844 rows while the older data/eval_datasets/378.csv holds
    # 5,000. The 5000 was an experimentation-era setting, never an intended part of the protocol.
    #
    # Default is now NO CAP, so exported datasets are their true size and match Table 2. Pass
    # --max-rows N to reinstate a cap. The value used is recorded in the manifest.
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Row cap per evaluation dataset. Default 0 = NO CAP (datasets keep their "
                         "true size). The historical experimentation value was 5000. Recorded in "
                         "the manifest.")
    args = ap.parse_args()

    ids = [t for t in args.ids.replace(",", " ").split() if t]
    os.makedirs(args.out_dir, exist_ok=True)

    online = _internet_available()
    roots = list(args.local_root) + list(DEFAULT_LOCAL_ROOTS)
    if args.openml_local_folder:
        roots.insert(0, args.openml_local_folder)
    local = _index_local_csvs(ids, roots)
    print(f"[env] OpenML reachable: {online} | local <id>.csv found for {len(local)}/{len(ids)} ids"
          + (f" (e.g. {sorted(local.items())[0][1]})" if local else ""))
    if not online and not local:
        print(f"[env] FATAL: no internet and no local CSVs under {roots}. Either enable Internet in "
              f"the notebook settings, or mount a dataset of <id>.csv files and pass "
              f"--openml-local-folder / --local-root.")

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

        local_dir = local.get(str(did)) or args.openml_local_folder
        src = "OpenML API"
        if local_dir:
            lpath = os.path.join(local_dir, f"{did}.csv")
            src = f"local {lpath}"
            if os.path.exists(lpath):
                # Surface the column the loader will treat as the label: for a local CSV it takes
                # target/class/label/y if present and otherwise THE LAST COLUMN, which is worth
                # eyeballing before a multi-hour run is scored against the wrong thing.
                try:
                    src += f" (label column: {_detect_target_column(pd.read_csv(lpath, nrows=0))!r})"
                except Exception:
                    pass
        print(f"[load] {did} from {src} ...", flush=True)
        # Declaring the ids as "test" selects the loader's max_samples_if_test branch, which is
        # how --max-rows takes effect; see the flag's help for why this was a silent 5000.
        if args.openml_backend == "gitlab":
            ds = load_gitlab_openml_dataset(
                did,
                test_dataset_ids=[str(d) for d in ids],
                verbose=args.verbose,
                local_data_folder=local_dir,
                cache_dir=args.gitlab_cache_dir or os.path.join(args.out_dir, ".gitlab_cache"),
                max_samples_if_test=(args.max_rows if args.max_rows > 0 else _NO_CAP),
            )
        else:
            ds = load_openml_dataset(
                did,
                test_dataset_ids=[str(d) for d in ids],
                verbose=args.verbose,
                local_data_folder=local_dir,
                # Offline, the API attempt only burns time on DNS/connect failures.
                use_direct_api=online,
                max_samples_if_test=(args.max_rows if args.max_rows > 0 else _NO_CAP),
            )
        if ds is None or "X" not in ds or "y" not in ds:
            why = ("no local CSV found and OpenML is unreachable — enable Internet or mount the "
                   "dataset" if (not online and not local_dir)
                   else "the loader could not parse it; rerun with --verbose for the traceback")
            print(f"[FAIL] {did}: {why}")
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
        json.dump({"datasets": manifest, "failed": failed,
                   "max_rows": args.max_rows or None}, f, indent=2)
    n_reg = sum(1 for m in manifest if m["task_type"] == "regression")
    print(f"\nWrote {len(manifest)} datasets to {args.out_dir} "
          f"({n_reg} regression, {len(manifest) - n_reg} classification); manifest -> {man_path}")
    if failed:
        print(f"FAILED ids: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
