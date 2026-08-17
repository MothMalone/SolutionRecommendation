#!/usr/bin/env python3
"""Convert DiffPrep's dataset folders into the ``<id>.csv`` layout the arms runner expects.

`run_arms.py` and `adp_bench.py` both address datasets by OpenML id and look for
``<data-dir>/<id>.csv`` with a ``target`` column. DiffPrep ships its 18 datasets as
``<name>/data.csv`` + ``<name>/info.json`` (the latter naming the label column), so 17 of the 30
evaluation datasets are unreachable without this translation.

Source: https://github.com/chu-data-lab/DiffPrep -- the `data/` folder. Either clone it and pass
``--diffprep-root``, or pass ``--download`` to pull the files through the GitHub API.

    # from a local checkout / Kaggle input
    python scripts/export_diffprep_datasets.py --diffprep-root /kaggle/input/diffprep-dataset \
        --out-dir /kaggle/working/eval_csv

    # merge with the OpenML CSVs so one --data-dir serves all 30
    python scripts/export_diffprep_datasets.py --diffprep-root ./DiffPrep/data \
        --out-dir data/eval_all --copy-openml-from /kaggle/input/.../openml

Four of the datasets (google, micro, uscensus, jungle_chess) are absent from
``data/openml/dataset_feats.csv``; their metafeatures are computed from the data at recommend
time (``data.metafeatures.compute_metafeatures_from_data``). `google` additionally has no OpenML
entry at all and uses the frozen synthetic id 100000 -- see ``eval_ids.DIFFPREP_SYNTHETIC_IDS``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from automl_aco.eval_ids import EVAL_DATASETS  # noqa: E402

# our short name -> DiffPrep's folder name (they differ enough that a heuristic would misfire)
DIFFPREP_FOLDERS = {
    "abalone": "abalone",
    "ada_prior": "ada_prior",
    "avila": "avila",
    "connect-4": "connect-4",
    "eeg": "eeg",
    "google": "google",
    "house": "house_prices",
    "jungle_chess": "jungle_chess_2pcs_raw_endgame_complete",
    "micro": "microaggregation2",
    "mozilla4": "mozilla4",
    "obesity": "obesity",
    "page-blocks": "page-blocks",
    "pbcseq": "pbcseq",
    "pol": "pol",
    "run_or_walk": "Run_or_walk_information",
    "uscensus": "USCensus",
    "wall-robot-nav": "wall-robot-navigation",
}
GH_REPO = "chu-data-lab/DiffPrep"


def _gh(path: str) -> bytes:
    """Read a file from the DiffPrep repo via the gh CLI (works for files >1MB)."""
    out = subprocess.run(
        ["gh", "api", f"repos/{GH_REPO}/contents/{path}", "-H", "Accept: application/vnd.github.raw"],
        capture_output=True, check=True,
    )
    return out.stdout


def index_dataset_dirs(root: Path) -> dict:
    """basename (lowercased) -> directory holding data.csv, found anywhere under root.

    Kaggle mounts nest unpredictably: the folders may sit at <root>/abalone/, or one or two levels
    down (<root>/data/abalone/, <root>/DiffPrep/data/abalone/). Requiring an exact layout produced
    17 FileNotFoundErrors against a correctly-attached dataset, so walk instead of assuming.
    """
    found = {}
    for path in root.rglob("data.csv"):
        found.setdefault(path.parent.name.lower(), path.parent)
    return found


def index_id_csvs(roots) -> dict:
    """'<id>' -> path of <id>.csv, found anywhere under any root."""
    found = {}
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        for path in Path(root).rglob("*.csv"):
            found.setdefault(path.stem, path)
    return found


def load_one(name: str, folder: str, root: Path | None, dir_index: dict | None = None) -> pd.DataFrame:
    if root is not None:
        src_dir = (dir_index or {}).get(folder.lower())
        if src_dir is None:
            raise FileNotFoundError(
                f"no directory named {folder!r} containing data.csv anywhere under {root}"
            )
        data_path, info_path = src_dir / "data.csv", src_dir / "info.json"
        df = pd.read_csv(data_path)
        info = json.loads(info_path.read_text())
    else:
        df = pd.read_csv(pd.io.common.BytesIO(_gh(f"data/{folder}/data.csv")))
        info = json.loads(_gh(f"data/{folder}/info.json"))

    label = info["label"]
    if label not in df.columns:
        raise ValueError(f"{name}: label {label!r} not among {list(df.columns)[:8]}...")
    # Only rename; do NOT encode. The recommender's preprocessing owns encoding, and the
    # metafeature computation needs the original dtypes to count symbolic columns correctly.
    if "target" in df.columns and label != "target":
        df = df.rename(columns={"target": "target__orig"})
    return df.rename(columns={label: "target"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diffprep-root", type=Path, default=None,
                    help="Folder containing the DiffPrep dataset directories (its data/).")
    ap.add_argument("--download", action="store_true",
                    help="Fetch from GitHub via the gh CLI instead of a local root.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to write <id>.csv")
    ap.add_argument("--copy-openml-from", type=Path, action="append", default=[],
                    help="Repeatable: root to search (recursively) for the 13 OpenML eval "
                         "<id>.csv files, so one dir serves all 30. The repo's own "
                         "data/eval_datasets is always searched as well.")
    ap.add_argument("--only", default="", help="Comma-separated subset of names.")
    args = ap.parse_args()

    if not args.download and args.diffprep_root is None:
        ap.error("pass --diffprep-root or --download")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dir_index = {}
    if not args.download:
        if not args.diffprep_root.is_dir():
            ap.error(f"--diffprep-root {args.diffprep_root} is not a directory")
        dir_index = index_dataset_dirs(args.diffprep_root)
        print(f"[scan] {len(dir_index)} dataset dir(s) with a data.csv under {args.diffprep_root}")
        if not dir_index:
            print("[scan] nothing found -- check the mount with: "
                  f"find {args.diffprep_root} -name data.csv | head")

    wanted = [n.strip() for n in args.only.split(",") if n.strip()] or list(DIFFPREP_FOLDERS)
    written, failed = [], []
    for name in wanted:
        folder = DIFFPREP_FOLDERS[name]
        did = EVAL_DATASETS[name]
        try:
            df = load_one(name, folder, args.diffprep_root if not args.download else None, dir_index)
        except Exception as exc:
            print(f"  FAIL {name:16} {type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        dest = args.out_dir / f"{did}.csv"
        df.to_csv(dest, index=False)
        n_sym = df.drop(columns=["target"]).select_dtypes(exclude="number").shape[1]
        print(f"  ok   {name:16} id={did:<7} {df.shape[0]:>6} x {df.shape[1]-1:<4} "
              f"({n_sym} symbolic, {df['target'].nunique()} classes) -> {dest.name}")
        written.append(name)

    # The 13 OpenML eval CSVs. Searched recursively across every supplied root plus the repo's own
    # data/eval_datasets, because no single Kaggle mount holds all of them (862 in particular is in
    # the repo but not in the mathurinache/openml dump).
    openml_roots = list(args.copy_openml_from) + [REPO / "data" / "eval_datasets"]
    id_index = index_id_csvs(openml_roots)
    openml_names = [n for n in EVAL_DATASETS if n not in DIFFPREP_FOLDERS]
    for name in openml_names:
        did = EVAL_DATASETS[name]
        dest = args.out_dir / f"{did}.csv"
        if dest.exists():
            continue
        src = id_index.get(did)
        if src is not None:
            shutil.copy(src, dest)
            n_rows = sum(1 for _ in open(dest)) - 1
            # Files exported before the cap was removed are exactly 5000 rows. Copying one silently
            # reinstates the truncation this pipeline just stopped doing, so say so. Of the 13
            # OpenML eval datasets only ipums-la-99 (378, true size 8,844) is actually affected --
            # the other 12 are smaller than the old cap.
            warn = ("  <-- WARNING: exactly 5000 rows, i.e. the OLD CAP. Re-export it with "
                    "scripts/export_eval_datasets.py (no cap by default) to get its true size."
                    if n_rows == 5000 else "")
            print(f"  ok   {name:16} id={did:<7} copied from {src.parent} ({n_rows} rows){warn}")
        else:
            print(f"  MISS {name:16} id={did} -- no {did}.csv under {[str(r) for r in openml_roots]}")
            failed.append(name)

    present = {p.stem for p in args.out_dir.glob("*.csv")}
    missing = [f"{n}({i})" for n, i in EVAL_DATASETS.items() if i not in present]
    print(f"\n{len(written)} DiffPrep CSV(s) written to {args.out_dir}")
    print(f"evaluation datasets present in out-dir: {30 - len(missing)}/30")
    if missing:
        print(f"MISSING: {', '.join(missing)}")

    # Fingerprint of the whole data directory.
    #
    # Sharded runs across separate notebooks each rebuild this folder, and row ORDER is the split
    # contract -- every stage re-derives the seed-42 0.6/0.2/0.2 split positionally from it. If two
    # notebooks ever build a dataset differently, their results silently stop being comparable and
    # nothing surfaces it. Print one digest per build and check that every shard agrees; that is
    # the same guarantee a shared mounted dataset would give, without the Kaggle plumbing.
    print(write_fingerprint(args.out_dir))
    return 1 if failed or missing else 0


def write_fingerprint(out_dir: Path) -> str:
    """Digest every <id>.csv (name + content) into one short, comparable string."""
    per_file = []
    for path in sorted(out_dir.glob("*.csv"), key=lambda p: p.name):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        n_rows = sum(1 for _ in open(path, "rb")) - 1
        per_file.append({"file": path.name, "n_rows": n_rows, "sha256": h.hexdigest()})

    combined = hashlib.sha256(
        "\n".join(f"{r['file']}:{r['sha256']}" for r in per_file).encode()
    ).hexdigest()
    (out_dir / "fingerprint.json").write_text(
        json.dumps({"combined_sha256": combined, "files": per_file}, indent=2)
    )
    lines = [f"\nDATA FINGERPRINT  {combined[:16]}  ({len(per_file)} files)",
             "Every shard must print this same value. If one differs, its results are not "
             "comparable -- diff fingerprint.json to find the file."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
