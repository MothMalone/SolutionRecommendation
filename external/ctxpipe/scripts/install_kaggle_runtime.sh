#!/usr/bin/env bash
set -euo pipefail

echo "[install] Python: $(python -V 2>&1)"
echo "[install] Upgrading pip tooling..."
python -m pip install -U pip setuptools wheel

echo "[install] Installing core runtime deps (Kaggle/Python3.12 compatible)..."
python -m pip install -r requirements-kaggle-py312.txt --no-cache-dir || {
  echo "[install] Bulk install failed, retrying AutoGluon with fallbacks..."
  python -m pip install \
    "numpy<2" \
    "pandas>=2.0,<2.3" \
    "scikit-learn>=1.3,<1.6" \
    "loguru>=0.7" \
    "peewee>=3.17" \
    "openml>=0.14" \
    "transformers>=4.36,<5" \
    --no-cache-dir

  python -m pip install "autogluon.tabular>=1.5" --no-cache-dir || \
  python -m pip install "autogluon.tabular>=1.4,<1.5" --no-cache-dir
}

echo "[install] Sanity checks..."
python - <<'PY'
import importlib
mods = ["torch", "pandas", "sklearn", "loguru", "transformers", "openml", "autogluon.tabular"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"[ok] {m}")
    except Exception as e:
        print(f"[missing] {m}: {e}")
        raise
PY

echo "[install] Done."
