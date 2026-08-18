"""Build a one-file Kaggle rescue notebook with shard 20 checkpoint embedded."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "build-performance-matrix-autodp.ipynb"
CHECKPOINT = ROOT / "historical_autodp36.part_0020_of_0032.txt"
OUTPUT = ROOT / "notebooks" / "resume-historical-shard20-autodp.ipynb"


def lines(text: str) -> list[str]:
    return (textwrap.dedent(text).strip("\n") + "\n").splitlines(keepends=True)


notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
checkpoint_bytes = CHECKPOINT.read_bytes()
checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
payload = base64.b64encode(zlib.compress(checkpoint_bytes, level=9)).decode("ascii")

controls_index = None
for index, cell in enumerate(notebook["cells"]):
    source = "".join(cell.get("source", []))
    if "ONLY_JOB_SHARDS = None" in source and "RESUME_FROM_DATASET_ID = None" in source:
        source = source.replace("ONLY_JOB_SHARDS = None", "ONLY_JOB_SHARDS = [20]")
        source = source.replace("RESUME_FROM_DATASET_ID = None", "RESUME_FROM_DATASET_ID = 1080")
        cell["source"] = source.splitlines(keepends=True)
        controls_index = index
        break
if controls_index is None:
    raise RuntimeError("Could not locate the execution-control cell")

rescue_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "embedded-shard20-checkpoint",
    "metadata": {},
    "outputs": [],
    "source": lines(
        f'''
        # Embedded rescue checkpoint: no Kaggle Input attachment is required.
        import base64 as _resume_base64
        import hashlib as _resume_hashlib
        import zlib as _resume_zlib

        _EMBEDDED_SHARD20 = "{payload}"
        _EMBEDDED_SHARD20_SHA256 = "{checkpoint_sha256}"
        _resume_path = OUTPUT_DIR / "historical_autodp36.part_0020_of_0032.csv"

        if _resume_path.exists():
            print(f"Keeping newer working checkpoint: {{_resume_path}}")
        else:
            _resume_bytes = _resume_zlib.decompress(
                _resume_base64.b64decode(_EMBEDDED_SHARD20)
            )
            if _resume_hashlib.sha256(_resume_bytes).hexdigest() != _EMBEDDED_SHARD20_SHA256:
                raise RuntimeError("Embedded shard-20 checkpoint failed its SHA-256 check")
            _resume_path.write_bytes(_resume_bytes)
            _resume_frame = pd.read_csv(_resume_path, index_col=0)
            print(
                f"Restored embedded shard-20 checkpoint: {{_resume_path}} | "
                f"shape={{_resume_frame.shape}} | "
                f"completed_cells={{int(_resume_frame.notna().sum().sum())}}"
            )
        '''
    ),
}
notebook["cells"].insert(controls_index + 1, rescue_cell)

intro = notebook["cells"][0]
intro_text = "".join(intro.get("source", []))
rescue_note = textwrap.dedent(
    """
    > **Shard-20 rescue edition.** This file embeds the downloaded partial
    > checkpoint, runs only historical shard 20, resumes at dataset 1080, and
    > needs no checkpoint file in Kaggle Input. Keep Kaggle Internet enabled for
    > GitLab dataset downloads.

    """
).lstrip()
intro["source"] = (rescue_note + intro_text).splitlines(keepends=True)

for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    else:
        cell.pop("execution_count", None)
    cell.setdefault("id", f"cell-{index:02d}")

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(OUTPUT)
print(f"Embedded checkpoint: {len(checkpoint_bytes)} bytes, sha256={checkpoint_sha256}")
