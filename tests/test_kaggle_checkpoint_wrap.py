"""kaggle_checkpoint_wrap.sh must survive a simulated session kill, using a fake `kaggle` CLI.

Cannot exercise the real Kaggle Datasets API (needs a phone-verified account and network), so
this pins the three paths that matter: fresh start creates a checkpoint, a second "session" with
an empty local --out resumes from what the first one pushed, and a missing kaggle.json degrades
to running uncheckpointed rather than blocking.
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WRAP = REPO / "scripts" / "kaggle_checkpoint_wrap.sh"


@pytest.fixture
def fake_kaggle_cli(tmp_path):
    """A `kaggle` stand-in backed by a local directory, keyed on dataset slug only."""
    remote = tmp_path / "remote"
    remote.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "kaggle"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        REMOTE={remote}
        case "$1 $2" in
          "datasets download")
            shift 2; slug=""; outdir=""
            while [ $# -gt 0 ]; do case "$1" in -d) slug="$2"; shift 2;; -p) outdir="$2"; shift 2;; *) shift;; esac; done
            f="$REMOTE/$(basename "$slug").jsonl"
            [ -f "$f" ] && cp "$f" "$outdir/arms.jsonl" && exit 0
            echo "404" >&2; exit 1 ;;
          "datasets version")
            shift 2; src=""
            while [ $# -gt 0 ]; do case "$1" in -p) src="$2"; shift 2;; *) shift;; esac; done
            id=$(python3 -c "import json;print(json.load(open('$src/dataset-metadata.json'))['id'])")
            slug=$(basename "$id")
            [ -f "$REMOTE/$slug.jsonl" ] && cp "$src/arms.jsonl" "$REMOTE/$slug.jsonl" && exit 0
            echo "does not exist" >&2; exit 1 ;;
          "datasets create")
            shift 2; src=""
            while [ $# -gt 0 ]; do case "$1" in -p) src="$2"; shift 2;; *) shift;; esac; done
            id=$(python3 -c "import json;print(json.load(open('$src/dataset-metadata.json'))['id'])")
            slug=$(basename "$id")
            cp "$src/arms.jsonl" "$REMOTE/$slug.jsonl"; exit 0 ;;
          *) echo "unhandled: $*" >&2; exit 1 ;;
        esac
    """))
    script.chmod(0o755)
    return bindir, remote


def _run(bindir, home, out, slug, cmd, interval=1):
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["HOME"] = str(home)
    return subprocess.run(
        ["bash", str(WRAP), "--out", str(out), "--slug", slug, "--interval", str(interval),
         "--", "bash", "-c", cmd],
        env=env, capture_output=True, text=True, timeout=30,
    )


def test_fresh_run_creates_checkpoint(fake_kaggle_cli, tmp_path):
    bindir, remote = fake_kaggle_cli
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text('{"username":"u","key":"k"}')
    out = tmp_path / "arms.jsonl"

    r = _run(bindir, home, out, "myckpt", f'echo "{{\\"row\\":1}}" >> {out}')
    assert r.returncode == 0, r.stderr
    assert out.read_text().strip() == '{"row":1}'
    assert (remote / "myckpt.jsonl").read_text().strip() == '{"row":1}'


def test_new_session_resumes_from_checkpoint(fake_kaggle_cli, tmp_path):
    bindir, remote = fake_kaggle_cli
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text('{"username":"u","key":"k"}')
    remote_file = remote / "myckpt.jsonl"
    remote_file.write_text('{"row":1}\n{"row":2}\n')

    out = tmp_path / "fresh_session_out.jsonl"   # simulates an empty /kaggle/working
    r = _run(bindir, home, out, "myckpt", f'echo "{{\\"row\\":3}}" >> {out}')
    assert r.returncode == 0, r.stderr
    assert "resumed:" in r.stdout
    assert out.read_text().splitlines() == ['{"row":1}', '{"row":2}', '{"row":3}']


def test_missing_credentials_runs_uncheckpointed(fake_kaggle_cli, tmp_path):
    bindir, _ = fake_kaggle_cli
    home = tmp_path / "empty_home"
    home.mkdir()
    out = tmp_path / "arms.jsonl"

    r = _run(bindir, home, out, "myckpt", f'echo "{{\\"row\\":1}}" >> {out}')
    assert r.returncode == 0
    assert "WARNING" in r.stderr
    assert out.read_text().strip() == '{"row":1}'
