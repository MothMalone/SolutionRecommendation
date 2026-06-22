from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rq3_hybrid_outer_holdout.py"
SPEC = importlib.util.spec_from_file_location("rq3_hybrid_outer_holdout_for_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
hybrid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid)


Candidate = Tuple[str, str, Dict[str, Any], float, str, str, str]


def _candidate(sig: str, source: str, score: float) -> Candidate:
    return (sig, source, {"name": sig}, score, "accuracy", "ok", "")


def test_hybrid_selector_exact_tie_defaults_to_no_search() -> None:
    chosen = hybrid._select_hybrid_candidate(
        [
            _candidate("no", "no_search_selected", 0.75),
            _candidate("search", "search_selected", 0.75),
        ],
        no_sig="no",
        search_sig="search",
        margin=0.0,
        tie_breaker="no_search",
    )

    assert chosen[0] == "no"
    assert chosen[5].endswith("within_margin_default_no_search")


def test_hybrid_selector_margin_requires_clear_search_win() -> None:
    chosen = hybrid._select_hybrid_candidate(
        [
            _candidate("no", "no_search_selected", 0.750),
            _candidate("search", "search_selected", 0.754),
        ],
        no_sig="no",
        search_sig="search",
        margin=0.005,
        tie_breaker="no_search",
    )

    assert chosen[0] == "no"
    assert chosen[5].endswith("within_margin_default_no_search")


def test_hybrid_selector_selects_search_when_margin_is_exceeded() -> None:
    chosen = hybrid._select_hybrid_candidate(
        [
            _candidate("no", "no_search_selected", 0.750),
            _candidate("search", "search_selected", 0.756),
        ],
        no_sig="no",
        search_sig="search",
        margin=0.005,
        tie_breaker="no_search",
    )

    assert chosen[0] == "search"
    assert chosen[5] == "search_beats_no_search_margin"


def test_hybrid_selector_inner_candidate_must_clear_baseline_margin() -> None:
    chosen = hybrid._select_hybrid_candidate(
        [
            _candidate("no", "no_search_selected", 0.750),
            _candidate("search", "search_selected", 0.751),
            _candidate("inner", "search_inner_topk", 0.754),
        ],
        no_sig="no",
        search_sig="search",
        margin=0.005,
        tie_breaker="no_search",
    )

    assert chosen[0] == "no"
    assert chosen[5].startswith("best_within_margin_default_baseline")

    chosen = hybrid._select_hybrid_candidate(
        [
            _candidate("no", "no_search_selected", 0.750),
            _candidate("search", "search_selected", 0.751),
            _candidate("inner", "search_inner_topk", 0.756),
        ],
        no_sig="no",
        search_sig="search",
        margin=0.005,
        tie_breaker="no_search",
    )

    assert chosen[0] == "inner"
    assert chosen[5] == "non_baseline_beats_baseline_margin"
