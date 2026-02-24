"""Utilities for searching pipeline step-type orderings under constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple
import random

Step = str
Constraint = Tuple[Step, Step]


@dataclass(frozen=True)
class OrderSearchConfig:
    steps: Tuple[Step, ...]
    constraints: Tuple[Constraint, ...]
    max_orders: int = 10
    strategy: str = "fixed"  # fixed | random | heuristic | scored | all
    seed: int = 42


def _build_graph(
    steps: Sequence[Step],
    constraints: Sequence[Constraint],
) -> Tuple[Dict[Step, Set[Step]], Dict[Step, int]]:
    adj: Dict[Step, Set[Step]] = {s: set() for s in steps}
    indeg: Dict[Step, int] = {s: 0 for s in steps}

    step_set = set(steps)
    for a, b in constraints:
        if a not in step_set or b not in step_set:
            raise ValueError(f"Constraint ({a}->{b}) uses step not in candidate steps.")
        if b not in adj[a]:
            adj[a].add(b)
            indeg[b] += 1
    return adj, indeg


def all_topological_orders(
    steps: Sequence[Step],
    constraints: Sequence[Constraint],
    limit: int | None = None,
) -> List[List[Step]]:
    """Enumerate all valid linearizations for the precedence graph."""
    adj, indeg0 = _build_graph(steps, constraints)
    steps = list(steps)
    used: Set[Step] = set()
    current: List[Step] = []
    results: List[List[Step]] = []

    def backtrack(indeg: Dict[Step, int]) -> None:
        if limit is not None and len(results) >= limit:
            return
        if len(current) == len(steps):
            results.append(current.copy())
            return

        available = [s for s in steps if s not in used and indeg[s] == 0]
        if not available:
            return

        for step in available:
            used.add(step)
            current.append(step)
            changed: List[Step] = []
            for nxt in adj[step]:
                indeg[nxt] -= 1
                changed.append(nxt)

            backtrack(indeg)

            for nxt in changed:
                indeg[nxt] += 1
            current.pop()
            used.remove(step)

    backtrack(indeg0.copy())
    if not results:
        raise ValueError("No valid step ordering found. Check precedence constraints for cycles.")
    return results


def heuristic_score_order(order: Sequence[Step]) -> float:
    """Simple static score for ranking candidate orders."""
    pos = {s: i for i, s in enumerate(order)}
    score = 0.0

    if "scaling" in pos and "dimensionality_reduction" in pos and pos["scaling"] < pos["dimensionality_reduction"]:
        score += 1.0

    for step in ("feature_selection", "dimensionality_reduction"):
        if "outlier_removal" in pos and step in pos and pos["outlier_removal"] < pos[step]:
            score += 0.5

    if "imputation" in pos:
        score += max(0.0, 1.0 - 0.1 * pos["imputation"])

    return score


def propose_orders(cfg: OrderSearchConfig) -> List[List[Step]]:
    strategy = cfg.strategy.lower().strip()
    rnd = random.Random(cfg.seed)

    if strategy == "fixed":
        return [list(cfg.steps)]

    pool = all_topological_orders(cfg.steps, cfg.constraints, limit=None)

    if strategy == "all":
        return pool[: cfg.max_orders]

    if strategy == "random":
        rnd.shuffle(pool)
        return pool[: cfg.max_orders]

    if strategy in {"heuristic", "scored"}:
        pool.sort(key=heuristic_score_order, reverse=True)
        top_count = max(1, int(cfg.max_orders * 0.7))
        top = pool[:top_count]
        rest = pool[top_count:]
        rnd.shuffle(rest)
        return (top + rest)[: cfg.max_orders]

    raise ValueError(f"Unknown order strategy: {cfg.strategy}")
