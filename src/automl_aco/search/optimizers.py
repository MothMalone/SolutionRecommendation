"""Alternative optimizers for pipeline config search under a fixed step order."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


EvalFn = Callable[
    [List[Dict[str, Any]]],
    Tuple[Any, float, List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]],
]


def _cfg_key(cfg: Mapping[str, Any], step_order: Sequence[str]) -> Tuple[Tuple[str, Any], ...]:
    return tuple((step, cfg.get(step)) for step in step_order)


def _random_cfg(options: Mapping[str, List[str]], rng: np.random.RandomState) -> Dict[str, Any]:
    return {step: vals[int(rng.randint(0, len(vals)))] for step, vals in options.items()}


def _neighbor_cfg(
    cfg: Mapping[str, Any],
    options: Mapping[str, List[str]],
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    step = list(options.keys())[int(rng.randint(0, len(options)))]
    vals = options[step]
    if len(vals) <= 1:
        return dict(cfg)
    cur = cfg[step]
    candidates = [v for v in vals if v != cur]
    nxt = candidates[int(rng.randint(0, len(candidates)))]
    out = dict(cfg)
    out[step] = nxt
    return out


def _append_history(history: List[Dict[str, Any]], n_eval: int, best_score: Optional[float]) -> None:
    history.append({"iteration": n_eval, "best_score": best_score})


def _evaluate_one(
    cfg: Dict[str, Any],
    evaluate_fn: EvalFn,
    cache: MutableMapping[Tuple[Tuple[str, Any], ...], float],
    step_order: Sequence[str],
) -> Optional[float]:
    key = _cfg_key(cfg, step_order)
    if key in cache:
        return cache[key]
    try:
        _best_cfg, _best_score, sorted_results, _unsorted = evaluate_fn([cfg])
    except Exception as exc:  # pragma: no cover
        logger.debug("Optimizer evaluation failed: %s", exc)
        return None
    if not sorted_results:
        return None
    score = float(sorted_results[0][1])
    cache[key] = score
    return score


@dataclass
class _MCTSNode:
    depth: int
    choices: List[str]
    parent: Optional["_MCTSNode"] = None
    action: Optional[str] = None
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self):
        self.children: Dict[str, _MCTSNode] = {}
        self.untried: List[str] = list(self.choices)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


def _ucb(node: _MCTSNode, parent_visits: int, c: float = 1.4) -> float:
    if node.visits == 0:
        return float("inf")
    return node.mean_value + c * sqrt(max(log(max(parent_visits, 1)), 0.0) / node.visits)


def search_pipelines_with_optimizer(
    optimizer: str,
    options: Mapping[str, List[str]],
    evaluate_fn: EvalFn,
    sample_budget: int = 100,
    seed: int = 42,
    n_pipelines: int = 5,
    verbose: bool = False,
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]], List[Dict[str, Any]]]:
    """Search configs with non-ACO optimizers under a fixed step order."""
    optimizer = optimizer.lower().strip()
    if optimizer not in {"random", "ga", "sa", "greedy", "mcts"}:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    rng = np.random.RandomState(seed)
    step_order = list(options.keys())
    cache: Dict[Tuple[Tuple[str, Any], ...], float] = {}
    cfg_cache: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []

    def eval_cfg(cfg: Dict[str, Any]) -> Optional[float]:
        score = _evaluate_one(cfg, evaluate_fn, cache, step_order)
        key = _cfg_key(cfg, step_order)
        if score is not None and key not in cfg_cache:
            cfg_cache[key] = dict(cfg)
            _append_history(history, len(cache), max(cache.values()) if cache else None)
        return score

    def best_ranked() -> List[Tuple[Dict[str, Any], float]]:
        ranked = [(cfg_cache[k], v) for k, v in cache.items() if k in cfg_cache]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def random_until_budget() -> None:
        while len(cache) < sample_budget:
            cfg = _random_cfg(options, rng)
            eval_cfg(cfg)

    if optimizer == "random":
        random_until_budget()

    elif optimizer == "sa":
        cur = _random_cfg(options, rng)
        cur_score = eval_cfg(cur)
        if cur_score is None:
            cur_score = -np.inf
        temp = 1.0
        while len(cache) < sample_budget:
            nxt = _neighbor_cfg(cur, options, rng)
            nxt_score = eval_cfg(nxt)
            if nxt_score is None:
                temp *= 0.97
                continue
            delta = nxt_score - cur_score
            if delta >= 0 or rng.rand() < exp(delta / max(temp, 1e-6)):
                cur = nxt
                cur_score = nxt_score
            temp *= 0.97

    elif optimizer == "greedy":
        cur = _random_cfg(options, rng)
        cur_score = eval_cfg(cur)
        if cur_score is None:
            cur_score = -np.inf
        while len(cache) < sample_budget:
            improved = False
            best_neighbor = cur
            best_neighbor_score = cur_score
            for step in step_order:
                for val in options[step]:
                    if len(cache) >= sample_budget:
                        break
                    if val == cur[step]:
                        continue
                    cand = dict(cur)
                    cand[step] = val
                    sc = eval_cfg(cand)
                    if sc is not None and sc > best_neighbor_score:
                        best_neighbor = cand
                        best_neighbor_score = sc
                        improved = True
                if len(cache) >= sample_budget:
                    break
            if improved:
                cur = best_neighbor
                cur_score = best_neighbor_score
            else:
                cur = _random_cfg(options, rng)
                sc = eval_cfg(cur)
                cur_score = sc if sc is not None else cur_score

    elif optimizer == "ga":
        pop_size = min(20, max(4, sample_budget // 5))
        population: List[Tuple[Dict[str, Any], float]] = []
        while len(population) < pop_size and len(cache) < sample_budget:
            cfg = _random_cfg(options, rng)
            sc = eval_cfg(cfg)
            if sc is not None:
                population.append((cfg, sc))

        while len(cache) < sample_budget and population:
            population.sort(key=lambda x: x[1], reverse=True)
            survivors = population[: max(2, pop_size // 2)]
            children: List[Tuple[Dict[str, Any], float]] = []
            while len(children) + len(survivors) < pop_size and len(cache) < sample_budget:
                p1 = survivors[int(rng.randint(0, len(survivors)))][0]
                p2 = survivors[int(rng.randint(0, len(survivors)))][0]
                child = {}
                for step in step_order:
                    child[step] = p1[step] if rng.rand() < 0.5 else p2[step]
                    if rng.rand() < 0.15:
                        vals = options[step]
                        child[step] = vals[int(rng.randint(0, len(vals)))]
                sc = eval_cfg(child)
                if sc is not None:
                    children.append((child, sc))
            population = survivors + children

        if len(cache) < sample_budget:
            random_until_budget()

    elif optimizer == "mcts":
        steps = step_order
        root = _MCTSNode(depth=0, choices=options[steps[0]])

        def node_cfg(node: _MCTSNode) -> Dict[str, Any]:
            actions = [None] * len(steps)
            cur = node
            while cur and cur.action is not None and cur.depth > 0:
                actions[cur.depth - 1] = cur.action
                cur = cur.parent
            cfg = {}
            for idx, step in enumerate(steps):
                if actions[idx] is not None:
                    cfg[step] = actions[idx]
                else:
                    vals = options[step]
                    cfg[step] = vals[int(rng.randint(0, len(vals)))]
            return cfg

        while len(cache) < sample_budget:
            node = root
            # Selection
            while not node.untried and node.children:
                node = max(node.children.values(), key=lambda ch: _ucb(ch, node.visits))
            # Expansion
            if node.untried and node.depth < len(steps):
                action = node.untried.pop(int(rng.randint(0, len(node.untried))))
                depth = node.depth + 1
                next_choices = options[steps[depth]] if depth < len(steps) else []
                child = _MCTSNode(depth=depth, choices=next_choices, parent=node, action=action)
                node.children[action] = child
                node = child
            # Simulation
            cfg = node_cfg(node)
            sc = eval_cfg(cfg)
            reward = sc if sc is not None else 0.0
            # Backprop
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.value_sum += reward
                cur = cur.parent

    ranked = best_ranked()
    if verbose:
        print(f"{optimizer.upper()} search done | evaluated={len(cache)} | best={ranked[0][1]:.4f}" if ranked else f"{optimizer.upper()} search done | no valid configs")
    return ranked[:n_pipelines], ranked, history
