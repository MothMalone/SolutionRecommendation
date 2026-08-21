"""Alternative optimizers for pipeline config search under a fixed step order."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from itertools import product
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


def _random_unseen_cfg(
    options: Mapping[str, List[str]],
    step_order: Sequence[str],
    rng: np.random.RandomState,
    cache: Mapping[Tuple[Tuple[str, Any], ...], float],
    max_tries: int = 500,
) -> Optional[Dict[str, Any]]:
    for _ in range(max_tries):
        cfg = _random_cfg(options, rng)
        if _cfg_key(cfg, step_order) not in cache:
            return cfg
    return None


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
    if optimizer not in {"random", "ga", "sa", "greedy", "mcts", "beam", "tpe", "exhaustive"}:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    rng = np.random.RandomState(seed)
    step_order = list(options.keys())
    cache: Dict[Tuple[Tuple[str, Any], ...], float] = {}
    cfg_cache: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []

    # `cache` is keyed by configuration, so len(cache) counts DISTINCT configs evaluated and can
    # never exceed the size of the search space. Every loop below is driven by that count, so a
    # sample_budget larger than the space made them spin forever: the caller asks for 12 draws
    # from a 9-point space and the loop never reaches 12. Cap the budget at what exists.
    space_size = 1
    for _step in step_order:
        space_size *= max(1, len(options[_step]))
    budget = max(0, min(int(sample_budget), int(space_size)))

    # Second, independent stall source: a configuration whose evaluation fails is never cached
    # (see _evaluate_one), so it is redrawn indefinitely. If every remaining config in the space
    # fails, the cap above is unreachable and the loop still hangs. Bound the real evaluation
    # work too. Only cache MISSES count -- cache hits are free, and greedy/beam revisit
    # neighbours constantly, so counting hits would cut those searches short.
    attempts = {"n": 0}
    max_attempts = max(50, 25 * budget)

    def budget_left() -> bool:
        return len(cache) < budget and attempts["n"] < max_attempts

    def eval_cfg(cfg: Dict[str, Any]) -> Optional[float]:
        key = _cfg_key(cfg, step_order)
        if key not in cache:
            attempts["n"] += 1
        score = _evaluate_one(cfg, evaluate_fn, cache, step_order)
        if score is not None and key not in cfg_cache:
            cfg_cache[key] = dict(cfg)
            _append_history(history, len(cache), max(cache.values()) if cache else None)
        return score

    def best_ranked() -> List[Tuple[Dict[str, Any], float]]:
        ranked = [(cfg_cache[k], v) for k, v in cache.items() if k in cfg_cache]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def random_until_budget() -> None:
        while budget_left():
            cfg = _random_cfg(options, rng)
            eval_cfg(cfg)

    if optimizer == "random":
        random_until_budget()

    elif optimizer == "exhaustive":
        all_combos = int(np.prod([len(options[s]) for s in step_order], dtype=np.int64))
        if all_combos <= budget:
            for values in product(*[options[s] for s in step_order]):
                cfg = {s: v for s, v in zip(step_order, values)}
                eval_cfg(cfg)
        else:
            # Exact for small spaces; fallback to unique random subset when full enumeration is too large.
            random_until_budget()

    elif optimizer == "sa":
        cur = _random_unseen_cfg(options, step_order, rng, cache) or _random_cfg(options, rng)
        cur_score = eval_cfg(cur)
        if cur_score is None:
            cur_score = -np.inf
        temp = 1.0
        while budget_left():
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
        cur = _random_unseen_cfg(options, step_order, rng, cache) or _random_cfg(options, rng)
        cur_score = eval_cfg(cur)
        if cur_score is None:
            cur_score = -np.inf
        while budget_left():
            improved = False
            best_neighbor = cur
            best_neighbor_score = cur_score
            for step in step_order:
                for val in options[step]:
                    if not budget_left():
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
                if not budget_left():
                    break
            if improved:
                cur = best_neighbor
                cur_score = best_neighbor_score
            else:
                cur = _random_unseen_cfg(options, step_order, rng, cache) or _random_cfg(options, rng)
                sc = eval_cfg(cur)
                cur_score = sc if sc is not None else cur_score

    elif optimizer == "ga":
        pop_size = min(20, max(4, budget // 5))
        population: List[Tuple[Dict[str, Any], float]] = []
        while len(population) < pop_size and budget_left():
            cfg = _random_unseen_cfg(options, step_order, rng, cache) or _random_cfg(options, rng)
            sc = eval_cfg(cfg)
            if sc is not None:
                population.append((cfg, sc))

        while budget_left() and population:
            population.sort(key=lambda x: x[1], reverse=True)
            survivors = population[: max(2, pop_size // 2)]
            children: List[Tuple[Dict[str, Any], float]] = []
            while len(children) + len(survivors) < pop_size and budget_left():
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

        if budget_left():
            random_until_budget()

    elif optimizer == "beam":
        beam_width = min(12, max(4, budget // 10))
        beam: List[Tuple[Dict[str, Any], float]] = []

        # Warm start the beam with random unique seeds.
        while len(beam) < beam_width and budget_left():
            cfg = _random_unseen_cfg(options, step_order, rng, cache)
            if cfg is None:
                break
            sc = eval_cfg(cfg)
            if sc is not None:
                beam.append((cfg, sc))

        while budget_left() and beam:
            neighborhood: List[Tuple[Dict[str, Any], float]] = []
            for cfg, _sc in beam:
                for step in step_order:
                    for val in options[step]:
                        if not budget_left():
                            break
                        if cfg.get(step) == val:
                            continue
                        cand = dict(cfg)
                        cand[step] = val
                        sc = eval_cfg(cand)
                        if sc is not None:
                            neighborhood.append((cand, sc))
                    if not budget_left():
                        break

            if not neighborhood:
                refill = _random_unseen_cfg(options, step_order, rng, cache)
                if refill is None:
                    break
                sc = eval_cfg(refill)
                if sc is None:
                    break
                beam = [(refill, sc)]
                continue

            merged = beam + neighborhood
            merged.sort(key=lambda x: x[1], reverse=True)
            next_beam: List[Tuple[Dict[str, Any], float]] = []
            seen_keys = set()
            for cfg, sc in merged:
                key = _cfg_key(cfg, step_order)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                next_beam.append((cfg, sc))
                if len(next_beam) >= beam_width:
                    break
            beam = next_beam

        if budget_left():
            random_until_budget()

    elif optimizer == "tpe":
        # Lightweight categorical TPE-style search:
        # model P(x|good) vs P(x|bad) with count statistics and sample by likelihood ratio.
        n_init = min(max(10, budget // 10), budget)
        # budget_left() as well as n_init: _random_unseen_cfg keeps finding fresh configs in a
        # large space, so if every evaluation fails the cache never grows and this warm-up alone
        # spins forever.
        while len(cache) < n_init and budget_left():
            cfg = _random_unseen_cfg(options, step_order, rng, cache)
            if cfg is None:
                break
            eval_cfg(cfg)

        gamma = 0.25
        n_candidates = 48
        prior = 1.0

        def _build_stats():
            ranked_items = sorted(cache.items(), key=lambda kv: kv[1], reverse=True)
            n_good = max(1, int(np.ceil(gamma * len(ranked_items))))
            good_keys = [k for k, _ in ranked_items[:n_good]]
            bad_keys = [k for k, _ in ranked_items[n_good:]] or [k for k, _ in ranked_items[:n_good]]

            good_counts = {s: {v: prior for v in options[s]} for s in step_order}
            bad_counts = {s: {v: prior for v in options[s]} for s in step_order}

            for key in good_keys:
                for s, v in key:
                    if s in good_counts and v in good_counts[s]:
                        good_counts[s][v] += 1.0
            for key in bad_keys:
                for s, v in key:
                    if s in bad_counts and v in bad_counts[s]:
                        bad_counts[s][v] += 1.0
            return good_counts, bad_counts

        def _score_cfg_with_stats(cfg: Mapping[str, Any], good_counts, bad_counts) -> float:
            score = 0.0
            for s in step_order:
                gv = good_counts[s][cfg[s]]
                bv = bad_counts[s][cfg[s]]
                gtot = sum(good_counts[s].values())
                btot = sum(bad_counts[s].values())
                pg = gv / gtot
                pb = bv / btot
                score += log(max(pg, 1e-12)) - log(max(pb, 1e-12))
            return score

        while budget_left():
            if len(cache) < 2:
                cfg = _random_unseen_cfg(options, step_order, rng, cache)
                if cfg is None:
                    break
                eval_cfg(cfg)
                continue

            good_counts, bad_counts = _build_stats()
            best_cfg = None
            best_model_score = -np.inf
            for _ in range(n_candidates):
                cand = {}
                for s in step_order:
                    vals = options[s]
                    probs = np.array([good_counts[s][v] for v in vals], dtype=float)
                    probs = probs / probs.sum()
                    cand[s] = vals[int(rng.choice(len(vals), p=probs))]
                key = _cfg_key(cand, step_order)
                if key in cache:
                    continue
                mscore = _score_cfg_with_stats(cand, good_counts, bad_counts)
                if mscore > best_model_score:
                    best_cfg = cand
                    best_model_score = mscore

            if best_cfg is None:
                best_cfg = _random_unseen_cfg(options, step_order, rng, cache)
                if best_cfg is None:
                    break
            eval_cfg(best_cfg)

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

        while budget_left():
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
