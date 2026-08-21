import pytest
from automl_aco.search.optimizers import search_pipelines_with_optimizer


def _dummy_eval_factory(options):
    def _eval(configs):
        results = []
        for cfg in configs:
            score = 0.0
            for step, vals in options.items():
                score += vals.index(cfg[step]) + 1
            results.append((cfg, score))
        if not results:
            return None, float("nan"), [], []
        unsorted = results.copy()
        results.sort(key=lambda x: x[1], reverse=True)
        best_cfg, best_score = results[0]
        return best_cfg, best_score, results, unsorted

    return _eval


def test_random_optimizer_budget_and_topk():
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}
    evaluate_fn = _dummy_eval_factory(options)
    top, all_results, history = search_pipelines_with_optimizer(
        optimizer="random",
        options=options,
        evaluate_fn=evaluate_fn,
        sample_budget=4,
        seed=42,
        n_pipelines=2,
        verbose=False,
    )
    assert len(top) == 2
    assert len(all_results) <= 4
    assert len(history) >= 1


def test_ga_optimizer_runs():
    options = {"imputation": ["none", "mean", "knn"], "scaling": ["none", "standard", "robust"]}
    evaluate_fn = _dummy_eval_factory(options)
    top, all_results, history = search_pipelines_with_optimizer(
        optimizer="ga",
        options=options,
        evaluate_fn=evaluate_fn,
        sample_budget=12,
        seed=7,
        n_pipelines=3,
        verbose=False,
    )
    assert len(top) >= 1
    assert len(all_results) >= 1
    assert len(history) >= 1


def test_tpe_optimizer_runs():
    options = {"imputation": ["none", "mean", "knn"], "scaling": ["none", "standard", "robust"]}
    evaluate_fn = _dummy_eval_factory(options)
    top, all_results, history = search_pipelines_with_optimizer(
        optimizer="tpe",
        options=options,
        evaluate_fn=evaluate_fn,
        sample_budget=12,
        seed=11,
        n_pipelines=3,
        verbose=False,
    )
    assert len(top) >= 1
    assert len(all_results) >= 1
    assert len(history) >= 1


def test_beam_optimizer_runs():
    options = {"imputation": ["none", "mean", "knn"], "scaling": ["none", "standard", "robust"]}
    evaluate_fn = _dummy_eval_factory(options)
    top, all_results, history = search_pipelines_with_optimizer(
        optimizer="beam",
        options=options,
        evaluate_fn=evaluate_fn,
        sample_budget=12,
        seed=13,
        n_pipelines=3,
        verbose=False,
    )
    assert len(top) >= 1
    assert len(all_results) >= 1
    assert len(history) >= 1


def test_exhaustive_is_exact_when_budget_covers_space():
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard", "robust"]}
    evaluate_fn = _dummy_eval_factory(options)
    # Search space size = 2 * 3 = 6
    top, all_results, history = search_pipelines_with_optimizer(
        optimizer="exhaustive",
        options=options,
        evaluate_fn=evaluate_fn,
        sample_budget=6,
        seed=5,
        n_pipelines=2,
        verbose=False,
    )
    assert len(all_results) == 6
    assert len(top) == 2
    assert len(history) >= 1


@pytest.mark.parametrize(
    "optimizer", ["random", "ga", "sa", "greedy", "mcts", "beam", "tpe", "exhaustive"])
def test_budget_larger_than_space_terminates(optimizer):
    """Every optimizer must stop when the budget exceeds the number of distinct configs.

    The evaluation cache is keyed by configuration, so it saturates at the size of the space.
    Loops written as `while len(cache) < sample_budget` then never exit -- not slowly, never.
    DEFAULT_PIPELINE_OPTIONS spans 1800 configs and real budgets sit well below that, so this
    exact shape is latent in production rather than observed there; it was reached from the
    tests. The sibling guard in the same fix is not latent: a configuration whose evaluation
    raises is never cached, so it is redrawn forever, and "onehot widens the frame, then
    outlier removal deletes every row" is a failure we have already hit on real datasets.
    """
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}   # 4 configs
    evaluate_fn = _dummy_eval_factory(options)
    top, all_results, history = search_pipelines_with_optimizer(
        optimizer=optimizer,
        options=options,
        evaluate_fn=evaluate_fn,
        sample_budget=50,          # way past the 4 that exist
        seed=3,
        n_pipelines=2,
        verbose=False,
    )
    assert len(all_results) <= 4, "evaluated a config twice, or invented one"
    assert len(top) >= 1


@pytest.mark.parametrize("optimizer", ["random", "ga", "sa", "greedy", "mcts", "beam", "tpe"])
def test_failing_evaluations_do_not_hang(optimizer):
    """A config that always fails is never cached, so nothing stops it being redrawn.

    Here every evaluation raises, which is the limit case of a real one: onehot widens a frame
    past what the proxy can fit, or outlier removal empties it. The search must give up and
    return, not spin. 4^4 = 256 configs, so the space cap alone cannot save it.
    """
    def _always_fails(configs):
        raise RuntimeError("proxy fit failed")

    options = {f"step{i}": ["a", "b", "c", "d"] for i in range(4)}
    top, all_results, _history = search_pipelines_with_optimizer(
        optimizer=optimizer,
        options=options,
        evaluate_fn=_always_fails,
        sample_budget=20,
        seed=17,
        n_pipelines=3,
        verbose=False,
    )
    assert all_results == []
    assert top == []


@pytest.mark.parametrize("optimizer", ["random", "ga", "sa", "greedy", "mcts", "beam", "tpe"])
def test_budget_is_respected_when_space_is_larger(optimizer):
    """The cap must not shrink a normal search: 4^4 = 256 configs, budget 20."""
    vals = ["a", "b", "c", "d"]
    options = {f"step{i}": list(vals) for i in range(4)}
    evaluate_fn = _dummy_eval_factory(options)
    _top, all_results, _history = search_pipelines_with_optimizer(
        optimizer=optimizer,
        options=options,
        evaluate_fn=evaluate_fn,
        sample_budget=20,
        seed=11,
        n_pipelines=3,
        verbose=False,
    )
    assert len(all_results) == 20
