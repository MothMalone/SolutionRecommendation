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
