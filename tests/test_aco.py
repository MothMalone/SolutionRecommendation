import numpy as np

from automl_aco.search.aco import search_pipelines_aco


def _make_eta(options):
    return {step: np.ones(len(vals), dtype=float) for step, vals in options.items()}


def _dummy_evaluate_factory(options):
    def _evaluate(configs):
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

    return _evaluate


def test_aco_returns_k_pipelines():
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}
    eta = _make_eta(options)
    evaluate_fn = _dummy_evaluate_factory(options)

    final, _unsorted = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate_fn,
        eta=eta,
        n_pipelines=2,
        n_ants=2,
        n_iterations=2,
        seed=42,
    )

    assert len(final) == 2
