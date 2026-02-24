from automl_aco.search.ordering import OrderSearchConfig, all_topological_orders, propose_orders


def _is_valid(order, constraints):
    pos = {s: i for i, s in enumerate(order)}
    return all(pos[a] < pos[b] for a, b in constraints)


def test_all_topological_orders_respect_constraints():
    steps = ("imputation", "encoding", "scaling")
    constraints = (("imputation", "encoding"),)
    orders = all_topological_orders(steps, constraints)
    assert len(orders) >= 1
    for order in orders:
        assert _is_valid(order, constraints)


def test_propose_orders_fixed():
    cfg = OrderSearchConfig(
        steps=("imputation", "encoding", "scaling"),
        constraints=(("imputation", "encoding"),),
        max_orders=5,
        strategy="fixed",
        seed=42,
    )
    orders = propose_orders(cfg)
    assert orders == [["imputation", "encoding", "scaling"]]


def test_propose_orders_random_is_deterministic_with_seed():
    cfg = OrderSearchConfig(
        steps=("imputation", "encoding", "scaling", "feature_selection"),
        constraints=(("imputation", "encoding"),),
        max_orders=4,
        strategy="random",
        seed=7,
    )
    orders_a = propose_orders(cfg)
    orders_b = propose_orders(cfg)
    assert orders_a == orders_b
