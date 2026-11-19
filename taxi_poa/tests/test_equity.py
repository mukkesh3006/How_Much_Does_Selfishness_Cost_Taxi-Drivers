"""Tests for experiment E (equity + convergence speed), game/sweep_equity.py."""
import numpy as np
import pytest

from game.sweep_equity import gini, cv, zero_share, equity_instance, configs
from game.instance import InstanceConfig, run_instance


# ---- inequality measures -------------------------------------------------------------------------------------------
def test_gini_equal_is_zero():
    assert gini([3.0, 3.0, 3.0, 3.0]) == pytest.approx(0.0)


def test_gini_one_takes_all_is_n_minus_1_over_n():
    for n in (2, 5, 10):
        x = np.zeros(n); x[-1] = 7.0
        assert gini(x) == pytest.approx((n - 1) / n)


def test_gini_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    x = rng.random(25) * 10
    brute = np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2 * x.mean())
    assert gini(x) == pytest.approx(brute)


def test_gini_scale_and_order_invariant():
    x = np.array([0.0, 1.0, 4.0, 9.0])
    assert gini(10 * x) == pytest.approx(gini(x))
    assert gini(x[::-1]) == pytest.approx(gini(x))


def test_gini_all_zero_defined_as_zero():
    assert gini([0, 0, 0]) == 0.0 and cv([0, 0, 0]) == 0.0


def test_cv_and_zero_share():
    assert cv([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert cv([0.0, 4.0]) == pytest.approx(1.0)          # std 2, mean 2
    assert zero_share([0.0, 0.0, 1.0, 2.0]) == pytest.approx(0.5)


# ---- the experiment replays the main sweep exactly ------------------------------------------------------------------
@pytest.mark.parametrize("topo,kw,alpha,n,seed", [("grid", {}, 1.0, 10, 0), ("hub_spoke", {"num_hubs": 3}, 1.5, 20, 3),
                                                  ("scale_free", {"m": 2}, 0.0, 40, 7), ("grid", {}, 2.0, 5, 11)])
def test_equity_replays_run_instance(topo, kw, alpha, n, seed):
    cfg = InstanceConfig(topology=topo, topology_kwargs=kw, num_nodes=36, num_zones=10, num_drivers=n,
                         regime="power_law", power_law_alpha=alpha, seed=seed)
    e = equity_instance(cfg)
    r = run_instance(cfg, n_runs=20, enumerate_limit=0)
    assert e["loss_mean"] == pytest.approx(r["loss_mean"], abs=1e-12)     # same instances, same dynamics runs
    assert e["w_opt"] == pytest.approx(r["w_opt"], rel=1e-9)             # Hungarian == optimizer module
    assert e["n_converged"] == r["n_converged"] == 20


def test_equity_basic_invariants():
    cfg = InstanceConfig(topology="grid", num_nodes=36, num_zones=10, num_drivers=20, regime="power_law",
                         power_law_alpha=1.0, seed=2)
    e = equity_instance(cfg)
    assert 0 <= e["eq_gini"] < 1 and 0 <= e["opt_gini"] < 1
    assert e["eq_min_u"] >= -1e-9                       # the idle option guarantees U_i >= 0 at any equilibrium
    assert e["opt_min_u"] >= -1e-9                      # the planner never makes a driver worse off than idle
    assert e["eq_mean_u"] <= e["opt_mean_u"] + 1e-9     # mean utility = W/n, and W_EQ <= W_OPT
    assert e["updates_per_driver"] > 0 and e["sweeps_rr"] >= 2   # at least one moving sweep + one confirming sweep


def test_zero_share_reflects_idling_of_the_planner():
    # abundant drivers: the optimum idles many of them, so its zero-earner share is large; equilibrium idles fewer
    cfg = InstanceConfig(topology="grid", num_nodes=36, num_zones=10, num_drivers=40, regime="power_law",
                         power_law_alpha=1.0, seed=0)
    e = equity_instance(cfg)
    assert e["opt_zero_share"] > e["eq_zero_share"]
    assert e["opt_zero_share"] >= 1 - 10 / 40 - 1e-9    # at most one driver per zone earns under the optimum


def test_config_grid_matches_experiment_A():
    from game.sweep import configs as A_configs
    mine = [(c.topology, c.power_law_alpha, c.num_drivers, c.seed) for c in configs(3)]
    theirs = [(c.topology, c.power_law_alpha, c.num_drivers, c.seed) for c, _ in A_configs("A", 3)]
    assert mine == theirs
