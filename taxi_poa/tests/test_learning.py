import numpy as np
import pytest
from game import InstanceConfig, build_game
from game.learning import run_learning
from game.benchmark import hungarian_optimum


def small(seed=0, n=6, zones=5, **kw):
    g, _ = build_game(InstanceConfig(num_nodes=16, num_zones=zones, num_drivers=n, seed=seed, regime="power_law", **kw))
    return g


@pytest.mark.parametrize("mode", ["planner", "selfish"])
def test_welfare_never_exceeds_optimum_and_is_deterministic(mode):
    g = small(1)
    a = run_learning(g, mode, 40, np.random.default_rng(5)); b = run_learning(g, mode, 40, np.random.default_rng(5))
    assert (a.welfare <= a.w_opt + 1e-9).all() and (a.welfare >= -1e-9).all()
    assert np.array_equal(a.welfare, b.welfare)


def test_planner_with_ucb_learns_the_optimum():
    g = small(2)
    tr = run_learning(g, "planner", 400, np.random.default_rng(0), c=0.5, noise_k=50.0)
    assert tr.welfare[-50:].mean() >= 0.97 * tr.w_opt


def test_first_epoch_explores_via_optimism():
    # all zones are believed to be worth 'cap', so with at least m drivers and cheap travel every zone gets visited early
    g = small(3, n=8, zones=4)
    tr = run_learning(g, "planner", 6, np.random.default_rng(0))
    assert (tr.visits > 0).all()


def test_visits_bounded_by_epochs_and_zero_noise_gives_exact_beliefs():
    g = small(4)
    tr = run_learning(g, "selfish", 30, np.random.default_rng(1), noise_k=1e9)   # essentially no noise
    assert (tr.visits <= 30).all()
    assert tr.est_error < 1e-3 or np.isnan(tr.est_error)


def test_selfish_final_profile_close_to_full_information_equilibrium_welfare():
    g = small(5, n=6, zones=4)
    tr = run_learning(g, "selfish", 300, np.random.default_rng(0), c=0.5, noise_k=50.0)
    from game import best_response_dynamics, initial_profile
    rng = np.random.default_rng(0)
    eq = np.mean([best_response_dynamics(g, initial_profile(g, "random", rng), rng).welfare for _ in range(30)])
    assert tr.welfare[-50:].mean() >= 0.8 * min(eq, tr.w_opt)
