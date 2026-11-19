import itertools
import numpy as np
import pytest

from game import TaxiGame, best_response_dynamics, enumerate_equilibria, initial_profile, InstanceConfig, run_instance, build_game
from optimizer import evaluate_welfare, optimize_welfare_exact


def rand_game(seed, n=None, m=None, lam=None):
    r = np.random.default_rng(seed)
    n = n or int(r.integers(2, 7)); m = m or int(r.integers(2, 5))
    D = r.uniform(0, 40, m) * (r.random(m) > 0.1)
    C = r.uniform(0, 12, (n, m))
    return TaxiGame(D, C, lam if lam is not None else float(r.choice([0.0, 0.3, 1.0, 3.0])))


def test_hand_example_poa():
    # two drivers, same spot, zero travel cost. D = [10, 4]. Unique Nash: both at zone 0 (5 each > 4). OPT: split.
    g = TaxiGame([10.0, 4.0], np.zeros((2, 2)), 1.0)
    en = enumerate_equilibria(g)
    assert en.opt_welfare == 14.0
    assert sorted(map(tuple, en.nash_profiles.tolist())) == [(0, 0)]
    assert en.poa() == pytest.approx(1.4) and en.pos() == pytest.approx(1.4)


@pytest.mark.parametrize("seed", range(40))
def test_exact_potential_identity(seed):
    g = rand_game(seed); r = np.random.default_rng(seed + 1000)
    a = r.integers(0, g.m, g.n)
    for _ in range(30):
        i = int(r.integers(g.n)); z = int(r.integers(g.m))
        b = a.copy(); b[i] = z
        du = g.utilities(b)[i] - g.utilities(a)[i]
        dphi = g.potential(b) - g.potential(a)
        assert du == pytest.approx(dphi, abs=1e-9)
        a = b


@pytest.mark.parametrize("seed", range(40))
def test_welfare_is_sum_of_utilities_and_matches_optimizer_module(seed):
    g = rand_game(seed); r = np.random.default_rng(seed)
    a = r.integers(0, g.m, g.n)
    assert g.welfare(a) == pytest.approx(g.utilities(a).sum(), abs=1e-9)
    drivers = list(range(g.n)); locs = {i: f"L{i}" for i in drivers}; zones = [f"Z{k}" for k in range(g.m)]
    dem = {zones[k]: g.demand[k] for k in range(g.m)}
    cost = {(f"L{i}", zones[k]): g.cost[i, k] for i in drivers for k in range(g.m)}
    w, *_ = evaluate_welfare({i: zones[a[i]] for i in drivers}, locs, dem, cost, g.lam)
    assert w == pytest.approx(g.welfare(a), abs=1e-9)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("schedule", ["round_robin", "random"])
def test_brd_always_converges_to_nash_and_potential_increases(seed, schedule):
    g = rand_game(seed, n=int(np.random.default_rng(seed).integers(2, 12)))
    rng = np.random.default_rng(seed)
    res = best_response_dynamics(g, initial_profile(g, "random", rng), rng, schedule=schedule, track_potential=True)
    assert res.status == "converged"                       # potential game: no cycles, no limit hits
    assert g.is_pure_nash(res.profile)
    assert all(b > a for a, b in zip(res.potential_trace, res.potential_trace[1:]))
    assert res.welfare == pytest.approx(g.welfare(res.profile))


@pytest.mark.parametrize("seed", range(40))
def test_enumeration_matches_bruteforce_nash_and_opt(seed):
    g = rand_game(seed, n=int(np.random.default_rng(seed).integers(2, 6)))
    en = enumerate_equilibria(g, chunk=37)  # odd chunk size exercises chunk boundaries
    brute = [p for p in itertools.product(range(g.m), repeat=g.n) if g.is_pure_nash(np.array(p), eps=1e-9)]
    assert sorted(map(tuple, en.nash_profiles.tolist())) == sorted(brute)
    assert len(brute) >= 1                                  # existence
    assert max(g.welfare(np.array(p)) for p in itertools.product(range(g.m), repeat=g.n)) == pytest.approx(en.opt_welfare)
    assert g.is_pure_nash(en.phi_max_profile)               # potential maximizer is an equilibrium
    # welfare of every equilibrium <= OPT  =>  PoA, PoS >= 1 when defined
    assert en.nash_welfare.max() <= en.opt_welfare + 1e-9
    if en.worst_nash_welfare > 0:
        assert 1 - 1e-9 <= en.pos() <= en.poa() + 1e-9
    # BRD outcomes must be members of the enumerated Nash set
    rng = np.random.default_rng(seed)
    nash = set(map(tuple, en.nash_profiles.tolist()))
    for _ in range(10):
        r = best_response_dynamics(g, initial_profile(g, "random", rng), rng)
        assert tuple(r.profile.tolist()) in nash


@pytest.mark.parametrize("seed", range(25))
def test_exact_optimizer_agrees_with_game_enumeration(seed):
    g = rand_game(seed, n=int(np.random.default_rng(seed).integers(2, 5)))
    drivers = list(range(g.n)); locs = {i: f"L{i}" for i in drivers}; zones = [f"Z{k}" for k in range(g.m)]
    dem = {zones[k]: g.demand[k] for k in range(g.m)}
    cost = {(f"L{i}", zones[k]): g.cost[i, k] for i in drivers for k in range(g.m)}
    res = optimize_welfare_exact(drivers, locs, zones, dem, cost, lambda_cost=g.lam)
    assert res.welfare == pytest.approx(enumerate_equilibria(g).opt_welfare, abs=1e-9)


@pytest.mark.parametrize("topo,kw", [("grid", {}), ("hub_spoke", {"num_hubs": 2}), ("scale_free", {"m": 2})])
@pytest.mark.parametrize("regime", ["uniform", "concentrated", "power_law", "degree_correlated"])
def test_run_instance_smoke_and_consistency(topo, kw, regime):
    cfg = InstanceConfig(topology=topo, topology_kwargs=kw, regime=regime, num_nodes=16, num_zones=4,
                         num_drivers=5, seed=3)
    out = run_instance(cfg, n_runs=12)
    assert out["n_converged"] == 12 and out["n_cycle"] == 0 and out["n_limit"] == 0
    assert out["opt_matches_enum"]                          # optimizer == brute-force optimum
    assert out["w_opt"] >= out["w_eq_max"] - 1e-9
    assert out["eff_best"] <= 1 + 1e-9
    assert out["exact_poa"] >= out["exact_pos"] - 1e-9 >= 1 - 2e-9 or np.isnan(out["exact_poa"])
    # sampled equilibria are real equilibria => sampled worst PoA can't exceed the true PoA
    if not np.isnan(out["exact_poa"]) and not np.isnan(out["poa_sampled_worst"]):
        assert out["poa_sampled_worst"] <= out["exact_poa"] + 1e-9


def test_run_instance_is_deterministic():
    cfg = InstanceConfig(num_nodes=16, num_zones=4, num_drivers=5, seed=11)
    assert run_instance(cfg, n_runs=6) == run_instance(cfg, n_runs=6)


def test_ilp_path_matches_enumeration_on_larger_than_exact_threshold():
    # (4 zones + idle)^8 = 390625 > 50_000 so method="auto" takes the ILP branch
    cfg = InstanceConfig(num_nodes=16, num_zones=4, num_drivers=8, seed=5)
    out = run_instance(cfg, n_runs=6, enumerate_limit=400_000)
    assert out["opt_method"].startswith("ilp") and out["opt_matches_enum"]


# ---------- outside option, efficiency metric, lambda calibration ----------
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("kappa", [0.1, 0.6, 3.0])
def test_outside_option_makes_equilibrium_welfare_nonnegative(seed, kappa):
    cfg = InstanceConfig(num_nodes=16, num_zones=4, num_drivers=5, seed=seed, kappa=kappa, regime="power_law")
    out = run_instance(cfg, n_runs=9)
    assert out["w_eq_min"] >= -1e-9 and out["w_opt"] >= -1e-9
    assert -1e-9 <= out["eff_worst"] <= out["eff_mean"] <= out["eff_best"] <= 1 + 1e-9
    assert out["exact_eff_worst"] <= out["eff_worst"] + 1e-9      # sampled equilibria can only look better than the worst


def test_idle_is_a_zero_utility_dummy_zone():
    g, meta = build_game(InstanceConfig(num_nodes=16, num_zones=4, num_drivers=5, seed=2))
    assert g.m == 5 and g.demand[-1] == 0 and (g.cost[:, -1] == 0).all()
    a = np.full(g.n, g.m - 1)
    assert g.utilities(a).tolist() == [0.0] * g.n and g.welfare(a) == 0.0


def test_calibrated_lambda_is_independent_of_density_and_concentration():
    base = dict(num_nodes=16, num_zones=4, seed=4, kappa=0.3)
    lams = {build_game(InstanceConfig(num_drivers=n, regime=r, **base))[0].lam
            for n in (2, 6, 12) for r in ("uniform", "power_law", "concentrated")}
    assert max(lams) - min(lams) < 1e-12
    explicit = build_game(InstanceConfig(num_drivers=3, lam=0.77, **base))[0].lam
    assert explicit == 0.77


def test_without_outside_option_still_supported():
    out = run_instance(InstanceConfig(num_nodes=16, num_zones=4, num_drivers=5, seed=1, outside_option=False), n_runs=6)
    assert out["opt_matches_enum"] and out["n_cycle"] == 0


from game.benchmark import hungarian_optimum


@pytest.mark.parametrize("seed", range(40))
def test_hungarian_optimum_equals_enumeration_with_idle(seed):
    r = np.random.default_rng(seed); n = int(r.integers(1, 6)); m = int(r.integers(1, 4))
    D = np.append(r.uniform(0, 40, m), 0.0); C = np.hstack([r.uniform(0, 12, (n, m)), np.zeros((n, 1))])
    g = TaxiGame(D, C, float(r.choice([0.0, 0.3, 1.0, 3.0])))
    w, prof = hungarian_optimum(g)
    assert w == pytest.approx(enumerate_equilibria(g).opt_welfare, abs=1e-9)
    assert g.welfare(prof) == pytest.approx(w, abs=1e-9)
