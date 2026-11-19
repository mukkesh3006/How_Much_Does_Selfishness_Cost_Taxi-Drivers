"""Build one experiment instance (city + zones + demand + drivers) and evaluate it: PoA estimates and exact values."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

from src.network import generate_city_network
from demand import generate_demand, compute_herfindahl_index
from optimizer import optimize_welfare
from game.model import TaxiGame
from game.dynamics import best_response_dynamics, initial_profile
from game.enumeration import enumerate_equilibria


@dataclass
class InstanceConfig:
    topology: str = "grid"                 # grid | hub_spoke | scale_free
    num_nodes: int = 25
    num_zones: int = 8
    num_drivers: int = 12
    regime: str = "concentrated"           # uniform | concentrated | degree_correlated | power_law
    kappa: float = 0.3                     # travel-cost scale: lam = kappa * (total_demand/num_zones) / mean_dist
    lam: float | None = None               # explicit override of the calibrated lam (kappa then ignored)
    outside_option: bool = True            # add an "idle" action worth exactly 0 (reservation utility)
    seed: int = 0
    total_demand: float = 100.0            # held fixed so that concentration sweeps do not change total demand
    hotspot_fraction: float = 0.2
    hotspot_share: float = 0.8
    power_law_alpha: float = 1.0
    driver_start: str = "random"           # random: each driver at a random node | depot: all at one node
    min_weight: float = 1.0
    max_weight: float = 10.0
    topology_kwargs: dict = field(default_factory=dict)


def _seeds(seed: int, k: int = 5) -> list[int]:
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(k)]


def build_game(cfg: InstanceConfig):
    """Returns (TaxiGame, meta). Graph, zone choice, demand and driver locations use independent seed streams."""
    if cfg.num_zones > cfg.num_nodes:
        raise ValueError("num_zones cannot exceed num_nodes")
    s_graph, s_zone, s_dem, s_drv, _ = _seeds(cfg.seed)
    city = generate_city_network(cfg.topology, num_nodes=cfg.num_nodes, seed=s_graph,
                                 min_weight=cfg.min_weight, max_weight=cfg.max_weight, **cfg.topology_kwargs)
    rng = np.random.default_rng(s_zone)
    zone_nodes = sorted(int(z) for z in rng.choice(cfg.num_nodes, size=cfg.num_zones, replace=False))
    dkw: dict[str, Any] = dict(seed=s_dem, total_demand=cfg.total_demand)
    if cfg.regime == "concentrated":
        dkw.update(hotspot_fraction=cfg.hotspot_fraction, hotspot_share=cfg.hotspot_share)
    elif cfg.regime == "power_law":
        dkw.update(power_law_alpha=cfg.power_law_alpha)
    elif cfg.regime == "degree_correlated":
        dkw.update(degrees={z: city.graph.degree(z) for z in zone_nodes})
    demand = generate_demand(zone_nodes, regime=cfg.regime, **dkw)
    rng = np.random.default_rng(s_drv)
    if cfg.driver_start == "random":
        driver_nodes = [int(x) for x in rng.integers(0, cfg.num_nodes, size=cfg.num_drivers)]
    elif cfg.driver_start == "depot":
        driver_nodes = [int(rng.integers(0, cfg.num_nodes))] * cfg.num_drivers
    else:
        raise ValueError(f"unknown driver_start '{cfg.driver_start}'")
    D = np.array([demand[z] for z in zone_nodes])
    C = np.array([[city.get_travel_cost(l, z) for z in zone_nodes] for l in driver_nodes])
    lam = cfg.lam if cfg.lam is not None else calibrated_lam(city, cfg)
    labels = list(zone_nodes)
    if cfg.outside_option:      # idle = a dummy zone with D=0 and c=0 for everyone: utility exactly 0
        D = np.append(D, 0.0)
        C = np.hstack([C, np.zeros((C.shape[0], 1))])
        labels.append(IDLE)
        demand = {**demand, IDLE: 0.0}
    game = TaxiGame(D, C, lam)
    return game, dict(city=city, zone_nodes=labels, driver_nodes=driver_nodes, demand=demand, lam=lam)


IDLE = "idle"


def mean_distance(city) -> float:
    n = city.graph.num_nodes
    return float(sum(city.get_travel_cost(u, v) for u in range(n) for v in range(n) if u != v) / (n * (n - 1)))


def calibrated_lam(city, cfg: "InstanceConfig") -> float:
    """lam such that an average-length trip costs kappa x (average zone demand).

    Dividing by the instance's mean shortest-path distance makes topologies comparable on 'how costly is travel
    relative to the prize' (otherwise hub-and-spoke looks different merely because its paths are short), and
    dividing by total_demand/num_zones makes it independent of driver density and of demand concentration.
    """
    return cfg.kappa * (cfg.total_demand / cfg.num_zones) / mean_distance(city)


def central_optimum(game: TaxiGame, meta: dict, method: str = "auto", seed: int = 0):
    """W_OPT via the teammates' optimizer. Cross-checked against game.welfare so a solver/objective mismatch
    fails loudly instead of silently corrupting PoA. Returns (welfare, profile as zone indices, method_used)."""
    city, zones, locs = meta["city"], meta["zone_nodes"], meta["driver_nodes"]
    res = optimize_welfare(list(range(game.n)), {i: l for i, l in enumerate(locs)}, zones, meta["demand"],
                           lambda u, v: 0.0 if v == IDLE else city.get_travel_cost(u, v), game.lam,
                           method=method, seed=seed)
    zi = {z: k for k, z in enumerate(zones)}
    prof = np.array([zi[res.assignment[i]] for i in range(game.n)])
    w = game.welfare(prof)
    if abs(w - res.welfare) > 1e-6 * max(1.0, abs(w)):
        raise RuntimeError(f"optimizer welfare {res.welfare} != recomputed {w} (method {res.method_used})")
    return w, prof, res.method_used


def run_instance(cfg: InstanceConfig, n_runs: int = 20, schedules=("round_robin", "random"),
                 inits=("random", "greedy", "nearest"), opt_method: str = "auto",
                 enumerate_limit: int = 300_000) -> dict:
    game, meta = build_game(cfg)
    w_opt, opt_prof, opt_used = central_optimum(game, meta, opt_method, seed=cfg.seed)
    rng = np.random.default_rng(_seeds(cfg.seed)[4])
    W, statuses, updates, active, maxload, idle_frac, trav_frac = [], [], [], [], [], [], []
    idle_i = game.m - 1 if cfg.outside_option else None
    for r in range(n_runs):
        init = initial_profile(game, inits[r % len(inits)], rng)
        out = best_response_dynamics(game, init, rng, schedule=schedules[(r // len(inits)) % len(schedules)])
        statuses.append(out.status); updates.append(out.updates)
        if out.status == "converged":
            W.append(out.welfare)
            loads = game.loads(out.profile)
            real = np.delete(loads, idle_i) if idle_i is not None else loads
            active.append(int((real > 0).sum())); maxload.append(int(real.max()))
            idle_frac.append(float(loads[idle_i] / game.n) if idle_i is not None else 0.0)
            trav_frac.append(float(game.lam * game.cost[np.arange(game.n), out.profile].sum() / max(out.welfare + game.lam * game.cost[np.arange(game.n), out.profile].sum(), 1e-12)))
    W = np.array(W)
    opt_loads = game.loads(opt_prof)
    idle = game.m - 1 if cfg.outside_option else None

    def eff(w):     # efficiency = W_EQ / W_OPT in [0, 1]; defined as 1 when W_OPT = 0 (nothing worth doing)
        return np.full_like(w, 1.0) if w_opt <= 1e-12 else w / w_opt

    E = eff(W)
    out = dict(
        **{k: v for k, v in asdict(cfg).items() if k != "topology_kwargs"},
        lam_used=game.lam, herfindahl=compute_herfindahl_index({z: d for z, d in meta["demand"].items() if z != IDLE}),
        rho=cfg.num_drivers / cfg.num_zones,
        w_opt=w_opt, opt_method=opt_used,
        w_eq_min=float(W.min()) if len(W) else np.nan, w_eq_max=float(W.max()) if len(W) else np.nan,
        w_eq_mean=float(W.mean()) if len(W) else np.nan,
        # primary metrics (welfare is >= 0 at every equilibrium when the outside option exists)
        eff_mean=float(E.mean()) if len(W) else np.nan,
        eff_worst=float(E.min()) if len(W) else np.nan,     # sampled worst equilibrium: upper bound on 1/PoA
        eff_best=float(E.max()) if len(W) else np.nan,
        loss_mean=float(1 - E.mean()) if len(W) else np.nan,
        gap_mean=float(w_opt - W.mean()) if len(W) else np.nan,
        poa_sampled_worst=float(1 / E.min()) if len(W) and E.min() > 1e-12 else np.nan,   # lower bound on true PoA
        n_converged=int(sum(s == "converged" for s in statuses)), n_runs=n_runs,
        n_cycle=int(sum(s == "cycle" for s in statuses)), n_limit=int(sum(s == "limit" for s in statuses)),
        mean_updates=float(np.mean(updates)),
        # coverage / congestion diagnostics (idle excluded from 'zones')
        eq_active_zones=float(np.mean(active)) if active else np.nan,
        opt_active_zones=int((np.delete(opt_loads, idle) > 0).sum()) if idle is not None else int((opt_loads > 0).sum()),
        eq_max_load=float(np.mean(maxload)) if maxload else np.nan,
        opt_max_load=int(np.delete(opt_loads, idle).max()) if idle is not None else int(opt_loads.max()),
        eq_idle_frac=float(np.mean(idle_frac)) if idle_frac else np.nan,
        opt_idle_frac=float(opt_loads[idle] / game.n) if idle is not None else 0.0,
        eq_travel_frac=float(np.mean(trav_frac)) if trav_frac else np.nan,
    )
    if game.m ** game.n <= enumerate_limit:
        en = enumerate_equilibria(game, max_profiles=enumerate_limit)
        out.update(exact_n_nash=len(en.nash_welfare), exact_poa=en.poa(), exact_pos=en.pos(),
                   exact_eff_worst=(en.worst_nash_welfare / en.opt_welfare) if en.opt_welfare > 1e-12 else 1.0,
                   exact_eff_best=(en.best_nash_welfare / en.opt_welfare) if en.opt_welfare > 1e-12 else 1.0,
                   exact_w_opt=en.opt_welfare, exact_w_phi_max=en.phi_max_welfare,
                   opt_matches_enum=bool(abs(en.opt_welfare - w_opt) < 1e-6 * max(1.0, abs(w_opt))))
    return out
