"""Experiment E: who earns what? Equity of the selfish equilibrium vs. the coordinated optimum, and speed of convergence.

    python -m game.sweep_equity [--seeds 100] [--jobs 2] [--outdir results]   ->  results/exp_E.csv

Why this experiment exists.  The idea review promised two metrics that the main sweeps do not report:
  * "number of best-response iterations to convergence"                       -> columns updates_per_driver, sweeps_rr
  * "variance in individual driver utility, to check whether the centralized
     optimum is efficient but inequitable"                                      -> columns *_gini, *_cv, *_zero_share

The instances are exactly those of experiment A (same grid, same seeds) and the dynamics runs are replayed with the
same random-number stream as game.instance.run_instance, so `loss_mean` here must equal `loss_mean` in exp_A.csv
(this is asserted in tests/test_equity.py).  Utilities are the model's own U_i (no side payments): the planner picks
WHO goes WHERE; it does not redistribute fares.
"""
from __future__ import annotations

import argparse, itertools, os, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from game.benchmark import hungarian_optimum
from game.dynamics import best_response_dynamics, initial_profile
from game.instance import InstanceConfig, build_game, _seeds

TOPO = {"grid": {}, "hub_spoke": {"num_hubs": 3}, "scale_free": {"m": 2}}
N_NODES, M_ZONES = 36, 10
SCHEDULES = ("round_robin", "random")
INITS = ("random", "greedy", "nearest")


def gini(x) -> float:
    """Gini coefficient of a nonnegative vector: 0 = everybody earns the same, (n-1)/n = one driver earns everything.
    Defined as 0 when nobody earns anything."""
    x = np.sort(np.asarray(x, dtype=float))
    n, s = len(x), x.sum()
    if n == 0 or s <= 1e-12:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * s))


def cv(x) -> float:
    """Coefficient of variation std/mean (scale-free version of 'variance in driver utility'); 0 if mean is 0."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return float(x.std() / m) if m > 1e-12 else 0.0


def zero_share(x, tol: float = 1e-9) -> float:
    """Fraction of drivers whose utility is (numerically) zero, i.e. who earn nothing this epoch."""
    return float((np.asarray(x) <= tol).mean())


def equity_instance(cfg: InstanceConfig, n_runs: int = 20) -> dict:
    """Replays run_instance's dynamics runs and records per-driver utility statistics and convergence speed."""
    game, _ = build_game(cfg)
    w_opt, opt_prof = hungarian_optimum(game)
    u_opt = game.utilities(opt_prof)
    rng = np.random.default_rng(_seeds(cfg.seed)[4])         # same stream as run_instance
    W, g, c, z, mn, upd, swp = [], [], [], [], [], [], []
    for r in range(n_runs):
        init = initial_profile(game, INITS[r % len(INITS)], rng)
        sched = SCHEDULES[(r // len(INITS)) % len(SCHEDULES)]
        out = best_response_dynamics(game, init, rng, schedule=sched)
        if out.status != "converged":
            continue
        u = game.utilities(out.profile)
        W.append(out.welfare); g.append(gini(u)); c.append(cv(u)); z.append(zero_share(u)); mn.append(float(u.min()))
        upd.append(out.updates / game.n)
        if sched == "round_robin":
            swp.append(out.sweeps)
    W = np.array(W)
    return dict(
        topology=cfg.topology, power_law_alpha=cfg.power_law_alpha, rho=cfg.num_drivers / cfg.num_zones,
        num_drivers=cfg.num_drivers, seed=cfg.seed, w_opt=w_opt,
        loss_mean=float(1 - (W / w_opt).mean()) if len(W) and w_opt > 1e-12 else np.nan,
        eq_mean_u=float(W.mean() / game.n) if len(W) else np.nan, opt_mean_u=w_opt / game.n,
        eq_gini=float(np.mean(g)) if g else np.nan, opt_gini=gini(u_opt),
        eq_cv=float(np.mean(c)) if c else np.nan, opt_cv=cv(u_opt),
        eq_zero_share=float(np.mean(z)) if z else np.nan, opt_zero_share=zero_share(u_opt),
        eq_min_u=float(np.mean(mn)) if mn else np.nan, opt_min_u=float(u_opt.min()),
        updates_per_driver=float(np.mean(upd)) if upd else np.nan,
        sweeps_rr=float(np.mean(swp)) if swp else np.nan,
        n_converged=len(W), n_runs=n_runs,
    )


def configs(seeds: int):
    for (t, kw), a, rho, s in itertools.product(TOPO.items(), [0.0, 0.5, 1.0, 1.5, 2.0], [0.5, 1, 2, 4], range(seeds)):
        yield InstanceConfig(topology=t, topology_kwargs=kw, num_nodes=N_NODES, num_zones=M_ZONES,
                             num_drivers=int(rho * M_ZONES), regime="power_law", power_law_alpha=a, seed=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(equity_instance, configs(a.seeds), chunksize=8))
    df = pd.DataFrame(rows); df.insert(0, "exp", "E")
    df.to_csv(f"{a.outdir}/exp_E.csv", index=False)
    print(f"exp E: {len(df)} instances in {time.time()-t:.0f}s, non-converged runs={int((df.n_runs - df.n_converged).sum())}", flush=True)


if __name__ == "__main__":
    main()
