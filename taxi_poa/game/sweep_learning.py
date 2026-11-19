"""Experiment L: learning unknown demand. python -m game.sweep_learning [--seeds 40] [--T 300] [--jobs 2]
Writes results/exp_L.csv (per instance/mode/c summary) and results/exp_L_curves.npz (efficiency trajectories)."""
from __future__ import annotations
import argparse, itertools, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

from game import InstanceConfig, build_game, best_response_dynamics, initial_profile
from game.learning import run_learning
from game.benchmark import hungarian_optimum

TOPO = {"grid": {}, "hub_spoke": {"num_hubs": 3}, "scale_free": {"m": 2}}
VARIANTS = [("planner", 1.0), ("planner", 0.0), ("selfish", 1.0), ("selfish", 0.0)]
WINDOWS = {"eff_001_010": (0, 10), "eff_011_050": (10, 50), "eff_051_150": (50, 150), "eff_151_end": (150, None)}


def one(args):
    topo, alpha, rho, seed, T = args
    cfg = InstanceConfig(topology=topo, topology_kwargs=TOPO[topo], num_nodes=36, num_zones=10,
                         num_drivers=max(1, int(round(rho * 10))), regime="power_law", power_law_alpha=alpha, seed=seed)
    g, _ = build_game(cfg)
    w_opt, _ = hungarian_optimum(g)
    rng = np.random.default_rng(seed + 12345)
    eq = np.mean([best_response_dynamics(g, initial_profile(g, "random", rng), rng).welfare for _ in range(20)])
    rows, curves = [], {}
    for mode, c in VARIANTS:
        tr = run_learning(g, mode, T, np.random.default_rng(seed + 777), c=c)
        e = tr.welfare / w_opt if w_opt > 0 else np.ones(T)
        row = dict(topology=topo, alpha=alpha, rho=rho, seed=seed, mode=mode, c=c, w_opt=w_opt,
                   eff_fullinfo_eq=eq / w_opt if w_opt > 0 else 1.0, zones_never_served=int((tr.visits == 0).sum()),
                   est_error=tr.est_error, cum_loss_vs_opt=float((1 - e).sum()))
        for k, (a, b) in WINDOWS.items():
            row[k] = float(e[a:b].mean())
        rows.append(row); curves[(topo, alpha, rho, mode, c, seed)] = e
    return rows, curves


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=40); ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--outdir", default="results"); a = ap.parse_args()
    jobs = list(itertools.product(TOPO, [1.0, 1.5], [0.3, 0.5, 1.0, 2.0], range(a.seeds), [a.T]))
    t = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        out = list(ex.map(one, jobs, chunksize=4))
    rows = [r for rr, _ in out for r in rr]
    pd.DataFrame(rows).to_csv(f"{a.outdir}/exp_L.csv", index=False)
    keys, arrs = [], []
    for _, cv in out:
        for k, v in cv.items(): keys.append("|".join(map(str, k))); arrs.append(v)
    np.savez_compressed(f"{a.outdir}/exp_L_curves.npz", keys=np.array(keys), curves=np.array(arrs))
    print(f"exp L: {len(jobs)} instances x {len(VARIANTS)} variants in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
