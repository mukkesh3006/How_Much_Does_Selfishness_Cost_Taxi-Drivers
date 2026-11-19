"""Parameter sweeps. Usage:  python -m game.sweep A B C D V [--jobs 2] [--seeds 40]  -> results/exp_<X>.csv

A  main      : topology x demand concentration (power-law alpha) x driver density rho        (H1, H2, H3)
B  travel    : topology x kappa (travel-cost scale)                                          (H4)
C  regimes   : topology x {uniform, concentrated, degree_correlated} x rho                    (report Sec 3.4)
D  start     : topology x {random starts, common depot}                                      (symmetry check)
V  validate  : small instances, exact enumeration of every equilibrium vs sampled dynamics
"""
from __future__ import annotations

import argparse, itertools, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from game.instance import InstanceConfig, run_instance

TOPO = {"grid": {}, "hub_spoke": {"num_hubs": 3}, "scale_free": {"m": 2}}
N_NODES, M_ZONES = 36, 10


def _run(args):
    cfg, kw = args
    return run_instance(cfg, **kw)


def configs(exp: str, seeds: int):
    out = []
    if exp == "A":
        for (t, kw), a, rho, s in itertools.product(TOPO.items(), [0.0, 0.5, 1.0, 1.5, 2.0], [0.5, 1, 2, 4], range(seeds)):
            out.append((InstanceConfig(topology=t, topology_kwargs=kw, num_nodes=N_NODES, num_zones=M_ZONES,
                                       num_drivers=int(rho * M_ZONES), regime="power_law", power_law_alpha=a, seed=s),
                        dict(n_runs=20, enumerate_limit=0)))
    elif exp == "B":
        for (t, kw), k, s in itertools.product(TOPO.items(), [0.1, 0.3, 0.6, 1.0, 2.0], range(seeds)):
            out.append((InstanceConfig(topology=t, topology_kwargs=kw, num_nodes=N_NODES, num_zones=M_ZONES,
                                       num_drivers=M_ZONES, regime="power_law", power_law_alpha=1.0, kappa=k, seed=s),
                        dict(n_runs=20, enumerate_limit=0)))
    elif exp == "C":
        for (t, kw), r, rho, s in itertools.product(TOPO.items(), ["uniform", "concentrated", "degree_correlated"], [1, 2], range(seeds)):
            out.append((InstanceConfig(topology=t, topology_kwargs=kw, num_nodes=N_NODES, num_zones=M_ZONES,
                                       num_drivers=int(rho * M_ZONES), regime=r, seed=s),
                        dict(n_runs=20, enumerate_limit=0)))
    elif exp == "D":
        for (t, kw), ds, s in itertools.product(TOPO.items(), ["random", "depot"], range(seeds)):
            out.append((InstanceConfig(topology=t, topology_kwargs=kw, num_nodes=N_NODES, num_zones=M_ZONES,
                                       num_drivers=M_ZONES, regime="power_law", power_law_alpha=1.0, driver_start=ds, seed=s),
                        dict(n_runs=20, enumerate_limit=0)))
    elif exp == "V":
        for (t, kw), a, n, s in itertools.product(TOPO.items(), [0.0, 1.0, 2.0], [3, 5, 6], range(max(10, seeds // 2))):
            out.append((InstanceConfig(topology=t, topology_kwargs=kw, num_nodes=16, num_zones=4, num_drivers=n,
                                       regime="power_law", power_law_alpha=a, seed=1000 + s),
                        dict(n_runs=20, enumerate_limit=400_000)))
    else:
        raise SystemExit(f"unknown experiment {exp}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+"); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=40); ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for exp in a.exps:
        cfgs = configs(exp, a.seeds); t = time.time()
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(_run, cfgs, chunksize=8))
        df = pd.DataFrame(rows); df.insert(0, "exp", exp)
        df.to_csv(f"{a.outdir}/exp_{exp}.csv", index=False)
        bad = int((df.n_cycle + df.n_limit).sum())
        print(f"exp {exp}: {len(df)} instances in {time.time()-t:.0f}s, non-converged runs={bad}", flush=True)


if __name__ == "__main__":
    main()
