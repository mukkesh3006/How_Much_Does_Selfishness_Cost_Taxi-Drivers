"""Pilot WITHOUT the idle option and with a fixed lam=1: shows why the raw ratio W_OPT/W_EQ is unusable. -> results/pilot_poa.txt"""
import numpy as np
from game import InstanceConfig, run_instance
worst = 0.0; n = 0; nonpos = 0
for topo, kw in [("grid", {}), ("hub_spoke", {"num_hubs": 3}), ("scale_free", {"m": 2})]:
    for regime in ["uniform", "concentrated"]:
        for rho in [0.5, 1.0, 2.0]:
            for seed in range(15):
                o = run_instance(InstanceConfig(topology=topo, topology_kwargs=kw, regime=regime, num_nodes=30, num_zones=8,
                                                num_drivers=int(rho * 8), lam=1.0, outside_option=False, seed=seed), n_runs=12, enumerate_limit=0)
                n += 1; nonpos += o["frac_nonpositive_eq"] > 0 if "frac_nonpositive_eq" in o else 0
                if not np.isnan(o["poa_sampled_worst"]): worst = max(worst, o["poa_sampled_worst"])
open("results/pilot_poa.txt", "w").write(f"instances={n}\nmax_sampled_poa={worst:.1f}\n"); print(n, round(worst, 1))
