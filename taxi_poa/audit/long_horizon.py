"""Selfish UCB learning over 6000 epochs (rho=1, alpha=1, 30 seeds) vs full-information equilibrium. -> results/long_horizon.csv"""
import numpy as np, pandas as pd
from game import InstanceConfig, build_game, best_response_dynamics, initial_profile
from game.learning import run_learning
from game.benchmark import hungarian_optimum
rows = []
for seed in range(30):
    g, _ = build_game(InstanceConfig(num_nodes=36, num_zones=10, num_drivers=10, regime="power_law", power_law_alpha=1.0, seed=seed))
    w, _ = hungarian_optimum(g); rng = np.random.default_rng(seed)
    eq = np.mean([best_response_dynamics(g, initial_profile(g, "random", rng), rng).welfare for _ in range(20)]) / w
    e = run_learning(g, "selfish", 6000, np.random.default_rng(seed), c=1.0).welfare / w
    rows.append(dict(seed=seed, eff_1_50=e[0:50].mean(), eff_250_350=e[250:350].mean(), eff_1000_1500=e[1000:1500].mean(),
                     eff_2500_3000=e[2500:3000].mean(), eff_5500_6000=e[5500:].mean(), eff_fullinfo_eq=eq))
pd.DataFrame(rows).to_csv("results/long_horizon.csv", index=False); print("ok")
