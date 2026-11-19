"""Fast exact centralized optimum when drivers have the idle option (last zone column, D=0, c=0).

With idle available, a second driver in a zone adds -lam*c <= 0 to welfare (the zone's demand is already counted),
so WLOG at most one driver serves each real zone and the optimum is a maximum-weight bipartite assignment with
weights max(D(z) - lam*c(i,z), 0). Solved exactly by the Hungarian algorithm in polynomial time.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from game.model import TaxiGame


def hungarian_optimum(game: TaxiGame):
    """Returns (W_OPT, profile). Assumes the last zone of `game` is the idle dummy zone."""
    gain = np.maximum(game.demand[None, :-1] - game.lam * game.cost[:, :-1], 0.0)
    rows, cols = linear_sum_assignment(-gain)
    profile = np.full(game.n, game.m - 1)
    for r, c in zip(rows, cols):
        if gain[r, c] > 0:
            profile[r] = c
    return float(gain[rows, cols].sum()), profile
