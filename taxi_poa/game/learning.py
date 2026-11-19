"""Extension: unknown demand learned online (course direction C).

Repeated game. Each epoch t = 1..T the stage game of the report is played, but the true mean demand D(z) is unknown.
Realised zone demand is R_z(t) = D(z) * Gamma(k, 1/k) (mean D(z), coefficient of variation 1/sqrt(k)). The realisation of a zone
is observed publicly iff at least one driver served it (a zone nobody visits is never observed: bandit feedback).
Known to everybody: the graph (so travel costs), lam, and the city's total demand (an upper bound on any single zone).

Beliefs are common (pooled observations). Index for zone z with N_z observations and sample mean m_z:
        idx_z = cap                                         if N_z = 0         (optimism for unexplored zones)
        idx_z = min(cap, m_z + c * scale * sqrt(ln(t+1) / N_z))   otherwise    (UCB; c = 0 -> greedy on the sample mean)
Each epoch the agents act as if idx were the true demand:
        'selfish'  : asynchronous best-response dynamics on the believed game (warm-started from last epoch's profile)
        'planner'  : Hungarian assignment on the believed game (centralised dispatcher)
and we score the profile played with the TRUE expected welfare (pseudo-regret: removes realisation noise from the metric).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from game.model import TaxiGame
from game.dynamics import best_response_dynamics
from game.benchmark import hungarian_optimum


@dataclass
class LearningTrace:
    welfare: np.ndarray          # true expected welfare of the profile played at each epoch, shape (T,)
    w_opt: float                 # full-information optimum
    visits: np.ndarray           # (m_real,) number of epochs each real zone was served
    est_error: float             # mean |idx - D| / D over real zones at the end (belief quality)


def run_learning(true_game: TaxiGame, mode: str, T: int, rng: np.random.Generator, c: float = 1.0,
                 noise_k: float = 4.0) -> LearningTrace:
    """`true_game` includes the idle dummy zone as its LAST column (D=0, c=0)."""
    n, m_all = true_game.n, true_game.m
    m = m_all - 1
    D = true_game.demand[:m]
    cap = float(D.sum())                       # total demand is public
    scale = cap / m
    N = np.zeros(m); S = np.zeros(m)
    prof = np.full(n, m)                       # everyone starts idle
    w_opt, _ = hungarian_optimum(true_game)
    W = np.empty(T); visits = np.zeros(m)
    for t in range(1, T + 1):
        mean = np.divide(S, N, out=np.zeros(m), where=N > 0)
        bonus = c * scale * np.sqrt(np.log(t + 1) / np.maximum(N, 1))
        idx = np.where(N == 0, cap, np.minimum(cap, mean + bonus))
        believed = TaxiGame(np.append(idx, 0.0), true_game.cost, true_game.lam)
        if mode == "planner":
            _, prof = hungarian_optimum(believed)
        elif mode == "selfish":
            prof = best_response_dynamics(believed, prof, rng, max_sweeps=2000).profile
        else:
            raise ValueError(mode)
        W[t - 1] = true_game.welfare(prof)
        served = np.bincount(prof, minlength=m_all)[:m] > 0
        obs = D * rng.gamma(noise_k, 1.0 / noise_k, size=m)
        N[served] += 1; S[served] += obs[served]; visits[served] += 1
    mean = np.divide(S, N, out=np.zeros(m), where=N > 0)
    err = float(np.mean(np.abs(mean - D)[N > 0] / D[N > 0])) if (N > 0).any() and (D[N > 0] > 0).all() else float("nan")
    return LearningTrace(W, w_opt, visits, err)
