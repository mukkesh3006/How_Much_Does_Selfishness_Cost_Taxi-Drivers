"""Asynchronous best-response dynamics (report Alg. 1).

Because the game has an exact potential (see model.py) every run reaches a pure Nash equilibrium; 'cycle'
and 'limit' statuses are kept as runtime assertions of that theorem, not as expected outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from game.model import TaxiGame


@dataclass
class BRDResult:
    profile: np.ndarray
    status: str                 # 'converged' | 'cycle' | 'limit'
    updates: int                # number of profitable unilateral moves performed
    sweeps: int
    welfare: float
    potential: float
    potential_trace: list = field(default_factory=list)   # Phi after each move (only if track_potential)


def initial_profile(game: TaxiGame, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "random":
        return rng.integers(0, game.m, size=game.n)
    if kind == "greedy":        # each driver takes its best zone as if alone
        return np.argmax(game.demand[None, :] - game.lam * game.cost, axis=1)
    if kind == "nearest":
        return np.argmin(game.cost, axis=1)
    raise ValueError(f"unknown init '{kind}'")


def best_response_dynamics(game: TaxiGame, init, rng: np.random.Generator | None = None,
                           schedule: str = "round_robin", max_sweeps: int = 10_000,
                           eps: float = 1e-12, track_potential: bool = False) -> BRDResult:
    """schedule:
        'round_robin'  : a random permutation of drivers is drawn once, drivers best-respond in that order,
                         sweep after sweep. Converged = a full sweep with no move. Profile repeating at a sweep
                         boundary = cycle (sound, since the schedule is deterministic after the draw).
        'random'       : each step activates a uniformly random driver. Convergence is checked explicitly with
                         is_pure_nash every n steps; budget is max_sweeps * n activations.
    """
    rng = rng or np.random.default_rng()
    a = np.array(init, dtype=int).copy()
    n = game.n
    trace, updates = [], 0
    seen = {tuple(a)}

    def move(i):
        nonlocal updates
        z = game.best_response(i, a, eps)
        if z != a[i]:
            a[i] = z
            updates += 1
            if track_potential:
                trace.append(game.potential(a))
            return True
        return False

    if schedule == "round_robin":
        order = rng.permutation(n)
        for sweep in range(1, max_sweeps + 1):
            moved = False
            for i in order:
                moved |= move(i)
            if not moved:
                return BRDResult(a, "converged", updates, sweep, game.welfare(a), game.potential(a), trace)
            key = tuple(a)
            if key in seen:
                return BRDResult(a, "cycle", updates, sweep, game.welfare(a), game.potential(a), trace)
            seen.add(key)
        return BRDResult(a, "limit", updates, max_sweeps, game.welfare(a), game.potential(a), trace)

    if schedule == "random":
        for step in range(1, max_sweeps * n + 1):
            move(int(rng.integers(n)))
            if step % n == 0 and game.is_pure_nash(a, eps=1e-9):
                return BRDResult(a, "converged", updates, step // n, game.welfare(a), game.potential(a), trace)
        return BRDResult(a, "limit", updates, max_sweeps, game.welfare(a), game.potential(a), trace)

    raise ValueError(f"unknown schedule '{schedule}'")
