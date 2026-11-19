"""The stage game (report Sec. 3.1).

Drivers i = 0..n-1, zones z = 0..m-1. Driver i at location l_i picking zone z gets

    U_i(a) = D[z] / n_z(a) - lam * c[i, z],       c[i, z] = c(l_i, zone z)

Key structural fact (not in the draft report): the payoff is  f_z(n_z) - lam * c[i, z]  where
f_z(k) = D[z]/k depends only on the zone and its load, and the player-specific part depends only on
(i, z) and is additive. That is Rosenthal's setting with player-specific constants, so the game has the
EXACT potential

    Phi(a) = sum_z D[z] * H(n_z(a))  -  lam * sum_i c[i, a_i],        H(k) = 1 + 1/2 + ... + 1/k

A unilateral move changes Phi by exactly the mover's change in utility. Consequences:
  * a pure Nash equilibrium exists (argmax of Phi is one);
  * every best-response / better-response path is finite -> asynchronous best-response dynamics CANNOT cycle
    (the cycles Milchtaich exhibits need payoffs that are not additive in this way).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TaxiGame:
    demand: np.ndarray   # shape (m,), D(z) >= 0
    cost: np.ndarray     # shape (n, m), c(l_i, z) >= 0
    lam: float = 1.0

    def __post_init__(self):
        d = np.asarray(self.demand, dtype=float)
        c = np.asarray(self.cost, dtype=float)
        if d.ndim != 1 or c.ndim != 2 or c.shape[1] != d.shape[0]:
            raise ValueError("demand must be (m,), cost must be (n, m)")
        if (d < 0).any() or (c < 0).any():
            raise ValueError("demand and cost must be nonnegative")
        if self.lam < 0:
            raise ValueError("lam must be >= 0")
        object.__setattr__(self, "demand", d)
        object.__setattr__(self, "cost", c)
        # harmonic numbers H[0..n]
        h = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, c.shape[0] + 1))])
        object.__setattr__(self, "_H", h)

    @property
    def n(self) -> int:
        return self.cost.shape[0]

    @property
    def m(self) -> int:
        return self.demand.shape[0]

    # ---- profile functions (a: int array of length n with entries in 0..m-1) ----
    def loads(self, a) -> np.ndarray:
        return np.bincount(np.asarray(a), minlength=self.m)

    def utilities(self, a) -> np.ndarray:
        a = np.asarray(a)
        n_z = self.loads(a)
        return self.demand[a] / n_z[a] - self.lam * self.cost[np.arange(self.n), a]

    def welfare(self, a) -> float:
        """W(a) = sum over ACTIVE zones of D(z)  -  lam * total travel cost (report Eq. 4)."""
        a = np.asarray(a)
        active = self.loads(a) > 0
        return float(self.demand[active].sum() - self.lam * self.cost[np.arange(self.n), a].sum())

    def potential(self, a) -> float:
        a = np.asarray(a)
        n_z = self.loads(a)
        return float((self.demand * self._H[n_z]).sum() - self.lam * self.cost[np.arange(self.n), a].sum())

    # ---- unilateral deviations ----
    def deviation_utilities(self, i: int, a) -> np.ndarray:
        """Utility driver i would get from each zone z, others fixed (entry a_i is its current utility)."""
        a = np.asarray(a)
        n_z = self.loads(a).astype(float)
        n_z[a[i]] -= 1.0          # remove i from its current zone
        return self.demand / (n_z + 1.0) - self.lam * self.cost[i]

    def best_response(self, i: int, a, eps: float = 1e-12) -> int:
        """Best zone for i. Stays put unless some zone is better by more than eps; ties -> lowest index."""
        u = self.deviation_utilities(i, a)
        cur = int(np.asarray(a)[i])
        best = int(np.argmax(u))            # argmax returns the lowest index among ties
        return best if u[best] > u[cur] + eps else cur

    def is_pure_nash(self, a, eps: float = 1e-9) -> bool:
        a = np.asarray(a)
        return all(self.best_response(i, a, eps) == a[i] for i in range(self.n))
