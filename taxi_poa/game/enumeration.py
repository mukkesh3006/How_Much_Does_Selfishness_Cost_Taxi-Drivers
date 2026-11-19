"""Exhaustive enumeration of all profiles for small games: exact set of pure Nash equilibria, true PoA / PoS."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from game.model import TaxiGame


@dataclass
class EnumerationResult:
    n_profiles: int
    nash_profiles: np.ndarray      # (k, n)
    nash_welfare: np.ndarray       # (k,)
    opt_welfare: float
    opt_profile: np.ndarray
    phi_max_profile: np.ndarray    # potential maximizer (is a Nash equilibrium)
    phi_max_welfare: float

    @property
    def worst_nash_welfare(self): return float(self.nash_welfare.min())
    @property
    def best_nash_welfare(self): return float(self.nash_welfare.max())

    def poa(self):
        """OPT / worst-equilibrium welfare; nan if the denominator is not positive (ratio meaningless)."""
        w = self.worst_nash_welfare
        return self.opt_welfare / w if w > 0 else float("nan")

    def pos(self):
        w = self.best_nash_welfare
        return self.opt_welfare / w if w > 0 else float("nan")


def enumerate_equilibria(game: TaxiGame, max_profiles: int = 3_000_000, chunk: int = 100_000,
                         tol: float = 1e-9) -> EnumerationResult:
    n, m = game.n, game.m
    total = m ** n
    if total > max_profiles:
        raise ValueError(f"{m}^{n} = {total} profiles exceeds max_profiles={max_profiles}")
    rows = np.arange(n)
    powers = m ** np.arange(n, dtype=np.int64)
    nash_p, nash_w = [], []
    best_w, best_p = -np.inf, None
    best_phi, phi_p = -np.inf, None
    Hs = game._H
    for s in range(0, total, chunk):
        idx = np.arange(s, min(total, s + chunk), dtype=np.int64)
        A = (idx[:, None] // powers[None, :]) % m                        # (P, n)
        P = A.shape[0]
        load = np.zeros((P, m), dtype=np.int64)
        np.add.at(load, (np.repeat(np.arange(P), n), A.ravel()), 1)
        trav = game.cost[rows[None, :], A]                                # (P, n)
        W = (game.demand[None, :] * (load > 0)).sum(1) - game.lam * trav.sum(1)
        Phi = (game.demand[None, :] * Hs[load]).sum(1) - game.lam * trav.sum(1)
        # utilities if staying vs deviating
        n_own = np.take_along_axis(load, A, axis=1)                       # (P, n) load of own zone
        u_stay = game.demand[A] / n_own - game.lam * trav                 # (P, n)
        u_dev = game.demand[None, None, :] / (load[:, None, :] + 1) - game.lam * game.cost[None, :, :]  # (P, n, m)
        u_dev[np.arange(P)[:, None], rows[None, :], A] = -np.inf          # deviating to own zone is staying
        is_nash = (u_dev.max(axis=2) <= u_stay + tol).all(axis=1)
        if is_nash.any():
            nash_p.append(A[is_nash]); nash_w.append(W[is_nash])
        k = int(W.argmax())
        if W[k] > best_w:
            best_w, best_p = float(W[k]), A[k].copy()
        k = int(Phi.argmax())
        if Phi[k] > best_phi + 1e-12:
            best_phi, phi_p = float(Phi[k]), A[k].copy()
    nash_p = np.concatenate(nash_p) if nash_p else np.zeros((0, n), dtype=int)
    nash_w = np.concatenate(nash_w) if nash_w else np.zeros(0)
    return EnumerationResult(total, nash_p, nash_w, best_w, best_p, phi_p, game.welfare(phi_p))
