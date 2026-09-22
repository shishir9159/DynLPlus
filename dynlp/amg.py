"""Aggregation-based algebraic multigrid for grounded graph Laplacians.

The operator is A = diag(s + rowsum(W)) - W with grounding s > 0. Aggregates
come from heavy-edge handshake matching (a GPU-friendly parallel matching).
Two matching passes per level coarsen by about 4x. The coarse operator
keeps the same (W_c, s_c) form: W_c sums the inter-aggregate weights and s_c
sums the grounding. This stays exact and avoids cancellation in float32.

DynLP's supernodes are one level of exactly this kind of aggregation; this
module makes it multilevel and uses the V-cycle as a CG preconditioner.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import backend
from .graphs import coalesce, csr_rows


@dataclass
class Level:
    W: object
    s: object
    diag: object
    agg: object = None   # fine -> coarse map
    PT: object = None    # restriction (coarse x fine) CSR

    def apply(self, X):
        return self.diag[:, None] * X - self.W @ X


def _handshake(n, rows, cols, w, rounds):
    """Heavy-edge handshake matching; unmatched vertices join a matched neighbour."""
    B = backend.get()
    xp = B.xp
    agg = xp.full(n, -1, dtype=xp.int64)
    matched = xp.zeros(n, dtype=bool)
    ar = xp.arange(n, dtype=xp.int64)

    def best_choice(er, ec, ew):
        best = xp.full(n, -xp.inf, dtype=ew.dtype)
        B.scatter_max(best, er, ew)
        tie = ew >= best[er]
        choice = xp.full(n, n, dtype=xp.int64)
        B.scatter_min(choice, er[tie], ec[tie].astype(xp.int64))
        return choice

    for _ in range(rounds):
        ok = ~matched[rows] & ~matched[cols]
        if not bool(ok.any()):
            break
        choice = best_choice(rows[ok], cols[ok], w[ok])
        has = choice < n
        partner = xp.where(has, choice, 0)
        mutual = has & (choice[partner] == ar)
        matched |= mutual
        agg = xp.where(mutual, xp.minimum(ar, partner), agg)
    ok = ~matched[rows] & matched[cols]
    if bool(ok.any()):
        choice = best_choice(rows[ok], cols[ok], w[ok])
        has = choice < n
        agg[has] = agg[choice[has]]
    left = agg < 0
    agg[left] = ar[left]
    uniq, inv = xp.unique(agg, return_inverse=True)
    return inv.ravel().astype(xp.int64), int(uniq.shape[0])


def _coarsen(W, s, agg, nc):
    B = backend.get()
    xp = B.xp
    rows = csr_rows(W)
    rc, cc = agg[rows], agg[W.indices]
    off = rc != cc
    Wc = coalesce(rc[off], cc[off], W.data[off], nc, W.data.dtype)
    sc = xp.zeros(nc, dtype=s.dtype)
    B.scatter_add(sc, agg, s)
    return Wc, sc


class AMG:
    """Symmetric V-cycle preconditioner (weighted-Jacobi smoothing, exact coarse solve)."""

    def __init__(self, W, s, *, max_coarse=1500, passes=2, rounds=3, omega=0.7, nu=1,
                 max_levels=25, min_ratio=0.85, coarse_sweeps=20):
        B = backend.get()
        xp = B.xp
        self.omega, self.nu, self.coarse_sweeps = omega, nu, coarse_sweeps
        self.levels: list[Level] = []
        dt = W.data.dtype
        n = W.shape[0]
        cur = Level(W, s, s + W @ xp.ones(n, dtype=dt))
        while n > max_coarse and len(self.levels) < max_levels:
            agg = xp.arange(n, dtype=xp.int64)
            nc, Wp = n, cur.W
            for _ in range(passes):
                a2, nc2 = _handshake(Wp.shape[0], csr_rows(Wp), Wp.indices, Wp.data, rounds)
                agg = a2[agg]
                Wp, _ = _coarsen(Wp, xp.zeros(Wp.shape[0], dtype=dt), a2, nc2)
                nc = nc2
            if nc > min_ratio * n:
                break
            Wc, sc = _coarsen(cur.W, cur.s, agg, nc)
            P = B.csr(xp.ones(n, dtype=dt), agg.astype(xp.int32), xp.arange(n + 1, dtype=xp.int32), (n, nc))
            cur.agg, cur.PT = agg, P.T.tocsr()
            self.levels.append(cur)
            cur = Level(Wc, sc, sc + Wc @ xp.ones(nc, dtype=dt))
            n = nc
        self.coarse = cur
        self.inv = None
        if n <= 4 * max_coarse:
            Ad = -cur.W.toarray().astype(xp.float64)
            Ad[xp.arange(n), xp.arange(n)] += cur.diag.astype(xp.float64)
            self.inv = xp.linalg.inv(Ad).astype(dt)
        # edge visits per V-cycle application (for hardware-independent work accounting)
        self.work_per_apply = sum(2 * nu * L.W.nnz + 2 * L.W.shape[0] for L in self.levels)
        self.work_per_apply += n * n if self.inv is not None else coarse_sweeps * cur.W.nnz
        self.sizes = [L.W.shape[0] for L in self.levels] + [n]

    def _smooth(self, L, X, R, sweeps):
        for _ in range(sweeps):
            X = X + self.omega * (R - L.apply(X)) / L.diag[:, None]
        return X

    def _cycle(self, lvl, R):
        if lvl == len(self.levels):
            L = self.coarse
            if self.inv is not None:
                return self.inv @ R
            X = self.omega * R / L.diag[:, None]
            return self._smooth(L, X, R, self.coarse_sweeps - 1)
        L = self.levels[lvl]
        X = self.omega * R / L.diag[:, None]
        X = self._smooth(L, X, R, self.nu - 1)
        Rc = L.PT @ (R - L.apply(X))
        X = X + self._cycle(lvl + 1, Rc)[L.agg]
        return self._smooth(L, X, R, self.nu)

    def __call__(self, R):
        return self._cycle(0, R)
