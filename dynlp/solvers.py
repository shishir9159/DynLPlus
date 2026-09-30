"""Label-propagation solvers for one batch of a stream.

Baselines: itlp (Jacobi from scratch), itlp-warm (warm-started), dynlp (DynLP's
Algorithm 2: component init + frontier Jacobi that stops on changes below delta).
DynLP+ (certified): warm start, known-neighbour supernode init, and a stop rule
that bounds the true error ||F - F*||_inf <= tol. Methods: push (local residual
relaxation), pcg (Jacobi-preconditioned block CG), amg (AMG-preconditioned CG),
auto (push when the residual is local, else PCG/AMG), inc (resident state).

Certificate: A = diag - W is a nonsingular M-matrix, so A^{-1} >= 0. If
|r| <= rho * diag then |F - F*| = |A^{-1} r| <= rho * h, with A h = diag (h_u:
expected walk visits before absorption). h rides along as one extra column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import backend
from .amg import AMG
from .backend import Timer
from .kernels import FrontierOps
from .stream import System


@dataclass
class Stats:
    iters: int = 0
    edges: int = 0              # edge visits: hardware-independent work
    ms: float = 0.0             # init + setup + solve
    setup_ms: float = 0.0       # AMG setup (included in ms)
    path: str = ""
    cert: float = float("nan")  # bound the solver certified itself
    hmax: float = float("nan")


def _xp():
    return backend.get().xp


def spmm(W, X):
    return (W @ X[:, 0])[:, None] if X.shape[1] == 1 else W @ X


def apply_A(sys: System, X):
    return sys.diag[:, None] * X - spmm(sys.W, X)


def rho_cols(sys: System, R):
    """max_u |R_uc| / diag_u per column (host floats)."""
    if R.shape[0] == 0:
        return np.zeros(R.shape[1])
    return backend.get().asnumpy((_xp().abs(R) / sys.diag[:, None]).max(axis=0)).astype(float)


def violation(sys: System, R, eps_d):
    """max |R_uc| / (eps_c * diag_u); <= 1 means converged."""
    return float((_xp().abs(R) / (eps_d[None, :] * sys.diag[:, None])).max()) if R.shape[0] else 0.0


def jacobi_M(sys):
    return lambda R: R / sys.diag[:, None]


def pcg(sys: System, Z, rhs, eps, M, max_iter, st: Stats, work_M=0, patience=400):
    """Block PCG (one CG per column) until |r_c| <= eps_c * diag. Stops if the
    violation stalls for `patience` iterations (precision floor; refinement follows)."""
    xp = _xp()
    eps_d = xp.asarray(eps, dtype=Z.dtype)
    R = rhs - apply_A(sys, Z)
    st.edges += sys.nnz
    best = violation(sys, R, eps_d)
    if best <= 1:
        return Z
    P = Y = M(R)
    rz, stall = (R * Y).sum(axis=0), 0
    for _ in range(max_iter):
        AP = apply_A(sys, P)
        pAp = (P * AP).sum(axis=0)
        alpha = xp.where(pAp > 0, rz / xp.where(pAp > 0, pAp, 1), 0)
        Z += alpha[None, :] * P
        R -= alpha[None, :] * AP
        st.iters, st.edges = st.iters + 1, st.edges + sys.nnz + work_M
        v = violation(sys, R, eps_d)
        if v <= 1:
            R = rhs - apply_A(sys, Z)  # guard against recurrence drift
            st.edges += sys.nnz
            if (v := violation(sys, R, eps_d)) <= 1:
                break
        best, stall = (v, 0) if v < 0.9 * best else (best, stall + 1)
        if stall >= patience:
            break
        Y = M(R)
        rz_new = (R * Y).sum(axis=0)
        P = Y + xp.where(rz > 0, rz_new / xp.where(rz > 0, rz, 1), 0)[None, :] * P
        rz = rz_new
    return Z


def push(sys: System, Z, R, eps, ops: FrontierOps, st: Stats, budget_edges=None, max_iter=10**7):
    """Residual push (Jacobi on the violating set) until |r_c| <= eps_c * diag.
    R stays exact up to rounding, so the stop test is a true residual test.
    Returns (Z, R, converged)."""
    xp = _xp()
    eps_d = xp.asarray(eps, dtype=Z.dtype)
    violating = lambda idx: (xp.abs(R[idx]) > eps_d[None, :] * sys.diag[idx][:, None]).any(axis=1)  # noqa: E731
    all_idx = xp.arange(sys.n_u, dtype=xp.int32)
    cand, e0, it, best, stall = all_idx[violating(all_idx)], int(ops.edges), 0, float("inf"), 0
    while cand.shape[0] and it < max_iter:
        D = R[cand] / sys.diag[cand][:, None]
        Z[cand] += D
        R[cand] = 0
        mark = xp.zeros(sys.n_u, dtype=xp.uint8)
        mark[cand] = 1
        ops.push(cand, D, R, mark)
        touched = xp.flatnonzero(mark).astype(xp.int32)
        cand, it = touched[violating(touched)], it + 1
        if it % 32 == 0:
            if budget_edges is not None and int(ops.edges) - e0 > budget_edges:
                break
            v = violation(sys, R, eps_d)
            best, stall = (v, 0) if v < 0.99 * best else (best, stall + 1)
            if stall >= 8:  # no progress in 256 rounds: precision floor
                break
    st.iters += it
    return Z, R, cand.shape[0] == 0


def _comp_avg(sys, S_rows, T_rows, fallback):
    """Average of S/T over each tau-component of the new vertices (None rows: T = 0)."""
    be, xp = backend.get(), _xp()
    S = xp.zeros((sys.n_comp,) + S_rows.shape[1:], dtype=S_rows.dtype)
    T = xp.zeros(sys.n_comp, dtype=S_rows.dtype)
    be.scatter_add(S, sys.comp, S_rows)
    be.scatter_add(T, sys.comp, T_rows)
    ok = T > 0
    Tn = xp.where(ok, T, 1)
    return xp.where(ok[:, None], S / Tn[:, None], fallback) if S.ndim == 2 else xp.where(ok, S / Tn, fallback)


def supernode_init(sys: System, X, known=None, cols=None):
    """Initialize new vertices per component (DynLP's supernodes). known=None: the
    share of the component's weight to each ground-truth class (DynLP's rule);
    known=mask: weighted average over all known neighbours (labels + last solution)."""
    new = _xp().flatnonzero(sys.is_new)
    if not new.shape[0]:
        return X
    cols = slice(0, sys.C) if cols is None else cols
    s_, t = sys.label_cols(sys.B)[new], sys.B[new].sum(axis=1)
    if known is not None:
        kf, Wn = known.astype(X.dtype), sys.W[new]
        s_, t = s_ + spmm(Wn, X[:, cols] * kf[:, None]), t + Wn @ kf
    X[new, cols] = _comp_avg(sys, s_, t, sys.prior[None, :])[sys.comp]
    return X


def init_h(sys: System, H, known, fallback):
    """New vertices: h ~ 1 + average h of known neighbours (labeled ones have h = 0)."""
    new = _xp().flatnonzero(sys.is_new)
    if new.shape[0]:
        kf, Wn = known.astype(H.dtype), sys.W[new]
        hc = _comp_avg(sys, Wn @ (H[:, 0] * kf), sys.B[new].sum(axis=1) + Wn @ kf, fallback - 1) + 1
        H[new, 0] = hc[sys.comp]
    return H


def predict(F, K):
    xp = _xp()
    return (F[:, 0] > 0.5).astype(xp.int32) if K == 2 else F.argmax(axis=1).astype(xp.int32)


class Solver:
    name = "base"

    def __init__(self, n_total, K, dtype="float32", store_dtype=None):
        self.n_total, self.K, self.dtype = n_total, K, np.dtype(dtype)
        self.store_dtype, self.C, self.F = np.dtype(store_dtype or dtype), 1 if K == 2 else K, None

    def _warm(self, sys: System):
        if self.F is None:
            self.F = _xp().empty((self.n_total, self.C), dtype=self.store_dtype)
            self.F[:] = sys.prior.astype(self.store_dtype)
        return self.F[sys.U].astype(self.dtype)

    def _store(self, sys: System, X):
        self.F[sys.U] = X

    def scores(self, sys: System):
        return self.F[sys.U]


class ItLP(Solver):
    def __init__(self, n_total, K, dtype="float32", delta=1e-4, warm=False, max_iter=200_000):
        super().__init__(n_total, K, dtype)
        self.delta, self.warm, self.max_iter = delta, warm, max_iter
        self.name = "itlp-warm" if warm else "itlp"

    def solve(self, sys):
        xp, st, sys = _xp(), Stats(path="jacobi"), sys.astype(self.dtype)
        with Timer() as tm:
            X = self._warm(sys)
            if self.warm:
                supernode_init(sys, X, known=~sys.is_new)
            else:
                X[:] = sys.prior
            for _ in range(self.max_iter):
                Y = (sys.rhs + spmm(sys.W, X)) / sys.diag[:, None]
                ch = float(xp.abs(Y - X).max()) if sys.n_u else 0.0
                X, st.iters, st.edges = Y, st.iters + 1, st.edges + sys.nnz
                if ch <= self.delta:
                    break
            self._store(sys, X)
        st.ms = tm.ms
        return st


class DynLP(Solver):
    """Faithful DynLP (Algorithm 2), with |.| in the line-29 change test."""

    def __init__(self, n_total, K, dtype="float32", delta=1e-4, group="auto", known_init=False, max_iter=10**7):
        super().__init__(n_total, K, dtype)
        self.delta, self.group, self.known_init, self.max_iter = delta, group, known_init, max_iter
        self.name = "dynlp-knowninit" if known_init else "dynlp"

    def solve(self, sys):
        xp, st, sys = _xp(), Stats(path="frontier-jacobi"), sys.astype(self.dtype)
        with Timer() as tm:
            X = supernode_init(sys, self._warm(sys), known=~sys.is_new if self.known_init else None)
            ops = FrontierOps(sys.W, self.C, self.group)
            frontier = xp.flatnonzero(sys.seed).astype(xp.int32)
            while frontier.shape[0] and st.iters < self.max_iter:
                Y, ch = ops.jacobi(frontier, X, sys.rhs, sys.diag, self.delta)
                st.iters += 1
                upd = frontier[ch]
                if not upd.shape[0]:
                    break
                X[upd] = Y[ch]
                mark = xp.zeros(sys.n_u, dtype=xp.uint8)
                mark[upd] = 1
                ops.mark_neighbors(upd, mark)
                frontier = xp.flatnonzero(mark).astype(xp.int32)
            self._store(sys, X)
        st.edges, st.ms = int(ops.edges), tm.ms
        return st


class DynLPPlus(Solver):
    """Certified DynLP+: guarantees ||F - F*||_inf <= tol on every batch."""

    def __init__(self, n_total, K, dtype="float32", method="auto", tol=1e-3, eps_h=0.05, group="auto",
                 push_frac=0.05, push_budget=30.0, amg_hmax=2000.0, max_iter=100_000, amg_opts=None):
        super().__init__(n_total, K, dtype, store_dtype="float64")  # float64 master copy of F and h
        self.method, self.tol, self.eps_h, self.group = method, tol, eps_h, group
        self.push_frac, self.push_budget, self.amg_hmax = push_frac, push_budget, amg_hmax
        self.max_iter, self.amg_opts = max_iter, amg_opts or {}
        self.name, self.H, self.hmax = f"dynlp+{method}", None, None

    def _use_amg(self):  # auto: AMG only when the problem is hard (large h)
        return self.method == "amg" or (self.method == "auto" and (self.hmax or float("inf")) > self.amg_hmax)

    def _precond(self, sys, st, amg_box):
        if not self._use_amg():
            return jacobi_M(sys), 0
        if amg_box[0] is None:
            with Timer() as t:
                amg_box[0] = AMG(sys.W, sys.s, **self.amg_opts)
            st.setup_ms += t.ms
            st.edges += 10 * sys.nnz  # rough setup cost: matching rounds + Galerkin
        return amg_box[0], amg_box[0].work_per_apply

    def _global(self, sys, Z, rhs, eps, st, amg_box):
        M, wm = self._precond(sys, st, amg_box)
        return pcg(sys, Z, rhs, eps, M, self.max_iter, st, wm)

    def _run(self, sys, Z, rhs, eps, ops, st, amg_box):
        xp = _xp()
        if self.method not in ("push", "auto"):
            st.path = self.method
            return self._global(sys, Z, rhs, eps, st, amg_box)
        R = rhs - apply_A(sys, Z)
        st.edges += sys.nnz
        if self.method == "auto" and sys.n_u and float(
                (xp.abs(R) > xp.asarray(eps, dtype=Z.dtype)[None, :] * sys.diag[:, None]).any(axis=1).mean()
        ) > self.push_frac:
            st.path = "amg" if self._use_amg() else "pcg"
            return self._global(sys, Z, rhs, eps, st, amg_box)
        budget = None if self.method == "push" else self.push_budget * max(sys.nnz, 1)
        Z, R, ok = push(sys, Z, R, eps, ops, st, budget)
        st.path = "push"
        if ok:
            return Z
        if self.method == "push":  # push stalled (slow global mixing): finish with Jacobi-PCG
            st.path = "push+pcg"
            return pcg(sys, Z, rhs, eps, jacobi_M(sys), self.max_iter, st, 0)
        st.path = "push+amg" if self._use_amg() else "push+pcg"
        return self._global(sys, Z, rhs, eps, st, amg_box)

    def _floor(self, scale=1.0):
        """Smallest residual ratio the working precision resolves for values of size `scale`."""
        u = 1e-7 if self.dtype == np.float32 else 2e-16
        return min(0.5, max(20 * u, 10 * u * scale))

    def _bound(self, rc, h_est):
        """(hmax, bound) from the column residual ratios; inf until h itself is converged."""
        C = self.C
        if rc[C] > self.eps_h:
            return h_est, float("inf")
        hmax = h_est / (1 - rc[C])
        return hmax, float(rc[:C].max()) * hmax

    def solve(self, sys):
        xp, st, C, tol = _xp(), Stats(), self.C, self.tol
        sw, s64 = sys.astype(self.dtype), sys.astype(np.float64)  # working / certification precision
        with Timer() as tm:
            if self.H is None:
                self.H = xp.zeros((self.n_total, 1), dtype=self.store_dtype)
            known = ~sw.is_new
            X = supernode_init(sw, self._warm(sw), known)
            Hc = init_h(sw, self.H[sw.U].astype(self.dtype), known, self.hmax or 1.0)
            Z = xp.ascontiguousarray(xp.concatenate([X, Hc], axis=1))
            rhs = xp.ascontiguousarray(xp.concatenate([sw.rhs, sw.diag[:, None]], axis=1))
            ops, amg_box, hmax = FrontierOps(sw.W, C + 1, self.group), [None], self.hmax
            if hmax is None and sw.n_u:  # first batch: get h before choosing the label tolerance
                M, wm = self._precond(sw, st, amg_box) if self.method == "amg" else (jacobi_M(sw), 0)
                Z[:, C:] = pcg(sw, Z[:, C:].copy(), rhs[:, C:], [self.eps_h], M, self.max_iter, st, wm)
                hmax = float(Z[:, C].max()) / (1 - self.eps_h)
            self.hmax = hmax = hmax or 1.0  # lets auto pick its global solver by difficulty
            eps = [max(tol / (1.25 * hmax), self._floor())] * C + [max(self.eps_h, self._floor(hmax))]
            Z = self._run(sw, Z, rhs, eps, ops, st, amg_box)
            # certify in float64; if the working precision fell short, refine: solve A D = R (scaled), Z += D
            Z64, rhs64, bound = Z.astype(np.float64), xp.concatenate([s64.rhs, s64.diag[:, None]], axis=1), 0.0
            for _ in range(10 if self.dtype == np.float32 else 2):
                if not sw.n_u:
                    break
                R64 = rhs64 - apply_A(s64, Z64)
                st.edges += s64.nnz
                rc, h_est = rho_cols(s64, R64), float(Z64[:, C].max())
                hmax, bound = self._bound(rc, h_est)
                if bound <= tol:
                    break
                scale = np.where(rc > 0, rc, 1.0)
                eps_c = np.clip(np.array([tol / (1.25 * hmax)] * C + [self.eps_h]) / scale, self._floor(h_est), 1.0)
                Rs = xp.ascontiguousarray((R64 / xp.asarray(scale)[None, :]).astype(self.dtype))
                D = self._run(sw, xp.zeros_like(Rs), Rs, list(eps_c), ops, st, amg_box)
                Z64 += D.astype(np.float64) * xp.asarray(scale)[None, :]
            st.cert, st.hmax, self.hmax = bound, hmax, hmax
            self._store(sw, Z64[:, :C])
            self.H[sw.U] = Z64[:, C:]
        st.edges += int(ops.edges)
        st.ms = tm.ms
        return st


class IncrementalLP(DynLPPlus):
    """Certified DynLP+ with resident state (needs an IncrementalStream system).

    Keeps Z = [F, h] and its float64 residual R between batches: a batch refreshes R
    only on its changed rows, small changes are repaired by the fused asynchronous
    push (GPU) or the vectorized push (CPU), and the certificate is read off the
    kept residual. Large changes and the first batch go through DynLP+. R is
    recomputed in full every `full_every` batches to bound rounding drift.
    """

    def __init__(self, n_total, K, dtype="float32", tol=1e-3, full_every=16, max_rounds=50_000, **kw):
        super().__init__(n_total, K, dtype, method="auto", tol=tol, **kw)
        self.name, self.full_every, self.max_rounds = "dynlp+inc", full_every, max_rounds
        self.Z = self.R = self._fused = None
        self.since_full = 0

    def _rhs64(self, s64):
        return _xp().concatenate([s64.rhs, s64.diag[:, None]], axis=1)

    def _park(self, Z, out):
        """Exact values of inert rows (diag 1, rhs 0; h's rhs is diag): F = 0, h = 1."""
        Z[out, :self.C] = 0
        Z[out, self.C] = 1

    def _full_residual(self, s64, st):
        self.R = _xp().ascontiguousarray(self._rhs64(s64) - apply_A(s64, self.Z))
        st.edges += s64.nnz
        self.since_full = 0

    def _push(self, s64, eps, st):
        """Repair the violating rows of R; True when every row satisfies eps."""
        be, xp = backend.get(), _xp()
        Z, R, eps_d = self.Z, self.R, xp.asarray(eps, dtype=xp.float64)
        if be.is_gpu and Z.shape[1] <= 32:
            if self._fused is None or self._fused.W is not s64.W:
                from .kernels import FusedPush
                self._fused = FusedPush(s64.W, Z.shape[1], self.group)
            cand = xp.flatnonzero((xp.abs(R) > eps_d[None, :] * s64.diag[:, None]).any(axis=1)).astype(xp.int32)
            rounds, ok, edges = self._fused.run(Z, R, s64.diag, eps_d, cand, self.max_rounds)
            st.iters, st.edges = st.iters + rounds, st.edges + edges
            return ok
        ops = FrontierOps(s64.W, Z.shape[1], self.group)
        ok = push(s64, Z, R, eps, ops, st, max_iter=self.max_rounds)[2]
        st.edges += int(ops.edges)
        return ok

    def solve(self, sys):
        xp = _xp()
        if sys.solved is None:
            raise ValueError("dynlp+inc needs an IncrementalStream system (bench --incremental)")
        s64, C, tol, st = sys.astype(np.float64), self.C, self.tol, Stats(path="inc-push")
        with Timer() as tm:
            small = self.Z is not None
            if small:
                Z, R = self.Z, self.R
                self._park(Z, ~s64.solved)
                R[~s64.solved] = 0
                known = s64.solved & ~s64.is_new
                supernode_init(s64, Z[:, :C], known)
                init_h(s64, Z[:, C:], known, self.hmax)
                if (Q := s64.changed).shape[0]:
                    WQ = s64.W[Q]
                    R[Q] = self._rhs64(s64)[Q] - s64.diag[Q][:, None] * Z[Q] + spmm(WQ, Z)
                    st.edges += int(WQ.nnz)
                self.since_full += 1
                if self.since_full >= self.full_every:  # bound rounding drift in the kept residual
                    self._full_residual(s64, st)
                eps = [tol / (1.25 * (self.hmax or 1.0))] * C + [self.eps_h]
                viol = (xp.abs(self.R) > xp.asarray(eps)[None, :] * s64.diag[:, None]).any(axis=1)
                small = float(viol.sum()) / max(s64.n_solved, 1) <= self.push_frac
            if small:
                bound = float("inf")
                for _ in range(4):
                    ok = self._push(s64, eps, st)
                    hmax, bound = self._bound(rho_cols(s64, self.R), float(self.Z[:, C].max()))
                    if ok and bound <= tol:
                        break
                    eps = [tol / (1.25 * hmax)] * C + [self.eps_h]
                small, st.cert, st.hmax, self.hmax = bound <= tol, bound, hmax, hmax
            if not small:  # first or large batch, or push did not certify: global solve
                if self.Z is not None:
                    self.F, self.H = self.Z[:, :C].copy(), self.Z[:, C:].copy()
                inner = DynLPPlus.solve(self, sys)
                st.iters, st.edges, st.setup_ms = st.iters + inner.iters, st.edges + inner.edges, inner.setup_ms
                st.cert, st.hmax, st.path = inner.cert, inner.hmax, "inc-" + inner.path
                self.Z = xp.ascontiguousarray(xp.concatenate([self.F, self.H], axis=1).astype(np.float64))
                self._park(self.Z, ~s64.solved)
                self._full_residual(s64, st)
            self.F, self.H = self.Z[:, :C], self.Z[:, C:]
        st.ms = tm.ms
        return st


class Reference:
    """High-accuracy float64 solution F* and absorption bound (metrics only)."""

    def __init__(self, n_total, K, tol=1e-8):
        self.inner = DynLPPlus(n_total, K, "float64", method="amg", tol=tol, eps_h=1e-6, max_iter=100_000)

    def solve(self, sys):
        st = self.inner.solve(sys)
        return self.inner.scores(sys), self.inner.hmax, st


def make(name, n_total, K, *, dtype, delta, tol, group, max_iter=None):
    kw = {} if max_iter is None else {"max_iter": max_iter}
    if name in ("itlp", "itlp-warm"):
        return ItLP(n_total, K, dtype, delta, warm=name == "itlp-warm", **kw)
    if name in ("dynlp", "dynlp-knowninit"):
        return DynLP(n_total, K, dtype, delta, group, known_init=name == "dynlp-knowninit", **kw)
    if name == "dynlp+inc":
        return IncrementalLP(n_total, K, dtype, tol=tol, group=group, **kw)
    if name.startswith("dynlp+"):
        return DynLPPlus(n_total, K, dtype, method=name.split("+", 1)[1], tol=tol, group=group, **kw)
    raise ValueError(f"unknown solver {name!r}")


SOLVERS = ["itlp", "itlp-warm", "dynlp", "dynlp-knowninit", "dynlp+push", "dynlp+pcg", "dynlp+amg",
           "dynlp+auto", "dynlp+inc"]
