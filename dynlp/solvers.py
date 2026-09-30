"""Label-propagation solvers for one batch of a stream.

Baselines
  itlp        Jacobi from scratch on every batch (the paper's ItLP).
  itlp-warm   Jacobi warm-started from the previous batch (a missing baseline).
  dynlp       Faithful DynLP (Algorithm 2): component init + frontier Jacobi with
              a stop-on-small-change rule (delta).

DynLP+ (certified): warm start, known-neighbour supernode init, and a
stopping rule that bounds the true error ||F - F*||_inf <= tol.
  push        residual-driven frontier relaxation (local; good for small batches)
  pcg         Jacobi-preconditioned block CG (all classes at once)
  amg         CG with an aggregation-AMG V-cycle preconditioner
  auto        push when the residual is local, else AMG-PCG

Error certificate. A = diag - W is a nonsingular M-matrix, so A^{-1} >= 0
entrywise. If |r| <= rho * diag (elementwise), then |F - F*| = |A^{-1} r| <=
rho * A^{-1} diag = rho * h. Here h solves A h = diag; h_u is the expected
number of random-walk visits before absorption at a labeled vertex. DynLP+
carries h as one extra column of the same block solve.
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
    edges: int = 0            # edge visits: hardware-independent work
    ms: float = 0.0           # wall time: init + setup + solve
    setup_ms: float = 0.0     # AMG setup (included in ms)
    path: str = ""
    cert: float = float("nan")  # certified bound reported by the solver itself
    hmax: float = float("nan")  # max expected visits before absorption (DynLP+ only)


# ----------------------------------------------------------------------------
# linear-algebra helpers
# ----------------------------------------------------------------------------

def spmm(W, X):
    if X.shape[1] == 1:
        return (W @ X[:, 0])[:, None]
    return W @ X


def apply_A(sys: System, X):
    return sys.diag[:, None] * X - spmm(sys.W, X)


def rho_cols(sys: System, R):
    """max_u |R_uc| / diag_u for every column c (host floats)."""
    xp = backend.get().xp
    if R.shape[0] == 0:
        return np.zeros(R.shape[1])
    return backend.get().asnumpy((xp.abs(R) / sys.diag[:, None]).max(axis=0)).astype(float)


def violation(sys: System, R, eps_d):
    """max over entries of |R_uc| / (eps_c * diag_u); <= 1 means converged."""
    xp = backend.get().xp
    if R.shape[0] == 0:
        return 0.0
    return float((xp.abs(R) / (eps_d[None, :] * sys.diag[:, None])).max())


def pcg(sys: System, Z, rhs, eps, M, max_iter, st: Stats, work_M=0, patience=400):
    """Block PCG, one independent CG per column, stopped when |r_c| <= eps_c * diag.

    Stops early if the violation stalls for `patience` iterations. That happens
    when eps is below what the working precision can resolve; the caller's
    float64 refinement then takes over.
    """
    xp = backend.get().xp
    eps_d = xp.asarray(eps, dtype=Z.dtype)
    R = rhs - apply_A(sys, Z)
    st.edges += sys.nnz
    best = violation(sys, R, eps_d)
    if best <= 1:
        return Z
    Y = M(R)
    P = Y.copy()
    rz = (R * Y).sum(axis=0)
    stall = 0
    for _ in range(max_iter):
        AP = apply_A(sys, P)
        pAp = (P * AP).sum(axis=0)
        alpha = xp.where(pAp > 0, rz / xp.where(pAp > 0, pAp, 1), 0)
        Z += alpha[None, :] * P
        R -= alpha[None, :] * AP
        st.iters += 1
        st.edges += sys.nnz + work_M
        v = violation(sys, R, eps_d)
        if v <= 1:
            R = rhs - apply_A(sys, Z)  # guard against recurrence drift
            st.edges += sys.nnz
            v = violation(sys, R, eps_d)
            if v <= 1:
                break
        if v < 0.9 * best:
            best, stall = v, 0
        else:
            stall += 1
            if stall >= patience:
                break
        Y = M(R)
        rz_new = (R * Y).sum(axis=0)
        beta = xp.where(rz > 0, rz_new / xp.where(rz > 0, rz, 1), 0)
        P = Y + beta[None, :] * P
        rz = rz_new
    return Z


def push(sys: System, Z, R, eps, ops: FrontierOps, st: Stats, budget_edges=None, max_iter=10**7):
    """Residual push (Jacobi on the violating set) until |r_c| <= eps_c * diag.

    Returns (Z, R, converged). R is kept exact up to rounding by scattering each
    update to the neighbours, so the stopping test is a true residual test.
    """
    xp = backend.get().xp
    n_u = sys.n_u
    eps_d = xp.asarray(eps, dtype=Z.dtype)

    def violating(idx):
        return (xp.abs(R[idx]) > eps_d[None, :] * sys.diag[idx][:, None]).any(axis=1)

    all_idx = xp.arange(n_u, dtype=xp.int32)
    cand = all_idx[violating(all_idx)]
    e0 = int(ops.edges)
    it, best, stall = 0, float("inf"), 0
    while cand.shape[0] > 0 and it < max_iter:
        D = R[cand] / sys.diag[cand][:, None]
        Z[cand] += D
        R[cand] = 0
        mark = xp.zeros(n_u, dtype=xp.uint8)
        mark[cand] = 1
        ops.push(cand, D, R, mark)
        touched = xp.flatnonzero(mark).astype(xp.int32)
        cand = touched[violating(touched)]
        it += 1
        if it % 32 == 0:
            if budget_edges is not None and int(ops.edges) - e0 > budget_edges:
                break
            v = float((xp.abs(R) / (eps_d[None, :] * sys.diag[:, None])).max())
            if v < 0.99 * best:
                best, stall = v, 0
            else:
                stall += 1
                if stall >= 8:  # no progress in 256 rounds: precision floor
                    break
    st.iters += it
    return Z, R, cand.shape[0] == 0


def supernode_init(sys: System, X, known=None, cols=None):
    """Initialize new vertices by component (DynLP's supernodes).

    known=None: DynLP's rule, the share of each component's weight that goes
    to each ground-truth class. known=mask: weighted average over all known
    neighbours (ground truth plus the previous batch's solution).
    """
    B = backend.get()
    xp = B.xp
    new = xp.flatnonzero(sys.is_new)
    if new.shape[0] == 0:
        return X
    C = sys.C
    cols = slice(0, C) if cols is None else cols
    s_ = sys.label_cols(sys.B)[new]
    t = sys.B[new].sum(axis=1)
    if known is not None:
        kf = known.astype(X.dtype)
        Wn = sys.W[new]
        s_ = s_ + spmm(Wn, X[:, cols] * kf[:, None])
        t = t + Wn @ kf
    S = xp.zeros((sys.n_comp, C), dtype=X.dtype)
    T = xp.zeros(sys.n_comp, dtype=X.dtype)
    B.scatter_add(S, sys.comp, s_)
    B.scatter_add(T, sys.comp, t)
    ok = T > 0
    Fc = xp.where(ok[:, None], S / xp.where(ok, T, 1)[:, None], sys.prior[None, :])
    X[new, cols] = Fc[sys.comp]
    return X


def init_h(sys: System, H, known, fallback):
    """New vertices: h ~ 1 + average h of known neighbours (labeled ones have h = 0)."""
    B = backend.get()
    xp = B.xp
    new = xp.flatnonzero(sys.is_new)
    if new.shape[0] == 0:
        return H
    kf = known.astype(H.dtype)
    Wn = sys.W[new]
    s_ = Wn @ (H[:, 0] * kf)
    t = sys.B[new].sum(axis=1) + Wn @ kf
    S = xp.zeros(sys.n_comp, dtype=H.dtype)
    T = xp.zeros(sys.n_comp, dtype=H.dtype)
    B.scatter_add(S, sys.comp, s_)
    B.scatter_add(T, sys.comp, t)
    hc = xp.where(T > 0, 1 + S / xp.where(T > 0, T, 1), fallback)
    H[new, 0] = hc[sys.comp]
    return H


def predict(F, K):
    xp = backend.get().xp
    return (F[:, 0] > 0.5).astype(xp.int32) if K == 2 else F.argmax(axis=1).astype(xp.int32)


# ----------------------------------------------------------------------------
# solvers
# ----------------------------------------------------------------------------

class Solver:
    name = "base"

    def __init__(self, n_total, K, dtype="float32", store_dtype=None):
        self.n_total, self.K, self.dtype = n_total, K, np.dtype(dtype)
        self.store_dtype = np.dtype(store_dtype or dtype)
        self.C = 1 if K == 2 else K
        self.F = None

    def _warm(self, sys: System):
        xp = backend.get().xp
        if self.F is None:
            self.F = xp.empty((self.n_total, self.C), dtype=self.store_dtype)
            self.F[:] = sys.prior.astype(self.store_dtype)
        return self.F[sys.U].astype(self.dtype)

    def _store(self, sys: System, X):
        self.F[sys.U] = X

    def scores(self, sys: System):
        return self.F[sys.U]

    def solve(self, sys: System) -> Stats:
        raise NotImplementedError


class ItLP(Solver):
    def __init__(self, n_total, K, dtype="float32", delta=1e-4, warm=False, max_iter=200_000):
        super().__init__(n_total, K, dtype)
        self.delta, self.warm, self.max_iter = delta, warm, max_iter
        self.name = "itlp-warm" if warm else "itlp"

    def solve(self, sys):
        xp = backend.get().xp
        st = Stats(path="jacobi")
        sys = sys.astype(self.dtype)
        with Timer() as tm:
            if self.warm:
                X = supernode_init(sys, self._warm(sys), known=~sys.is_new)
            else:
                self._warm(sys)
                X = xp.empty((sys.n_u, self.C), dtype=self.dtype)
                X[:] = sys.prior
            for _ in range(self.max_iter):
                Y = (sys.rhs + spmm(sys.W, X)) / sys.diag[:, None]
                ch = float(xp.abs(Y - X).max()) if sys.n_u else 0.0
                X = Y
                st.iters += 1
                st.edges += sys.nnz
                if ch <= self.delta:
                    break
            self._store(sys, X)
        st.ms = tm.ms
        return st


class DynLP(Solver):
    """Faithful DynLP (Algorithm 2), with |.| in the line-29 change test."""

    def __init__(self, n_total, K, dtype="float32", delta=1e-4, group="auto",
                 known_init=False, max_iter=10**7):
        super().__init__(n_total, K, dtype)
        self.delta, self.group, self.known_init, self.max_iter = delta, group, known_init, max_iter
        self.name = "dynlp-knowninit" if known_init else "dynlp"

    def solve(self, sys):
        xp = backend.get().xp
        st = Stats(path="frontier-jacobi")
        sys = sys.astype(self.dtype)
        with Timer() as tm:
            X = self._warm(sys)
            X = supernode_init(sys, X, known=(~sys.is_new) if self.known_init else None)
            ops = FrontierOps(sys.W, self.C, self.group)
            frontier = xp.flatnonzero(sys.seed).astype(xp.int32)
            while frontier.shape[0] > 0 and st.iters < self.max_iter:
                Y, ch = ops.jacobi(frontier, X, sys.rhs, sys.diag, self.delta)
                st.iters += 1
                upd = frontier[ch]
                if upd.shape[0] == 0:
                    break
                X[upd] = Y[ch]
                mark = xp.zeros(sys.n_u, dtype=xp.uint8)
                mark[upd] = 1
                ops.mark_neighbors(upd, mark)
                frontier = xp.flatnonzero(mark).astype(xp.int32)
            self._store(sys, X)
        st.edges = int(ops.edges)
        st.ms = tm.ms
        return st


class DynLPPlus(Solver):
    """Certified DynLP+: guarantees ||F - F*||_inf <= tol on every batch."""

    def __init__(self, n_total, K, dtype="float32", method="auto", tol=1e-3, eps_h=0.05,
                 group="auto", push_frac=0.05, push_budget=30.0, amg_hmax=2000.0,
                 max_iter=100_000, amg_opts=None):
        # float64 master copy of F and h; the solves run in `dtype`
        super().__init__(n_total, K, dtype, store_dtype="float64")
        self.method, self.tol, self.eps_h = method, tol, eps_h
        self.group, self.push_frac, self.push_budget = group, push_frac, push_budget
        self.amg_hmax = amg_hmax  # auto: use AMG only when the problem is hard (large h)
        self.max_iter, self.amg_opts = max_iter, amg_opts or {}
        self.name = f"dynlp+{method}"
        self.H, self.hmax = None, None

    def _amg(self, sys, st):
        with Timer() as t:
            amg = AMG(sys.W, sys.s, **self.amg_opts)
        st.setup_ms += t.ms
        st.edges += 10 * sys.nnz  # rough setup cost: matching rounds + Galerkin
        return amg

    def _use_amg(self):
        if self.method == "amg":
            return True
        if self.method == "auto":
            return (self.hmax or float("inf")) > self.amg_hmax
        return False

    def _precond(self, sys, st, amg_box):
        if not self._use_amg():
            return (lambda R: R / sys.diag[:, None]), 0
        if amg_box[0] is None:
            amg_box[0] = self._amg(sys, st)
        return amg_box[0], amg_box[0].work_per_apply

    def _run(self, sys, Z, rhs, eps, ops, st, amg_box):
        xp = backend.get().xp
        if self.method in ("push", "auto"):
            R = rhs - apply_A(sys, Z)
            st.edges += sys.nnz
            if self.method == "auto":
                frac = float((xp.abs(R) > xp.asarray(eps, dtype=Z.dtype)[None, :] * sys.diag[:, None])
                             .any(axis=1).mean()) if sys.n_u else 0.0
                if frac > self.push_frac:
                    st.path = "amg" if self._use_amg() else "pcg"
                    M, wm = self._precond(sys, st, amg_box)
                    return pcg(sys, Z, rhs, eps, M, self.max_iter, st, wm)
            budget = None if self.method == "push" else self.push_budget * max(sys.nnz, 1)
            Z, R, ok = push(sys, Z, R, eps, ops, st, budget)
            st.path = "push"
            if ok:
                return Z
            if self.method == "push":  # push stalled (slow global mixing): finish with Jacobi-PCG
                st.path = "push+pcg"
                return pcg(sys, Z, rhs, eps, lambda R_: R_ / sys.diag[:, None], self.max_iter, st, 0)
            st.path = "push+amg" if self._use_amg() else "push+pcg"
        else:
            st.path = self.method
        M, wm = self._precond(sys, st, amg_box)
        return pcg(sys, Z, rhs, eps, M, self.max_iter, st, wm)

    def _floor(self, scale=1.0):
        """Smallest residual ratio the working precision resolves for values of size `scale`."""
        u = 1e-7 if self.dtype == np.float32 else 2e-16
        return min(0.5, max(20 * u, 10 * u * scale))

    def solve(self, sys):
        xp = backend.get().xp
        st = Stats()
        sw = sys.astype(self.dtype)       # working precision
        s64 = sys.astype(np.float64)      # certification precision
        C, tol = self.C, self.tol
        with Timer() as tm:
            X = self._warm(sw)
            if self.H is None:
                self.H = xp.zeros((self.n_total, 1), dtype=self.store_dtype)
            known = ~sw.is_new
            X = supernode_init(sw, X, known)
            Hc = init_h(sw, self.H[sw.U].astype(self.dtype), known, self.hmax or 1.0)
            Z = xp.ascontiguousarray(xp.concatenate([X, Hc], axis=1))
            rhs = xp.ascontiguousarray(xp.concatenate([sw.rhs, sw.diag[:, None]], axis=1))
            ops = FrontierOps(sw.W, C + 1, self.group)
            amg_box = [None]
            hmax = self.hmax
            if hmax is None and sw.n_u:  # first batch: get h before choosing the label tolerance
                M, wm = self._precond(sw, st, amg_box) if self.method == "amg" else                     ((lambda R: R / sw.diag[:, None]), 0)
                Z[:, C:] = pcg(sw, Z[:, C:].copy(), rhs[:, C:], [self.eps_h], M, self.max_iter, st, wm)
                hmax = float(Z[:, C].max()) / (1 - self.eps_h)
            hmax = hmax or 1.0
            self.hmax = hmax  # lets `auto` pick its global solver by difficulty
            eps = [max(tol / (1.25 * hmax), self._floor())] * C + [max(self.eps_h, self._floor(hmax))]
            Z = self._run(sw, Z, rhs, eps, ops, st, amg_box)

            # Certify in float64. If the working precision could not reach the
            # target, refine: solve A D = R (scaled) in working precision, Z += D.
            Z64 = Z.astype(np.float64)
            rhs64 = xp.concatenate([s64.rhs, s64.diag[:, None]], axis=1)
            bound = 0.0
            for _ in range(10 if self.dtype == np.float32 else 2):
                if sw.n_u == 0:
                    break
                R64 = rhs64 - apply_A(s64, Z64)
                st.edges += s64.nnz
                rc = rho_cols(s64, R64)
                h_est = float(Z64[:, C].max())
                h_ok = rc[C] <= self.eps_h
                hmax = h_est / (1 - rc[C]) if h_ok else h_est
                bound = float(rc[:C].max()) * hmax if h_ok else float("inf")
                if bound <= tol:
                    break
                target = np.array([tol / (1.25 * hmax)] * C + [self.eps_h])
                scale = np.where(rc > 0, rc, 1.0)
                eps_c = np.clip(target / scale, self._floor(h_est), 1.0)
                Rs = xp.ascontiguousarray((R64 / xp.asarray(scale)[None, :]).astype(self.dtype))
                D = xp.zeros_like(Rs)
                D = self._run(sw, D, Rs, list(eps_c), ops, st, amg_box)
                Z64 += D.astype(np.float64) * xp.asarray(scale)[None, :]
            st.cert, st.hmax = bound, hmax
            self._store(sw, Z64[:, :C])
            self.H[sw.U] = Z64[:, C:]
            self.hmax = hmax
        st.edges += int(ops.edges)
        st.ms = tm.ms
        return st


class IncrementalLP(DynLPPlus):
    """Certified DynLP+ with resident state (needs an IncrementalStream system).

    Keeps Z = [F, h] and its float64 residual R between batches:
      - a batch refreshes R only on the rows whose equation changed (sys.changed);
      - small changes are repaired by the fused asynchronous push, in float64, on
        the GPU (kernels.FusedPush), or by the vectorized push on CPU;
      - the certificate is read off the maintained residual (an O(n) scan) instead
        of a fresh full residual;
      - large changes (and the first batch) go through DynLP+ and re-initialize R.
    R is recomputed in full every `full_every` batches to bound rounding drift.
    """

    def __init__(self, n_total, K, dtype="float32", tol=1e-3, full_every=16, max_rounds=50_000, **kw):
        super().__init__(n_total, K, dtype, method="auto", tol=tol, **kw)
        self.name = "dynlp+inc"
        self.full_every, self.max_rounds = full_every, max_rounds
        self.Z = self.R = None
        self.since_full = 0
        self._fused = None

    def _rhs64(self, s64):
        xp = backend.get().xp
        return xp.concatenate([s64.rhs, s64.diag[:, None]], axis=1)

    def _park(self, Z, out):
        """Exact values of inert rows (diag 1, rhs 0; h's rhs is diag): F = 0, h = 1."""
        Z[out, :self.C] = 0
        Z[out, self.C] = 1

    def _reset_state(self, s64, st):
        """Adopt DynLP+'s solution as the resident state and compute its residual in full."""
        xp = backend.get().xp
        Z = xp.ascontiguousarray(xp.concatenate([self.F, self.H], axis=1).astype(xp.float64))
        self._park(Z, ~s64.solved)
        self.Z = Z
        self.R = xp.ascontiguousarray(self._rhs64(s64) - apply_A(s64, Z))
        st.edges += s64.nnz
        self.since_full = 0

    def _push(self, s64, eps, st):
        """Repair the violating rows of R; returns True when every row satisfies eps."""
        B = backend.get()
        xp = B.xp
        Z, R = self.Z, self.R
        eps_d = xp.asarray(eps, dtype=xp.float64)
        viol = (xp.abs(R) > eps_d[None, :] * s64.diag[:, None]).any(axis=1)
        cand = xp.flatnonzero(viol).astype(xp.int32)
        if B.is_gpu and Z.shape[1] <= 32:
            if self._fused is None or self._fused.W is not s64.W:
                from .kernels import FusedPush
                self._fused = FusedPush(s64.W, Z.shape[1], self.group)
            rounds, ok, edges = self._fused.run(Z, R, s64.diag, eps_d, cand, self.max_rounds)
            st.iters += rounds
            st.edges += edges
            return ok
        ops = FrontierOps(s64.W, Z.shape[1], self.group)
        _, _, ok = push(s64, Z, R, eps, ops, st, max_iter=self.max_rounds)
        st.edges += int(ops.edges)
        return ok

    def solve(self, sys):
        xp = backend.get().xp
        if sys.solved is None:
            raise ValueError("dynlp+inc needs an IncrementalStream system (bench --incremental)")
        s64 = sys.astype(np.float64)
        C, tol = self.C, self.tol
        st = Stats(path="inc-push")
        with Timer() as tm:
            small = self.Z is not None
            if small:
                Z, R = self.Z, self.R
                out = ~s64.solved
                self._park(Z, out)
                R[out] = 0
                known = s64.solved & ~s64.is_new
                supernode_init(s64, Z[:, :C], known)
                init_h(s64, Z[:, C:], known, self.hmax)
                Q = s64.changed
                if Q.shape[0]:
                    WQ = s64.W[Q]
                    R[Q] = self._rhs64(s64)[Q] - s64.diag[Q][:, None] * Z[Q] + spmm(WQ, Z)
                    st.edges += int(WQ.nnz)
                self.since_full += 1
                if self.since_full >= self.full_every:  # bound rounding drift in the kept residual
                    self.R = R = xp.ascontiguousarray(self._rhs64(s64) - apply_A(s64, Z))
                    st.edges += s64.nnz
                    self.since_full = 0
                hmax = self.hmax or 1.0
                eps = [tol / (1.25 * hmax)] * C + [self.eps_h]
                eps_d = xp.asarray(eps)
                frac = float((xp.abs(R) > eps_d[None, :] * s64.diag[:, None]).any(axis=1).sum()) \
                    / max(s64.n_solved, 1)
                small = frac <= self.push_frac
            if small:
                bound = float("inf")
                for _ in range(4):
                    ok = self._push(s64, eps, st)
                    rc = rho_cols(s64, self.R)
                    h_est = float(self.Z[:, C].max())
                    h_ok = rc[C] <= self.eps_h
                    hmax = h_est / (1 - rc[C]) if h_ok else h_est
                    bound = float(rc[:C].max()) * hmax if h_ok else float("inf")
                    if ok and bound <= tol:
                        break
                    eps = [tol / (1.25 * hmax)] * C + [self.eps_h]
                small = bound <= tol
                st.cert, st.hmax = bound, hmax
                self.hmax = hmax
            if not small:  # first or large batch, or push did not certify: global solve
                if self.Z is not None:
                    self.F = self.Z[:, :C].copy()
                    self.H = self.Z[:, C:].copy()
                inner = DynLPPlus.solve(self, sys)
                st.iters += inner.iters
                st.edges += inner.edges
                st.setup_ms += inner.setup_ms
                st.cert, st.hmax, st.path = inner.cert, inner.hmax, "inc-" + inner.path
                self._reset_state(s64, st)
            self.F = self.Z[:, :C]
            self.H = self.Z[:, C:]
        st.ms = tm.ms
        return st


class Reference:
    """High-accuracy float64 solution F* and absorption bound (for metrics only)."""

    def __init__(self, n_total, K, tol=1e-8):
        self.inner = DynLPPlus(n_total, K, "float64", method="amg", tol=tol, eps_h=1e-6,
                               max_iter=100_000)

    def solve(self, sys):
        st = self.inner.solve(sys)
        return self.inner.scores(sys), self.inner.hmax, st


def make(name, n_total, K, *, dtype, delta, tol, group, max_iter=None):
    kw = {}
    if max_iter is not None:
        kw["max_iter"] = max_iter
    if name == "itlp":
        return ItLP(n_total, K, dtype, delta, **kw)
    if name == "itlp-warm":
        return ItLP(n_total, K, dtype, delta, warm=True, **kw)
    if name == "dynlp":
        return DynLP(n_total, K, dtype, delta, group, **kw)
    if name == "dynlp-knowninit":
        return DynLP(n_total, K, dtype, delta, group, known_init=True, **kw)
    if name == "dynlp+inc":
        return IncrementalLP(n_total, K, dtype, tol=tol, group=group, **kw)
    if name.startswith("dynlp+"):
        return DynLPPlus(n_total, K, dtype, method=name.split("+", 1)[1], tol=tol, group=group, **kw)
    raise ValueError(f"unknown solver {name!r}")


SOLVERS = ["itlp", "itlp-warm", "dynlp", "dynlp-knowninit",
           "dynlp+push", "dynlp+pcg", "dynlp+amg", "dynlp+auto", "dynlp+inc"]
