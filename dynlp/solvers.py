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
