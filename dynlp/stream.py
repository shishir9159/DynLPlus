"""Streaming protocol and the per-batch linear system.

For each batch we form the harmonic system on the active unlabeled set U:

    (diag - W_UU) F_U = rhs,   diag = s + rowsum(W_UU),   s = w(u, L) + eta

``s`` is the "grounding" of each vertex: its total edge weight to labeled
vertices plus Zhu et al.'s dongle regularizer eta (which pulls components
without any labeled vertex to the class prior). Binary problems solve one
column (score of class 1); K-class problems solve K columns at once (SpMM).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import backend
from .cc import connected_components
from .graphs import Dataset, csr_rows, row_positions


@dataclass
class System:
    t: int
    U: object          # global ids of active unlabeled vertices (int64, sorted)
    W: object          # CSR n_u x n_u (U-U edges)
    s: object          # (n_u,) grounding: weight to labeled + eta
    diag: object       # (n_u,) s + rowsum(W)
    B: object          # (n_u, K) weight to labeled neighbours of each class
    rhs: object        # (n_u, C)
    prior: object      # (C,)
    is_new: object     # (n_u,) bool: not in U at the previous batch
    seed: object       # (n_u,) bool: DynLP's affected set (new, N(inserted), N(deleted))
    comp: object       # (n_new,) component id of each new vertex (tau-sparsified graph)
    n_comp: int
    K: int
    C: int
    eta: float
    n_inserted: int
    n_deleted: int
    n_active: int
    n_ungrounded: int = 0
    # Global-index systems (IncrementalStream) only: rows that are really solved
    # (the others are inert: diag 1, rhs 0, no edges), and rows whose equation
    # changed in this batch.
    solved: object = None
    changed: object = None
    _conv: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def n_u(self) -> int:
        return int(self.U.shape[0])

    @property
    def n_solved(self) -> int:
        return self.n_u if self.solved is None else int(self.solved.sum())

    @property
    def nnz(self) -> int:
        return int(self.W.nnz)

    def label_cols(self, M):
        """Columns of a (., K) class matrix that are solved for (class 1 if binary)."""
        return M[:, 1:2] if self.K == 2 else M

    def astype(self, dtype) -> "System":
        """Same system in another precision (cached; index arrays are shared)."""
        dtype = np.dtype(dtype)
        if dtype == self.W.data.dtype:
            return self
        if dtype in self._conv:
            return self._conv[dtype]
        B = backend.get()
        W = B.csr(self.W.data.astype(dtype), self.W.indices, self.W.indptr, self.W.shape)
        self._conv[dtype] = System(self.t, self.U, W, self.s.astype(dtype), self.diag.astype(dtype),
                      self.B.astype(dtype), self.rhs.astype(dtype), self.prior.astype(dtype),
                      self.is_new, self.seed, self.comp, self.n_comp, self.K, self.C, self.eta,
                      self.n_inserted, self.n_deleted, self.n_active, self.n_ungrounded,
                      self.solved, self.changed)
        return self._conv[dtype]


class Stream:
    """Reveals a dataset in batches of insertions and deletions.

    Mirrors the paper's protocol: an initial snapshot, then batches of new
    vertices (a ``label_frac`` fraction carries ground truth) plus deletions of
    ``del_frac`` x batch-size random active unlabeled vertices.
    """

    def __init__(self, ds: Dataset, *, label_frac=0.01, init_frac=0.1, n_batches=10,
                 del_frac=0.1, eta_rel=0.0, seed=0, dtype="float32"):
        B = backend.get()
        xp = B.xp
        self.ds, self.dtype = ds, dtype
        n, K = ds.n, ds.K
        rs = np.random.RandomState(seed + 7)
        y_host = B.asnumpy(ds.y)
        labeled = np.zeros(n, bool)
        cand = np.flatnonzero(y_host >= 0)
        labeled[rs.choice(cand, size=max(K, int(label_frac * n)), replace=False)] = True
        order = ds.order.copy()
        n0 = max(int(init_frac * n), 2 * K)
        # the initial snapshot must contain a labeled vertex of every class
        pos = np.empty(n, np.int64)
        pos[order] = np.arange(n)
        for c in range(K):
            lc = np.flatnonzero(labeled & (y_host == c))
            if lc.size == 0:
                lc = np.flatnonzero(y_host == c)[:1]
                labeled[lc] = True
            if lc.size and pos[lc].min() >= n0:
                v = lc[0]
                j = rs.choice(np.flatnonzero(~labeled[order[:n0]]))
                a, pv = order[j], pos[v]
                order[j], order[pv] = v, a
                pos[v], pos[a] = j, pv
        self.labeled_host = labeled
        self.order = order
        self.n0 = n0
        self.chunks = np.array_split(order[n0:], n_batches) if n_batches > 0 else []
        self.del_frac = del_frac
        self.rs = rs
        # device-side constants
        self.labeled = xp.asarray(labeled)
        self.y = ds.y
        self.rows = csr_rows(ds.A)
        self.tau = float(ds.A.data.mean())
        self.eta = float(eta_rel * ds.A.data.sum() / n)
        self.prior = xp.full(1 if K == 2 else K, 0.5 if K == 2 else 1.0 / K, dtype=dtype)
        # The whole insert/delete schedule is drawn up front, so every stream
        # class sees identical batches and sampling is not billed as assembly.
        self.schedule = self._make_schedule()

    def _make_schedule(self):
        n = self.ds.n
        active_h = np.zeros(n, bool)
        sched = []
        for t in range(1 + len(self.chunks)):
            if t == 0:
                ins = self.order[: self.n0]
                dele = np.zeros(0, np.int64)
            else:
                ins = self.chunks[t - 1]
                pool = np.flatnonzero(active_h & ~self.labeled_host)
                nd = min(int(self.del_frac * len(ins)), max(pool.size - 1, 0))
                dele = self.rs.choice(pool, size=nd, replace=False) if nd > 0 else np.zeros(0, np.int64)
            active_h[ins] = True
            active_h[dele] = False
            sched.append((np.asarray(ins, np.int64), np.asarray(dele, np.int64)))
        return sched

    def __len__(self):
        return len(self.schedule)

    def batches(self):
        B = backend.get()
        xp = B.xp
        n = self.ds.n
        active = xp.zeros(n, dtype=bool)
        prevU = xp.zeros(n, dtype=bool)
        for t, (ins_h, dele_h) in enumerate(self.schedule):
            ins, dele = xp.asarray(ins_h), xp.asarray(dele_h)
            active[ins] = True
            active[dele] = False
            sys = self._build(t, active, prevU, ins, dele)
            prevU = xp.zeros(n, dtype=bool)
            prevU[sys.U] = True
            yield sys

    def _build(self, t, active, prevU, ins, dele) -> System:
        B = backend.get()
        xp = B.xp
        A, K, dt = self.ds.A, self.ds.K, self.dtype
        n = A.shape[0]
        Lm = active & self.labeled
        Um = active & ~self.labeled
        # class-wise weight to labeled neighbours: (A @ onehot(L)) restricted to U
        onehot = xp.zeros((n, K), dtype=dt)
        li = xp.flatnonzero(Lm)
        onehot[li, self.y[li]] = 1
        Bfull = A @ onehot

        # Components of U with no path to a labeled vertex have the prior as their
        # harmonic value (and an unbounded h), so they are set aside, not solved.
        U0 = xp.flatnonzero(Um).astype(xp.int64)
        loc = xp.full(n, -1, dtype=xp.int32)
        loc[U0] = xp.arange(U0.shape[0], dtype=xp.int32)
        keep = Um[self.rows] & Um[A.indices]
        r0, c0 = loc[self.rows[keep]], loc[A.indices[keep]]
        lab, ncomp = connected_components(int(U0.shape[0]), r0, c0)
        gw = xp.zeros(ncomp, dtype=dt)
        B.scatter_add(gw, lab, Bfull[U0].sum(axis=1).astype(dt))
        grounded = gw[lab] > 0
        Um = Um.copy()
        Um[U0[~grounded]] = False
        n_ungrounded = int((~grounded).sum())

        U = xp.flatnonzero(Um).astype(xp.int64)
        n_u = int(U.shape[0])
        loc[:] = -1
        loc[U] = xp.arange(n_u, dtype=xp.int32)

        # U-U block: rows of A are sorted, so filtered entries stay in CSR order
        keep = Um[self.rows] & Um[A.indices]
        r = loc[self.rows[keep]]
        c = loc[A.indices[keep]]
        w = A.data[keep].astype(dt)
        indptr = xp.zeros(n_u + 1, dtype=xp.int32)
        indptr[1:] = xp.cumsum(xp.bincount(r, minlength=n_u))
        W = B.csr(w, c, indptr, (n_u, n_u))

        Bm = Bfull[U].astype(dt)
        s = Bm.sum(axis=1) + dt_scalar(self.eta, dt)
        rowsum = _rowsum(W, n_u, dt)
        diag = s + rowsum
        C = 1 if K == 2 else K
        rhs = (Bm[:, 1:2] if K == 2 else Bm) + self.eta * self.prior[None, :]

        is_new = ~prevU[U]
        # DynLP affected set: new vertices and neighbours of inserted/deleted ones
        touch = xp.zeros(n, dtype=dt)
        touch[ins] = 1
        touch[dele] = 1
        seed = is_new | ((A @ touch)[U] > 0)

        # DynLP step 1: components of new vertices over edges heavier than tau
        new_idx = xp.flatnonzero(is_new)
        newloc = xp.full(n_u, -1, dtype=xp.int32)
        newloc[new_idx] = xp.arange(new_idx.shape[0], dtype=xp.int32)
        rW = csr_rows(W)
        e = is_new[rW] & is_new[W.indices] & (W.data > self.tau)
        comp, n_comp = connected_components(int(new_idx.shape[0]), newloc[rW[e]], newloc[W.indices[e]])

        return System(t, U, W, s.astype(dt), diag.astype(dt), Bm, rhs.astype(dt),
                      self.prior.astype(dt), is_new, seed, comp, n_comp, K, C, self.eta,
                      int(ins.shape[0]), int(dele.shape[0]), int(active.sum()), n_ungrounded)


def dt_scalar(v, dt):
    return np.dtype(dt).type(v)


def _rowsum(W, n, dt):
    xp = backend.get().xp
    return (W @ xp.ones(n, dtype=dt)).astype(dt)
