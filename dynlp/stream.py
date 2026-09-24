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


class IncrementalStream(Stream):
    """The same stream, with the batch system kept resident and updated in place.

    Vertices keep their global ids. A vertex outside U (inactive, labeled, or in a
    component with no labeled vertex) gets an inert row: diag = 1, rhs = 0, no edges.
    Every solver therefore runs on it unchanged, and all state stays put between
    batches:
      - edge weights are masked in place (W_eff = A on U x U, 0 elsewhere), using a
        precomputed reverse-entry index, so a toggle costs the vertex's degree;
      - row sums and class weights to labeled neighbours are updated by increments;
      - groundedness is re-checked locally: a CC over new and ungrounded vertices for
        insertions, and a bounded ball around deleted vertices for deletions, with a
        full CC only when the ball can't decide.
    Per batch the work is the edges around the change plus a few O(n) vector
    passes, instead of a rebuild. Arrays in the returned System are live and stay
    valid until the next batch is built.
    """

    def __init__(self, ds: Dataset, *, ball_hops=3, **kw):
        super().__init__(ds, **kw)
        xp = backend.get().xp
        A = ds.A
        n = A.shape[0]
        self.ball_hops = ball_hops
        # rev[p] = position of the reverse entry (v, u) of entry p = (u, v); A is symmetric
        key = A.indices.astype(xp.int64) * n + self.rows.astype(xp.int64)
        perm = xp.argsort(key)
        self.rev = xp.empty(A.nnz, dtype=xp.int32)
        self.rev[perm] = xp.arange(A.nnz, dtype=xp.int32)
        del key, perm
        self.counters = {"ball_checks": 0, "full_ground": 0}

    # ------------------------------------------------------------------ edits
    def _edges(self, V):
        pos, seg = row_positions(self.ds.A.indptr, V)
        return pos, V.astype(backend.get().xp.int64)[seg]

    def _toggle(self, T, entering):
        """T enters U (Umask already includes T) or leaves U (Umask already excludes T)."""
        B = backend.get()
        xp = B.xp
        if T.shape[0] == 0:
            return
        A = self.ds.A
        pos, rows = self._edges(T)
        cols = A.indices[pos]
        inT = xp.zeros(A.shape[0], dtype=bool)
        inT[T] = True
        if entering:
            on = self.Umask[cols]
            w = xp.where(on, A.data[pos].astype(xp.float64), 0.0)
            self.W64.data[pos] = w
            self.Wd.data[pos] = w.astype(self.dtype)
            other = on & ~inT[cols]
            q = self.rev[pos[other]]
            self.W64.data[q] = w[other]
            self.Wd.data[q] = w[other].astype(self.dtype)
            self.rowsum[T] = 0
            B.scatter_add(self.rowsum, rows, w)
            B.scatter_add(self.rowsum, cols[other], w[other])
        else:
            was = self.Umask[cols]
            old = self.W64.data[pos]
            self.W64.data[pos] = 0
            self.Wd.data[pos] = 0
            q = self.rev[pos[was]]
            self.W64.data[q] = 0
            self.Wd.data[q] = 0
            B.scatter_add(self.rowsum, cols[was], -old[was])
            self.rowsum[T] = 0

    def _label_update(self, L, sign):
        """Class weights to labeled neighbours, for every vertex: B[v, y_l] += sign * w(v, l)."""
        B = backend.get()
        xp = B.xp
        if L.shape[0] == 0:
            return
        A = self.ds.A
        pos, rows = self._edges(L)
        cols = A.indices[pos].astype(xp.int64)
        K = self.ds.K
        B.scatter_add(self.Bm.reshape(-1), cols * K + self.y[rows].astype(xp.int64),
                      sign * A.data[pos].astype(xp.float64))

    def _cc_with_sink(self, m, r, c):
        """CC over m local nodes plus a sink node m; returns (labels, sink label)."""
        comp, nc = connected_components(m + 1, r, c)
        return comp, comp[m], nc

    # ------------------------------------------------------------ groundedness
    def _deletion_check(self, dele):
        """Vertices of U that deletions cut off from every labeled vertex."""
        xp = backend.get().xp
        A, n = self.ds.A, self.ds.A.shape[0]
        empty = xp.zeros(0, dtype=xp.int64)
        if dele.shape[0] == 0:
            return empty
        pos, _ = self._edges(dele)
        nb = A.indices[pos]
        src = xp.unique(nb[self.Umask[nb]]).astype(xp.int64)
        if src.shape[0] == 0:
            return empty
        self.counters["ball_checks"] += 1
        inball = xp.zeros(n, dtype=bool)
        inball[src] = True
        frontier = src
        for _ in range(self.ball_hops):
            if frontier.shape[0] == 0:
                break
            pos, _ = self._edges(frontier)
            nb = A.indices[pos]
            nb = xp.unique(nb[self.active[nb] & ~inball[nb]]).astype(xp.int64)
            inball[nb] = True
            frontier = nb[~self.labeled[nb]]  # labeled vertices are sinks: not expanded
        boundary = frontier
        ball = xp.flatnonzero(inball)
        m = int(ball.shape[0])
        loc = xp.full(n, -1, dtype=xp.int32)
        loc[ball] = xp.arange(m, dtype=xp.int32)
        lab = self.labeled[ball]
        ub = ball[~lab]
        pos, rows = self._edges(ub)
        cols = A.indices[pos]
        keep = inball[cols] & self.active[cols]
        r = loc[rows[keep]]
        c = loc[cols[keep]]
        c = xp.where(lab[c], m, c)
        comp, sink, nc = self._cc_with_sink(m, r, c)
        open_c = xp.zeros(nc, dtype=bool)
        if boundary.shape[0]:
            open_c[comp[loc[boundary]]] = True
        cs = comp[loc[src]]
        if bool((~(cs == sink) & open_c[cs]).any()):  # the ball could not decide
            return self._full_ground_check()
        closed = ~open_c
        closed[sink] = False
        ungr = ball[(~lab) & closed[comp[:m]]]
        return ungr[self.Umask[ungr]].astype(xp.int64)

    def _full_ground_check(self):
        xp = backend.get().xp
        A, n = self.ds.A, self.ds.A.shape[0]
        self.counters["full_ground"] += 1
        rows, cols = self.rows, A.indices
        keep = self.active[rows] & self.active[cols] & ~self.labeled[rows]
        c = xp.where(self.labeled[cols[keep]], n, cols[keep])
        comp, sink, _ = self._cc_with_sink(n, rows[keep], c)
        return xp.flatnonzero(self.Umask & (comp[:n] != sink)).astype(xp.int64)

    def _insertion_ground(self):
        """Vertices outside U (new or ungrounded) that now reach a labeled vertex."""
        xp = backend.get().xp
        A, n = self.ds.A, self.ds.A.shape[0]
        X = xp.flatnonzero(self.active & ~self.labeled & ~self.Umask).astype(xp.int64)
        m = int(X.shape[0])
        if m == 0:
            return X
        loc = xp.full(n, -1, dtype=xp.int32)
        loc[X] = xp.arange(m, dtype=xp.int32)
        pos, rows = self._edges(X)
        cols = A.indices[pos]
        cl = loc[cols]
        sink_edge = self.active[cols] & (self.labeled[cols] | self.Umask[cols])
        keep = (cl >= 0) | sink_edge
        c = xp.where(cl[keep] >= 0, cl[keep], m)
        comp, sink, _ = self._cc_with_sink(m, loc[rows[keep]], c)
        return X[comp[:m] == sink]

    # ------------------------------------------------------------------ batches
    def batches(self):
        B = backend.get()
        xp = B.xp
        A, K, dt = self.ds.A, self.ds.K, self.dtype
        n = A.shape[0]
        f64 = xp.float64
        self.active = xp.zeros(n, dtype=bool)
        self.Umask = xp.zeros(n, dtype=bool)
        self.W64 = B.csr(xp.zeros(A.nnz, dtype=f64), A.indices, A.indptr, A.shape)
        self.Wd = B.csr(xp.zeros(A.nnz, dtype=dt), A.indices, A.indptr, A.shape)
        self.rowsum = xp.zeros(n, dtype=f64)
        self.Bm = xp.zeros((n, K), dtype=f64)
        ids = xp.arange(n, dtype=xp.int64)
        prior64 = self.prior.astype(f64)
        C = 1 if K == 2 else K
        prevU = xp.zeros(n, dtype=bool)
        empty = xp.zeros(0, dtype=xp.int64)
        for t, (ins_h, dele_h) in enumerate(self.schedule):
            ins, dele = xp.asarray(ins_h), xp.asarray(dele_h)
            # 1. deletions
            leaving = empty
            if dele.shape[0]:
                self.active[dele] = False
                leaving = dele[self.Umask[dele]]
                self.Umask[leaving] = False
                self._toggle(leaving, entering=False)
                dl = dele[self.labeled[dele]]
                self._label_update(dl, -1.0)
                cut = self._deletion_check(dele)
                if cut.shape[0]:
                    self.Umask[cut] = False
                    self._toggle(cut, entering=False)
                    leaving = xp.concatenate([leaving, cut])
            else:
                dl = empty
            # 2. insertions
            self.active[ins] = True
            il = ins[self.labeled[ins]]
            self._label_update(il, 1.0)
            entering = self._insertion_ground()
            self.Umask[entering] = True
            self._toggle(entering, entering=True)

            # rows whose equation changed: entering rows, and U-neighbours of every toggle
            Um = self.Umask
            touched = xp.concatenate([entering, leaving, il.astype(xp.int64), dl.astype(xp.int64)])
            mark = xp.zeros(n, dtype=bool)
            mark[entering] = True
            if touched.shape[0]:
                pos, _ = self._edges(touched)
                mark[A.indices[pos]] = True
            changed = xp.flatnonzero(mark & Um)

            # per-batch views (O(n) vector passes)
            s64 = xp.where(Um, self.Bm.sum(axis=1) + self.eta, 1.0)
            diag64 = xp.where(Um, s64 + self.rowsum, 1.0)
            rhs64 = (self.Bm[:, 1:2] if K == 2 else self.Bm) + self.eta * prior64[None, :]
            rhs64 = xp.where(Um[:, None], rhs64, 0.0)
            is_new = Um & ~prevU
            # DynLP's affected set
            seedm = is_new.copy()
            tv = xp.concatenate([ins, dele])
            if tv.shape[0]:
                pos, _ = self._edges(tv)
                seedm[A.indices[pos]] = True
            seedm &= Um
            # DynLP step 1: components of new vertices over edges heavier than tau
            new_idx = xp.flatnonzero(is_new).astype(xp.int64)
            locN = xp.full(n, -1, dtype=xp.int32)
            locN[new_idx] = xp.arange(new_idx.shape[0], dtype=xp.int32)
            pos, rows = self._edges(new_idx)
            cols = A.indices[pos]
            e = (locN[cols] >= 0) & (self.Wd.data[pos] > self.tau)
            comp, n_comp = connected_components(int(new_idx.shape[0]), locN[rows[e]], locN[cols[e]])

            n_ungr = int((self.active & ~self.labeled & ~Um).sum())
            common = (is_new, seedm, comp, n_comp, K, C, self.eta, int(ins.shape[0]), int(dele.shape[0]),
                      int(self.active.sum()), n_ungr, Um.copy(), changed)
            sys = System(t, ids, self.Wd, s64.astype(dt), diag64.astype(dt), self.Bm.astype(dt),
                         rhs64.astype(dt), self.prior, *common)
            if np.dtype(dt) != np.float64:
                sys._conv[np.dtype(np.float64)] = System(t, ids, self.W64, s64, diag64, self.Bm, rhs64,
                                                         prior64, *common)
            prevU = Um.copy()
            yield sys


def dt_scalar(v, dt):
    return np.dtype(dt).type(v)


def _rowsum(W, n, dt):
    xp = backend.get().xp
    return (W @ xp.ones(n, dtype=dt)).astype(dt)
