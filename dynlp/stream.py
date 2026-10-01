"""Streaming protocol and the per-batch linear system.

Per batch, the harmonic system on the active unlabeled set U:
    (diag - W_UU) F_U = rhs,   diag = s + rowsum(W_UU),   s = w(u, L) + eta
s is each vertex's grounding (weight to labeled vertices plus the dongle eta).
Binary problems solve one column (class 1); K classes solve K columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .backend import csr, scatter_add, xp
from .cc import adaptive_components, connected_components
from .graphs import Dataset, csr_rows, row_positions


@dataclass
class System:
    t: int
    U: object          # global ids of the rows (sorted)
    W: object          # CSR U x U
    s: object          # grounding
    diag: object       # s + rowsum(W)
    B: object          # (n_u, K) weight to labeled neighbours per class
    rhs: object        # (n_u, C)
    prior: object      # (C,)
    is_new: object     # rows not in U at the previous batch
    seed: object       # DynLP's affected set: new, N(inserted), N(deleted)
    comps: dict        # new vertices' components: {"tau": (labels, n), "fh": (labels, n)}
    K: int
    C: int
    eta: float
    n_inserted: int
    n_deleted: int
    n_active: int
    n_ungrounded: int = 0
    solved: object = None   # global-index systems: rows really solved (others inert)
    changed: object = None  # global-index systems: rows whose equation changed
    _conv: dict = field(default_factory=dict, repr=False, compare=False)

    n_u = property(lambda self: int(self.U.shape[0]))
    nnz = property(lambda self: int(self.W.nnz))
    n_solved = property(lambda self: self.n_u if self.solved is None else int(self.solved.sum()))

    def label_cols(self, M):
        """Columns of a (., K) class matrix that are solved (class 1 if binary)."""
        return M[:, 1:2] if self.K == 2 else M

    def astype(self, dtype) -> "System":
        """Same system in another precision (cached; index arrays shared)."""
        dt = xp.dtype(dtype)
        if dt == self.W.data.dtype:
            return self
        if dt not in self._conv:
            cast = {k: getattr(self, k).astype(dt) for k in ("s", "diag", "B", "rhs", "prior")}
            W = csr(self.W.data.astype(dt), self.W.indices, self.W.indptr, self.W.shape)
            self._conv[dt] = replace(self, W=W, _conv={}, **cast)
        return self._conv[dt]


def _local(ids, n):
    loc = xp.full(n, -1, dtype=xp.int32)
    loc[ids] = xp.arange(ids.shape[0], dtype=xp.int32)
    return loc


def new_components(m, lr, lc, w, tau, wmax, ground):
    """DynLP step 1 for the m new vertices, from every stored entry (lr -> lc, w) of their
    rows (lc = -1 when the neighbour isn't new): the paper's components over edges heavier
    than the global mean tau, and adaptive ones (cc.adaptive_components), whose local
    scale is each vertex's mean dissimilarity over all its edges; ground = label weight."""
    on = w > 0
    s, cnt = xp.zeros(m, dtype=xp.float32), xp.zeros(m, dtype=xp.float32)
    scatter_add(s, lr, xp.where(on, 1 - w / wmax, 0).astype(xp.float32))
    scatter_add(cnt, lr, on.astype(xp.float32))
    s = s / xp.maximum(cnt, 1) + 1e-6
    e = on & (lc >= 0)
    r, c, w = lr[e], lc[e], w[e]
    heavy = w > tau
    return {"tau": connected_components(m, r[heavy], c[heavy]),
            "fh": adaptive_components(m, r, c, w, s[r], s[c], wmax, ground.astype(xp.float32))}


class Stream:
    """Reveals a dataset in batches: an initial snapshot, then batches of new vertices
    (a label_frac share with ground truth) plus deletions of del_frac x batch-size random
    active unlabeled vertices (the paper's protocol). The schedule is drawn up front."""

    def __init__(self, ds: Dataset, *, label_frac=0.01, init_frac=0.1, n_batches=10,
                 del_frac=0.1, eta_rel=0.0, seed=0, dtype="float32"):
        self.ds, self.dtype, self.del_frac = ds, dtype, del_frac
        n, K, y = ds.n, ds.K, ds.y
        rs = self.rs = xp.random.RandomState(seed + 7)
        cand = xp.flatnonzero(y >= 0)
        lab = xp.zeros(n, dtype=bool)
        lab[cand[rs.permutation(cand.shape[0])[: max(K, int(label_frac * n))]]] = True
        order, n0 = ds.order.copy(), max(int(init_frac * n), 2 * K)
        pos = xp.empty(n, dtype=xp.int64)
        pos[order] = xp.arange(n)
        for c in range(K):  # the initial snapshot needs a labeled vertex of every class
            lc = xp.flatnonzero(lab & (y == c))
            if not lc.shape[0]:
                lc = xp.flatnonzero(y == c)[:1]
                lab[lc] = True
            if lc.shape[0] and int(pos[lc].min()) >= n0:
                free = xp.flatnonzero(~lab[order[:n0]])
                v, j = int(lc[0]), int(free[int(rs.randint(0, free.shape[0]))])
                a, pv = int(order[j]), int(pos[v])
                order[j], order[pv], pos[v], pos[a] = v, a, j, pv
        self.labeled, self.order, self.n0, self.y = lab, order, n0, y
        self.chunks = xp.array_split(order[n0:], n_batches) if n_batches > 0 else []
        self.rows, self.tau, self.wmax = csr_rows(ds.A), float(ds.A.data.mean()), float(ds.A.data.max())
        self.eta = float(eta_rel * ds.A.data.sum() / n)
        self.prior = xp.full(1 if K == 2 else K, 0.5 if K == 2 else 1.0 / K, dtype=dtype)
        self.schedule = self._make_schedule()

    def _make_schedule(self):
        active, sched, none = xp.zeros(self.ds.n, dtype=bool), [], xp.zeros(0, dtype=xp.int64)
        for t in range(1 + len(self.chunks)):
            ins, dele = (self.order[: self.n0] if t == 0 else self.chunks[t - 1]), none
            if t:
                pool = xp.flatnonzero(active & ~self.labeled)
                nd = min(int(self.del_frac * ins.shape[0]), max(pool.shape[0] - 1, 0))
                if nd > 0:
                    dele = pool[self.rs.permutation(pool.shape[0])[:nd]]
            active[ins], active[dele] = True, False
            sched.append((ins.astype(xp.int64), dele.astype(xp.int64)))
        return sched

    def __len__(self):
        return len(self.schedule)

    def batches(self):
        active, prevU = xp.zeros(self.ds.n, dtype=bool), xp.zeros(self.ds.n, dtype=bool)
        for t, (ins, dele) in enumerate(self.schedule):
            active[ins], active[dele] = True, False
            sys = self._build(t, active, prevU, ins, dele)
            prevU = xp.zeros(self.ds.n, dtype=bool)
            prevU[sys.U] = True
            yield sys

    def _build(self, t, active, prevU, ins, dele) -> System:
        A, K, dt = self.ds.A, self.ds.K, self.dtype
        n, rows, cols = A.shape[0], self.rows, A.indices
        onehot = xp.zeros((n, K), dtype=dt)
        li = xp.flatnonzero(active & self.labeled)
        onehot[li, self.y[li]] = 1
        Bfull = A @ onehot
        # components of U with no labeled vertex (value = prior, unbounded h) are set aside
        Um = active & ~self.labeled
        U0 = xp.flatnonzero(Um).astype(xp.int64)
        loc, keep = _local(U0, n), Um[rows] & Um[cols]
        lab, ncomp = connected_components(int(U0.shape[0]), loc[rows[keep]], loc[cols[keep]])
        gw = xp.zeros(ncomp, dtype=dt)
        scatter_add(gw, lab, Bfull[U0].sum(axis=1).astype(dt))
        grounded = gw[lab] > 0
        Um = Um.copy()
        Um[U0[~grounded]] = False
        U = xp.flatnonzero(Um).astype(xp.int64)
        n_u, loc, keep = int(U.shape[0]), _local(U, n), Um[rows] & Um[cols]
        indptr = xp.zeros(n_u + 1, dtype=xp.int32)  # CSR order survives the filter
        indptr[1:] = xp.cumsum(xp.bincount(loc[rows[keep]], minlength=n_u))
        W = csr(A.data[keep].astype(dt), loc[cols[keep]], indptr, (n_u, n_u))
        Bm = Bfull[U].astype(dt)
        s = (Bm.sum(axis=1) + self.eta).astype(dt)
        diag = (s + W @ xp.ones(n_u, dtype=dt)).astype(dt)
        rhs = ((Bm[:, 1:2] if K == 2 else Bm) + self.eta * self.prior[None, :]).astype(dt)
        is_new = ~prevU[U]
        touch = xp.zeros(n, dtype=dt)
        touch[ins], touch[dele] = 1, 1
        seed = is_new | ((A @ touch)[U] > 0)
        nl, rW = _local(xp.flatnonzero(is_new), n_u), csr_rows(W)
        on = is_new[rW]
        comps = new_components(int(is_new.sum()), nl[rW[on]], nl[W.indices[on]], W.data[on], self.tau, self.wmax,
                               Bm[is_new].sum(axis=1))
        return System(t, U, W, s, diag, Bm, rhs, self.prior.astype(dt), is_new, seed, comps, K, 1 if K == 2 else K,
                      self.eta, int(ins.shape[0]), int(dele.shape[0]), int(active.sum()), int((~grounded).sum()))


class IncrementalStream(Stream):
    """The same stream with the batch system resident and updated in place.

    Vertices keep global ids; rows outside U (inactive, labeled, ungrounded) are inert
    (diag 1, rhs 0, no edges), so every solver runs unchanged. Edge weights are masked in
    place via a reverse-entry index, row sums and class weights are updated by increments,
    and groundedness is re-checked locally (new/ungrounded vertices on insertion, a bounded
    ball around deletions, full CC as fallback). Returned arrays are live until the next batch.
    """

    def __init__(self, ds: Dataset, *, ball_hops=3, **kw):
        super().__init__(ds, **kw)
        A = ds.A
        self.ball_hops, self.counters = ball_hops, {"ball_checks": 0, "full_ground": 0}
        perm = xp.argsort(A.indices.astype(xp.int64) * A.shape[0] + self.rows)  # A is symmetric
        self.rev = xp.empty(A.nnz, dtype=xp.int32)  # rev[p]: entry (v, u) of entry p = (u, v)
        self.rev[perm] = xp.arange(A.nnz, dtype=xp.int32)

    def _edges(self, V):
        pos, seg = row_positions(self.ds.A.indptr, V)
        return pos, V.astype(xp.int64)[seg]

    def _set_w(self, pos, w):
        self.W64.data[pos] = w
        self.Wd.data[pos] = w.astype(self.dtype)

    def _toggle(self, T, entering):
        """T enters U (Umask already includes T) or leaves it (Umask already excludes T)."""
        if not T.shape[0]:
            return
        pos, rows = self._edges(T)
        cols = self.ds.A.indices[pos]
        on = self.Umask[cols]
        if entering:
            inT = xp.zeros(self.ds.n, dtype=bool)
            inT[T] = True
            w = xp.where(on, self.ds.A.data[pos].astype(xp.float64), 0.0)
            other = on & ~inT[cols]
            self._set_w(pos, w)
            self._set_w(self.rev[pos[other]], w[other])
            self.rowsum[T] = 0
            scatter_add(self.rowsum, rows, w)
            scatter_add(self.rowsum, cols[other], w[other])
        else:
            old = self.W64.data[pos]
            self._set_w(pos, xp.zeros_like(old))
            self._set_w(self.rev[pos[on]], xp.zeros(int(on.sum())))
            scatter_add(self.rowsum, cols[on], -old[on])
            self.rowsum[T] = 0

    def _label_update(self, L, sign):
        """B[v, y_l] += sign * w(v, l) for every neighbour v of the labeled vertices L."""
        if L.shape[0]:
            pos, rows = self._edges(L)
            idx = self.ds.A.indices[pos].astype(xp.int64) * self.ds.K + self.y[rows].astype(xp.int64)
            scatter_add(self.Bm.reshape(-1), idx, sign * self.ds.A.data[pos].astype(xp.float64))

    def _deletion_check(self, dele):
        """Vertices of U that the deletions cut off from every labeled vertex."""
        A, n, empty = self.ds.A, self.ds.n, xp.zeros(0, dtype=xp.int64)
        if not dele.shape[0]:
            return empty
        nb = A.indices[self._edges(dele)[0]]
        src = xp.unique(nb[self.Umask[nb]]).astype(xp.int64)
        if not src.shape[0]:
            return empty
        self.counters["ball_checks"] += 1
        inball = xp.zeros(n, dtype=bool)
        inball[src] = True
        frontier = src
        for _ in range(self.ball_hops):
            if not frontier.shape[0]:
                break
            nb = A.indices[self._edges(frontier)[0]]
            nb = xp.unique(nb[self.active[nb] & ~inball[nb]]).astype(xp.int64)
            inball[nb] = True
            frontier = nb[~self.labeled[nb]]  # labeled vertices are sinks: not expanded
        ball = xp.flatnonzero(inball)
        m, loc, lab = int(ball.shape[0]), _local(ball, n), self.labeled[ball]
        pos, rows = self._edges(ball[~lab])
        cols = A.indices[pos]
        keep = inball[cols] & self.active[cols]
        c = loc[cols[keep]]
        comp, nc = connected_components(m + 1, loc[rows[keep]], xp.where(lab[c], m, c))
        sink, open_c = comp[m], xp.zeros(nc, dtype=bool)
        if frontier.shape[0]:
            open_c[comp[loc[frontier]]] = True
        cs = comp[loc[src]]
        if bool(((cs != sink) & open_c[cs]).any()):  # the ball can't decide
            return self._full_ground_check()
        closed = ~open_c
        closed[sink] = False
        ungr = ball[~lab & closed[comp[:m]]]
        return ungr[self.Umask[ungr]].astype(xp.int64)

    def _full_ground_check(self):
        rows, cols, n = self.rows, self.ds.A.indices, self.ds.n
        self.counters["full_ground"] += 1
        keep = self.active[rows] & self.active[cols] & ~self.labeled[rows]
        comp, _ = connected_components(n + 1, rows[keep], xp.where(self.labeled[cols[keep]], n, cols[keep]))
        return xp.flatnonzero(self.Umask & (comp[:n] != comp[n])).astype(xp.int64)

    def _insertion_ground(self):
        """Vertices outside U (new or ungrounded) that now reach a labeled vertex."""
        X = xp.flatnonzero(self.active & ~self.labeled & ~self.Umask).astype(xp.int64)
        m = int(X.shape[0])
        if not m:
            return X
        loc = _local(X, self.ds.n)
        pos, rows = self._edges(X)
        cols = self.ds.A.indices[pos]
        cl = loc[cols]
        keep = (cl >= 0) | (self.active[cols] & (self.labeled[cols] | self.Umask[cols]))
        comp, _ = connected_components(m + 1, loc[rows[keep]], xp.where(cl[keep] >= 0, cl[keep], m))
        return X[comp[:m] == comp[m]]

    def _neighbours(self, V, mask):
        mask[self.ds.A.indices[self._edges(V)[0]]] = True
        return mask

    def batches(self):
        A, K, dt, n = self.ds.A, self.ds.K, self.dtype, self.ds.n
        f64, ids = xp.float64, xp.arange(n, dtype=xp.int64)
        self.active, self.Umask = xp.zeros(n, dtype=bool), xp.zeros(n, dtype=bool)
        self.W64 = csr(xp.zeros(A.nnz, dtype=f64), A.indices, A.indptr, A.shape)
        self.Wd = csr(xp.zeros(A.nnz, dtype=dt), A.indices, A.indptr, A.shape)
        self.rowsum, self.Bm = xp.zeros(n, dtype=f64), xp.zeros((n, K), dtype=f64)
        prior64, prevU = self.prior.astype(f64), xp.zeros(n, dtype=bool)
        for t, (ins, dele) in enumerate(self.schedule):
            self.active[dele] = False  # 1. deletions
            leaving = dele[self.Umask[dele]]
            self.Umask[leaving] = False
            self._toggle(leaving, False)
            dl = dele[self.labeled[dele]]
            self._label_update(dl, -1.0)
            cut = self._deletion_check(dele)
            self.Umask[cut] = False
            self._toggle(cut, False)
            self.active[ins] = True  # 2. insertions
            il = ins[self.labeled[ins]]
            self._label_update(il, 1.0)
            entering = self._insertion_ground()
            self.Umask[entering] = True
            self._toggle(entering, True)
            Um = self.Umask
            touched = xp.concatenate([entering, leaving, cut, il, dl]).astype(xp.int64)
            mark = xp.zeros(n, dtype=bool)
            mark[entering] = True
            changed = xp.flatnonzero(self._neighbours(touched, mark) & Um)
            s64 = xp.where(Um, self.Bm.sum(axis=1) + self.eta, 1.0)
            diag64 = xp.where(Um, s64 + self.rowsum, 1.0)
            rhs64 = xp.where(Um[:, None], (self.Bm[:, 1:2] if K == 2 else self.Bm) + self.eta * prior64, 0.0)
            is_new = Um & ~prevU
            seed = self._neighbours(xp.concatenate([ins, dele]), is_new.copy()) & Um
            new = xp.flatnonzero(is_new).astype(xp.int64)
            locN = _local(new, n)
            pos, rows = self._edges(new)
            comps = new_components(int(new.shape[0]), locN[rows], locN[A.indices[pos]], self.Wd.data[pos],
                                   self.tau, self.wmax, self.Bm[new].sum(axis=1))
            common = (is_new, seed, comps, K, 1 if K == 2 else K, self.eta, int(ins.shape[0]), int(dele.shape[0]),
                      int(self.active.sum()), int((self.active & ~self.labeled & ~Um).sum()), Um.copy(), changed)
            sys = System(t, ids, self.Wd, s64.astype(dt), diag64.astype(dt), self.Bm.astype(dt), rhs64.astype(dt),
                         self.prior, *common)
            if xp.dtype(dt) != f64:
                sys._conv[xp.dtype(f64)] = System(t, ids, self.W64, s64, diag64, self.Bm, rhs64, prior64, *common)
            prevU = Um.copy()
            yield sys
