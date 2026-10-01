"""Components of new vertices (DynLP step 1): the paper's tau-threshold CC, and
component-adaptive merging (Felzenszwalb-Huttenlocher on locally scaled weights)."""
from __future__ import annotations

from .backend import scatter_add, scatter_max, scatter_min, xp


def connected_components(n, rows, cols):
    """(labels in [0, n_comp), n_comp) for an undirected edge list: hook + jump."""
    if n == 0:
        return xp.zeros(0, dtype=xp.int32), 0
    lab, rows, cols = xp.arange(n, dtype=xp.int32), rows.astype(xp.int32), cols.astype(xp.int32)
    while True:
        lr, lc = lab[rows], lab[cols]
        diff = lr != lc
        if not bool(diff.any()):
            break
        lr, lc = lr[diff], lc[diff]
        scatter_min(lab, xp.maximum(lr, lc), xp.minimum(lr, lc))  # hook the larger root on the smaller
        while not bool(((nxt := lab[lab]) == lab).all()):  # jump: full path compression
            lab = nxt
    uniq, inv = xp.unique(lab, return_inverse=True)
    return inv.astype(xp.int32).ravel(), int(uniq.shape[0])


def _lightest(lab, nc, cu, cv, de):
    """Per component: (components with an outgoing edge, the other side, its weight); ties -> smallest id."""
    best = xp.full(nc, xp.inf, dtype=xp.float32)
    scatter_min(best, cu, de)
    nb = xp.full(nc, nc, dtype=xp.int32)
    sel = de <= best[cu]
    scatter_min(nb, cu[sel], cv[sel])
    c = xp.flatnonzero(nb < nc).astype(xp.int32)
    return c, nb[c], best[c]


def fh_components(m, rows, cols, d, k, ground=None):
    """Felzenszwalb-Huttenlocher merging in parallel (Borůvka) rounds. Every component
    takes its lightest outgoing edge e and merges across it if
        d(e) <= min(Int(C1) + k/|C1|, Int(C2) + k/|C2|),
    where Int(C) is C's largest internal dissimilarity: an edge must be strong relative
    to each side's own internal links, so weak bridges between classes stay cut.
    ground (per vertex, e.g. weight to labeled vertices): afterwards, components with
    none join their lightest-edge neighbour, so none is left with a 0/0 init."""
    lab = xp.arange(m, dtype=xp.int32)
    Int, size = xp.zeros(m, dtype=xp.float32), xp.ones(m, dtype=xp.float32)
    rows, cols, d = rows.astype(xp.int32), cols.astype(xp.int32), d.astype(xp.float32)
    nc, fh = m, True
    while rows.shape[0]:
        cu, cv = lab[rows], lab[cols]
        ext = cu != cv
        if not fh:  # ground pass: only components without ground pick an edge
            g = xp.zeros(nc, dtype=xp.float32)
            scatter_add(g, lab, ground)
            ext &= g[cu] == 0
        if not bool(ext.any()):
            break
        c, t, dc = _lightest(lab, nc, cu[ext], cv[ext], d[ext])
        if fh:
            ok = dc <= xp.minimum(Int[c] + k / size[c], Int[t] + k / size[t])
            if not bool(ok.any()):
                if ground is None:
                    break
                fh = False
                continue
            c, t, dc = c[ok], t[ok], dc[ok]
        root, nc = connected_components(nc, c, t)
        Int2, size2 = xp.zeros(nc, dtype=xp.float32), xp.zeros(nc, dtype=xp.float32)
        scatter_max(Int2, root, Int)
        scatter_max(Int2, root[c], dc)
        scatter_add(size2, root, size)
        lab, Int, size = root[lab], Int2, size2
    return lab, nc


def adaptive_components(m, rows, cols, w, s_r, s_c, wmax, ground=None):
    """Component-adaptive merging of the new vertices. Dissimilarity 1 - w/wmax is scaled
    by each endpoint's own mean dissimilarity s (self-tuning), so dense and sparse regions
    compare on one scale; k is the mean scaled dissimilarity (no tuning)."""
    if m == 0:
        return xp.zeros(0, dtype=xp.int32), 0
    d = (1 - w / wmax) / xp.sqrt(s_r * s_c)
    return fh_components(m, rows, cols, d, float(d.mean()) if d.shape[0] else 0.0, ground)
