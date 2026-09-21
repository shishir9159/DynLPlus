"""Connected components (DynLP step 1): Shiloach-Vishkin-style hook + jump."""
from __future__ import annotations

from . import backend


def connected_components(n, rows, cols):
    """Return (labels in [0, n_comp), n_comp) for an undirected edge list."""
    B = backend.get()
    xp = B.xp
    if n == 0:
        return xp.zeros(0, dtype=xp.int32), 0
    if not B.is_gpu:
        import numpy as np
        import scipy.sparse as ssp
        from scipy.sparse.csgraph import connected_components as scc
        g = ssp.coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
        nc, lab = scc(g, directed=False)
        return lab.astype(np.int32), int(nc)
    lab = xp.arange(n, dtype=xp.int32)
    rows = rows.astype(xp.int32)
    cols = cols.astype(xp.int32)
    while True:
        lr, lc = lab[rows], lab[cols]
        diff = lr != lc
        if not bool(diff.any()):
            break
        lr, lc = lr[diff], lc[diff]
        m = xp.minimum(lr, lc)
        # hook: point the larger root at the smaller one
        B.scatter_min(lab, xp.maximum(lr, lc), m)
        # jump: full path compression
        while True:
            nxt = lab[lab]
            if bool((nxt == lab).all()):
                break
            lab = nxt
    uniq, inv = xp.unique(lab, return_inverse=True)
    return inv.astype(xp.int32).ravel(), int(uniq.shape[0])
