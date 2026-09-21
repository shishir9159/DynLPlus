"""Graph datasets: synthetic generators and loaders.

Every dataset is a symmetric, non-negative, loop-free CSR matrix ``A`` with
true labels ``y`` in [0, K) and an arrival ``order``. The streaming protocol
(see stream.py) reveals vertices in that order.
"""
from __future__ import annotations

import gzip
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

from . import backend


@dataclass
class Dataset:
    name: str
    A: object          # backend CSR (n x n), symmetric
    y: object          # backend int32 (n,)
    K: int
    order: np.ndarray  # host int64 permutation, arrival order

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def nnz(self) -> int:
        return int(self.A.nnz)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def coalesce(rows, cols, vals, n, dtype):
    """Build a canonical CSR: duplicates summed, self-loops dropped."""
    B = backend.get()
    xp = B.xp
    keep = rows != cols
    rows, cols, vals = rows[keep], cols[keep], vals[keep].astype(dtype)
    coo = B.sp.coo_matrix((vals, (rows.astype(xp.int32), cols.astype(xp.int32))), shape=(n, n))
    coo.sum_duplicates()
    A = coo.tocsr()
    A.sum_duplicates()
    return A


def symmetric_from_pairs(u, v, w, n, dtype):
    xp = backend.get().xp
    return coalesce(xp.concatenate([u, v]), xp.concatenate([v, u]), xp.concatenate([w, w]), n, dtype)


def csr_rows(A):
    """Row index of every stored entry (COO rows) for a CSR matrix."""
    xp = backend.get().xp
    n = A.shape[0]
    counts = xp.diff(A.indptr)
    return expand_counts(counts, n)


def expand_counts(counts, nseg):
    """Segment id for each element given per-segment counts (like np.repeat(arange, counts))."""
    B = backend.get()
    xp = B.xp
    total = int(counts.sum())
    if total == 0:
        return xp.zeros(0, dtype=xp.int32)
    starts = xp.cumsum(counts) - counts
    marks = xp.zeros(total + 1, dtype=xp.int32)
    nz = counts > 0
    B.scatter_add(marks, starts[nz].astype(xp.int64), xp.int32(1))
    seg = xp.cumsum(marks[:total]) - 1
    # map "k-th non-empty segment" back to segment ids
    nonempty = xp.flatnonzero(nz).astype(xp.int32)
    return nonempty[seg]


def row_positions(indptr, rows):
    """Positions of all stored entries in the given CSR rows, and each one's segment index."""
    xp = backend.get().xp
    rows = rows.astype(xp.int64)
    counts = (indptr[rows + 1] - indptr[rows]).astype(xp.int64)
    seg = expand_counts(counts, int(rows.shape[0]))
    if seg.shape[0] == 0:
        return xp.zeros(0, dtype=xp.int64), seg
    excl = xp.cumsum(counts) - counts
    pos = indptr[rows].astype(xp.int64)[seg] + (xp.arange(seg.shape[0], dtype=xp.int64) - excl[seg])
    return pos, seg


def _rng(seed):
    return backend.get().xp.random.RandomState(seed)


# ----------------------------------------------------------------------------
# synthetic graphs
# ----------------------------------------------------------------------------

def sbm(n, K=2, deg=10.0, p_in=0.85, seed=0, dtype="float32") -> Dataset:
    """Planted-partition graph: a fraction p_in of edges stays inside a class.

    Intra-class edges get heavier weights than inter-class ones, as in a
    similarity graph, so DynLP's tau-sparsification behaves sensibly.
    """
    xp = backend.get().xp
    rs = _rng(seed)
    y = rs.randint(0, K, size=n).astype(xp.int32)
    members = xp.argsort(y).astype(xp.int64)
    sizes = xp.bincount(y, minlength=K).astype(xp.int64)
    offs = xp.cumsum(sizes) - sizes
    m = int(n * deg / 2)
    u = rs.randint(0, n, size=m).astype(xp.int64)
    intra = rs.random_sample(m) < p_in
    yu = y[u]
    v_in = members[offs[yu] + (rs.random_sample(m) * sizes[yu]).astype(xp.int64)]
    v_out = rs.randint(0, n, size=m).astype(xp.int64)
    v = xp.where(intra, v_in, v_out)
    same = y[u] == y[v]
    w = xp.where(same, 0.4 + 0.6 * rs.random_sample(m), 0.1 + 0.6 * rs.random_sample(m))
    A = symmetric_from_pairs(u, v, w, n, dtype)
    order = np.random.RandomState(seed + 1).permutation(n)
    return Dataset(f"sbm-n{n}-K{K}-d{deg:g}", A, y, K, order)


def erdos_renyi(n, K=2, deg=5.0, seed=0, dtype="float32") -> Dataset:
    """The paper's 'Random' dataset: uniform edges, labels carry no structure."""
    xp = backend.get().xp
    rs = _rng(seed)
    m = int(n * deg / 2)
    u = rs.randint(0, n, size=m).astype(xp.int64)
    v = rs.randint(0, n, size=m).astype(xp.int64)
    w = 0.1 + 0.9 * rs.random_sample(m)
    A = symmetric_from_pairs(u, v, w, n, dtype)
    y = rs.randint(0, K, size=n).astype(xp.int32)
    order = np.random.RandomState(seed + 1).permutation(n)
    return Dataset(f"er-n{n}-d{deg:g}", A, y, K, order)


def gmm_knn(n, K=2, dim=32, k=10, sep=2.0, seed=0, dtype="float32") -> Dataset:
    """kNN cosine-similarity graph over a Gaussian mixture (brute force; n <= ~2e5)."""
    xp = backend.get().xp
    rs = _rng(seed)
    y = rs.randint(0, K, size=n).astype(xp.int32)
    means = rs.standard_normal((K, dim)) * sep / np.sqrt(dim) * 3.0
    X = (means[y] + rs.standard_normal((n, dim))).astype(xp.float32)
    X /= xp.linalg.norm(X, axis=1, keepdims=True)
    chunk = max(1, min(n, (1 << 26) // n))
    nbr = xp.empty((n, k), dtype=xp.int64)
    sim = xp.empty((n, k), dtype=xp.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        S = X[s:e] @ X.T
        S[xp.arange(e - s), xp.arange(s, e)] = -xp.inf
        idx = xp.argpartition(-S, k, axis=1)[:, :k]
        nbr[s:e] = idx
        sim[s:e] = xp.take_along_axis(S, idx, axis=1)
    u = xp.repeat(xp.arange(n, dtype=xp.int64), k)
    w = xp.clip(sim.ravel(), 1e-3, None)
    A = symmetric_from_pairs(u, nbr.ravel(), w, n, dtype)
    order = np.random.RandomState(seed + 1).permutation(n)
    return Dataset(f"gmmknn-n{n}-K{K}-k{k}", A, y, K, order)


# ----------------------------------------------------------------------------
# real graphs
# ----------------------------------------------------------------------------

_OGB = {
    "ogbn-arxiv": "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
    "ogbn-products": "http://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
}


def _read_csv_gz(zf, member, dtype):
    with zf.open(member) as fh:
        raw = gzip.decompress(fh.read())
    try:
        import pandas as pd
        return pd.read_csv(io.BytesIO(raw), header=None).to_numpy(dtype=dtype)
    except ImportError:
        return np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=dtype, ndmin=2)


def ogbn(name, data_dir="data", seed=0, dtype="float32") -> Dataset:
    """ogbn-arxiv (temporal: arrival by publication year) or ogbn-products.

    Downloads the raw OGB zip once and caches a compact .npz next to it.
    Edges are unweighted (w = 1) and symmetrized.
    """
    xp = backend.get().xp
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"{name}.npz")
    if not os.path.exists(cache):
        url = _OGB[name]
        zpath = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(zpath):
            print(f"[data] downloading {url} ...", flush=True)
            urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            pick = lambda s: next(m for m in names if m.endswith(s))  # noqa: E731
            edges = _read_csv_gz(zf, pick("raw/edge.csv.gz"), np.int64)
            labels = _read_csv_gz(zf, pick("raw/node-label.csv.gz"), np.float64)[:, 0]
            year = None
            if any(m.endswith("raw/node_year.csv.gz") for m in names):
                year = _read_csv_gz(zf, pick("raw/node_year.csv.gz"), np.int64)[:, 0]
        labels = np.nan_to_num(labels, nan=-1).astype(np.int32)
        np.savez(cache, src=edges[:, 0], dst=edges[:, 1], y=labels,
                 year=year if year is not None else np.zeros(0, np.int64))
    z = np.load(cache)
    n = int(z["y"].shape[0])
    src, dst = xp.asarray(z["src"]), xp.asarray(z["dst"])
    A = symmetric_from_pairs(src, dst, xp.ones(src.shape[0], dtype=dtype), n, dtype)
    A.data[:] = 1.0  # duplicates (u->v and v->u) collapse to weight 1
    y = xp.asarray(z["y"])
    K = int(z["y"].max()) + 1
    rs = np.random.RandomState(seed + 1)
    if z["year"].size:
        order = np.lexsort((rs.random_sample(n), z["year"]))  # by year, random within year
    else:
        order = rs.permutation(n)
    return Dataset(name, A, y, K, order)


def from_npz(path, dtype="float32", seed=0) -> Dataset:
    """Bring your own graph: npz with src, dst, y and optional w, order."""
    xp = backend.get().xp
    z = np.load(path)
    n = int(z["y"].shape[0])
    w = z["w"] if "w" in z else np.ones(z["src"].shape[0])
    A = symmetric_from_pairs(xp.asarray(z["src"]), xp.asarray(z["dst"]), xp.asarray(w), n, dtype)
    y = xp.asarray(z["y"].astype(np.int32))
    order = z["order"] if "order" in z else np.random.RandomState(seed + 1).permutation(n)
    return Dataset(os.path.basename(path), A, y, int(z["y"].max()) + 1, order)


def load(spec: str, *, n=100_000, K=2, deg=10.0, p_in=0.85, knn=10, dim=32,
         seed=0, dtype="float32", data_dir="data") -> Dataset:
    if spec == "sbm":
        return sbm(n, K, deg, p_in, seed, dtype)
    if spec == "er":
        return erdos_renyi(n, K, deg, seed, dtype)
    if spec == "gmm-knn":
        return gmm_knn(n, K, dim, knn, seed=seed, dtype=dtype)
    if spec in _OGB:
        return ogbn(spec, data_dir, seed, dtype)
    if spec.endswith(".npz"):
        return from_npz(spec, dtype, seed)
    raise ValueError(f"unknown dataset {spec!r}")
