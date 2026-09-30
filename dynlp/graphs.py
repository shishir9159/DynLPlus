"""Graph datasets: synthetic generators and loaders.

A dataset is a symmetric, non-negative, loop-free CSR ``A`` with labels ``y``
in [0, K) and an arrival ``order`` that the stream reveals vertices in.
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

    n = property(lambda self: self.A.shape[0])
    nnz = property(lambda self: int(self.A.nnz))


def coalesce(rows, cols, vals, n, dtype):
    """Canonical CSR: duplicates summed, self-loops dropped."""
    be, keep = backend.get(), rows != cols
    i32 = be.xp.int32
    coo = be.sp.coo_matrix((vals[keep].astype(dtype), (rows[keep].astype(i32), cols[keep].astype(i32))), shape=(n, n))
    coo.sum_duplicates()
    A = coo.tocsr()
    A.sum_duplicates()
    return A


def symmetric_from_pairs(u, v, w, n, dtype):
    xp = backend.get().xp
    return coalesce(xp.concatenate([u, v]), xp.concatenate([v, u]), xp.concatenate([w, w]), n, dtype)


def expand_counts(counts, nseg):
    """Segment id of each element given per-segment counts (np.repeat(arange, counts))."""
    be = backend.get()
    xp = be.xp
    total = int(counts.sum())
    if not total:
        return xp.zeros(0, dtype=xp.int32)
    nz = counts > 0
    marks = xp.zeros(total + 1, dtype=xp.int32)
    be.scatter_add(marks, (xp.cumsum(counts) - counts)[nz].astype(xp.int64), xp.int32(1))
    return xp.flatnonzero(nz).astype(xp.int32)[xp.cumsum(marks[:total]) - 1]  # k-th non-empty -> id


def csr_rows(A):
    """Row index of every stored entry (COO rows) of a CSR matrix."""
    return expand_counts(backend.get().xp.diff(A.indptr), A.shape[0])


def row_positions(indptr, rows):
    """Positions of all stored entries in the given CSR rows, and each one's segment index."""
    xp = backend.get().xp
    rows = rows.astype(xp.int64)
    counts = (indptr[rows + 1] - indptr[rows]).astype(xp.int64)
    seg = expand_counts(counts, int(rows.shape[0]))
    if not seg.shape[0]:
        return xp.zeros(0, dtype=xp.int64), seg
    excl = xp.cumsum(counts) - counts
    return indptr[rows].astype(xp.int64)[seg] + (xp.arange(seg.shape[0], dtype=xp.int64) - excl[seg]), seg


def _setup(seed):
    xp = backend.get().xp
    return xp, xp.random.RandomState(seed), lambda n: np.random.RandomState(seed + 1).permutation(n)


def sbm(n, K=2, deg=10.0, p_in=0.85, seed=0, dtype="float32") -> Dataset:
    """Planted partition: a p_in share of edges stays inside a class; intra-class
    edges are heavier (like a similarity graph), so tau-sparsification behaves."""
    xp, rs, order = _setup(seed)
    y = rs.randint(0, K, size=n).astype(xp.int32)
    members, sizes = xp.argsort(y).astype(xp.int64), xp.bincount(y, minlength=K).astype(xp.int64)
    offs, m = xp.cumsum(sizes) - sizes, int(n * deg / 2)
    u = rs.randint(0, n, size=m).astype(xp.int64)
    intra = rs.random_sample(m) < p_in
    yu = y[u]
    v_in = members[offs[yu] + (rs.random_sample(m) * sizes[yu]).astype(xp.int64)]
    v = xp.where(intra, v_in, rs.randint(0, n, size=m).astype(xp.int64))
    w = xp.where(y[u] == y[v], 0.4 + 0.6 * rs.random_sample(m), 0.1 + 0.6 * rs.random_sample(m))
    return Dataset(f"sbm-n{n}-K{K}-d{deg:g}", symmetric_from_pairs(u, v, w, n, dtype), y, K, order(n))


def erdos_renyi(n, K=2, deg=5.0, seed=0, dtype="float32") -> Dataset:
    """The paper's 'Random' dataset: uniform edges, labels without structure."""
    xp, rs, order = _setup(seed)
    m = int(n * deg / 2)
    u, v = rs.randint(0, n, size=m).astype(xp.int64), rs.randint(0, n, size=m).astype(xp.int64)
    A = symmetric_from_pairs(u, v, 0.1 + 0.9 * rs.random_sample(m), n, dtype)
    return Dataset(f"er-n{n}-d{deg:g}", A, rs.randint(0, K, size=n).astype(xp.int32), K, order(n))


def gmm_knn(n, K=2, dim=32, k=10, sep=2.0, seed=0, dtype="float32") -> Dataset:
    """kNN cosine-similarity graph over a Gaussian mixture (brute force; n <= ~2e5)."""
    xp, rs, order = _setup(seed)
    y = rs.randint(0, K, size=n).astype(xp.int32)
    means = rs.standard_normal((K, dim)) * sep / np.sqrt(dim) * 3.0
    X = (means[y] + rs.standard_normal((n, dim))).astype(xp.float32)
    X /= xp.linalg.norm(X, axis=1, keepdims=True)
    nbr, sim = xp.empty((n, k), dtype=xp.int64), xp.empty((n, k), dtype=xp.float32)
    chunk = max(1, min(n, (1 << 26) // n))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        S = X[s:e] @ X.T
        S[xp.arange(e - s), xp.arange(s, e)] = -xp.inf
        nbr[s:e] = idx = xp.argpartition(-S, k, axis=1)[:, :k]
        sim[s:e] = xp.take_along_axis(S, idx, axis=1)
    u = xp.repeat(xp.arange(n, dtype=xp.int64), k)
    A = symmetric_from_pairs(u, nbr.ravel(), xp.clip(sim.ravel(), 1e-3, None), n, dtype)
    return Dataset(f"gmmknn-n{n}-K{K}-k{k}", A, y, K, order(n))


_OGB = {name: f"http://snap.stanford.edu/ogb/data/nodeproppred/{name[5:]}.zip"
        for name in ("ogbn-arxiv", "ogbn-products")}


def _read_csv_gz(zf, member, dtype):
    with zf.open(member) as fh:
        raw = io.BytesIO(gzip.decompress(fh.read()))
    try:
        import pandas as pd
        return pd.read_csv(raw, header=None).to_numpy(dtype=dtype)
    except ImportError:
        return np.loadtxt(raw, delimiter=",", dtype=dtype, ndmin=2)


def ogbn(name, data_dir="data", seed=0, dtype="float32") -> Dataset:
    """ogbn-arxiv (arrival by publication year) or ogbn-products; unweighted, symmetrized.
    Downloads the raw OGB zip once and caches a compact .npz next to it."""
    xp = backend.get().xp
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"{name}.npz")
    if not os.path.exists(cache):
        zpath = os.path.join(data_dir, os.path.basename(_OGB[name]))
        if not os.path.exists(zpath):
            print(f"[data] downloading {_OGB[name]} ...", flush=True)
            urllib.request.urlretrieve(_OGB[name], zpath)
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            read = lambda s, dt: _read_csv_gz(zf, next(m for m in names if m.endswith(s)), dt)[:, :2]  # noqa: E731
            edges = read("raw/edge.csv.gz", np.int64)
            labels = np.nan_to_num(read("raw/node-label.csv.gz", np.float64)[:, 0], nan=-1).astype(np.int32)
            has_year = any(m.endswith("raw/node_year.csv.gz") for m in names)
            year = read("raw/node_year.csv.gz", np.int64)[:, 0] if has_year else np.zeros(0, np.int64)
        np.savez(cache, src=edges[:, 0], dst=edges[:, 1], y=labels, year=year)
    z = np.load(cache)
    n, src = int(z["y"].shape[0]), xp.asarray(z["src"])
    A = symmetric_from_pairs(src, xp.asarray(z["dst"]), xp.ones(src.shape[0], dtype=dtype), n, dtype)
    A.data[:] = 1.0  # u->v and v->u collapse to weight 1
    rs = np.random.RandomState(seed + 1)
    order = np.lexsort((rs.random_sample(n), z["year"])) if z["year"].size else rs.permutation(n)
    return Dataset(name, A, xp.asarray(z["y"]), int(z["y"].max()) + 1, order)


def from_npz(path, dtype="float32", seed=0) -> Dataset:
    """Bring your own graph: npz with src, dst, y and optional w, order."""
    xp, z = backend.get().xp, np.load(path)
    n = int(z["y"].shape[0])
    w = z["w"] if "w" in z else np.ones(z["src"].shape[0])
    A = symmetric_from_pairs(xp.asarray(z["src"]), xp.asarray(z["dst"]), xp.asarray(w), n, dtype)
    order = z["order"] if "order" in z else np.random.RandomState(seed + 1).permutation(n)
    return Dataset(os.path.basename(path), A, xp.asarray(z["y"].astype(np.int32)), int(z["y"].max()) + 1, order)


def load(spec: str, *, n=100_000, K=2, deg=10.0, p_in=0.85, knn=10, dim=32, seed=0, dtype="float32",
         data_dir="data") -> Dataset:
    makers = {"sbm": lambda: sbm(n, K, deg, p_in, seed, dtype), "er": lambda: erdos_renyi(n, K, deg, seed, dtype),
              "gmm-knn": lambda: gmm_knn(n, K, dim, knn, seed=seed, dtype=dtype)}
    if spec in makers:
        return makers[spec]()
    if spec in _OGB:
        return ogbn(spec, data_dir, seed, dtype)
    if spec.endswith(".npz"):
        return from_npz(spec, dtype, seed)
    raise ValueError(f"unknown dataset {spec!r}")
