"""Graph datasets. A dataset is a symmetric, non-negative, loop-free CSR ``A`` with labels
``y`` in [0, K) and an arrival ``order`` that the stream reveals vertices in.

imdb      the paper's IMDB graph: the 50K labeled reviews (Maas et al. 2011, Hugging Face
          copy), TF-IDF, cosine kNN (k = 5), binary
synth{K}  50K-vertex cosine kNN (k = 5) over a mixture with dense and sparse
          sub-clusters, dense ones of different classes overlapping (K = 2 or 10)
sbm, er   planted partition and Erdos-Renyi graphs for scale runs
"""
from __future__ import annotations

import math
import os
import re
import shutil
import urllib.request
from array import array
from collections import Counter
from dataclasses import dataclass

from .backend import scatter_add, sp, xp

IMDB_URL = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/{}-00000-of-00001.parquet"


@dataclass
class Dataset:
    name: str
    A: object      # CSR (n x n), symmetric
    y: object      # int32 (n,)
    K: int
    order: object  # int64 permutation: arrival order

    n = property(lambda self: self.A.shape[0])
    nnz = property(lambda self: int(self.A.nnz))


def coalesce(rows, cols, vals, n, dtype):
    """Canonical CSR: duplicates summed, self-loops dropped."""
    keep = rows != cols
    coo = sp.coo_matrix((vals[keep].astype(dtype), (rows[keep].astype(xp.int32), cols[keep].astype(xp.int32))),
                        shape=(n, n))
    coo.sum_duplicates()
    A = coo.tocsr()
    A.sum_duplicates()
    return A


def symmetric_from_pairs(u, v, w, n, dtype):
    return coalesce(xp.concatenate([u, v]), xp.concatenate([v, u]), xp.concatenate([w, w]), n, dtype)


def expand_counts(counts, nseg):
    """Segment id of each element given per-segment counts (repeat(arange, counts))."""
    total = int(counts.sum())
    if not total:
        return xp.zeros(0, dtype=xp.int32)
    nz = counts > 0
    marks = xp.zeros(total + 1, dtype=xp.int32)
    scatter_add(marks, (xp.cumsum(counts) - counts)[nz].astype(xp.int64), xp.int32(1))
    return xp.flatnonzero(nz).astype(xp.int32)[xp.cumsum(marks[:total]) - 1]  # k-th non-empty -> id


def csr_rows(A):
    """Row index of every stored entry of a CSR matrix."""
    return expand_counts(xp.diff(A.indptr), A.shape[0])


def row_positions(indptr, rows):
    """Positions of all stored entries in the given CSR rows, and each one's segment index."""
    rows = rows.astype(xp.int64)
    counts = (indptr[rows + 1] - indptr[rows]).astype(xp.int64)
    seg = expand_counts(counts, int(rows.shape[0]))
    if not seg.shape[0]:
        return xp.zeros(0, dtype=xp.int64), seg
    excl = xp.cumsum(counts) - counts
    return indptr[rows].astype(xp.int64)[seg] + (xp.arange(seg.shape[0], dtype=xp.int64) - excl[seg]), seg


def knn_graph(X, k, dtype):
    """Symmetric cosine kNN graph of the L2-normalized rows of X (dense or CSR); a pair
    found from both sides is kept once, so weights stay cosine similarities."""
    n = X.shape[0]
    chunk = max(1, min(n, (1 << 26) // n))
    nbr, sim = xp.empty((n, k), dtype=xp.int64), xp.empty((n, k), dtype=xp.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        S = (X @ X[s:e].toarray().T).T if sp.issparse(X) else X[s:e] @ X.T
        S = xp.ascontiguousarray(S, dtype=xp.float32)
        S[xp.arange(e - s), xp.arange(s, e)] = -xp.inf
        nbr[s:e] = idx = xp.argpartition(-S, k, axis=1)[:, :k]
        sim[s:e] = xp.take_along_axis(S, idx, axis=1)
    u, v = xp.repeat(xp.arange(n, dtype=xp.int64), k), nbr.ravel()
    _, first = xp.unique(xp.minimum(u, v) * n + xp.maximum(u, v), return_index=True)
    return symmetric_from_pairs(u[first], v[first], xp.clip(sim.ravel()[first], 1e-3, None), n, dtype)


def _rng(seed):
    return xp.random.RandomState(seed), xp.random.RandomState(seed + 1).permutation


def sbm(n, K=2, deg=10.0, p_in=0.85, seed=0, dtype="float32") -> Dataset:
    """Planted partition: a p_in share of edges stays inside a class; intra-class edges are heavier."""
    rs, perm = _rng(seed)
    y = rs.randint(0, K, size=n).astype(xp.int32)
    members, sizes = xp.argsort(y).astype(xp.int64), xp.bincount(y, minlength=K).astype(xp.int64)
    offs, m = xp.cumsum(sizes) - sizes, int(n * deg / 2)
    u = rs.randint(0, n, size=m).astype(xp.int64)
    intra = rs.random_sample(m) < p_in
    yu = y[u]
    v_in = members[offs[yu] + (rs.random_sample(m) * sizes[yu]).astype(xp.int64)]
    v = xp.where(intra, v_in, rs.randint(0, n, size=m).astype(xp.int64))
    w = xp.where(y[u] == y[v], 0.4 + 0.6 * rs.random_sample(m), 0.1 + 0.6 * rs.random_sample(m))
    return Dataset(f"sbm-n{n}-K{K}-d{deg:g}", symmetric_from_pairs(u, v, w, n, dtype), y, K, perm(n))


def erdos_renyi(n, K=2, deg=5.0, seed=0, dtype="float32") -> Dataset:
    """The paper's 'Random' dataset: uniform edges, labels without structure."""
    rs, perm = _rng(seed)
    m = int(n * deg / 2)
    u, v = rs.randint(0, n, size=m).astype(xp.int64), rs.randint(0, n, size=m).astype(xp.int64)
    A = symmetric_from_pairs(u, v, 0.1 + 0.9 * rs.random_sample(m), n, dtype)
    return Dataset(f"er-n{n}-d{deg:g}", A, rs.randint(0, K, size=n).astype(xp.int32), K, perm(n))


def synth(n=50_000, K=10, k=5, dim=16, subs=5, seed=0, dtype="float32") -> Dataset:
    """Mixture built to stress DynLP step 1. Each class has `subs` sub-clusters with spreads
    spanning 8x (dense and sparse regions, so similarity scales differ by region), and a
    quarter of the densest sub-clusters overlap a dense one of another class (so a global
    threshold chains them). Cosine kNN, k = 5, like the paper's graphs."""
    rs, perm = _rng(seed)
    S = K * subs
    centers = rs.standard_normal((S, dim)).astype(xp.float32)
    centers *= 10 / xp.linalg.norm(centers, axis=1, keepdims=True)
    spread = (0.4 * 8 ** rs.random_sample(S)).astype(xp.float32)
    dense = xp.argsort(spread)[: max(2, S // 4)]
    for a in dense.tolist():  # a dense sub-cluster of another class overlapping `a`
        b = (a + subs * int(rs.randint(1, K))) % S if K > 1 else a
        step = rs.standard_normal(dim).astype(xp.float32)
        spread[b] = spread[a]
        centers[b] = centers[a] + step / xp.linalg.norm(step) * spread[a] * dim ** 0.5
    sub = rs.randint(0, S, size=n)
    X = centers[sub] + spread[sub, None] * rs.standard_normal((n, dim)).astype(xp.float32)
    X /= xp.linalg.norm(X, axis=1, keepdims=True)
    return Dataset(f"synth{K}", knn_graph(X, k, dtype), (sub // subs).astype(xp.int32), K, perm(n))


def _download(url, path, timeout=60):
    """Fetch to path.part, rename only when complete: an interrupted download never looks finished."""
    print(f"[data] downloading {url} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(path + ".part", "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
        os.replace(path + ".part", path)
    finally:
        if os.path.exists(path + ".part"):
            os.remove(path + ".part")


def tfidf(docs, min_df=5, max_df=0.5):
    """L2-normalized TF-IDF CSR (sublinear tf, smooth idf) of tokenized documents."""
    df = Counter(t for d in docs for t in set(d))
    vocab = {t: i for i, t in enumerate(t for t, c in df.items() if min_df <= c <= max_df * len(docs))}
    rows, cols, vals = array("i"), array("i"), array("f")
    for r, d in enumerate(docs):
        for t, c in Counter(t for t in d if t in vocab).items():
            rows.append(r), cols.append(vocab[t])
            vals.append((1 + math.log(c)) * (math.log((1 + len(docs)) / (1 + df[t])) + 1))
    rows, cols, vals = xp.asarray(rows), xp.asarray(cols), xp.asarray(vals)
    norm = xp.zeros(len(docs), dtype=xp.float32)
    scatter_add(norm, rows, vals * vals)
    vals /= xp.sqrt(xp.maximum(norm, 1e-12))[rows]
    return sp.coo_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab))).tocsr()


def _imdb_reviews(data_dir):
    """(tokens, labels) of the 25K train + 25K test labeled reviews (downloads ~41 MB once)."""
    import pandas as pd
    word, docs, ys = re.compile(r"\b\w\w+\b"), [], []
    for split in ("train", "test"):
        path = os.path.join(data_dir, f"imdb-{split}.parquet")
        if not os.path.exists(path):
            _download(IMDB_URL.format(split), path)
        df = pd.read_parquet(path)
        docs += [word.findall(t.lower().replace("<br />", " ")) for t in df.text]
        ys += df.label.tolist()
    return docs, ys


def _cached(path, build):
    """Load an edge-list cache (src, dst, w, y), building it first if missing."""
    if not os.path.exists(path):
        A, y = build()
        r = csr_rows(A)
        up = r < A.indices
        xp.savez(path, src=r[up], dst=A.indices[up], w=A.data[up], y=y)
    z = xp.load(path)
    return z["src"], z["dst"], z["w"], z["y"]


def from_npz(path, dtype="float32", seed=0, name=None) -> Dataset:
    """Edge-list npz with src, dst, y and optional w, order (bring your own graph too)."""
    z = xp.load(path)
    keys = set(z.npz_file.files)
    n, y = int(z["y"].shape[0]), z["y"].astype(xp.int32)
    w = z["w"] if "w" in keys else xp.ones(z["src"].shape[0])
    A = symmetric_from_pairs(z["src"].astype(xp.int64), z["dst"].astype(xp.int64), w, n, dtype)
    order = z["order"] if "order" in keys else xp.random.RandomState(seed + 1).permutation(n)
    return Dataset(name or os.path.basename(path), A, y, int(y.max()) + 1, order)


def build_cached(spec, data_dir="data"):
    """Build data/<spec>.npz if missing: imdb (downloads ~41 MB once), synth2, synth10."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{spec}.npz")
    if spec == "imdb":
        def build():
            docs, y = _imdb_reviews(data_dir)
            return knn_graph(tfidf(docs), 5, "float32"), xp.asarray(y, dtype=xp.int32)
    elif spec.startswith("synth"):
        def build():
            ds = synth(K=int(spec[5:]))
            return ds.A, ds.y
    else:
        raise ValueError(f"unknown dataset {spec!r}")
    _cached(path, build)
    return path


def load(spec: str, *, n=100_000, K=2, deg=10.0, p_in=0.85, seed=0, dtype="float32", data_dir="data") -> Dataset:
    if spec == "sbm":
        return sbm(n, K, deg, p_in, seed, dtype)
    if spec == "er":
        return erdos_renyi(n, K, deg, seed, dtype)
    if spec.endswith(".npz"):
        return from_npz(spec, dtype, seed)
    return from_npz(build_cached(spec, data_dir), dtype, seed, name=spec)
