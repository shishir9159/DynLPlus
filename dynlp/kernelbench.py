"""Frontier-kernel micro-benchmark (GPU only).

Compares lanes-per-row mappings for the frontier Jacobi kernel, including
DynLP's one-block-per-row mapping (group=128/256), against a full cuSPARSE SpMM.

  python -m dynlp.kernelbench --dataset sbm --n 10000000 --deg 10 --cols 1,2,8 --out results/kernels.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from . import backend, graphs
from .kernels import FrontierOps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, typ, default in [("--dataset", str, "sbm"), ("--n", int, 1_000_000), ("--classes", int, 2),
                               ("--deg", float, 10.0), ("--dtype", str, "float32"), ("--cols", str, "1,2,8"),
                               ("--fracs", str, "0.001,0.01,0.1,1.0"), ("--groups", str, "1,2,4,8,16,32,128"),
                               ("--reps", int, 20), ("--seed", int, 0), ("--data-dir", str, "data"), ("--out", str, None)]:
        ap.add_argument(flag, type=typ, default=default)
    a = ap.parse_args(argv)

    be = backend.set_backend("cupy")
    import cupy as cp
    ds = graphs.load(a.dataset, n=a.n, K=a.classes, deg=a.deg, seed=a.seed, dtype=a.dtype, data_dir=a.data_dir)
    W, n, real = ds.A, ds.A.shape[0], np.dtype(a.dtype).itemsize
    deg = np.diff(be.asnumpy(W.indptr))
    print(f"[data] {ds.name}: n={n:,} nnz={W.nnz:,} mean deg={deg.mean():.1f} max deg={deg.max():,}")
    rs = cp.random.RandomState(a.seed)
    rows = []
    start, end = cp.cuda.Event(), cp.cuda.Event()

    def timeit(fn):
        fn()
        start.record()
        for _ in range(a.reps):
            fn()
        end.record()
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end) / a.reps

    for C in [int(c) for c in a.cols.split(",")]:
        X, rhs = (rs.random_sample((n, C)).astype(a.dtype) for _ in range(2))
        diag = (W @ cp.ones(n, dtype=a.dtype) + 1).astype(a.dtype)
        full_ms = timeit(lambda: W @ X if C > 1 else W @ X[:, 0])
        bytes_full = W.nnz * (4 + real + C * real) + n * (4 + 2 * C * real)
        print(f"\nC={C}: cuSPARSE full SpMM {full_ms:.3f} ms  ({bytes_full / full_ms / 1e6:.0f} GB/s effective)")
        print(f"{'frontier':>10} {'group':>6} {'ms':>9} {'Gedge/s':>8} {'GB/s':>7} {'vs best':>8}")
        for frac in [float(f) for f in a.fracs.split(",")]:
            nf = max(1, int(frac * n))
            fr = cp.sort(rs.choice(n, nf, replace=False)).astype(cp.int32) if frac < 1 else \
                cp.arange(n, dtype=cp.int32)
            e = int((W.indptr[fr + 1] - W.indptr[fr]).sum())
            byts = e * (4 + real + C * real) + nf * (4 + 8 + 3 * C * real + 1)
            res = []
            for gs in [int(g) for g in a.groups.split(",")]:
                ops = FrontierOps(W, C, gs)
                if not ops.use_kernel:
                    continue
                ms = timeit(lambda: ops.jacobi(fr, X, rhs, diag, 1e-4))
                res.append((gs, ms))
            best = min(m for _, m in res)
            for gs, ms in res:
                label = f"{gs}" if gs <= 32 else f"blk{gs}"
                print(f"{nf:>10,} {label:>6} {ms:9.3f} {e / ms / 1e6:8.2f} {byts / ms / 1e6:7.0f} {ms / best:7.1f}x")
                rows.append(dict(dataset=ds.name, n=n, nnz=W.nnz, C=C, dtype=a.dtype, frontier=nf, edges=e,
                                 group=gs, ms=ms, gedges_s=e / ms / 1e6, gb_s=byts / ms / 1e6,
                                 full_spmm_ms=full_ms))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
