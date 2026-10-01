"""Kernel micro-benchmark over all rows: cuSPARSE SpMM, our row-major SpMM, and the frontier
Jacobi kernel with each lane mapping (lanes per row; 128 = DynLP's block per row; cols =
class-parallel).

  python -m dynlp.kernelbench --dataset sbm --n 5000000 --cols 2,16,32,64 --out results/x/kernels.csv
"""
from __future__ import annotations

import argparse
import csv
import os

from . import graphs
from .backend import xp
from .kernels import FrontierOps, spmm_axpy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, typ, default in [("--dataset", str, "sbm"), ("--n", int, 1_000_000), ("--deg", float, 10.0),
                               ("--dtype", str, "float32"), ("--cols", str, "2,16,32,64"),
                               ("--groups", str, "8,32,128,cols"), ("--reps", int, 10), ("--seed", int, 0),
                               ("--data-dir", str, "data"), ("--out", str, None)]:
        ap.add_argument(flag, type=typ, default=default)
    a = ap.parse_args(argv)
    ds = graphs.load(a.dataset, n=a.n, deg=a.deg, seed=a.seed, dtype=a.dtype, data_dir=a.data_dir)
    W, n, real = ds.A, ds.A.shape[0], xp.dtype(a.dtype).itemsize
    print(f"[data] {ds.name}: n={n:,} nnz={W.nnz:,} mean deg={W.nnz / n:.1f}")
    rs, rows, (t0, t1) = xp.random.RandomState(a.seed), [], (xp.cuda.Event(), xp.cuda.Event())

    def timeit(fn):
        fn()
        t0.record()
        for _ in range(a.reps):
            fn()
        t1.record()
        t1.synchronize()
        return xp.cuda.get_elapsed_time(t0, t1) / a.reps

    allrows = xp.arange(n, dtype=xp.int32)
    for C in [int(c) for c in a.cols.split(",")]:
        X, rhs = (rs.random_sample((n, C)).astype(a.dtype) for _ in range(2))
        diag = (W @ xp.ones(n, dtype=a.dtype) + 1).astype(a.dtype)
        res = [("cusparse", timeit(lambda: W @ X)), ("rowmajor", timeit(lambda: spmm_axpy(W, X)))]
        for g in a.groups.split(","):
            ops = FrontierOps(W, C, mapping="cols") if g == "cols" else FrontierOps(W, C, int(g), mapping="rows")
            if ops.use_kernel:
                res.append((g if g == "cols" else f"lanes{g}" if int(g) <= 32 else f"block{g}",
                            timeit(lambda: ops.jacobi(allrows, X, rhs, diag, 1e-4))))
        best = min(ms for _, ms in res)
        gb = W.nnz * (4 + real + C * real) / 1e6
        print(f"\nC={C}\n{'kernel':>9} {'ms':>9} {'GB/s':>7} {'vs best':>8}")
        for k, ms in res:
            print(f"{k:>9} {ms:9.3f} {gb / ms:7.0f} {ms / best:7.1f}x")
            rows.append(dict(dataset=ds.name, n=n, nnz=W.nnz, C=C, dtype=a.dtype, kernel=k, ms=ms, gb_s=gb / ms))
        del X, rhs
        xp.get_default_memory_pool().free_all_blocks()  # keep the peak to one column count
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
