"""Streaming benchmark: run solvers over a batch stream and compare against F*.

  python -m dynlp.bench --dataset sbm --n 1000000 --solvers itlp,dynlp,dynlp+auto --out results/sbm.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys as _sys
import time

import numpy as np

from . import backend, graphs
from .backend import Timer
from .solvers import SOLVERS, Reference, make, predict, rho_cols
from .stream import IncrementalStream, Stream

ARGS = {  # group: [(flag, type, default, help)]; type None = store_true
    "dataset": [("--dataset", str, "sbm", "sbm | er | gmm-knn | ogbn-arxiv | ogbn-products | path/to/graph.npz"),
                ("--n", int, 100_000, None), ("--classes", int, 2, None), ("--deg", float, 10.0, None),
                ("--p-in", float, 0.85, "SBM: fraction of intra-class edges"), ("--knn", int, 10, None),
                ("--dim", int, 32, None), ("--data-dir", str, "data", None)],
    "stream": [("--label-frac", float, 0.01, None), ("--init-frac", float, 0.1, None), ("--batches", int, 10, None),
               ("--del-frac", float, 0.1, None), ("--eta-rel", float, 0.0, "dongle regularizer (x mean degree)"),
               ("--incremental", None, False, "keep the batch system resident (needed by dynlp+inc)")],
    "solvers": [("--solvers", str, "itlp,dynlp,dynlp+auto", ",".join(SOLVERS)),
                ("--delta", float, 1e-4, "ItLP/DynLP change threshold"), ("--tol", float, 1e-3, "certified max error"),
                ("--group", str, "auto", "lanes per row for frontier kernels (1..32, 128=block/row)"),
                ("--dtype", str, "float32", None), ("--max-iter", int, None, None)],
    "run": [("--backend", str, "auto", "auto | cupy | numpy"), ("--no-reference", None, False, "skip F* (no errors)"),
            ("--seed", int, 0, None), ("--out", str, None, "CSV output (a .json config is written next to it)"),
            ("--tag", str, "", "free-form tag stored in every row"), ("--no-warmup", None, False, "skip the JIT warm-up")],
}
FIELDS = ["tag", "dataset", "K", "batch", "solver", "n_active", "n_u", "nnz", "inserted", "deleted", "iters", "edges",
          "sweeps", "ms", "setup_ms", "path", "err_max", "cert_self", "cert_post", "hmax", "hmax_ref", "agree",
          "acc_true", "build_ms", "ref_ms", "delta", "tol", "dtype", "group"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, opts in ARGS.items():
        g = ap.add_argument_group(name)
        for flag, typ, default, hlp in opts:
            if typ is None:
                g.add_argument(flag, action="store_true", help=hlp)
            else:
                g.add_argument(flag, type=typ, default=default, help=hlp)
    return ap.parse_args(argv)


def env_info(be):
    info = {"backend": be.name, "python": platform.python_version()}
    if be.is_gpu:
        cp = be.xp
        p = cp.cuda.runtime.getDeviceProperties(0)
        info.update(gpu=p["name"].decode(), cc=f'{p["major"]}.{p["minor"]}', mem_gb=round(p["totalGlobalMem"] / 1e9, 1),
                    cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
    return info


def metrics(sys, sys64, Fall, Fstar, hmax, y_u, K, pick):
    """acc (true labels), and err / agree / cert_post against F* when a reference ran."""
    xp, nan = backend.get().xp, float("nan")
    F = pick(Fall)
    pred, known = predict(F, K), y_u >= 0
    acc = float((pred[known] == y_u[known]).mean()) if bool(known.any()) else nan
    if Fstar is None or not sys.n_solved:
        return acc, nan, nan, nan
    Fa = Fall.astype(xp.float64)
    if sys.solved is not None:
        Fa[~sys.solved] = 0  # inert rows hold 0 in the exact solution
    R = sys64.rhs - (sys64.diag[:, None] * Fa - sys64.W @ Fa)
    return (acc, float(xp.abs(F.astype(xp.float64) - Fstar).max()), float((pred == predict(Fstar, K)).mean()),
            float(rho_cols(sys64, R).max()) * hmax)


def main(argv=None):
    a = parse(argv)
    be = backend.set_backend(a.backend)
    info = env_info(be)
    print("[env]", json.dumps(info), flush=True)
    t0 = time.perf_counter()
    ds = graphs.load(a.dataset, n=a.n, K=a.classes, deg=a.deg, p_in=a.p_in, knn=a.knn, dim=a.dim, seed=a.seed,
                     dtype=a.dtype, data_dir=a.data_dir)
    be.sync()
    print(f"[data] {ds.name}: n={ds.n:,} nnz={ds.nnz:,} K={ds.K} ({time.perf_counter() - t0:.1f}s, "
          f"gpu mem {be.mem_used_gb():.1f} GB)", flush=True)
    stream = (IncrementalStream if a.incremental else Stream)(
        ds, label_frac=a.label_frac, init_frac=a.init_frac, n_batches=a.batches, del_frac=a.del_frac,
        eta_rel=a.eta_rel, seed=a.seed, dtype=a.dtype)
    names = [s.strip() for s in a.solvers.split(",") if s.strip()]
    if "dynlp+inc" in names and not a.incremental:
        raise SystemExit("dynlp+inc needs --incremental")
    if not a.no_warmup:
        warmup(names, ds.K, a)
    solvers = [make(nm, ds.n, ds.K, dtype=a.dtype, delta=a.delta, tol=a.tol, group=a.group, max_iter=a.max_iter)
               for nm in names]
    ref, rows, it = None if a.no_reference else Reference(ds.n, ds.K), [], stream.batches()
    for _ in range(len(stream)):
        with Timer() as tb:
            sys = next(it)
        ref_ms, Fstar, hmax, sys64 = float("nan"), None, None, None
        pick = (lambda v: v) if sys.solved is None else (lambda v: v[sys.solved])  # noqa: E731  score solved rows
        if ref is not None:
            with Timer() as tr:
                Fstar, hmax, _ = ref.solve(sys)
            ref_ms, sys64, Fstar = tr.ms, sys.astype("float64"), pick(Fstar)
        y_u = pick(ds.y[sys.U])
        print(f"[batch {sys.t:>3}] active={sys.n_active:,} |U|={sys.n_solved:,} nnz(W)={sys.nnz:,} "
              f"+{sys.n_inserted:,} -{sys.n_deleted:,} comps={sys.n_comp:,} ungrounded={sys.n_ungrounded:,} "
              f"build={tb.ms:.0f}ms ref={ref_ms:.0f}ms" + (f" max(h)={hmax:.3g}" if hmax is not None else ""),
              flush=True)
        for s in solvers:
            st = s.solve(sys)
            acc, err, agree, cert = metrics(sys, sys64, s.scores(sys), Fstar, hmax, y_u, ds.K, pick)
            row = dict(tag=a.tag, dataset=ds.name, K=ds.K, batch=sys.t, solver=s.name, n_active=sys.n_active,
                       n_u=sys.n_solved, nnz=sys.nnz, inserted=sys.n_inserted, deleted=sys.n_deleted, iters=st.iters,
                       edges=st.edges, sweeps=st.edges / max(sys.nnz, 1), ms=st.ms, setup_ms=st.setup_ms,
                       path=st.path, err_max=err, cert_self=st.cert, cert_post=cert, hmax=st.hmax,
                       hmax_ref=float("nan") if hmax is None else hmax, agree=agree, acc_true=acc, build_ms=tb.ms,
                       ref_ms=ref_ms, delta=a.delta, tol=a.tol, dtype=a.dtype, group=a.group)
            rows.append(row)
            print(f"    {s.name:<16} {st.ms:9.1f} ms  iters={st.iters:<7} sweeps={row['sweeps']:8.1f} "
                  f"err={err:.2e} bound={cert:.2e} agree={agree:.4f} acc={acc:.4f} {st.path}", flush=True)
    summarize(rows, names)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
            json.dump({"args": vars(a), "env": info, "argv": _sys.argv}, fh, indent=2)
        print(f"[out] {len(rows)} rows -> {a.out}")
    return rows


def warmup(names, K, a):
    """Compile the CUDA kernels on a tiny stream (large first batch, then small ones) so they are not timed."""
    ds = graphs.sbm(3000, K, 8.0, seed=123, dtype=a.dtype)
    st = (IncrementalStream if a.incremental else Stream)(ds, init_frac=0.9, n_batches=3, dtype=a.dtype,
                                                          label_frac=0.02)
    solvers = [make(nm, ds.n, K, dtype=a.dtype, delta=a.delta, tol=a.tol, group=a.group) for nm in names]
    ref = None if a.no_reference else Reference(ds.n, K)
    for sys in st.batches():
        for s in ([ref] if ref else []) + solvers:
            s.solve(sys)
    backend.get().sync()


def summarize(rows, names):
    agg = lambda f, v: f(v) if np.isfinite(v).any() else float("nan")  # noqa: E731
    hdr = f"{'solver':<16} {'total ms':>10} {'sweeps':>9} {'iters':>8} {'max err':>9} {'max bound':>10} " \
          f"{'min agree':>9} {'acc':>7}"
    print(f"\nsummary (all batches)\n{hdr}\n{'-' * len(hdr)}")
    for nm in names:
        rs = [r for r in rows if r["solver"] == nm]
        if rs:
            f = lambda k: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
            print(f"{nm:<16} {f('ms').sum():10.1f} {f('sweeps').sum():9.1f} {int(f('iters').sum()):8d} "
                  f"{agg(np.nanmax, f('err_max')):9.2e} {agg(np.nanmax, f('cert_post')):10.2e} "
                  f"{agg(np.nanmin, f('agree')):9.4f} {np.nanmean(f('acc_true')):7.4f}")


if __name__ == "__main__":
    main()
