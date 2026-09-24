"""Streaming benchmark: run solvers over a batch stream and compare against F*.

Example:
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


def env_info(be):
    info = {"backend": be.name, "python": platform.python_version()}
    if be.is_gpu:
        import cupy
        p = cupy.cuda.runtime.getDeviceProperties(0)
        info.update(gpu=p["name"].decode(), cc=f'{p["major"]}.{p["minor"]}',
                    mem_gb=round(p["totalGlobalMem"] / 1e9, 1), cupy=cupy.__version__,
                    cuda_runtime=cupy.cuda.runtime.runtimeGetVersion())
    return info


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("dataset")
    g.add_argument("--dataset", default="sbm",
                   help="sbm | er | gmm-knn | ogbn-arxiv | ogbn-products | path/to/graph.npz")
    g.add_argument("--n", type=int, default=100_000)
    g.add_argument("--classes", type=int, default=2)
    g.add_argument("--deg", type=float, default=10.0)
    g.add_argument("--p-in", type=float, default=0.85, help="SBM: fraction of intra-class edges")
    g.add_argument("--knn", type=int, default=10)
    g.add_argument("--dim", type=int, default=32)
    g.add_argument("--data-dir", default="data")
    g = ap.add_argument_group("stream")
    g.add_argument("--label-frac", type=float, default=0.01)
    g.add_argument("--init-frac", type=float, default=0.1)
    g.add_argument("--batches", type=int, default=10)
    g.add_argument("--del-frac", type=float, default=0.1)
    g.add_argument("--eta-rel", type=float, default=0.0, help="dongle regularizer (x mean degree)")
    g.add_argument("--incremental", action="store_true",
                   help="keep the batch system resident and update it in place (needed by dynlp+inc)")
    g = ap.add_argument_group("solvers")
    g.add_argument("--solvers", default="itlp,dynlp,dynlp+auto", help=",".join(SOLVERS))
    g.add_argument("--delta", type=float, default=1e-4, help="ItLP/DynLP change threshold")
    g.add_argument("--tol", type=float, default=1e-3, help="DynLP+ certified max error")
    g.add_argument("--group", default="auto", help="lanes per row for frontier kernels (1..32, 128=block/row)")
    g.add_argument("--dtype", default="float32")
    g.add_argument("--max-iter", type=int, default=None)
    g = ap.add_argument_group("run")
    g.add_argument("--backend", default="auto", help="auto | cupy | numpy")
    g.add_argument("--no-reference", action="store_true", help="skip F* (no error metrics)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default=None, help="CSV output path (a .json with the config is written next to it)")
    g.add_argument("--tag", default="", help="free-form tag stored in every CSV row")
    g.add_argument("--no-warmup", action="store_true", help="skip the untimed JIT warm-up run")
    return ap.parse_args(argv)


FIELDS = ["tag", "dataset", "K", "batch", "solver", "n_active", "n_u", "nnz", "inserted", "deleted",
          "iters", "edges", "sweeps", "ms", "setup_ms", "path", "err_max", "cert_self", "cert_post", "hmax", "hmax_ref",
          "agree", "acc_true", "build_ms", "ref_ms", "delta", "tol", "dtype", "group"]


def main(argv=None):
    a = parse(argv)
    be = backend.set_backend(a.backend)
    xp = be.xp
    info = env_info(be)
    print("[env]", json.dumps(info), flush=True)

    t0 = time.perf_counter()
    ds = graphs.load(a.dataset, n=a.n, K=a.classes, deg=a.deg, p_in=a.p_in, knn=a.knn, dim=a.dim,
                     seed=a.seed, dtype=a.dtype, data_dir=a.data_dir)
    be.sync()
    print(f"[data] {ds.name}: n={ds.n:,} nnz={ds.nnz:,} K={ds.K} "
          f"({time.perf_counter() - t0:.1f}s, gpu mem {be.mem_used_gb():.1f} GB)", flush=True)

    stream_cls = IncrementalStream if a.incremental else Stream
    stream = stream_cls(ds, label_frac=a.label_frac, init_frac=a.init_frac, n_batches=a.batches,
                        del_frac=a.del_frac, eta_rel=a.eta_rel, seed=a.seed, dtype=a.dtype)
    names = [s.strip() for s in a.solvers.split(",") if s.strip()]
    if "dynlp+inc" in names and not a.incremental:
        raise SystemExit("dynlp+inc needs --incremental")
    if not a.no_warmup:
        warmup(names, ds.K, a)
    solvers = [make(nm, ds.n, ds.K, dtype=a.dtype, delta=a.delta, tol=a.tol, group=a.group,
                    max_iter=a.max_iter) for nm in names]
    ref = None if a.no_reference else Reference(ds.n, ds.K)

    rows = []
    it = stream.batches()
    for _ in range(len(stream)):
        with Timer() as tb:
            sys = next(it)
        ref_ms, Fstar, hmax, sys64 = float("nan"), None, None, None
        if ref is not None:
            with Timer() as tr:
                Fstar, hmax, _ = ref.solve(sys)
            ref_ms = tr.ms
            sys64 = sys.astype("float64")
        # global-index systems carry inert rows: score only the rows that are solved
        pick = (lambda v: v) if sys.solved is None else (lambda v: v[sys.solved])  # noqa: E731
        y_u = pick(ds.y[sys.U])
        if Fstar is not None:
            Fstar = pick(Fstar)
        print(f"[batch {sys.t:>3}] active={sys.n_active:,} |U|={sys.n_solved:,} nnz(W)={sys.nnz:,} "
              f"+{sys.n_inserted:,} -{sys.n_deleted:,} comps={sys.n_comp:,} ungrounded={sys.n_ungrounded:,} "
              f"build={tb.ms:.0f}ms ref={ref_ms:.0f}ms"
              + (f" max(h)={hmax:.3g}" if hmax is not None else ""), flush=True)
        for s in solvers:
            st = s.solve(sys)
            Fall = s.scores(sys)
            F = pick(Fall)
            pred = predict(F, ds.K)
            known = y_u >= 0
            acc = float((pred[known] == y_u[known]).mean()) if bool(known.any()) else float("nan")
            err = agree = cert_post = float("nan")
            if Fstar is not None and sys.n_solved:
                F64 = F.astype(xp.float64)
                err = float(xp.abs(F64 - Fstar).max())
                agree = float((pred == predict(Fstar, ds.K)).mean())
                Fa = Fall.astype(xp.float64)
                if sys.solved is not None:
                    Fa[~sys.solved] = 0  # inert rows hold 0 in the exact solution
                R = sys64.rhs - (sys64.diag[:, None] * Fa - (sys64.W @ Fa))
                cert_post = float(rho_cols(sys64, R).max()) * hmax
            row = dict(tag=a.tag, dataset=ds.name, K=ds.K, batch=sys.t, solver=s.name,
                       n_active=sys.n_active, n_u=sys.n_solved, nnz=sys.nnz, inserted=sys.n_inserted,
                       deleted=sys.n_deleted, iters=st.iters, edges=st.edges,
                       sweeps=st.edges / max(sys.nnz, 1), ms=st.ms, setup_ms=st.setup_ms, path=st.path,
                       err_max=err, cert_self=st.cert, cert_post=cert_post, hmax=st.hmax,
                       hmax_ref=hmax if hmax is not None else float("nan"), agree=agree, acc_true=acc,
                       build_ms=tb.ms, ref_ms=ref_ms, delta=a.delta, tol=a.tol, dtype=a.dtype, group=a.group)
            rows.append(row)
            print(f"    {s.name:<16} {st.ms:9.1f} ms  iters={st.iters:<7} sweeps={row['sweeps']:8.1f} "
                  f"err={err:.2e} bound={cert_post:.2e} agree={agree:.4f} acc={acc:.4f} {st.path}",
                  flush=True)

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
    """Compile CUDA/NVRTC kernels on a tiny stream so they are not timed."""
    ds = graphs.sbm(3000, K, 8.0, seed=123, dtype=a.dtype)
    # small batches after a large first one, so both the global and the push paths compile
    st = (IncrementalStream if a.incremental else Stream)(ds, init_frac=0.9, n_batches=3, dtype=a.dtype,
                                                          label_frac=0.02)
    solvers = [make(nm, ds.n, K, dtype=a.dtype, delta=a.delta, tol=a.tol, group=a.group) for nm in names]
    ref = None if a.no_reference else Reference(ds.n, K)
    for sys in st.batches():
        if ref is not None:
            ref.solve(sys)
        for s in solvers:
            s.solve(sys)
    backend.get().sync()


def summarize(rows, names):
    print("\nsummary (all batches)")
    hdr = f"{'solver':<16} {'total ms':>10} {'sweeps':>9} {'iters':>8} {'max err':>9} {'max bound':>10} " \
          f"{'min agree':>9} {'acc':>7}"
    print(hdr)
    print("-" * len(hdr))
    for nm in names:
        rs = [r for r in rows if r["solver"] == nm]
        if not rs:
            continue
        f = lambda k: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
        print(f"{nm:<16} {f('ms').sum():10.1f} {f('sweeps').sum():9.1f} {int(f('iters').sum()):8d} "
              f"{np.nanmax(f('err_max')) if np.isfinite(f('err_max')).any() else float('nan'):9.2e} "
              f"{np.nanmax(f('cert_post')) if np.isfinite(f('cert_post')).any() else float('nan'):10.2e} "
              f"{np.nanmin(f('agree')) if np.isfinite(f('agree')).any() else float('nan'):9.4f} "
              f"{np.nanmean(f('acc_true')):7.4f}")


if __name__ == "__main__":
    main()
