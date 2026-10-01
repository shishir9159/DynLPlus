"""Streaming benchmark: run solvers over a batch stream and compare them against F*.

  python -m dynlp.bench --dataset imdb --solvers dynlp,dynlp-fh,dynlp+auto --out results/x/imdb.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys as _sys
import time

from . import graphs
from .backend import Timer, mem_used_gb, sync, xp
from .solvers import SOLVERS, Reference, make, predict, rho_cols
from .stream import IncrementalStream, Stream

ARGS = [  # (flag, type, default, help); type None = store_true
    ("--dataset", str, "imdb", "imdb | synth2 | synth10 | sbm | er | path/to/graph.npz"),
    ("--n", int, 100_000, "sbm/er size"), ("--classes", int, 2, "sbm/er classes"), ("--deg", float, 10.0, None),
    ("--p-in", float, 0.85, "SBM: fraction of intra-class edges"), ("--data-dir", str, "data", None),
    ("--label-frac", float, 0.01, None), ("--init-frac", float, 0.1, None), ("--batches", int, 10, None),
    ("--del-frac", float, 0.1, None), ("--eta-rel", float, 0.0, "dongle regularizer (x mean degree)"),
    ("--incremental", None, False, "keep the batch system resident (needed by dynlp+inc)"),
    ("--solvers", str, "dynlp,dynlp-fh,dynlp+auto", ",".join(SOLVERS)),
    ("--delta", float, 1e-4, "DynLP/ItLP change threshold (the paper's)"), ("--tol", float, 1e-3, "certified max error"),
    ("--group", str, "auto", "lanes per row for frontier kernels (1..32, 128 = block per row)"),
    ("--dtype", str, "float32", None), ("--max-iter", int, None, None),
    ("--no-reference", None, False, "skip F* (no error columns)"), ("--seed", int, 0, None),
    ("--out", str, None, "CSV output (a .json config is written next to it)"), ("--tag", str, "", "label for this run"),
    ("--no-warmup", None, False, "skip the JIT warm-up"),
]
FIELDS = ["tag", "dataset", "K", "batch", "solver", "n_active", "n_u", "nnz", "inserted", "deleted", "iters", "edges",
          "sweeps", "ms", "setup_ms", "path", "err_max", "cert_self", "cert_post", "hmax", "hmax_ref", "agree",
          "acc_true", "build_ms", "ref_ms", "run_s", "delta", "tol", "dtype", "group"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, typ, default, hlp in ARGS:
        if typ is None:
            ap.add_argument(flag, action="store_true", help=hlp)
        else:
            ap.add_argument(flag, type=typ, default=default, help=hlp)
    return ap.parse_args(argv)


def env_info():
    p = xp.cuda.runtime.getDeviceProperties(0)
    return {"python": platform.python_version(), "gpu": p["name"].decode(), "cc": f'{p["major"]}.{p["minor"]}',
            "mem_gb": round(p["totalGlobalMem"] / 1e9, 1), "cupy": xp.__version__,
            "cuda_runtime": xp.cuda.runtime.runtimeGetVersion()}


def metrics(sys, sys64, Fall, Fstar, hmax, y_u, K, pick):
    """acc (true labels), and err / agree / cert_post against F* when a reference ran."""
    nan, F = float("nan"), pick(Fall)
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
    a, t_run = parse(argv), time.perf_counter()
    info = env_info()
    print("[env]", json.dumps(info), flush=True)
    ds = graphs.load(a.dataset, n=a.n, K=a.classes, deg=a.deg, p_in=a.p_in, seed=a.seed, dtype=a.dtype,
                     data_dir=a.data_dir)
    sync()
    print(f"[data] {ds.name}: n={ds.n:,} nnz={ds.nnz:,} K={ds.K} ({time.perf_counter() - t_run:.1f}s, "
          f"gpu mem {mem_used_gb():.1f} GB)", flush=True)
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
        print(f"[batch {sys.t:>3}] active={sys.n_active:,} |U|={sys.n_solved:,} +{sys.n_inserted:,} "
              f"-{sys.n_deleted:,} comps tau/fh={sys.comps['tau'][1]:,}/{sys.comps['fh'][1]:,} "
              f"build={tb.ms:.0f}ms ref={ref_ms:.0f}ms" + (f" max(h)={hmax:.3g}" if hmax is not None else ""),
              flush=True)
        for s in solvers:
            st = s.solve(sys)
            acc, err, agree, cert = metrics(sys, sys64, s.scores(sys), Fstar, hmax, y_u, ds.K, pick)
            row = dict(tag=a.tag or ds.name, dataset=ds.name, K=ds.K, batch=sys.t, solver=s.name,
                       n_active=sys.n_active, n_u=sys.n_solved, nnz=sys.nnz, inserted=sys.n_inserted,
                       deleted=sys.n_deleted, iters=st.iters, edges=st.edges, sweeps=st.edges / max(sys.nnz, 1),
                       ms=st.ms, setup_ms=st.setup_ms, path=st.path, err_max=err, cert_self=st.cert, cert_post=cert,
                       hmax=st.hmax, hmax_ref=float("nan") if hmax is None else hmax, agree=agree, acc_true=acc,
                       build_ms=tb.ms, ref_ms=ref_ms, delta=a.delta, tol=a.tol, dtype=a.dtype, group=a.group)
            rows.append(row)
            print(f"    {s.name:<11} {st.ms:9.1f} ms  iters={st.iters:<7} err={err:.2e} bound={cert:.2e} "
                  f"acc={acc:.4f} {st.path}", flush=True)
    run_s = time.perf_counter() - t_run
    for r in rows:
        r["run_s"] = run_s
    summarize(rows, names, run_s)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
            json.dump({"args": vars(a), "env": info, "argv": _sys.argv, "run_s": run_s}, fh, indent=2)
        print(f"[out] {len(rows)} rows -> {a.out}")
    return rows


def warmup(names, K, a):
    """Compile the CUDA kernels on a tiny stream (large first batch, then small ones) so they are not timed."""
    ds = graphs.sbm(3000, K, 8.0, seed=123, dtype=a.dtype)
    st = (IncrementalStream if a.incremental else Stream)(ds, init_frac=0.9, n_batches=3, dtype=a.dtype,
                                                          label_frac=0.02)
    solvers = [make(nm, ds.n, K, dtype=a.dtype, delta=a.delta, tol=a.tol, group=a.group) for nm in names]
    for sys in st.batches():
        for s in ([] if a.no_reference else [Reference(ds.n, K)]) + solvers:
            s.solve(sys)
    sync()


def summarize(rows, names, run_s):
    """Per method over all batches: time, speedup vs dynlp, iterations, worst error and bound, accuracy."""
    fin = lambda v: [x for x in v if not math.isnan(x)]  # noqa: E731
    tot = {nm: sum(r["ms"] for r in rows if r["solver"] == nm) for nm in names}
    base, nan = tot.get("dynlp"), float("nan")
    hdr = f"{'solver':<11} {'time s':>8} {'x dynlp':>8} {'iters':>8} {'max err':>9} {'max bound':>10} {'acc':>7}"
    print(f"\nsummary ({run_s:.0f}s wall)\n{hdr}\n{'-' * len(hdr)}")
    for nm in names:
        rs = [r for r in rows if r["solver"] == nm]
        err, bnd, acc = (fin([r[k] for r in rs]) for k in ("err_max", "cert_post", "acc_true"))
        print(f"{nm:<11} {tot[nm] / 1e3:8.2f} {base / tot[nm] if base and tot[nm] else nan:8.2f} "
              f"{sum(r['iters'] for r in rs):8d} {max(err, default=nan):9.2e} {max(bnd, default=nan):10.2e} "
              f"{sum(acc) / len(acc) if acc else nan:7.4f}")


if __name__ == "__main__":
    main()
