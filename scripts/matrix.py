"""Method matrix for a results directory: methods x datasets, time and accuracy first.

  python scripts/matrix.py results/sanity     # -> results/sanity/matrix.md and matrix.csv
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

ORDER = ["itlp", "dynlp", "dynlp-fh", "dynlp+pcg", "dynlp+amg", "dynlp+auto", "dynlp+inc"]


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def main(d):
    csvs = [f for f in sorted(glob.glob(os.path.join(d, "*.csv"))) if not f.endswith("matrix.csv")]
    df = pd.concat([x for x in map(pd.read_csv, csvs) if "solver" in x.columns], ignore_index=True)
    m = df.groupby(["tag", "solver"]).agg(
        time_s=("ms", lambda v: v.sum() / 1e3), iters=("iters", "sum"), sweeps=("sweeps", "sum"),
        err=("err_max", "max"), bound=("cert_post", "max"), acc=("acc_true", "mean"), run_s=("run_s", "max"),
        path=("path", lambda v: v.mode().iat[0])).reset_index()
    build = df.drop_duplicates(["tag", "batch"]).groupby("tag").build_ms.sum() / 1e3
    base = m[m.solver == "dynlp"].set_index("tag").time_s
    m["x_dynlp"] = m.tag.map(base) / m.time_s
    m["order"] = m.solver.map({s: i for i, s in enumerate(ORDER)}).fillna(len(ORDER))
    m = m.sort_values(["tag", "order"]).drop(columns="order")
    m.to_csv(os.path.join(d, "matrix.csv"), index=False)

    tags, solvers = list(dict.fromkeys(m.tag)), list(dict.fromkeys(m.sort_values("solver", key=lambda s: s.map(
        {x: i for i, x in enumerate(ORDER)}).fillna(99)).solver))
    cell = {(r.tag, r.solver): r for r in m.itertuples()}
    f = lambda v, fmt: "–" if pd.isna(v) else format(v, fmt)  # noqa: E731
    out = [f"# Method matrix: `{d}`", "",
           f"δ = {df.delta.iat[0]:g} (DynLP/ItLP), tol = {df.tol.iat[0]:g} (DynLP+). Cell: solve time / accuracy.", "",
           table(["method"] + tags, [[s] + [f"{f(c.time_s, '.2f')} s / {f(c.acc, '.3f')}" if (c := cell.get((t, s)))
                                            else "" for t in tags] for s in solvers])]
    for t in tags:
        rs = [cell[(t, s)] for s in solvers if (t, s) in cell]
        out += ["", f"## {t}", f"wall {rs[0].run_s:.0f} s per run, batch assembly {build[t]:.2f} s", "",
                table(["method", "time s", "× dynlp", "iters", "sweeps", "max err", "bound", "acc", "path"],
                      [[r.solver, f(r.time_s, ".2f"), f(r.x_dynlp, ".2f"), str(r.iters), f(r.sweeps, ".0f"),
                        f(r.err, ".1e"), f(r.bound, ".1e"), f(r.acc, ".4f"), r.path] for r in rs])]
    open(os.path.join(d, "matrix.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sanity")
