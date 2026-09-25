"""Figures from bench / kernelbench CSVs.

  python scripts/plot_results.py results/h100_sbm10m.csv --outdir results/fig
  python scripts/plot_results.py results/sweep_*.csv --pareto --outdir results/fig
  python scripts/plot_results.py --kernels results/h100_kernels.csv --outdir results/fig
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order; color follows the solver, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDER = ["itlp", "itlp-warm", "dynlp", "dynlp-knowninit", "dynlp+push", "dynlp+pcg", "dynlp+amg", "dynlp+auto"]
COLOR = dict(zip(ORDER, SERIES))
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # ordinal steps
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e6e5e0", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False,
        "axes.axisbelow": True,
    })


def solver_order(names):
    return [s for s in ORDER if s in names] + sorted(set(names) - set(ORDER))


def summary(df, outdir, tag):
    g = df.groupby("solver").agg(ms=("ms", "sum"), sweeps=("sweeps", "sum"), err=("err_max", "max"),
                                 bound=("cert_post", "max"), agree=("agree", "min"))
    names = solver_order(list(g.index))
    g = g.loc[names]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(names) + 1.6), sharey=True)
    for ax, col, title, fmt in [(axes[0], "ms", "Total time (ms, log)", "{:,.0f}"),
                                (axes[1], "sweeps", "Work (edge sweeps, log)", "{:,.0f}"),
                                (axes[2], "err", "Max error vs exact F* (log)", "{:.1e}")]:
        vals = g[col].to_numpy(dtype=float)
        y = np.arange(len(names))
        ax.barh(y, vals, color=[COLOR.get(s, INK2) for s in names], height=0.62, edgecolor=SURFACE, linewidth=2)
        ax.set_xscale("log")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", color=INK, fontsize=9)
        ax.margins(x=0.35)
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].invert_yaxis()
    fig.suptitle(f"{tag}: all batches", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, f"{tag}_summary.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def per_batch(df, outdir, tag):
    names = solver_order(df.solver.unique())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s in names:
        d = df[df.solver == s].sort_values("batch")
        ax.plot(d.batch, d.ms, color=COLOR.get(s, INK2), marker="o", markersize=5, label=s)
    ax.set_yscale("log")
    ax.set_xlabel("batch")
    ax.set_ylabel("time per batch (ms)")
    ax.set_title(f"{tag}: time per batch", loc="left")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, f"{tag}_per_batch.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def pareto(df, outdir, tag):
    """Total time vs worst true error; one point per (solver, run tag)."""
    g = df.groupby(["tag", "solver"]).agg(ms=("ms", "sum"), err=("err_max", "max")).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in solver_order(g.solver.unique()):
        d = g[g.solver == s].sort_values("err")
        ax.plot(d.err, d.ms, color=COLOR.get(s, INK2), marker="o", markersize=8, label=s,
                markeredgecolor=SURFACE, markeredgewidth=2)
        last = d.iloc[-1]
        ax.annotate(s, (last.err, last.ms), textcoords="offset points", xytext=(6, 4), fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("worst error vs exact harmonic solution (max over batches)")
    ax.set_ylabel("total time (ms)")
    ax.set_title(f"{tag}: time vs achieved accuracy (lower-left is better)", loc="left")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, f"{tag}_pareto.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def kernels(df, outdir, tag):
    cols = sorted(df.C.unique())
    fig, axes = plt.subplots(1, len(cols), figsize=(5.5 * len(cols), 4), sharey=True, squeeze=False)
    groups = sorted(df.group.unique())
    labels = [str(g) if g <= 32 else f"block/row ({g})" for g in groups]
    for ax, C in zip(axes[0], cols):
        d = df[df.C == C]
        fronts = sorted(d.frontier.unique())
        ramp = BLUE_RAMP[-len(fronts):] if len(fronts) <= len(BLUE_RAMP) else BLUE_RAMP
        for f, colr in zip(fronts, ramp):
            e = d[d.frontier == f].set_index("group").loc[groups]
            ax.plot(range(len(groups)), e.gedges_s, color=colr, marker="o", markersize=6, label=f"{f:,} rows")
        ax.set_xticks(range(len(groups)), labels, rotation=30, ha="right")
        ax.set_title(f"C = {C} column(s)", loc="left")
        ax.set_xlabel("lanes per frontier row")
    axes[0][0].set_ylabel("throughput (G edges/s)")
    axes[0][-1].legend(title="frontier size", fontsize=9)
    fig.suptitle(f"{tag}: frontier Jacobi kernel mapping", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, f"{tag}_kernels.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", help="bench CSV files (globs ok)")
    ap.add_argument("--kernels", default=None, help="kernelbench CSV")
    ap.add_argument("--pareto", action="store_true", help="treat CSVs as a delta/tol sweep")
    ap.add_argument("--outdir", default="results/fig")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    style()
    os.makedirs(a.outdir, exist_ok=True)
    files = [f for pat in a.csv for f in glob.glob(pat)]
    out = []
    if files:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df["tag"] = df["tag"].fillna("").astype(str)
        tag = a.tag or os.path.splitext(os.path.basename(files[0]))[0]
        if a.pareto:
            out.append(pareto(df, a.outdir, tag))
        else:
            out += [summary(df, a.outdir, tag), per_batch(df, a.outdir, tag)]
    if a.kernels:
        kdf = pd.read_csv(a.kernels)
        out.append(kernels(kdf, a.outdir, a.tag or os.path.splitext(os.path.basename(a.kernels))[0]))
    for p in out:
        print("[fig]", p)


if __name__ == "__main__":
    main()
