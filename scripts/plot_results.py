"""Figures for a results directory: per dataset, total time / work / error per method, and
time per batch.

  python scripts/plot_results.py results/sanity     # -> results/sanity/fig/*.png
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# validated categorical palette; color follows the method, never its rank
ORDER = ["itlp", "dynlp", "dynlp-fh", "dynlp+pcg", "dynlp+amg", "dynlp+auto", "dynlp+inc"]
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False})


def figures(df, tag, outdir):
    names = [s for s in ORDER if s in set(df.solver)]
    g = df.groupby("solver").agg(ms=("ms", "sum"), sweeps=("sweeps", "sum"), err=("err_max", "max")).loc[names]
    fig, axes = plt.subplots(1, 4, figsize=(17, 0.45 * len(names) + 1.8))
    for ax, col, title, fmt in [(axes[0], "ms", "total time (ms, log)", "{:,.0f}"),
                                (axes[1], "sweeps", "work (edge sweeps, log)", "{:,.0f}"),
                                (axes[2], "err", "max error vs F* (log)", "{:.1e}")]:
        vals = g[col].tolist()
        ax.barh(range(len(names)), vals, color=[COLOR[s] for s in names], height=0.62, edgecolor=SURFACE, linewidth=2)
        ax.set(xscale="log", title=title, yticks=range(len(names)), yticklabels=names if col == "ms" else [])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(vals):
            if v == v:  # skip NaN
                ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9)
        ax.margins(x=0.35)
    for s in names:
        d = df[df.solver == s].sort_values("batch")
        axes[3].plot(d.batch, d.ms, color=COLOR[s], marker="o", markersize=4, label=s)
    axes[3].set(yscale="log", xlabel="batch", title="time per batch (ms)")
    axes[3].legend(fontsize=8)
    fig.suptitle(tag, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, f"{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(d):
    csvs = [f for f in glob.glob(os.path.join(d, "*.csv")) if not f.endswith("matrix.csv")]
    df = pd.concat([x for x in map(pd.read_csv, csvs) if "solver" in x.columns])
    os.makedirs(out := os.path.join(d, "fig"), exist_ok=True)
    for tag, part in df.groupby("tag"):
        print("[fig]", figures(part, tag, out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sanity")
