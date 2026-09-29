"""Offline checks for two KV-cache-inspired ideas (plan.md §5, K5 and K7).

codes: compressed class space (MLA-style). Solve r < K columns B·P instead of K.
       By linearity the result is F*·P. Rows of F* are probability vectors, so the
       decoded score <y_u, p_c> is within mu of F*_uc (mu = code coherence), and a
       label is certified when the decoded margin exceeds 2*mu.
       Reports the certified fraction and raw agreement against r.
elim:  exact eviction by elimination. Counts vertices removable by exact Gaussian
       elimination of degree <= 2 (tree parts outside the 2-core, plus degree-2
       vertices inside it).

  uv run --no-sync python scripts/kv_offline_checks.py codes --dataset ogbn-arxiv --init-frac 0.5
  uv run --no-sync python scripts/kv_offline_checks.py codes --dataset sbm --n 200000 --classes 172
  uv run --no-sync python scripts/kv_offline_checks.py elim --dataset ogbn-arxiv --batches 5
"""
from __future__ import annotations

import argparse

import numpy as np

from dynlp import backend, graphs
from dynlp.graphs import csr_rows
from dynlp.solvers import Reference
from dynlp.stream import Stream


def load(a):
    ds = graphs.load(a.dataset, n=a.n, K=a.classes, deg=a.deg, seed=a.seed, data_dir=a.data_dir)
    st = Stream(ds, init_frac=a.init_frac, n_batches=a.batches, label_frac=a.label_frac, seed=a.seed)
    return ds, st


def best_codes(K, r, draws, rs):
    """Unit code vectors with the lowest coherence among `draws` Gaussian draws."""
    best, best_mu = None, np.inf
    for _ in range(draws):
        P = rs.standard_normal((K, r))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        G = np.abs(P @ P.T)
        np.fill_diagonal(G, 0)
        mu = G.max()
        if mu < best_mu:
            best, best_mu = P, mu
    return best, best_mu


def check_codes(a):
    be = backend.get()
    ds, st = load(a)
    if ds.K <= 2:
        raise SystemExit("codes needs K > 2")
    sys = next(st.batches())
    F, hmax, _ = Reference(ds.n, ds.K).solve(sys)
    F = be.asnumpy(F).astype(np.float64)
    top = F.argmax(1)
    srt = np.sort(F, axis=1)
    print(f"{ds.name}: |U|={sys.n_u:,} K={ds.K} max(h)={hmax:.3g}; "
          f"exact margin (top1-top2) median={np.median(srt[:, -1] - srt[:, -2]):.3f}")
    welch = lambda r: np.sqrt(max(ds.K - r, 0) / (r * (ds.K - 1)))  # noqa: E731
    print(f"{'r':>4} {'r/K':>5} {'mu':>6} {'Welch':>6} {'certified':>10} {'agree':>7}")
    rs = np.random.RandomState(a.seed)
    for r in [int(x) for x in a.ranks.split(",")]:
        if r >= ds.K:
            continue
        P, mu = best_codes(ds.K, r, a.draws, rs)
        S = (F @ P) @ P.T                     # decoded scores
        s = np.sort(S, axis=1)
        cert = (s[:, -1] - s[:, -2]) > 2 * mu
        agree = S.argmax(1) == top
        print(f"{r:>4} {r / ds.K:5.2f} {mu:6.3f} {welch(r):6.3f} {cert.mean():10.3f} {agree.mean():7.3f}")


def two_core(n, rows, cols):
    """Mask of vertices in the 2-core (iterated removal of degree <= 1)."""
    xp = backend.get().xp
    alive = xp.ones(n, dtype=bool)
    while True:
        e = alive[rows] & alive[cols]
        deg = xp.bincount(rows[e], minlength=n)
        drop = alive & (deg <= 1)
        if not bool(drop.any()):
            return alive, deg
        alive &= ~drop


def check_elim(a):
    ds, st = load(a)
    print(f"{ds.name}: n={ds.n:,} nnz={ds.nnz:,}")
    print(f"{'batch':>5} {'|U|':>10} {'deg<=2':>8} {'outside 2-core':>15} {'eliminable':>11}")
    for sys in st.batches():
        n = sys.n_u
        rows, cols = csr_rows(sys.W), sys.W.indices
        deg = np.diff(backend.get().asnumpy(sys.W.indptr))
        core, cdeg = two_core(n, rows, cols)
        core_h = backend.get().asnumpy(core)
        cdeg_h = backend.get().asnumpy(cdeg)
        elim = (~core_h).sum() + (core_h & (cdeg_h == 2)).sum()
        print(f"{sys.t:>5} {n:>10,} {np.mean(deg <= 2):8.3f} {np.mean(~core_h):15.3f} {elim / max(n, 1):11.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["codes", "elim"])
    ap.add_argument("--dataset", default="sbm")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--deg", type=float, default=10.0)
    ap.add_argument("--init-frac", type=float, default=0.5)
    ap.add_argument("--batches", type=int, default=0)
    ap.add_argument("--label-frac", type=float, default=0.01)
    ap.add_argument("--ranks", default="4,8,16,32,64")
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    backend.set_backend(a.backend)
    check_codes(a) if a.check == "codes" else check_elim(a)


if __name__ == "__main__":
    main()
