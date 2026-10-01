"""Correctness tests (GPU). Run: uv run pytest -q"""
import random

import pytest

from dynlp import graphs
from dynlp.amg import AMG
from dynlp.backend import scatter_add, xp
from dynlp.cc import connected_components
from dynlp.kernels import FrontierOps, spmm_axpy
from dynlp.solvers import DynLPPlus, Reference, Stats, make, pcg
from dynlp.stream import Stream, new_components

close = xp.testing.assert_allclose


def dense_system(sys):
    A = xp.diag(sys.diag.astype(xp.float64)) - sys.W.toarray().astype(xp.float64)
    return A, sys.rhs.astype(xp.float64)


def same_partition(a, b):
    a, b = [int(x) for x in a], [int(x) for x in b]
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


def union_find(n, pairs):
    p = list(range(n))

    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for a, b in pairs:
        p[f(a)] = f(b)
    return [f(x) for x in range(n)]


def test_connected_components():
    rnd = random.Random(0)
    pairs = [(rnd.randrange(400), rnd.randrange(400)) for _ in range(300)]
    lab, nc = connected_components(400, xp.asarray([a for a, _ in pairs]), xp.asarray([b for _, b in pairs]))
    ref = union_find(400, pairs)
    assert nc == len(set(ref)) and same_partition(lab.get(), ref)


def _components(edges, m, tau, ground=None):
    """new_components on an explicit undirected edge list (all vertices new)."""
    a, b, w = zip(*edges)
    g = xp.ones(m) if ground is None else xp.asarray(ground, dtype=xp.float64)
    return new_components(m, xp.asarray(a + b), xp.asarray(b + a), xp.asarray(w + w, dtype=xp.float32), tau, 1.0, g)


def test_adaptive_components_cut_a_bridge_that_tau_chains():
    """Two tight cliques joined by an edge above the global mean but weak relative to both."""
    edges = [(a + o, b + o, 0.95 + 0.01 * ((a + b) % 3)) for o in (0, 6) for a in range(6) for b in range(a + 1, 6)]
    edges.append((5, 6, 0.8))
    comps = _components(edges, 12, tau=0.5)
    assert comps["tau"][1] == 1
    assert comps["fh"][1] == 2 and same_partition(comps["fh"][0].get(), [0] * 6 + [1] * 6)


def test_adaptive_components_merge_a_sparse_cluster():
    """A weak chain (all edges below tau) is left as singletons by tau but merged by fh,
    and a component without labeled contact joins its strongest neighbour."""
    edges = [(i, i + 1, 0.3 + 0.02 * (i % 3)) for i in range(7)]
    comps = _components(edges, 8, tau=0.5, ground=[1, 0, 0, 0, 0, 0, 0, 0])
    assert comps["tau"][1] == 8
    assert comps["fh"][1] == 1


def test_system_matches_dense_definition():
    ds = graphs.sbm(1500, K=3, deg=6, seed=1, dtype="float64")
    st = Stream(ds, init_frac=0.2, n_batches=3, label_frac=0.03, dtype="float64")
    A, y, lab = ds.A.toarray(), ds.y, st.labeled
    act = xp.zeros(ds.n, dtype=bool)
    act[st.order[: st.n0]] = True
    for sys in st.batches():
        if sys.t:
            act[st.chunks[sys.t - 1]] = True  # labeled vertices are never deleted
        U = sys.U
        assert not bool(lab[U].any())
        W = sys.W.toarray()
        close(W, A[U][:, U], rtol=1e-12)
        al = act & lab
        close(sys.B, xp.stack([A[U][:, al & (y == k)].sum(1) for k in range(3)], axis=1), rtol=1e-10)
        close(sys.diag, sys.s + W.sum(1), rtol=1e-12)
        coo = sys.W.tocoo()  # every solved component touches a labeled vertex
        cl, nc = connected_components(sys.n_u, coo.row, coo.col)
        g = xp.zeros(nc)
        scatter_add(g, cl, sys.B.sum(1))
        assert bool((g > 0).all())


@pytest.mark.parametrize("K", [2, 4])
def test_reference_matches_direct_solve(K):
    ds = graphs.sbm(2000, K=K, deg=6, seed=2, dtype="float64")
    st = Stream(ds, init_frac=0.3, n_batches=2, label_frac=0.02, dtype="float64")
    ref = Reference(ds.n, K)
    for sys in st.batches():
        F, hmax, _ = ref.solve(sys)
        A, rhs = dense_system(sys)
        assert float(xp.abs(F - xp.linalg.solve(A, rhs)).max()) < 1e-7
        assert hmax >= float(xp.linalg.solve(A, sys.diag).max()) * (1 - 1e-6)


@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("method", ["pcg", "amg", "auto"])
def test_dynlp_plus_certificate_holds(K, method):
    ds = graphs.sbm(3000, K=K, deg=8, seed=3, dtype="float32")
    st = Stream(ds, init_frac=0.2, n_batches=3, label_frac=0.02, dtype="float32")
    solver = DynLPPlus(ds.n, K, "float32", method=method, tol=1e-3)
    for sys in st.batches():
        stats = solver.solve(sys)
        A, rhs = dense_system(sys)
        err = float(xp.abs(solver.scores(sys).astype(xp.float64) - xp.linalg.solve(A, rhs)).max())
        assert stats.cert <= 1e-3 * 1.0001
        assert err <= stats.cert * 1.05 + 1e-6, (err, stats.cert)


def test_baselines_run_and_are_close():
    ds = graphs.sbm(2000, K=2, deg=8, seed=4, dtype="float32")
    st = Stream(ds, init_frac=0.3, n_batches=2, label_frac=0.02, dtype="float32")
    solvers = [make(nm, ds.n, 2, dtype="float32", delta=1e-5, tol=1e-3, group="auto")
               for nm in ["itlp", "dynlp", "dynlp-fh"]]
    for sys in st.batches():
        A, rhs = dense_system(sys)
        Fd = xp.linalg.solve(A, rhs)
        for s in solvers:
            s.solve(sys)
            assert float(xp.abs(s.scores(sys) - Fd).max()) < 0.05, s.name


def test_certificate_bound_is_valid_for_any_iterate():
    ds = graphs.sbm(1500, K=2, deg=6, seed=5, dtype="float64")
    sys = next(Stream(ds, init_frac=0.5, n_batches=0, label_frac=0.03, dtype="float64").batches())
    A, rhs = dense_system(sys)
    Fd, h = xp.linalg.solve(A, rhs), xp.linalg.solve(A, sys.diag)
    rs = xp.random.RandomState(0)
    for scale in (1e-1, 1e-3):
        F = Fd + scale * rs.standard_normal(Fd.shape)
        rho = float((xp.abs(rhs - A @ F) / sys.diag[:, None]).max())
        assert bool((xp.abs(F - Fd) <= rho * h[:, None] + 1e-12).all())


def test_multiclass_columns_consistent_with_binary():
    """Solving both class columns jointly (K-class path) matches the 1-column binary path."""
    ds = graphs.sbm(2000, K=2, deg=6, seed=6, dtype="float64")
    sys = next(Stream(ds, init_frac=0.4, n_batches=0, label_frac=0.03, dtype="float64").batches())
    M = lambda R: R / sys.diag[:, None]  # noqa: E731
    Z2 = pcg(sys, xp.zeros((sys.n_u, 2)), sys.B.copy(), [1e-12, 1e-12], M, 10_000, Stats())
    Z1 = pcg(sys, xp.zeros((sys.n_u, 1)), sys.rhs.copy(), [1e-12], M, 10_000, Stats())
    close(Z2[:, 1], Z1[:, 0], atol=1e-8)
    close(Z2.sum(1), 1.0, atol=1e-8)


def test_amg_is_symmetric_positive_definite():
    ds = graphs.synth(1200, K=2, k=6, subs=2, seed=7, dtype="float64")
    sys = next(Stream(ds, init_frac=1.0, n_batches=0, label_frac=0.02, dtype="float64").batches())
    amg = AMG(sys.W, sys.s, max_coarse=50)
    assert len(amg.levels) >= 2
    Mi = amg(xp.eye(sys.n_u))
    close(Mi, Mi.T, atol=1e-10)
    assert float(xp.linalg.eigvalsh((Mi + Mi.T) / 2).min()) > 0


def _is_symmetric(A):
    return float(abs(A - A.T).max()) == 0


def test_synth_graph():
    ds = graphs.synth(3000, K=10, seed=1)
    assert ds.n == 3000 and ds.K == 10 and _is_symmetric(ds.A)
    assert float(ds.A.data.min()) > 0 and float(ds.A.data.max()) <= 1
    assert int(xp.diff(ds.A.indptr).min()) >= 5  # every vertex keeps its k = 5 neighbours


def test_imdb_pipeline_on_tiny_parquets(tmp_path):
    """Reviews -> TF-IDF -> cosine 5-NN, on fake train/test parquets (no download)."""
    import pandas as pd
    rnd = random.Random(0)
    pos, neg, common = (["great", "superb", "loved", "wonderful", "brilliant", "moving"],
                        ["awful", "boring", "hated", "terrible", "dull", "waste"], ["movie", "film", "the", "plot"])
    for split in ("train", "test"):
        text = [" ".join(rnd.choices(pos if i % 2 else neg, k=6) + common + ["<br />"]) for i in range(20)]
        pd.DataFrame({"text": text, "label": [i % 2 for i in range(20)]}).to_parquet(tmp_path / f"imdb-{split}.parquet")
    ds = graphs.load("imdb", data_dir=str(tmp_path))
    assert ds.n == 40 and ds.K == 2 and _is_symmetric(ds.A) and int(ds.y.sum()) == 20
    coo = ds.A.tocoo()
    assert float((ds.y[coo.row] == ds.y[coo.col]).mean()) > 0.9


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("C,gs,mapping", [(C, gs, "rows") for C in (1, 3) for gs in (1, 2, 8, 32, 128)]
                         + [(C, 8, "cols") for C in (1, 3, 8, 40)])
def test_cuda_kernels_match_fallback(dtype, C, gs, mapping):
    ds = graphs.sbm(3000, K=2, deg=12, seed=8, dtype=dtype)
    W, n, rs = ds.A, ds.n, xp.random.RandomState(0)
    frontier = xp.sort(rs.choice(n, 700, replace=False)).astype(xp.int32)
    X, rhs = rs.random_sample((n, C)).astype(dtype), rs.random_sample((n, C)).astype(dtype)
    diag = (W @ xp.ones(n, dtype=dtype) + 1).astype(dtype)
    kops, fops = FrontierOps(W, C, gs, mapping=mapping), FrontierOps(W, C, gs, mapping=mapping)
    fops.use_kernel = False
    assert kops.use_kernel
    Yk, ck = kops.jacobi(frontier, X, rhs, diag, 1e-2)
    Yf, cf = fops.jacobi(frontier, X, rhs, diag, 1e-2)
    rtol = 1e-5 if dtype == "float32" else 1e-12
    close(Yk, Yf, rtol=rtol)
    D = rs.standard_normal((frontier.size, C)).astype(dtype)
    Rk, Rf = xp.zeros((n, C), dtype=dtype), xp.zeros((n, C), dtype=dtype)
    mk, mf = xp.zeros(n, dtype=xp.uint8), xp.zeros(n, dtype=xp.uint8)
    kops.push(frontier, D, Rk, mk)
    fops.push(frontier, D, Rf, mf)
    close(Rk, Rf, rtol=rtol, atol=1e-5 if dtype == "float32" else 1e-12)
    xp.testing.assert_array_equal(mk, mf)
    mk[:], mf[:] = 0, 0
    kops.mark_neighbors(frontier, mk)
    fops.mark_neighbors(frontier, mf)
    xp.testing.assert_array_equal(mk, mf)
    assert int(kops.edges) == int(fops.edges)


@pytest.mark.parametrize("method", ["pcg", "amg", "auto"])
def test_refinement_certifies_beyond_float32(method):
    """A path graph labeled only at its ends has h ~ 1e4-1e5: float32 alone cannot
    certify 1e-3 there, so the float64 refinement rounds must do it."""
    n = 600
    u = xp.arange(n - 1, dtype=xp.int64)
    A = graphs.symmetric_from_pairs(u, u + 1, xp.ones(n - 1), n, "float32")
    ds = graphs.Dataset("path", A, (xp.arange(n) >= n // 2).astype(xp.int32), 2, xp.arange(n))
    st = Stream(ds, init_frac=1.0, n_batches=0, label_frac=0.0, dtype="float32")
    st.labeled[:] = False
    st.labeled[xp.asarray([0, n - 1])] = True
    sys = next(st.batches())
    A_, rhs = dense_system(sys)
    Fd, h = xp.linalg.solve(A_, rhs), xp.linalg.solve(A_, sys.diag.astype(xp.float64))
    assert float(h.max()) > 1e4
    solver = DynLPPlus(n, 2, "float32", method=method, tol=1e-3)
    stats = solver.solve(sys)
    assert stats.cert <= 1e-3
    assert float(xp.abs(solver.scores(sys) - Fd).max()) <= stats.cert * 1.05 + 1e-9


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("C", [2, 8, 40])
def test_rowmajor_spmm_matches_cusparse(dtype, C):
    W = graphs.sbm(3000, K=2, deg=12, seed=8, dtype=dtype).A
    rs = xp.random.RandomState(1)
    X, a = rs.random_sample((3000, C)).astype(dtype), rs.random_sample(3000).astype(dtype)
    tol = 1e-5 if dtype == "float32" else 1e-12
    close(spmm_axpy(W, X), W @ X, rtol=tol)
    close(spmm_axpy(W, X, a=a, b=-1.0), a[:, None] * X - W @ X, rtol=tol, atol=tol)
