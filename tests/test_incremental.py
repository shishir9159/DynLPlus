"""Tests for the resident-state path: IncrementalStream, IncrementalLP, FusedPush (GPU)."""
import pytest

from dynlp import graphs
from dynlp.backend import xp
from dynlp.kernels import FusedPush
from dynlp.solvers import IncrementalLP, apply_A
from dynlp.stream import IncrementalStream, Stream

close, equal = xp.testing.assert_allclose, xp.testing.assert_array_equal


def same_partition(a, b):
    a, b = a.tolist(), b.tolist()
    return len(a) == len(b) and len(set(zip(a, b))) == len(set(a)) == len(set(b))


@pytest.mark.parametrize("hops", [1, 3])
def test_incremental_stream_matches_rebuilt_system(hops):
    """Sparse graph + heavy deletions: exercises splits, regrounding and the full-CC fallback."""
    kw = dict(init_frac=0.3, n_batches=6, del_frac=0.4, label_frac=0.03, dtype="float64", seed=3)
    ds = graphs.sbm(4000, K=3, deg=3.0, seed=5, dtype="float64")
    for c, g in zip(Stream(ds, **kw).batches(), IncrementalStream(ds, ball_hops=hops, **kw).batches()):
        U, solved = c.U, g.solved
        equal(xp.flatnonzero(solved), U)
        assert g.n_ungrounded == c.n_ungrounded
        close(g.W[U][:, U].toarray(), c.W.toarray(), rtol=1e-12, atol=1e-12)
        for f in ("s", "diag", "rhs", "B"):
            close(getattr(g, f)[U], getattr(c, f), rtol=1e-10, atol=1e-12, err_msg=f)
        equal(g.is_new[U], c.is_new)
        equal(g.seed[U], c.seed)
        for k in ("tau", "fh"):
            assert g.comps[k][1] == c.comps[k][1] and same_partition(g.comps[k][0], c.comps[k][0]), k
        out = ~solved  # inert rows: no edges, diag 1, rhs 0
        assert float(abs(g.W[out]).sum()) == 0 and float(abs(g.W[:, out]).sum()) == 0
        equal(g.diag[out], 1.0)
        equal(g.rhs[out], 0.0)
        if c.t > 0:  # every row whose equation changed is listed in `changed`
            ch = xp.zeros(ds.n, dtype=bool)
            ch[g.changed] = True
            assert bool(ch[U[c.is_new]].all())


@pytest.mark.parametrize("K", [2, 3])
def test_incremental_solver_is_certified_on_small_batches(K):
    tol = 1e-3
    ds = graphs.sbm(3000, K=K, deg=6.0, seed=6, dtype="float32")
    st = IncrementalStream(ds, init_frac=0.85, n_batches=8, del_frac=0.1, label_frac=0.02, dtype="float32")
    solver = IncrementalLP(ds.n, K, "float32", tol=tol, full_every=4, push_frac=0.5)  # small graph: ~12% rows/batch
    paths = []
    for sys in st.batches():
        stats = solver.solve(sys)
        paths.append(stats.path)
        s64 = sys.astype(xp.float64)
        U = xp.flatnonzero(sys.solved)
        A = xp.diag(s64.diag[U]) - s64.W[U][:, U].toarray()
        err = float(xp.abs(solver.scores(sys)[U] - xp.linalg.solve(A, s64.rhs[U])).max())
        assert stats.cert <= tol * 1.0001, (sys.t, stats)
        assert err <= stats.cert * 1.05 + 1e-9, (sys.t, err, stats.cert)
        R_true = xp.concatenate([s64.rhs, s64.diag[:, None]], axis=1) - apply_A(s64, solver.Z)
        drift = float((xp.abs(R_true - solver.R) / s64.diag[:, None]).max())  # kept residual == fresh one
        assert drift < 1e-9, drift
    assert "inc-push" in paths, paths


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("C", [2, 4])
def test_fused_push_reaches_tolerance(dtype, C):
    ds = graphs.sbm(4000, K=2, deg=8.0, seed=9, dtype=dtype)
    s = next(IncrementalStream(ds, init_frac=1.0, n_batches=0, label_frac=0.03, dtype=dtype).batches()).astype(dtype)
    rs = xp.random.RandomState(0)
    rhs = xp.ascontiguousarray(rs.random_sample((ds.n, C)).astype(dtype) * s.solved[:, None])
    Z = xp.zeros((ds.n, C), dtype=dtype)
    R = xp.ascontiguousarray(rhs - apply_A(s, Z))
    eps = xp.full(C, 1e-5 if dtype == "float64" else 1e-4, dtype=dtype)
    rounds, ok, edges = FusedPush(s.W, C).run(Z, R, s.diag, eps,
                                              xp.flatnonzero((xp.abs(R) > 0).any(axis=1)).astype(xp.int32), 100_000)
    assert ok and rounds > 0 and edges > 0
    R_true = rhs - apply_A(s, Z)  # the kept residual is the true one (up to rounding) and meets eps everywhere
    close(R, R_true, atol=1e-4 if dtype == "float32" else 1e-10)
    assert float((xp.abs(R_true) / s.diag[:, None]).max() / eps[0]) <= 1.0 + (0.05 if dtype == "float32" else 1e-6)
