"""Tests for the resident-state path: IncrementalStream, IncrementalLP, FusedPush."""
import numpy as np
import pytest
import scipy.sparse as ssp

from dynlp import backend, graphs
from dynlp.solvers import IncrementalLP, apply_A
from dynlp.stream import IncrementalStream, Stream


def _gpu_ok():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


BACKENDS = ["numpy"] + (["cupy"] if _gpu_ok() else [])


@pytest.fixture(params=BACKENDS)
def be(request):
    return backend.set_backend(request.param)


def host(x):
    b = backend.get()
    if hasattr(x, "indptr"):
        return ssp.csr_matrix((b.asnumpy(x.data), b.asnumpy(x.indices), b.asnumpy(x.indptr)), shape=x.shape)
    return b.asnumpy(x)


def partition_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and len(set(zip(a.tolist(), b.tolist()))) == len(set(a.tolist())) == len(set(b.tolist()))


@pytest.mark.parametrize("hops", [1, 3])
def test_incremental_stream_matches_rebuilt_system(be, hops):
    """Sparse graph + heavy deletions: exercises splits, regrounding and the full-CC fallback."""
    kw = dict(init_frac=0.3, n_batches=6, del_frac=0.4, label_frac=0.03, dtype="float64", seed=3)
    ds = graphs.sbm(4000, K=3, deg=3.0, seed=5, dtype="float64")
    ref = Stream(ds, **kw)
    inc = IncrementalStream(ds, ball_hops=hops, **kw)
    for c, g in zip(ref.batches(), inc.batches()):
        U = host(c.U)
        solved = host(g.solved)
        np.testing.assert_array_equal(np.flatnonzero(solved), U)
        assert g.n_ungrounded == c.n_ungrounded
        Wg = host(g.W)
        np.testing.assert_allclose(Wg[U][:, U].toarray(), host(c.W).toarray(), rtol=1e-12, atol=1e-12)
        for f in ("s", "diag", "rhs", "B"):
            np.testing.assert_allclose(host(getattr(g, f))[U], host(getattr(c, f)), rtol=1e-10, atol=1e-12, err_msg=f)
        np.testing.assert_array_equal(host(g.is_new)[U], host(c.is_new))
        np.testing.assert_array_equal(host(g.seed)[U], host(c.seed))
        assert g.n_comp == c.n_comp and partition_equal(host(g.comp), host(c.comp))
        # inert rows: no edges, diag 1, rhs 0
        out = ~solved
        assert abs(Wg[out]).sum() == 0 and abs(Wg[:, out]).sum() == 0
        np.testing.assert_array_equal(host(g.diag)[out], 1.0)
        np.testing.assert_array_equal(host(g.rhs)[out], 0.0)
        # every row whose equation changed is listed in `changed`
        if c.t > 0:
            ch = np.zeros(ds.n, bool)
            ch[host(g.changed)] = True
            assert ch[U[host(c.is_new)]].all()


@pytest.mark.parametrize("K", [2, 3])
def test_incremental_solver_is_certified_on_small_batches(be, K):
    tol = 1e-3
    ds = graphs.sbm(3000, K=K, deg=6.0, seed=6, dtype="float32")
    st = IncrementalStream(ds, init_frac=0.85, n_batches=8, del_frac=0.1, label_frac=0.02, dtype="float32")
    solver = IncrementalLP(ds.n, K, "float32", tol=tol, full_every=4)
    paths = []
    for sys in st.batches():
        stats = solver.solve(sys)
        paths.append(stats.path)
        s64 = sys.astype(np.float64)
        U = np.flatnonzero(host(sys.solved))
        A = np.diag(host(s64.diag)[U]) - host(s64.W)[U][:, U].toarray()
        Fd = np.linalg.solve(A, host(s64.rhs)[U])
        err = np.abs(host(solver.scores(sys))[U] - Fd).max()
        assert stats.cert <= tol * 1.0001, (sys.t, stats)
        assert err <= stats.cert * 1.05 + 1e-9, (sys.t, err, stats.cert)
        # the kept residual matches a fresh one
        xp = backend.get().xp
        rhs = xp.concatenate([s64.rhs, s64.diag[:, None]], axis=1)
        R_true = rhs - apply_A(s64, solver.Z)
        drift = host(xp.abs(R_true - solver.R) / s64.diag[:, None]).max()
        assert drift < 1e-9, drift
    assert any(p == "inc-push" for p in paths), paths


@pytest.mark.skipif(not _gpu_ok(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("C", [2, 4])
def test_fused_push_reaches_tolerance(dtype, C):
    be = backend.set_backend("cupy")
    xp = be.xp
    from dynlp.kernels import FusedPush
    ds = graphs.sbm(4000, K=2, deg=8.0, seed=9, dtype=dtype)
    st = IncrementalStream(ds, init_frac=1.0, n_batches=0, label_frac=0.03, dtype=dtype)
    sys = next(st.batches())
    s = sys.astype(dtype)
    rs = xp.random.RandomState(0)
    rhs = xp.ascontiguousarray(rs.random_sample((ds.n, C)).astype(dtype) * s.solved[:, None])
    Z = xp.zeros((ds.n, C), dtype=dtype)
    R = xp.ascontiguousarray(rhs - apply_A(s, Z))
    eps = xp.full(C, 1e-5 if dtype == "float64" else 1e-4, dtype=dtype)
    fp = FusedPush(s.W, C)
    rounds, ok, edges = fp.run(Z, R, s.diag, eps, xp.flatnonzero((xp.abs(R) > 0).any(axis=1)).astype(xp.int32),
                               max_rounds=100_000)
    assert ok and rounds > 0 and edges > 0
    R_true = rhs - apply_A(s, Z)
    # the kept residual is the true residual (up to rounding) and satisfies eps everywhere
    tol = 1e-4 if dtype == "float32" else 1e-10
    xp.testing.assert_allclose(R, R_true, atol=tol)
    assert float((xp.abs(R_true) / s.diag[:, None]).max() / eps[0]) <= 1.0 + (0.05 if dtype == "float32" else 1e-6)
