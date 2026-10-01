"""Correctness tests. Run: python -m pytest -q  (GPU tests run when CuPy sees a GPU)."""
import numpy as np
import pytest
import scipy.sparse as ssp
import scipy.sparse.linalg as sla

from dynlp import backend, graphs
from dynlp.amg import AMG
from dynlp.cc import connected_components
from dynlp.kernels import FrontierOps
from dynlp.solvers import DynLPPlus, Reference, Stats, make, pcg, rho_cols
from dynlp.stream import Stream


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
    if ssp.issparse(x) or hasattr(x, "tocsr") and not isinstance(x, np.ndarray):
        return ssp.csr_matrix((b.asnumpy(x.data), b.asnumpy(x.indices), b.asnumpy(x.indptr)), shape=x.shape)
    return b.asnumpy(x)


def dense_system(sys):
    W = host(sys.W).toarray().astype(np.float64)
    A = np.diag(host(sys.diag).astype(np.float64)) - W
    return A, host(sys.rhs).astype(np.float64)


def test_connected_components(be):
    rs = np.random.RandomState(0)
    n, m = 400, 300
    r, c = rs.randint(0, n, m), rs.randint(0, n, m)
    lab, nc = connected_components(n, be.xp.asarray(r), be.xp.asarray(c))
    ref_nc, ref = ssp.csgraph.connected_components(ssp.coo_matrix((np.ones(m), (r, c)), (n, n)), directed=False)
    lab = host(lab)
    assert nc == ref_nc
    # same partition up to relabeling
    pairs = set(zip(lab.tolist(), ref.tolist()))
    assert len(pairs) == nc


def test_system_matches_dense_definition(be):
    ds = graphs.sbm(1500, K=3, deg=6, seed=1, dtype="float64")
    st = Stream(ds, init_frac=0.2, n_batches=3, label_frac=0.03, dtype="float64")
    A = host(ds.A).toarray()
    y = host(ds.y)
    lab = st.labeled_host
    active = np.zeros(ds.n, bool)
    for sys in st.batches():
        U = host(sys.U)
        assert not lab[U].any()
        # W is the U-U block of A
        W = host(sys.W).toarray()
        np.testing.assert_allclose(W, A[np.ix_(U, U)], rtol=1e-12)
        # B = class-wise weight to active labeled neighbours
        active[U] = True
        B = host(sys.B)
        Lnb = A[U] * (lab & (A[U].sum(0) >= 0))[None, :]  # all labeled columns
        act_lab = np.zeros(ds.n, bool)
        act_lab[st.order[: st.n0]] = True
        for ch in st.chunks[: sys.t]:
            act_lab[ch] = True
        act_lab &= lab
        Bd = np.stack([(Lnb * (act_lab & (y == k))[None, :]).sum(1) for k in range(3)], axis=1)
        np.testing.assert_allclose(B, Bd, rtol=1e-10)
        np.testing.assert_allclose(host(sys.diag), host(sys.s) + W.sum(1), rtol=1e-12)
        # every solved component touches a labeled vertex
        ncomp, comp = ssp.csgraph.connected_components(ssp.csr_matrix(W), directed=False)
        g = np.zeros(ncomp)
        np.add.at(g, comp, B.sum(1))
        assert (g > 0).all()


@pytest.mark.parametrize("K", [2, 4])
def test_reference_matches_direct_solve(be, K):
    ds = graphs.sbm(2000, K=K, deg=6, seed=2, dtype="float64")
    st = Stream(ds, init_frac=0.3, n_batches=2, label_frac=0.02, dtype="float64")
    ref = Reference(ds.n, K)
    for sys in st.batches():
        F, hmax, _ = ref.solve(sys)
        A, rhs = dense_system(sys)
        Fd = np.linalg.solve(A, rhs)
        assert np.abs(host(F) - Fd).max() < 1e-7
        h = np.linalg.solve(A, host(sys.diag))
        assert hmax >= h.max() * (1 - 1e-6)


@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("method", ["push", "pcg", "amg", "auto"])
def test_dynlp_plus_certificate_holds(be, K, method):
    tol = 1e-3
    ds = graphs.sbm(3000, K=K, deg=8, seed=3, dtype="float32")
    st = Stream(ds, init_frac=0.2, n_batches=3, label_frac=0.02, dtype="float32")
    solver = DynLPPlus(ds.n, K, "float32", method=method, tol=tol)
    for sys in st.batches():
        stats = solver.solve(sys)
        A, rhs = dense_system(sys)
        Fd = np.linalg.solve(A, rhs)
        err = np.abs(host(solver.scores(sys)).astype(np.float64) - Fd).max()
        assert stats.cert <= tol * 1.0001
        assert err <= stats.cert * 1.05 + 1e-6, (err, stats.cert)


def test_baselines_run_and_are_close(be):
    ds = graphs.sbm(2000, K=2, deg=8, seed=4, dtype="float32")
    st = Stream(ds, init_frac=0.3, n_batches=2, label_frac=0.02, dtype="float32")
    solvers = [make(nm, ds.n, 2, dtype="float32", delta=1e-5, tol=1e-3, group="auto")
               for nm in ["itlp", "itlp-warm", "dynlp", "dynlp-knowninit"]]
    for sys in st.batches():
        A, rhs = dense_system(sys)
        Fd = np.linalg.solve(A, rhs)
        for s in solvers:
            s.solve(sys)
            assert np.abs(host(s.scores(sys)) - Fd).max() < 0.05, s.name


def test_certificate_bound_is_valid_for_any_iterate(be):
    ds = graphs.sbm(1500, K=2, deg=6, seed=5, dtype="float64")
    st = Stream(ds, init_frac=0.5, n_batches=0, label_frac=0.03, dtype="float64")
    sys = next(st.batches())
    A, rhs = dense_system(sys)
    Fd = np.linalg.solve(A, rhs)
    h = np.linalg.solve(A, host(sys.diag))
    rs = np.random.RandomState(0)
    for scale in (1e-1, 1e-3):
        F = Fd + scale * rs.standard_normal(Fd.shape)
        R = rhs - A @ F
        rho = (np.abs(R) / host(sys.diag)[:, None]).max()
        assert (np.abs(F - Fd) <= rho * h[:, None] + 1e-12).all()


def test_multiclass_columns_consistent_with_binary(be):
    """Solving both class columns jointly (K-class path) matches the 1-column binary path."""
    ds = graphs.sbm(2000, K=2, deg=6, seed=6, dtype="float64")
    st = Stream(ds, init_frac=0.4, n_batches=0, label_frac=0.03, dtype="float64")
    sys = next(st.batches())
    xp = be.xp
    M = lambda R: R / sys.diag[:, None]  # noqa: E731
    Z2 = pcg(sys, xp.zeros((sys.n_u, 2)), sys.B.copy(), [1e-12, 1e-12], M, 10_000, Stats())
    Z1 = pcg(sys, xp.zeros((sys.n_u, 1)), sys.rhs.copy(), [1e-12], M, 10_000, Stats())
    np.testing.assert_allclose(host(Z2[:, 1]), host(Z1[:, 0]), atol=1e-8)
    np.testing.assert_allclose(host(Z2.sum(1)), 1.0, atol=1e-8)


def test_amg_is_symmetric_positive_definite():
    be = backend.set_backend("numpy")
    ds = graphs.gmm_knn(1200, K=2, k=6, seed=7, dtype="float64")
    st = Stream(ds, init_frac=1.0, n_batches=0, label_frac=0.02, dtype="float64")
    sys = next(st.batches())
    amg = AMG(sys.W, sys.s, max_coarse=50)
    assert len(amg.levels) >= 2
    Mi = amg(be.xp.eye(sys.n_u))
    np.testing.assert_allclose(Mi, Mi.T, atol=1e-10)
    assert np.linalg.eigvalsh((Mi + Mi.T) / 2).min() > 0


@pytest.mark.skipif(not _gpu_ok(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("C,gs,mapping", [(C, gs, "rows") for C in (1, 3) for gs in (1, 2, 8, 32, 128)]
                         + [(C, 8, "cols") for C in (1, 3, 8, 40)])
def test_cuda_kernels_match_fallback(dtype, C, gs, mapping):
    be = backend.set_backend("cupy")
    xp = be.xp
    ds = graphs.sbm(3000, K=2, deg=12, seed=8, dtype=dtype)
    W = ds.A
    n = W.shape[0]
    rs = xp.random.RandomState(0)
    frontier = xp.sort(rs.choice(n, 700, replace=False)).astype(xp.int32)
    X = rs.random_sample((n, C)).astype(dtype)
    rhs = rs.random_sample((n, C)).astype(dtype)
    diag = (W @ xp.ones(n, dtype=dtype) + 1).astype(dtype)
    kops, fops = FrontierOps(W, C, gs, mapping=mapping), FrontierOps(W, C, gs, mapping=mapping)
    fops.use_kernel = False
    assert kops.use_kernel
    Yk, ck = kops.jacobi(frontier, X, rhs, diag, 1e-2)
    Yf, cf = fops.jacobi(frontier, X, rhs, diag, 1e-2)
    rtol = 1e-5 if dtype == "float32" else 1e-12
    xp.testing.assert_allclose(Yk, Yf, rtol=rtol)
    D = rs.standard_normal((frontier.size, C)).astype(dtype)
    Rk, Rf = xp.zeros((n, C), dtype=dtype), xp.zeros((n, C), dtype=dtype)
    mk, mf = xp.zeros(n, dtype=xp.uint8), xp.zeros(n, dtype=xp.uint8)
    kops.push(frontier, D, Rk, mk)
    fops.push(frontier, D, Rf, mf)
    xp.testing.assert_allclose(Rk, Rf, rtol=rtol, atol=1e-5 if dtype == "float32" else 1e-12)
    xp.testing.assert_array_equal(mk, mf)
    mk[:] = 0
    mf[:] = 0
    kops.mark_neighbors(frontier, mk)
    fops.mark_neighbors(frontier, mf)
    xp.testing.assert_array_equal(mk, mf)
    assert int(kops.edges) == int(fops.edges)


@pytest.mark.parametrize("method", ["pcg", "amg", "auto", "push"])
def test_refinement_certifies_beyond_float32(be, method):
    """A path graph labeled only at its ends has h ~ 1e4-1e5: float32 alone cannot
    certify 1e-3 there, so the float64 refinement rounds must do it."""
    xp = be.xp
    n = 600
    u = xp.arange(n - 1, dtype=xp.int64)
    A = graphs.symmetric_from_pairs(u, u + 1, xp.ones(n - 1), n, "float32")
    y = xp.asarray((np.arange(n) >= n // 2).astype(np.int32))
    ds = graphs.Dataset("path", A, y, 2, np.arange(n))
    st = Stream(ds, init_frac=1.0, n_batches=0, label_frac=0.0, dtype="float32")
    st.labeled[:] = False
    st.labeled[xp.asarray([0, n - 1])] = True
    st.labeled_host[:] = False
    st.labeled_host[[0, n - 1]] = True
    sys = next(st.batches())
    A_, rhs = dense_system(sys)
    Fd = np.linalg.solve(A_, rhs)
    h = np.linalg.solve(A_, host(sys.diag))
    assert h.max() > 1e4
    solver = DynLPPlus(n, 2, "float32", method=method, tol=1e-3)
    stats = solver.solve(sys)
    err = np.abs(host(solver.scores(sys)) - Fd).max()
    assert stats.cert <= 1e-3
    assert err <= stats.cert * 1.05 + 1e-9


@pytest.mark.skipif(not _gpu_ok(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("C", [2, 8, 40])
def test_rowmajor_spmm_matches_cusparse(dtype, C):
    from dynlp.kernels import spmm_axpy
    xp = backend.set_backend("cupy").xp
    W = graphs.sbm(3000, K=2, deg=12, seed=8, dtype=dtype).A
    rs = xp.random.RandomState(1)
    X, a = rs.random_sample((3000, C)).astype(dtype), rs.random_sample(3000).astype(dtype)
    tol = 1e-5 if dtype == "float32" else 1e-12
    xp.testing.assert_allclose(spmm_axpy(W, X), W @ X, rtol=tol)
    xp.testing.assert_allclose(spmm_axpy(W, X, a=a, b=-1.0), a[:, None] * X - W @ X, rtol=tol, atol=tol)
