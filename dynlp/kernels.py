"""Frontier kernels: Jacobi on a vertex subset, residual push, neighbour marking.

GPU: hand-written CUDA (NVRTC via cupy.RawModule). ``GS`` lanes cooperate on
one frontier row (sub-warp CSR-vector). ``GS > 32`` gives one thread block per
row, which is DynLP's published mapping, kept for comparison.
CPU (and GPU with many columns): vectorized SciPy/CuPy fallbacks.
"""
from __future__ import annotations

import numpy as np

from . import backend

_SRC = r"""
#define GS {GS}
#define NC {NC}
typedef {REAL} real;

__device__ __forceinline__ real group_sum(real v) {{
#if GS > 1 && GS <= 32
  #pragma unroll
  for (int off = GS / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off, GS);
#endif
  return v;
}}

__device__ __forceinline__ real warp_sum(real v) {{
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}}

// Y[g] = (rhs[row] + sum_j w_j X[col_j]) / diag[row] for row = frontier[g]
extern "C" __global__ void frontier_jacobi(
    const int* __restrict__ indptr, const int* __restrict__ indices, const real* __restrict__ w,
    const int* __restrict__ frontier, const int nf,
    const real* __restrict__ X, const real* __restrict__ rhs, const real* __restrict__ diag,
    const real delta, real* __restrict__ Y, unsigned char* __restrict__ changed)
{{
#if GS <= 32
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long g = tid / GS;
  const int lane = (int)(tid % GS);
  const bool valid = g < nf;
  const int row = valid ? frontier[g] : 0;
  const int beg = valid ? indptr[row] : 0;
  const int end = valid ? indptr[row + 1] : 0;
  real acc[NC];
  #pragma unroll
  for (int c = 0; c < NC; ++c) acc[c] = 0;
  for (int j = beg + lane; j < end; j += GS) {{
    const long long v = indices[j];
    const real wv = w[j];
    #pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] += wv * X[v * NC + c];
  }}
  #pragma unroll
  for (int c = 0; c < NC; ++c) acc[c] = group_sum(acc[c]);
  if (valid && lane == 0) {{
    const real inv = (real)1 / diag[row];
    bool ch = false;
    #pragma unroll
    for (int c = 0; c < NC; ++c) {{
      const real yv = (rhs[(long long)row * NC + c] + acc[c]) * inv;
      Y[g * NC + c] = yv;
      ch |= fabs(yv - X[(long long)row * NC + c]) > delta;
    }}
    changed[g] = ch;
  }}
#else
  // one thread block per frontier row (DynLP's mapping); blockDim.x == GS
  __shared__ real sm[NC][GS / 32];
  const int g = blockIdx.x;
  const int row = frontier[g];
  const int beg = indptr[row], end = indptr[row + 1];
  real acc[NC];
  #pragma unroll
  for (int c = 0; c < NC; ++c) acc[c] = 0;
  for (int j = beg + threadIdx.x; j < end; j += GS) {{
    const long long v = indices[j];
    const real wv = w[j];
    #pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] += wv * X[v * NC + c];
  }}
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  #pragma unroll
  for (int c = 0; c < NC; ++c) {{
    real s = warp_sum(acc[c]);
    if (lane == 0) sm[c][wid] = s;
  }}
  __syncthreads();
  if (threadIdx.x == 0) {{
    const real inv = (real)1 / diag[row];
    bool ch = false;
    for (int c = 0; c < NC; ++c) {{
      real s = 0;
      for (int k = 0; k < GS / 32; ++k) s += sm[c][k];
      const real yv = (rhs[(long long)row * NC + c] + s) * inv;
      Y[(long long)g * NC + c] = yv;
      ch |= fabs(yv - X[(long long)row * NC + c]) > delta;
    }}
    changed[g] = ch;
  }}
#endif
}}

// R[col_j] += w_j * D[g] for every edge of row = frontier[g]; mark[col_j] = 1
extern "C" __global__ void push_scatter(
    const int* __restrict__ indptr, const int* __restrict__ indices, const real* __restrict__ w,
    const int* __restrict__ frontier, const int nf, const real* __restrict__ D,
    real* __restrict__ R, unsigned char* __restrict__ mark)
{{
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long g = tid / GS;
  const int lane = (int)(tid % GS);
  if (g >= nf) return;
  const int row = frontier[g];
  real d[NC];
  #pragma unroll
  for (int c = 0; c < NC; ++c) d[c] = D[g * NC + c];
  for (int j = indptr[row] + lane; j < indptr[row + 1]; j += GS) {{
    const long long v = indices[j];
    const real wv = w[j];
    #pragma unroll
    for (int c = 0; c < NC; ++c) atomicAdd(&R[v * NC + c], wv * d[c]);
    mark[v] = 1;
  }}
}}

extern "C" __global__ void mark_neighbors(
    const int* __restrict__ indptr, const int* __restrict__ indices,
    const int* __restrict__ rows, const int nr, unsigned char* __restrict__ mark)
{{
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long g = tid / GS;
  const int lane = (int)(tid % GS);
  if (g >= nr) return;
  const int row = rows[g];
  for (int j = indptr[row] + lane; j < indptr[row + 1]; j += GS) mark[indices[j]] = 1;
}}
"""

# Fused, asynchronous push: two kernels per round, no host sync per round.
# Round r reads frontier slot p = r & 1 and writes the next frontier into slot 1 - p;
# each kernel resets the counter the *other* kernel appends to next, so no extra
# launches are needed. The residual is read-and-zeroed atomically (atomicExch), so
# pushes from neighbours that land during the round are kept for the next one.
_FUSED_SRC = r"""
#define GS {GS}
#define NC {NC}
typedef {REAL} real;

__device__ __forceinline__ real take_residual(real* p) {{
#if {IS_DOUBLE}
  return __longlong_as_double((long long)atomicExch((unsigned long long*)p, 0ull));
#else
  return atomicExch(p, 0.0f);
#endif
}}

extern "C" __global__ void push_round(
    const int* __restrict__ indptr, const int* __restrict__ indices, const real* __restrict__ w,
    const real* __restrict__ diag, const int* __restrict__ frontier, const int* __restrict__ fcount,
    int* __restrict__ fcount_next, real* __restrict__ Z, real* __restrict__ R,
    int* __restrict__ mark, int* __restrict__ touched, int* __restrict__ tcount,
    unsigned long long* __restrict__ edges)
{{
  const int nf = *fcount;
  if (blockIdx.x == 0 && threadIdx.x == 0) *fcount_next = 0;
  const int lane_w = threadIdx.x & 31;
  const int lane = lane_w % GS;
  const long long gpw = 32 / GS;
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const long long stride = (((long long)gridDim.x * blockDim.x) >> 5) * gpw;
  for (long long g0 = warp * gpw; g0 < nf; g0 += stride) {{     // warp-uniform loop
    const long long g = g0 + lane_w / GS;
    const bool valid = g < nf;
    const int u = valid ? frontier[g] : 0;
    real d[NC];
    #pragma unroll
    for (int c = 0; c < NC; ++c) d[c] = 0;
    if (valid && lane == 0) {{
      const real inv = (real)1 / diag[u];
      #pragma unroll
      for (int c = 0; c < NC; ++c) {{
        d[c] = take_residual(&R[(long long)u * NC + c]) * inv;
        Z[(long long)u * NC + c] += d[c];
      }}
    }}
    #pragma unroll
    for (int c = 0; c < NC; ++c) d[c] = __shfl_sync(0xffffffffu, d[c], 0, GS);
    if (valid) {{
      const int beg = indptr[u], end = indptr[u + 1];
      if (lane == 0) atomicAdd(edges, (unsigned long long)(end - beg));
      for (int j = beg + lane; j < end; j += GS) {{
        const real wv = w[j];
        if (wv == (real)0) continue;                 // inert entries (outside U)
        const long long v = indices[j];
        #pragma unroll
        for (int c = 0; c < NC; ++c) atomicAdd(&R[v * NC + c], wv * d[c]);
        if (atomicExch(&mark[v], 1) == 0) touched[atomicAdd(tcount, 1)] = (int)v;
      }}
    }}
  }}
}}

extern "C" __global__ void compact_frontier(
    const int* __restrict__ touched, const int* __restrict__ tcount, int* __restrict__ tcount_next,
    int* __restrict__ mark, const real* __restrict__ R, const real* __restrict__ diag,
    const real* __restrict__ eps, int* __restrict__ out, int* __restrict__ fcount_out)
{{
  const int nt = *tcount;
  if (blockIdx.x == 0 && threadIdx.x == 0) *tcount_next = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nt;
       i += (long long)gridDim.x * blockDim.x) {{
    const int v = touched[i];
    mark[v] = 0;
    const real dv = diag[v];
    bool viol = false;
    #pragma unroll
    for (int c = 0; c < NC; ++c) viol |= fabs(R[(long long)v * NC + c]) > eps[c] * dv;
    if (viol) out[atomicAdd(fcount_out, 1)] = v;
  }}
}}
"""

_MODULES: dict = {}
MAX_KERNEL_COLS = 32


def _module(gs, nc, real):
    key = (gs, nc, real)
    if key not in _MODULES:
        import cupy
        src = _SRC.format(GS=gs, NC=nc, REAL=real)
        _MODULES[key] = cupy.RawModule(code=src, options=("--std=c++14",))
    return _MODULES[key]


class FusedPush:
    """Asynchronous residual push with resident buffers (GPU only, C <= 32).

    Solves A z = rhs in place given z and its residual r: repeatedly takes the
    residual of every violating vertex (|r_u| > eps_c * diag_u), adds r_u / diag_u
    to z_u, and scatters it to the neighbours. Converges for this M-matrix even
    with the asynchronous updates (chaotic relaxation).
    """

    def __init__(self, W, C, group="auto", threads=256, blocks_per_sm=8):
        import cupy
        n = W.shape[0]
        self.W, self.C, self.n = W, C, n
        avg = W.nnz / max(n, 1)
        self.gs = min(auto_group(avg) if group == "auto" else int(group), 32)
        double = W.data.dtype == np.float64
        key = ("fused", self.gs, C, double)
        if key not in _MODULES:
            src = _FUSED_SRC.format(GS=self.gs, NC=C, REAL="double" if double else "float",
                                    IS_DOUBLE=1 if double else 0)
            _MODULES[key] = cupy.RawModule(code=src, options=("--std=c++14",))
        mod = _MODULES[key]
        self.k_push = mod.get_function("push_round")
        self.k_compact = mod.get_function("compact_frontier")
        self.front = cupy.empty((2, n), dtype=cupy.int32)
        self.touched = cupy.empty(n, dtype=cupy.int32)
        self.mark = cupy.zeros(n, dtype=cupy.int32)
        self.fcnt = cupy.zeros(2, dtype=cupy.int32)
        self.tcnt = cupy.zeros(2, dtype=cupy.int32)
        self.edges = cupy.zeros(1, dtype=cupy.uint64)
        sms = cupy.cuda.Device().attributes["MultiProcessorCount"]
        self.grid, self.threads = sms * blocks_per_sm, threads

    def run(self, Z, R, diag, eps, cand, max_rounds, check_every=8):
        """Push until no vertex violates; returns (rounds, converged, edge visits)."""
        import cupy
        n0 = int(cand.shape[0])
        self.edges[:] = 0
        if n0 == 0:
            return 0, True, 0
        W = self.W
        self.front[0, :n0] = cand
        self.fcnt[0] = n0
        self.fcnt[1] = 0
        self.tcnt[:] = 0
        eps = cupy.ascontiguousarray(eps, dtype=Z.dtype)
        rounds, converged = 0, False
        g, b = (self.grid,), (self.threads,)
        while rounds < max_rounds:
            p, q = rounds & 1, 1 - (rounds & 1)
            self.k_push(g, b, (W.indptr, W.indices, W.data, diag, self.front[p], self.fcnt[p:p + 1],
                               self.fcnt[q:q + 1], Z, R, self.mark, self.touched, self.tcnt[p:p + 1],
                               self.edges))
            self.k_compact(g, b, (self.touched, self.tcnt[p:p + 1], self.tcnt[q:q + 1], self.mark, R, diag,
                                  eps, self.front[q], self.fcnt[q:q + 1]))
            rounds += 1
            if rounds % check_every == 0 and int(self.fcnt[q]) == 0:
                converged = True
                break
        if not converged:
            converged = int(self.fcnt[rounds & 1]) == 0
        return rounds, converged, int(self.edges[0])


def auto_group(avg_deg: float) -> int:
    """Power-of-two lanes per row close to the mean degree, in [2, 32]."""
    g = 2
    while g < 32 and g * 2 <= max(avg_deg, 1.0):
        g *= 2
    return g


class FrontierOps:
    """Frontier primitives bound to one CSR matrix and column count."""

    def __init__(self, W, C, group="auto", threads=256):
        B = backend.get()
        self.B, self.W, self.C = B, W, C
        n = W.shape[0]
        self.avg_deg = (W.nnz / n) if n else 0.0
        self.gs = auto_group(self.avg_deg) if group == "auto" else int(group)
        self.threads = threads
        self.real = "double" if W.data.dtype == np.float64 else "float"
        self.use_kernel = B.is_gpu and C <= MAX_KERNEL_COLS
        self.edges = B.xp.zeros((), dtype=B.xp.int64)  # device-side edge-visit counter

    # ---------------------------------------------------------------- utils
    def _launch(self, name, ngroups, args):
        mod = _module(self.gs, self.C, self.real)
        k = mod.get_function(name)
        if self.gs > 32 and name == "frontier_jacobi":
            k((ngroups,), (self.gs,), args)
        else:
            tpb = self.threads
            blocks = (ngroups * self.gs + tpb - 1) // tpb
            k((blocks,), (tpb,), args)

    def count(self, rows):
        ip = self.W.indptr
        self.edges += (ip[rows + 1] - ip[rows]).sum(dtype=self.B.xp.int64)

    def _scalar(self, v):
        return np.float64(v) if self.real == "double" else np.float32(v)

    # ------------------------------------------------------------- kernels
    def jacobi(self, frontier, X, rhs, diag, delta):
        """New values for frontier rows and a changed-by-more-than-delta flag."""
        xp = self.B.xp
        nf = int(frontier.shape[0])
        self.count(frontier)
        if self.use_kernel:
            Y = xp.empty((nf, self.C), dtype=X.dtype)
            ch = xp.empty(nf, dtype=xp.uint8)
            W = self.W
            self._launch("frontier_jacobi", nf,
                         (W.indptr, W.indices, W.data, frontier.astype(xp.int32), np.int32(nf),
                          xp.ascontiguousarray(X), xp.ascontiguousarray(rhs), diag,
                          self._scalar(delta), Y, ch))
            return Y, ch.astype(bool)
        acc = self.W[frontier] @ X
        Y = (rhs[frontier] + acc) / diag[frontier][:, None]
        return Y, (xp.abs(Y - X[frontier]) > delta).any(axis=1)

    def push(self, frontier, D, R, mark):
        """R += W[:, frontier] @ D (W symmetric) and mark touched vertices."""
        xp = self.B.xp
        nf = int(frontier.shape[0])
        self.count(frontier)
        if self.use_kernel:
            W = self.W
            self._launch("push_scatter", nf,
                         (W.indptr, W.indices, W.data, frontier.astype(xp.int32), np.int32(nf),
                          xp.ascontiguousarray(D), R, mark))
            return
        sub = self.W[frontier]
        seg = xp.searchsorted(sub.indptr[1:], xp.arange(sub.nnz), side="right")
        self.B.scatter_add(R, sub.indices, sub.data[:, None] * D[seg])
        mark[sub.indices] = 1

    def mark_neighbors(self, rows, mark):
        xp = self.B.xp
        nr = int(rows.shape[0])
        self.count(rows)
        if self.use_kernel:
            W = self.W
            self._launch("mark_neighbors", nr,
                         (W.indptr, W.indices, rows.astype(xp.int32), np.int32(nr), mark))
            return
        mark[self.W[rows].indices] = 1
