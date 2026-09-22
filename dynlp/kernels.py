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

_MODULES: dict = {}
MAX_KERNEL_COLS = 32


def _module(gs, nc, real):
    key = (gs, nc, real)
    if key not in _MODULES:
        import cupy
        src = _SRC.format(GS=gs, NC=nc, REAL=real)
        _MODULES[key] = cupy.RawModule(code=src, options=("--std=c++14",))
    return _MODULES[key]




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
