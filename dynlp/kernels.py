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

# Class-parallel mapping (multi-class), after sparse-attention kernels that spread the
# head dimension across lanes: one warp per row, lanes split into LE edge groups x LC
# class lanes, so a neighbour's row X[v, :] is read as one coalesced vector and atomics
# on R[v, :] hit contiguous addresses. Any number of classes (CPL per lane).
_COLS_SRC = r"""
#define NC {NC}
#define LC {LC}
#define LE (32 / LC)
#define CPL ((NC + LC - 1) / LC)
typedef {REAL} real;

#define ROW_SETUP \
  const long long wid = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; \
  const int lane = threadIdx.x & 31, l = lane % LC, e = lane / LC; \
  if (wid >= nf) return; \
  const int row = frontier[wid], beg = indptr[row], end = indptr[row + 1];

extern "C" __global__ void frontier_jacobi(
    const int* __restrict__ indptr, const int* __restrict__ indices, const real* __restrict__ w,
    const int* __restrict__ frontier, const int nf,
    const real* __restrict__ X, const real* __restrict__ rhs, const real* __restrict__ diag,
    const real delta, real* __restrict__ Y, unsigned char* __restrict__ changed)
{{
  ROW_SETUP
  real acc[CPL];
  #pragma unroll
  for (int k = 0; k < CPL; ++k) acc[k] = 0;
  for (int j = beg + e; j < end; j += LE) {{
    const long long v = indices[j];
    const real wv = w[j];
    #pragma unroll
    for (int k = 0; k < CPL; ++k) if (l + k * LC < NC) acc[k] += wv * X[v * NC + l + k * LC];
  }}
  #pragma unroll
  for (int k = 0; k < CPL; ++k)
    for (int off = LC; off < 32; off <<= 1) acc[k] += __shfl_xor_sync(0xffffffffu, acc[k], off);
  bool ch = false;
  if (e == 0) {{
    const real inv = (real)1 / diag[row];
    #pragma unroll
    for (int k = 0; k < CPL; ++k) {{
      const int c = l + k * LC;
      if (c < NC) {{
        const real yv = (rhs[(long long)row * NC + c] + acc[k]) * inv;
        Y[wid * NC + c] = yv;
        ch |= fabs(yv - X[(long long)row * NC + c]) > delta;
      }}
    }}
  }}
  const bool any = __any_sync(0xffffffffu, ch);
  if (lane == 0) changed[wid] = any;
}}

extern "C" __global__ void push_scatter(
    const int* __restrict__ indptr, const int* __restrict__ indices, const real* __restrict__ w,
    const int* __restrict__ frontier, const int nf, const real* __restrict__ D,
    real* __restrict__ R, unsigned char* __restrict__ mark)
{{
  ROW_SETUP
  real d[CPL];
  #pragma unroll
  for (int k = 0; k < CPL; ++k) d[k] = l + k * LC < NC ? D[wid * NC + l + k * LC] : 0;
  for (int j = beg + e; j < end; j += LE) {{
    const long long v = indices[j];
    const real wv = w[j];
    #pragma unroll
    for (int k = 0; k < CPL; ++k) if (l + k * LC < NC) atomicAdd(&R[v * NC + l + k * LC], wv * d[k]);
    if (l == 0) mark[v] = 1;
  }}
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

MAX_KERNEL_COLS = 32  # row mapping keeps C accumulators per lane
COLS_MIN = 17         # auto: class-parallel mapping above 16 columns (measured 1.5-2x at 24-32)
_MODULES: dict = {}


def _module(src, **fmt):
    key = (id(src),) + tuple(sorted(fmt.items()))
    if key not in _MODULES:
        import cupy
        _MODULES[key] = cupy.RawModule(code=src.format(**fmt), options=("--std=c++14",))
    return _MODULES[key]


def auto_group(avg_deg: float) -> int:
    """Power-of-two lanes per row close to the mean degree, in [2, 32]."""
    g = 2
    while g < 32 and g * 2 <= max(avg_deg, 1.0):
        g *= 2
    return g


def _group(W, group):
    return auto_group(W.nnz / max(W.shape[0], 1)) if group == "auto" else int(group)


class FusedPush:
    """Asynchronous residual push with resident buffers (GPU, C <= 32).

    Takes the residual of every violating vertex (|r_u| > eps_c * diag_u), adds
    r_u / diag_u to z_u and scatters it to the neighbours, keeping r exact.
    Converges for this M-matrix despite the asynchrony (chaotic relaxation).
    """

    def __init__(self, W, C, group="auto", threads=256, blocks_per_sm=8):
        import cupy
        n, double = W.shape[0], W.data.dtype == np.float64
        self.W, self.C, self.gs = W, C, min(_group(W, group), 32)
        mod = _module(_FUSED_SRC, GS=self.gs, NC=C, REAL="double" if double else "float", IS_DOUBLE=int(double))
        self.k_push, self.k_compact = mod.get_function("push_round"), mod.get_function("compact_frontier")
        z = lambda *s: cupy.zeros(s, dtype=cupy.int32)  # noqa: E731
        self.front, self.touched, self.mark, self.fcnt, self.tcnt = z(2, n), z(n), z(n), z(2), z(2)
        self.edges = cupy.zeros(1, dtype=cupy.uint64)
        self.grid = (cupy.cuda.Device().attributes["MultiProcessorCount"] * blocks_per_sm,)
        self.block = (threads,)

    def run(self, Z, R, diag, eps, cand, max_rounds, check_every=8):
        """Push until no vertex violates; returns (rounds, converged, edge visits)."""
        import cupy
        n0, W = int(cand.shape[0]), self.W
        self.edges[:] = 0
        if not n0:
            return 0, True, 0
        self.front[0, :n0], self.fcnt[:], self.tcnt[:] = cand, 0, 0
        self.fcnt[0] = n0
        eps, rounds = cupy.ascontiguousarray(eps, dtype=Z.dtype), 0
        while rounds < max_rounds:
            p, q = rounds & 1, 1 - (rounds & 1)
            fp, fq, tp, tq = self.fcnt[p:p + 1], self.fcnt[q:q + 1], self.tcnt[p:p + 1], self.tcnt[q:q + 1]
            self.k_push(self.grid, self.block, (W.indptr, W.indices, W.data, diag, self.front[p], fp, fq, Z, R,
                                                self.mark, self.touched, tp, self.edges))
            self.k_compact(self.grid, self.block, (self.touched, tp, tq, self.mark, R, diag, eps, self.front[q], fq))
            rounds += 1
            if rounds % check_every == 0 and int(self.fcnt[q]) == 0:
                break
        return rounds, int(self.fcnt[rounds & 1]) == 0, int(self.edges[0])


class FrontierOps:
    """Frontier primitives bound to one CSR matrix and column count."""

    def __init__(self, W, C, group="auto", threads=256, mapping="auto"):
        self.B, self.W, self.C, self.threads = backend.get(), W, C, threads
        self.gs = _group(W, group)
        self.real = np.float64 if W.data.dtype == np.float64 else np.float32
        self.mapping = mapping if mapping != "auto" else ("cols" if C >= COLS_MIN else "rows")
        self.use_kernel = self.B.is_gpu and (self.mapping == "cols" or C <= MAX_KERNEL_COLS)
        self.edges = self.B.xp.zeros((), dtype=self.B.xp.int64)  # device-side edge-visit counter

    def _launch(self, name, ngroups, *args):
        real = "double" if self.real is np.float64 else "float"
        if self.mapping == "cols" and name != "mark_neighbors":  # one warp per row
            lc = min(32, 1 << max(self.C - 1, 0).bit_length())
            k = _module(_COLS_SRC, NC=self.C, LC=lc, REAL=real).get_function(name)
            return k(((ngroups * 32 + self.threads - 1) // self.threads,), (self.threads,), args)
        k = _module(_SRC, GS=self.gs, NC=self.C, REAL=real).get_function(name)
        if self.gs > 32 and name == "frontier_jacobi":  # one block per row (DynLP's mapping)
            k((ngroups,), (self.gs,), args)
        else:
            k(((ngroups * self.gs + self.threads - 1) // self.threads,), (self.threads,), args)

    def _rows(self, rows):
        ip = self.W.indptr
        self.edges += (ip[rows + 1] - ip[rows]).sum(dtype=self.B.xp.int64)
        return rows.astype(self.B.xp.int32), np.int32(rows.shape[0])

    def jacobi(self, frontier, X, rhs, diag, delta):
        """New values for frontier rows and a changed-by-more-than-delta flag."""
        xp, W = self.B.xp, self.W
        f, nf = self._rows(frontier)
        if self.use_kernel:
            Y, ch = xp.empty((int(nf), self.C), dtype=X.dtype), xp.empty(int(nf), dtype=xp.uint8)
            self._launch("frontier_jacobi", int(nf), W.indptr, W.indices, W.data, f, nf, xp.ascontiguousarray(X),
                         xp.ascontiguousarray(rhs), diag, self.real(delta), Y, ch)
            return Y, ch.astype(bool)
        Y = (rhs[frontier] + W[frontier] @ X) / diag[frontier][:, None]
        return Y, (xp.abs(Y - X[frontier]) > delta).any(axis=1)

    def push(self, frontier, D, R, mark):
        """R += W[:, frontier] @ D (W symmetric) and mark touched vertices."""
        xp, W = self.B.xp, self.W
        f, nf = self._rows(frontier)
        if self.use_kernel:
            return self._launch("push_scatter", int(nf), W.indptr, W.indices, W.data, f, nf,
                                xp.ascontiguousarray(D), R, mark)
        sub = W[frontier]
        seg = xp.searchsorted(sub.indptr[1:], xp.arange(sub.nnz), side="right")
        self.B.scatter_add(R, sub.indices, sub.data[:, None] * D[seg])
        mark[sub.indices] = 1

    def mark_neighbors(self, rows, mark):
        r, nr = self._rows(rows)
        if self.use_kernel:
            return self._launch("mark_neighbors", int(nr), self.W.indptr, self.W.indices, r, nr, mark)
        mark[self.W[rows].indices] = 1
