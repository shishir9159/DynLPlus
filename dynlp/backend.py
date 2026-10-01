"""GPU backend: CuPy arrays and cuSPARSE matrices; everything runs on the device."""
from __future__ import annotations

import time

import cupy as xp
import cupyx
import cupyx.scipy.sparse as sp


def _at(ufunc, legacy, a, idx, v):
    """Unbuffered scatter: ufunc.at on recent CuPy, cupyx.scatter_* on older CuPy."""
    try:
        ufunc.at(a, idx, v)
    except (AttributeError, NotImplementedError, TypeError):
        getattr(cupyx, legacy)(a, idx, v)


def scatter_add(a, idx, v):
    _at(xp.add, "scatter_add", a, idx, v)


def scatter_max(a, idx, v):
    _at(xp.maximum, "scatter_max", a, idx, v)


def scatter_min(a, idx, v):
    _at(xp.minimum, "scatter_min", a, idx, v)


def sync():
    xp.cuda.Device().synchronize()


def csr(data, indices, indptr, shape):
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def mem_used_gb() -> float:
    return xp.get_default_memory_pool().used_bytes() / 1e9


class Timer:
    """Wall-clock timer that synchronizes the device on enter and exit."""

    def __init__(self):
        self.ms = 0.0

    def __enter__(self):
        sync()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.ms += (time.perf_counter() - self._t) * 1e3
