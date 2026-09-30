"""Array backend: NumPy/SciPy on CPU, CuPy/cuSPARSE on GPU; modules use get().xp / .sp."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Backend:
    name: str
    xp: Any  # numpy or cupy
    sp: Any  # scipy.sparse or cupyx.scipy.sparse
    is_gpu: bool

    def _at(self, ufunc, legacy, a, idx, v):
        """Unbuffered scatter: ufunc.at on NumPy and recent CuPy, cupyx.scatter_* on older CuPy."""
        try:
            return ufunc.at(a, idx, v)
        except (AttributeError, NotImplementedError, TypeError):
            if not self.is_gpu:
                raise
        import cupyx
        getattr(cupyx, legacy)(a, idx, v)

    def scatter_add(self, a, idx, v):
        self._at(self.xp.add, "scatter_add", a, idx, v)

    def scatter_max(self, a, idx, v):
        self._at(self.xp.maximum, "scatter_max", a, idx, v)

    def scatter_min(self, a, idx, v):
        self._at(self.xp.minimum, "scatter_min", a, idx, v)

    def sync(self):
        if self.is_gpu:
            self.xp.cuda.Device().synchronize()

    def asnumpy(self, x):
        return self.xp.asnumpy(x) if self.is_gpu else x

    def csr(self, data, indices, indptr, shape):
        return self.sp.csr_matrix((data, indices, indptr), shape=shape)

    def mem_used_gb(self) -> float:
        return self.xp.get_default_memory_pool().used_bytes() / 1e9 if self.is_gpu else 0.0


_current: Backend | None = None


def set_backend(name: str = "auto") -> Backend:
    """Select 'numpy', 'cupy', or 'auto' (CuPy if a GPU is usable)."""
    global _current
    if name in ("auto", "cupy"):
        try:
            import cupy
            import cupyx.scipy.sparse as csp
            cupy.cuda.runtime.getDeviceCount()
            _current = Backend("cupy", cupy, csp, True)
            return _current
        except Exception:
            if name == "cupy":
                raise
    import numpy
    import scipy.sparse as ssp
    _current = Backend("numpy", numpy, ssp, False)
    return _current


def get() -> Backend:
    return _current or set_backend("auto")


class Timer:
    """Wall-clock timer that synchronizes the device on enter and exit."""

    def __init__(self):
        self.ms = 0.0

    def __enter__(self):
        get().sync()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        get().sync()
        self.ms += (time.perf_counter() - self._t) * 1e3
