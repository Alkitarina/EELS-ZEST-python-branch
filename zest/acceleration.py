"""Per-solve Float64 workspaces; no global data cache or extra worker threads."""
import os
import numpy as np
from .numeric import EPS

try:
    from . import _kernels
except ImportError:
    _kernels = None


def native_enabled():
    mode = os.environ.get("ZEST_CPU_KERNELS", "auto").lower()
    if mode not in ("auto", "numpy", "native"):
        raise ValueError("ZEST_CPU_KERNELS must be auto, numpy or native")
    if mode == "native" and _kernels is None:
        raise RuntimeError("C++ kernels unavailable; run tools/build_native.py")
    return _kernels is not None and mode != "numpy"


def execution_info(device):
    kernels = ("cpp" if native_enabled() else "numpy") if device == "cpu" else "not_used"
    return dict(cpu_kernels=kernels,
                precision="float64", per_solve_cache=True, solver="unchanged")


class ShiftPlan:
    """Interpolation geometry for an immutable energy axis and mode grid."""
    def __init__(self, e, em, backend):
        self.b = backend
        xp = backend.xp
        self.e = backend.array(e)
        self.em = backend.array(em)
        self.shape = (self.e.size, self.em.size)
        self.native = backend.device == "cpu" and native_enabled()
        self.uniform = False
        if not self.em.size:
            return
        de = xp.diff(self.e)
        step = float(xp.median(de))
        self.uniform = (np.isfinite(step) and step != 0 and
                        float(xp.max(xp.abs(de - step))) <= max(1e-9, 1e-7 * max(abs(step), EPS)))
        if self.uniform:
            f = (self.e[:, None] - self.em[None, :] - self.e[0]) / step + 1
            lo = xp.floor(f)
            self.alpha = f - lo
            self.valid = (lo >= 1) & (lo < self.e.size)
            self.lo = xp.where(self.valid, lo, 1).astype(xp.int64) - 1
        else:
            # Keep np/cp.interp's exact arithmetic and endpoint semantics.
            self.queries = self.e[:, None] - self.em[None, :]

    def apply(self, kernel):
        b, xp = self.b, self.b.xp
        b.gpu_operations += int(b.device == "gpu")
        k = b.array(kernel)
        if not self.em.size:
            return xp.zeros(self.shape)
        if self.uniform:
            if self.native:
                M = np.empty(self.shape)
                _kernels.shift(np.ascontiguousarray(k), self.lo, self.alpha, self.valid, M)
            else:
                M = xp.where(self.valid, (1 - self.alpha) * k[self.lo] +
                             self.alpha * k[self.lo + 1], 0)
                M = xp.where(xp.isfinite(M) & (M >= 0), M, 0)
        else:
            M = xp.column_stack([xp.interp(self.queries[:, j], self.e, k, left=0, right=0)
                                 for j in range(self.em.size)])
            M = xp.where(xp.isfinite(M) & (M >= 0), M, 0)
        total = M.sum(axis=0)
        # Preserve reduction layout and arithmetic of the original path.
        return M / xp.where(total > 0, total, 1)[None, :]


class LinearWorkspace:
    """Only assemble C and d; all bounds, solvers and tolerances stay unchanged."""
    def __init__(self, y, w, rhs, columns, backend):
        self.b = backend
        self.w = backend.array(w)
        self.C = backend.xp.empty((len(y) + len(rhs), columns), dtype=backend.xp.float64)
        self.d = backend.xp.concatenate([backend.array(y * w), backend.array(rhs)])
        self.native = backend.device == "cpu" and native_enabled()

    def assemble(self, A, R):
        if self.native and A.flags.c_contiguous and R.flags.c_contiguous:
            _kernels.weighted_matrix(A, self.w, R, self.C)
        else:
            self.b.xp.multiply(A, self.w[:, None], out=self.C[:len(self.w)])
            self.C[len(self.w):] = R
        return self.C, self.d


def residual(model, y, w, reg, rhs, backend):
    if backend.device == "cpu" and native_enabled():
        return _kernels.residual(model, y, w, reg, rhs)
    return np.r_[(model - y) * w, backend.host(reg - rhs)]


def update_hard_columns(stokes, anti, q, out, backend):
    if backend.device == "cpu" and native_enabled():
        _kernels.hard_columns(stokes, anti, q, out)
    else:
        out[:, :len(q)] = stokes + anti * backend.array(q)[None, :]
