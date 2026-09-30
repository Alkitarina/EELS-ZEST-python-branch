"""Float64 NumPy/CuPy execution. Optimizer control remains on the CPU.

GPU mode executes kernel construction and dense matrix products on CUDA. The
constrained QP is solved on the CPU, preserving all bounds and inequalities.
This is an explicit hybrid backend, not a claim of a GPU-native lsqlin clone.
"""

import numpy as np
from scipy.optimize import nnls, lsq_linear, minimize, LinearConstraint, Bounds
from .numeric import EPS


class Cancelled(Exception):
    pass


class Backend:
    def __init__(self, device="cpu", cancel=None):
        self.device = device
        self.cancel = cancel
        self.gpu_operations = 0
        if device == "gpu":
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("没有检测到可用 CUDA GPU")
            self.xp = cp
            self.name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            self.array([1.0]).sum().get()
        elif device == "cpu":
            self.xp = np
            self.name = "CPU · NumPy / SciPy"
        else:
            raise ValueError("device 必须是 cpu 或 gpu")

    def check(self):
        if self.cancel and self.cancel():
            raise Cancelled("用户已停止任务")

    def array(self, x):
        return self.xp.asarray(x, dtype=self.xp.float64)

    def host(self, x):
        return (
            self.xp.asnumpy(x)
            if self.device == "gpu" and isinstance(x, self.xp.ndarray)
            else np.asarray(x)
        )

    def sync(self):
        if self.device == "gpu":
            self.xp.cuda.Stream.null.synchronize()

    def matmul(self, a, b):
        self.check()
        self.gpu_operations += int(self.device == "gpu")
        return self.array(a) @ self.array(b)

    def solve(
        self, C, d, lb=None, ub=None, G=None, h=None, l1=None, max_iter=300, ridge=0.0
    ):
        self.check()
        C = self.array(C)
        d = self.array(d)
        n = C.shape[1]
        lb = np.zeros(n) if lb is None else np.asarray(lb, dtype=float)
        ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float)
        l1 = np.zeros(n) if l1 is None else np.asarray(l1, dtype=float)
        constrained = G is not None and np.size(G) > 0
        if not constrained and not np.any(l1):
            # Same convex bounded least-squares problem as lsqlin. Library
            # iterations differ; never substitute clipping for constrained LS.
            Ch = self.host(C)
            dh = self.host(d)
            if np.all(lb == 0) and np.all(np.isinf(ub)):
                try:
                    x, _ = nnls(Ch, dh, maxiter=int(max_iter))
                    return x, dict(solver="scipy_nnls", success=True)
                except RuntimeError:
                    pass
            r = lsq_linear(
                Ch, dh, bounds=(lb, ub), method="bvls", tol=1e-8, max_iter=int(max_iter)
            )
            if not r.success:
                raise RuntimeError("有界最小二乘未收敛: " + r.message)
            return r.x, dict(solver="scipy_bvls", success=True)
        # General bounds / tail inequalities / L1 use the identical convex QP.
        # CUDA forms the products in double precision; CPU SLSQP controls steps.
        H = self.host(self.matmul(C.T, C))
        g = -self.host(self.matmul(C.T, d)) + l1
        H = (H + H.T) * 0.5 + ridge * np.eye(n)
        x0 = np.maximum(lb, np.minimum(ub, np.zeros(n)))
        cons = (
            []
            if not constrained
            else [LinearConstraint(np.asarray(G), -np.inf, np.asarray(h))]
        )

        def fun(x):
            self.check()
            return 0.5 * x @ H @ x + g @ x

        r = minimize(
            fun,
            x0,
            jac=lambda x: H @ x + g,
            method="SLSQP",
            bounds=Bounds(lb, ub),
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": int(max_iter)},
        )
        violation = max(
            float(np.max(lb - r.x)),
            float(np.max(r.x - ub)),
            float(np.max(np.asarray(G) @ r.x - h)) if constrained else 0,
        )
        if not r.success or violation > 1e-7:
            raise RuntimeError(
                f"约束求解未通过: {r.message}; violation={violation:.3g}"
            )
        return r.x, dict(
            solver="scipy_slsqp",
            success=True,
            max_constraint_violation=max(violation, 0),
        )


def hardware():
    info = {
        "cpu": True,
        "gpu": False,
        "gpu_name": None,
        "gpu_memory_bytes": None,
        "gpu_error": None,
        "precision": "float64",
        "gpu_mode": "hybrid",
    }
    try:
        b = Backend("gpu")
        props = b.xp.cuda.runtime.getDeviceProperties(0)
        info.update(
            gpu=True, gpu_name=b.name, gpu_memory_bytes=int(props["totalGlobalMem"])
        )
    except Exception as e:
        info["gpu_error"] = str(e)
    return info
