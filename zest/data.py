"""MAT/MCOS and NPZ import plus the existing align / centered SVD workflow."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import svds, ArpackError
from .numeric import col, interp
from .settings import defaults
from .stage1 import align


def unwrap(value):
    if hasattr(value, "properties"):
        return {k: unwrap(v) for k, v in value.properties.items()}
    if isinstance(value, dict):
        return {k: unwrap(v) for k, v in value.items() if not k.startswith("__")}
    if hasattr(value, "_fieldnames"):
        return {k: unwrap(getattr(value, k)) for k in value._fieldnames}
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            if value.size == 1:
                return {k: unwrap(value[k].ravel()[0]) for k in value.dtype.names}
        if value.dtype == object and value.size == 1:
            return unwrap(value.ravel()[0])
    return value


def load(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("文件不存在")
    if path.stat().st_size > 2 * 1024**3:
        raise ValueError("本地工作台单文件上限为 2 GiB")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as f:
            values = {k: f[k] for k in f.files}
    elif path.suffix.lower() == ".mat":
        try:
            values = loadmat(path, simplify_cells=True)
            if "__function_workspace__" in values:
                raise ValueError("MCOS")
        except (ValueError, NotImplementedError, TypeError):
            from matio import load_from_mat

            values = load_from_mat(path, raw_data=True)
    else:
        raise ValueError("支持 .mat 和 .npz 数据")
    values = unwrap(values)
    candidates = []

    def walk(x, depth=0):
        if depth > 5 or not isinstance(x, dict):
            return
        energy = next(
            (
                x[k]
                for k in ("ene", "energy", "energy_meV", "energy_axis", "E")
                if k in x
                and isinstance(x[k], np.ndarray)
                and x[k].size > 1
                and (x[k].ndim == 1 or (x[k].ndim == 2 and 1 in x[k].shape))
            ),
            None,
        )
        if energy is not None:
            for key in ("data", "cube", "spectra", "spectrum", "y"):
                if (
                    key in x
                    and isinstance(x[key], np.ndarray)
                    and np.issubdtype(x[key].dtype, np.number)
                ):
                    candidates.append((x[key], energy, x.get("info", {})))
        for v in x.values():
            if isinstance(v, dict):
                walk(v, depth + 1)

    walk(values)
    if not candidates:
        raise ValueError("没有找到数值 data/cube/spectra 和 ene/energy 能量轴")
    cube, energy, info = max(candidates, key=lambda t: t[0].size)
    energy = col(energy)
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim == 1:
        cube = cube[None, None, :]
    if cube.shape[-1] != len(energy):
        axes = [i for i, n in enumerate(cube.shape) if n == len(energy)]
        if len(axes) != 1:
            raise ValueError("能量轴维度不唯一，不能自动判断谱轴")
        cube = np.moveaxis(cube, axes[0], -1)
    if cube.ndim == 2:
        cube = cube[:, None, :]
    if cube.ndim != 3 or len(energy) < 12:
        raise ValueError("数据必须为 Ny × Nx × Ne，且至少 12 个能量通道")
    if not np.all(np.isfinite(energy)) or np.any(np.diff(energy) == 0):
        raise ValueError("能量轴存在重复或非有限值")
    if np.all(np.diff(energy) < 0):
        energy = energy[::-1]
        cube = cube[..., ::-1]
    if np.any(np.diff(energy) <= 0):
        raise ValueError("能量轴必须单调")
    if not np.any(np.isfinite(cube)):
        raise ValueError("数据中没有有效强度")
    return (
        np.ascontiguousarray(cube),
        energy,
        dict(
            source=str(path),
            name=path.name,
            shape=list(cube.shape),
            tag=str(info.get("tag", "")) if isinstance(info, dict) else "",
            history=[],
        ),
    )


def preprocess(
    cube, energy, backend, do_align=True, rank=12, smooth_points=1, progress=None,
    timer=None,
):
    from .timing import PhaseTimer
    timer = timer or PhaseTimer()
    s = defaults({"stage1_zlp_align_enabled": True})
    out = np.array(cube, dtype=np.float64, copy=True)
    shifts = np.zeros(cube.shape[:2])
    backend.check()
    if do_align:
        with timer.phase("align"):
            for count, (y, x) in enumerate(np.ndindex(cube.shape[:2])):
                backend.check()
                spec = out[y, x]
                norm = np.nansum(abs(spec)) or 1.0
                _, info = align(energy, spec / norm, s)
                shift = info["applied_shift_meV"]
                aligned = interp(energy - shift, spec, energy, fill=np.nan)
                bad = ~np.isfinite(aligned)
                good = ~bad
                if np.any(bad) and np.any(good):
                    aligned[bad] = interp(
                        energy[good], aligned[good], energy[bad], "nearest", "extrapolate"
                    )
                out[y, x] = aligned
                shifts[y, x] = shift
                if progress and count % 100 == 0:
                    progress(f"ZLP 对齐 · {count + 1}/{cube.shape[0] * cube.shape[1]}")
    if rank:
        with timer.phase("svd"):
            X = out.reshape(-1, len(energy))
            finite = np.isfinite(X)
            ref = np.nanmedian(X, axis=0)
            ref[~np.isfinite(ref)] = 0
            X = np.where(finite, X, ref)
            mu = np.mean(X, axis=0)
            Xc = X - mu
            k = min(int(rank), min(Xc.shape))
            if progress:
                progress(f"SVD/PCA · rank {k} · {backend.name}")
            backend.check()
            if backend.device == "gpu":
                a = backend.array(Xc)
                # CuPy's iterative svds lost the dominant components on a
                # rank-deficient test matrix. Economy SVD followed by rank-k
                # reconstruction computes the same PCA model robustly on CUDA.
                U, S, V = backend.xp.linalg.svd(a, full_matrices=False)
                U = U[:, :k]
                S = S[:k]
                V = V[:k]
                Xden = backend.host((U * S) @ V) + mu
                backend.gpu_operations += 1
            else:
                try:
                    if k < min(Xc.shape) - 1:
                        U, S, V = svds(Xc, k=k, rng=1)
                    else:
                        U, S, V = np.linalg.svd(Xc, full_matrices=False)
                        U = U[:, :k]
                        S = S[:k]
                        V = V[:k]
                except ArpackError:
                    # Identical spectra center to a zero matrix (ARPACK -9).
                    U, S, V = np.linalg.svd(Xc, full_matrices=False)
                    U = U[:, :k]
                    S = S[:k]
                    V = V[:k]
                Xden = (U * S) @ V + mu
            out = Xden.reshape(out.shape)
    if smooth_points > 1:
        with timer.phase("smooth"):
            from .numeric import mov

            for y, x in np.ndindex(cube.shape[:2]):
                out[y, x] = mov(out[y, x], smooth_points)
    backend.sync()
    backend.check()
    return out, dict(
        zlp_alignment=bool(do_align),
        svd_rank=int(rank),
        smoothing_points=int(smooth_points),
        shift_min=float(np.min(shifts)),
        shift_max=float(np.max(shifts)),
        backend=backend.device,
    )
