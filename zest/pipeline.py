"""Original serial reference path, plus an explicitly selected CPU scheduler."""

import time
import numpy as np
from .settings import defaults
from .backend import Cancelled
from . import calibration, hierarchy, stage1, stage2, maps
from .parallel import DEFAULT_WORKERS
from .timing import PhaseTimer


def coordinates(shape, scope="pixel", pixel=(0, 0), roi=None):
    ny, nx = shape[:2]
    if scope == "pixel":
        y, x = map(int, pixel)
        if not (0 <= y < ny and 0 <= x < nx):
            raise ValueError("像素坐标超出数据范围")
        return [(y, x)]
    if scope == "all":
        return [(y, x) for x in range(nx) for y in range(ny)]
    if scope == "roi":
        if roi is None or len(roi) != 4:
            raise ValueError("ROI 需要 y0, x0, y1, x1 四个坐标")
        y0, x0, y1, x1 = map(int, roi)
        if not (0 <= y0 <= y1 < ny and 0 <= x0 <= x1 < nx):
            raise ValueError("ROI 坐标超出数据范围或顺序不正确")
        return [(y, x) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    raise ValueError("未知的像素选择方式")


def run(
    cube,
    energy,
    coords,
    settings,
    backend,
    state,
    progress=None,
    checkpoint=None,
    stage2_only=False,
    execution_mode="serial",
    workers=DEFAULT_WORKERS,
    stripe_count=0,
    timer=None,
):
    if execution_mode == "stripes" and len(coords) > 1:
        from .parallel_pipeline import run as parallel_run

        return parallel_run(cube, energy, coords, settings, backend, state,
                            progress, checkpoint, stage2_only, workers, stripe_count, timer)
    if execution_mode not in ("serial", "stripes"):
        raise ValueError("执行模式必须是 serial 或 stripes")
    owns_timer = timer is None
    timer = timer or PhaseTimer()
    state["execution"] = dict(
        mode="serial", requested_mode=execution_mode, workers=1,
        blas_threads_per_worker=2, traversal="column_major",
        hierarchy_scope="whole_image", worker_pids=[], stripes=[],
        note="单像素直接计算" if execution_mode == "stripes" else "原版串行对照",
    )
    s = defaults(settings)
    n = len(coords)
    start = time.perf_counter()
    requested_stage2 = bool(s["run_stage2"])
    # state is observable by the job manager; append only fully solved pixels.
    state.update(
        coords=coords, shape=cube.shape[:2], settings=s, maps={}, elapsed_seconds=0
    )
    results = state.setdefault("results", [])
    h = state.setdefault("hierarchy", dict(enabled=False, status="disabled"))

    def notify(phase, done, total, message):
        state["elapsed_seconds"] = time.perf_counter() - start
        if progress:
            progress(phase, done, total, message)

    def interim(i, r):
        if checkpoint:
            checkpoint(i, r)

    if not stage2_only:
        results.clear()
        # Datacube calibration is enabled only for a datacube operation; the
        # standalone single-spectrum path mirrors solve_stage1 directly.
        with timer.phase("calibration"):
            if n > 1:
                s = calibration.calibrate(
                    cube,
                    energy,
                    coords,
                    s,
                    backend,
                    lambda msg: notify("calibration", 0, n, msg),
                )
                state["settings"] = s
        active = bool(s["stage1_hierarchical_enabled"]) and n >= max(
            8, min(s["stage1_cluster_min_pixels"], cube.shape[0] * cube.shape[1])
        )
        fs = dict(s, run_stage2=False)
        if active:
            fs.update(
                stage1_zlp_gain_heavy_init_enabled=False,
                stage1_zlp_gain_tail_prior_enabled=False,
                stage1_zlp_gain_asym_shape_prior_enabled=False,
                stage1_hierarchy_context={},
            )
        previous = None
        for i, (y, x) in enumerate(coords):
            with timer.phase("stage1"):
                backend.check()
                notify("stage1", i, n, f"Stage 1 · 像素 ({y + 1}, {x + 1}) · {i + 1}/{n}")
                r = None
                for attempt in range(1 + int(s["datacube_fit_retries"])):
                    ps = dict(fs)
                    if attempt and s["datacube_retry_with_strict_nnls"]:
                        ps["nonlinear_inner_solver"] = "lsqnonneg"
                    try:
                        r = stage1.solve(
                            energy,
                            cube[y, x],
                            ps,
                            backend,
                            previous=previous
                            if attempt == 0 and s["datacube_use_warm_start"]
                            else None,
                            progress=lambda msg: notify(
                                "stage1", i, n, f"({y + 1}, {x + 1}) · {msg}"
                            ),
                        )
                        r["fit_attempts"] = attempt + 1
                        break
                    except Cancelled:
                        raise
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        r = dict(error=str(exc), fit_attempts=attempt + 1)
                r.update(
                    pixel_y=y + 1, pixel_x=x + 1, pixel_index=y + x * cube.shape[0] + 1
                )
                if "error" not in r:
                    previous = r["theta"]
                results.append(r)
                interim(i, r)
            if requested_stage2 and not active and "error" not in r:
                with timer.phase("stage2"):
                    try:
                        results[i] = stage2.promote(
                            r,
                            r["settings"],
                            backend,
                            lambda msg: notify(
                                "stage2", i, n, f"({y + 1}, {x + 1}) · {msg}"
                            ),
                        )
                    except Cancelled:
                        raise
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        results[i]["stage2"] = dict(status="error", error=str(exc))
                    interim(i, results[i])
            notify("stage1", i + 1, n, f"完成 {i + 1}/{n} 像素")
        if active:
            with timer.phase("hierarchy"):
                results, h = hierarchy.run(
                    cube,
                    energy,
                    coords,
                    results,
                    fs,
                    backend,
                    lambda msg: notify("hierarchy", 0, n, msg),
                    timer=timer,
                )
            state.update(results=results, hierarchy=h)
            for i, r in enumerate(results):
                interim(i, r)
    else:
        active = True
        if not results:
            raise ValueError("没有可以继续计算的 Stage 1 结果")
    if requested_stage2 and active:
        with timer.phase("stage2"):
            for i, r in enumerate(results):
                backend.check()
                if "error" in r:
                    continue
                ps = dict(r["settings"], run_stage2=True)
                ps.update({k: v for k, v in s.items() if k.startswith("stage2_")})
                y, x = coords[i]
                try:
                    results[i] = stage2.promote(
                        r,
                        ps,
                        backend,
                        lambda msg: notify("stage2", i, n, f"({y + 1}, {x + 1}) · {msg}"),
                    )
                except Cancelled:
                    raise
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    results[i]["stage2"] = dict(status="error", error=str(exc))
                interim(i, results[i])
                notify("stage2", i + 1, n, f"Stage 2 完成 {i + 1}/{n}")
    with timer.phase("maps"):
        state["maps"] = maps.build(cube.shape[:2], coords, results, s, h)
    state.update(
        elapsed_seconds=time.perf_counter() - start,
        gpu_operations=backend.gpu_operations,
    )
    state["timings"] = timer.finish() if owns_timer else timer.snapshot()
    return state
