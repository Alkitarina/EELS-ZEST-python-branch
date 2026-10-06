"""Dynamically claimed vertical chains, then one whole-image hierarchy."""

import time
import numpy as np
from . import calibration, hierarchy, maps
from .parallel import ComputePool, vertical_stripes, scheduling_options, DEFAULT_WORKERS
from .settings import defaults
from .timing import PhaseTimer
from .acceleration import execution_info


def run(cube, energy, coords, settings, backend, state, progress=None,
        checkpoint=None, stage2_only=False, workers=DEFAULT_WORKERS, stripe_count=0,
        timer=None):
    if backend.device != "cpu":
        raise ValueError("竖条并行使用 CPU；GPU 混合计算请选择串行模式")
    s = defaults(settings)
    coords = [tuple(p) for p in coords]
    n = len(coords)
    if not n:
        raise ValueError("没有要计算的像素")
    workers, target_stripes = scheduling_options(workers, stripe_count)
    stripes = vertical_stripes(coords, target_stripes)
    count = min(workers, n)
    owns_timer = timer is None
    timer = timer or PhaseTimer()
    started = time.perf_counter()
    execution = dict(
        mode="stripes", requested_workers=workers, workers=count,
        requested_stripes=stripe_count, stripe_count=len(stripes) if not stage2_only else 0,
        scheduling="dynamic_queue", stage1_workers=min(count, len(stripes)),
        blas_threads_per_worker=1, traversal="vertical_snake",
        hierarchy_scope="whole_image", hierarchy_update="four_color_barriers",
        worker_pids=[], phase_seconds={},
        acceleration=execution_info(backend.device),
        stripes=[dict(id=st["id"] + 1, columns=[x + 1 for x in st["columns"]],
                      total=len(st["indices"]), done=0, pid=None, status="queued",
                      current_pixel=None, message="等待领取", elapsed_seconds=0.0,
                      wall_seconds=0.0)
                 for st in stripes] if not stage2_only else [],
    )
    state.update(coords=coords, shape=cube.shape[:2], settings=s, maps={},
                 elapsed_seconds=0, execution=execution)
    h = state.setdefault("hierarchy", dict(enabled=False, status="disabled"))
    requested_stage2 = bool(s["run_stage2"])

    def notify(phase, done, total, message):
        state["elapsed_seconds"] = time.perf_counter() - started
        if progress:
            progress(phase, done, total, message)

    def save(i, r):
        if checkpoint:
            checkpoint(i, r)

    def record_pid(pid):
        if pid not in execution["worker_pids"]:
            execution["worker_pids"].append(pid)
            execution["worker_pids"].sort()

    if not stage2_only:
        results = [None] * n  # Canonical coordinate slots, never completion order.
        state["results"] = results
        phase_start = time.perf_counter()
        with timer.phase("calibration"):
            if n > 1:
                s = calibration.calibrate(
                    cube, energy, coords, s, backend,
                    lambda msg: notify("calibration", 0, n, msg),
                )
                state["settings"] = s
        execution["phase_seconds"]["calibration"] = time.perf_counter() - phase_start
        active = bool(s["stage1_hierarchical_enabled"]) and n >= max(
            8, min(s["stage1_cluster_min_pixels"], cube.shape[0] * cube.shape[1])
        )
        fs = dict(s, run_stage2=False)
        if active:
            fs.update(stage1_zlp_gain_heavy_init_enabled=False,
                      stage1_zlp_gain_tail_prior_enabled=False,
                      stage1_zlp_gain_asym_shape_prior_enabled=False,
                      stage1_hierarchy_context={})
    else:
        results = state.get("results", [])
        if len(results) != n or any(r is None for r in results):
            raise ValueError("没有完整的 Stage 1 结果可以继续计算")
        active = False

    backend.check()
    notify("workers", 0, count, f"启动 {count} 个 CPU 工作进程 · 每进程 1 个数值线程")
    with ComputePool(cube, energy, count, backend, timer) as pool:
        for pid in pool.worker_pids:
            record_pid(pid)
        if not stage2_only:
            phase_start = time.perf_counter()
            done = 0

            def received(message):
                nonlocal done
                record_pid(message["pid"])
                strip = execution["stripes"][message["stripe"]]
                strip["pid"] = message["pid"]
                if message["kind"] == "stripe_done":
                    strip.update(status="completed", current_pixel=None, message="条带完成",
                                 wall_seconds=message["wall_seconds"])
                elif message["kind"] == "stripe_started":
                    strip.update(status="running", message="已领取条带")
                elif message["kind"] == "progress":
                    strip.update(status="running", current_pixel=[v + 1 for v in message["point"]],
                                 message=message["message"])
                elif message["kind"] == "pixel":
                    i, r = message["index"], message["result"]
                    if results[i] is None:
                        done += 1
                        strip["done"] += 1
                    results[i] = r
                    strip["elapsed_seconds"] += message["elapsed_seconds"]
                    save(i, r)
                notify("stage1", done, n,
                       f"Stage 1 · {count} 进程 / {len(stripes)} 条带 · 完成 {done}/{n} · 条带 {strip['id']}")

            try:
                with timer.phase("stage1"):
                    pool.stripes(stripes, coords, fs, received)
            finally:
                execution["phase_seconds"]["stage1"] = time.perf_counter() - phase_start
            if any(r is None for r in results):
                raise RuntimeError("工作进程未返回全部像素结果")
            if active:
                phase_start = time.perf_counter()
                # Global labels, templates and cross-stripe neighbours are
                # calculated once. Only the frozen-context refits are mapped.
                with timer.phase("hierarchy"):
                    results, h = hierarchy.run(
                        cube, energy, coords, results, fs, backend,
                        lambda msg: notify("hierarchy", 0, n, msg),
                        executor=pool, on_result=save, timer=timer,
                    )
                state.update(results=results, hierarchy=h)
                execution["phase_seconds"]["hierarchy"] = time.perf_counter() - phase_start
                for i, r in enumerate(results):
                    save(i, r)

        if requested_stage2:
            phase_start = time.perf_counter()

            def tasks():
                for i, r in enumerate(results):
                    if "error" in r:
                        continue
                    ps = dict(r["settings"], run_stage2=True)
                    ps.update({k: v for k, v in s.items() if k.startswith("stage2_")})
                    yield i, r, ps

            done = sum("error" in r for r in results)
            notify("stage2", done, n, f"Stage 2 · {count} 进程并行温度扫描")
            with timer.phase("stage2"):
                for i, r, pid in pool.stage2(tasks()):
                    record_pid(pid)
                    results[i] = r
                    save(i, r)
                    done += 1
                    notify("stage2", done, n, f"Stage 2 · 完成 {done}/{n}")
            execution["phase_seconds"]["stage2"] = time.perf_counter() - phase_start

    with timer.phase("maps"):
        state["maps"] = maps.build(cube.shape[:2], coords, results, s, h)
    state.update(elapsed_seconds=time.perf_counter() - started,
                 gpu_operations=backend.gpu_operations)
    state["timings"] = timer.finish() if owns_timer else timer.snapshot()
    return state
