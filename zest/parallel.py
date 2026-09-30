"""Spawn-based CPU workers. Share spectra, never share mutable fit results.

One long task owns each vertical stripe during Stage 1. The same pool then
serves independent refits, with barriers managed by the whole-image hierarchy.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context, shared_memory
from multiprocessing.util import Finalize
from queue import Empty
from threading import Thread
import os
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .backend import Backend, Cancelled
from . import stage1, stage2
from .timing import PhaseTimer

_WORKER = {}
MAX_WORKERS = 61  # Windows ProcessPoolExecutor limit.
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)


def scheduling_options(workers, stripe_count):
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"工作进程数需要为 1–{MAX_WORKERS} 的整数")
    if type(stripe_count) is not int or not 0 <= stripe_count <= 10000:
        raise ValueError("条带数需要为 0–10000 的整数；0 表示自动")
    return workers, stripe_count or 3 * workers


def vertical_stripes(coords, count=6):
    """Partition whole columns; return canonical indices in snake order."""
    if count < 1:
        raise ValueError("条带数必须大于零")
    points = [tuple(map(int, p)) for p in coords]
    if len(points) != len(set(points)):
        raise ValueError("像素坐标不能重复")
    columns = sorted({x for _, x in points})
    if not columns:
        return []
    lookup = {p: i for i, p in enumerate(points)}
    rows = {x: sorted(y for y, xx in points if xx == x) for x in columns}
    stripes = []
    for number, group in enumerate(np.array_split(columns, min(count, len(columns)))):
        indices = []
        for k, x in enumerate(group.tolist()):
            ys = rows[x] if k % 2 == 0 else rows[x][::-1]
            indices.extend(lookup[(y, x)] for y in ys)
        stripes.append(dict(id=number, columns=group.tolist(), indices=indices))
    return stripes


def _close_worker():
    _WORKER.pop("cube", None)
    _WORKER.pop("energy", None)
    for handle in _WORKER.pop("handles", []):
        handle.close()


def _initialize(specs, cancel, messages, ready=None):
    # Limits apply only to new worker processes, not the original running app.
    _WORKER["limits"] = threadpool_limits(limits=1)
    _WORKER["handles"] = []
    for key, (name, shape, dtype) in specs.items():
        handle = shared_memory.SharedMemory(name=name)
        array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=handle.buf)
        array.flags.writeable = False
        _WORKER[key] = array
        _WORKER["handles"].append(handle)
    _WORKER.update(cancel=cancel, messages=messages, ready=ready,
                   backend=Backend("cpu", cancel.is_set))
    Finalize(None, _close_worker, exitpriority=10)
    # A still-running 0.2 parent can spawn this module before the user restarts
    # the service. Keep its three-argument initializer and message protocol.
    if ready is not None:
        _emit("worker_ready")


def _ping():
    while not _WORKER["ready"].wait(0.1):
        _WORKER["backend"].check()
    return os.getpid()


def _emit(kind, **values):
    _WORKER["messages"].put(dict(kind=kind, pid=os.getpid(), **values))


def _fit_stripe(stripe, coords, settings):
    b = _WORKER["backend"]
    previous, previous_point = None, None
    stripe_id = stripe["id"]
    stripe_started = time.perf_counter()
    _emit("stripe_started", stripe=stripe_id)
    for step, i in enumerate(stripe["indices"]):
        b.check()
        y, x = coords[i]
        point = (y, x)
        # With a sparse selection or a failed predecessor, restart locally.
        adjacent = previous_point is not None and sum(
            abs(a - c) for a, c in zip(point, previous_point)
        ) == 1
        seed = previous if adjacent and settings["datacube_use_warm_start"] else None
        _emit("progress", stripe=stripe_id, index=i, point=point, message="开始拟合")
        started = time.perf_counter()
        used_seed = False
        for attempt in range(1 + int(settings["datacube_fit_retries"])):
            ps = dict(settings)
            if attempt and settings["datacube_retry_with_strict_nnls"]:
                ps["nonlinear_inner_solver"] = "lsqnonneg"
            current_seed = seed if attempt == 0 else None
            if current_seed is None:
                # Even a user-supplied initial_theta is only a guess at a new
                # stripe anchor: keep all original optimization rounds.
                ps["stage1_warm_start_skip_coarse"] = False
            try:
                r = stage1.solve(
                    _WORKER["energy"], _WORKER["cube"][y, x], ps, b,
                    previous=current_seed,
                    progress=lambda msg: _emit(
                        "progress", stripe=stripe_id, index=i, point=point, message=msg
                    ),
                )
                r["fit_attempts"] = attempt + 1
                used_seed = current_seed is not None
                break
            except Cancelled:
                raise
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                r = dict(error=str(exc), fit_attempts=attempt + 1)
        r.update(
            pixel_y=y + 1, pixel_x=x + 1,
            pixel_index=y + x * _WORKER["cube"].shape[0] + 1,
            execution=dict(
                mode="stripes", stripe_id=stripe_id + 1, worker_pid=os.getpid(),
                stripe_step=step + 1,
                warm_start_from=[v + 1 for v in previous_point] if used_seed else None,
                full_initialization=not used_seed,
            ),
        )
        if "error" not in r:
            previous, previous_point = r["theta"], point
        else:
            previous, previous_point = None, None
        _emit("pixel", stripe=stripe_id, index=i, point=point, result=r,
              elapsed_seconds=time.perf_counter() - started)
    _emit("stripe_done", stripe=stripe_id,
          wall_seconds=time.perf_counter() - stripe_started)
    return os.getpid()


def _refit(task):
    from .hierarchy import refit_one

    i, point, old, ctx, settings, iterations, phase = task
    y, x = point
    candidate, error = refit_one(
        _WORKER["energy"], _WORKER["cube"][y, x], old, ctx, settings,
        iterations, phase, _WORKER["backend"],
    )
    return i, candidate, error, os.getpid()


def _stage2(task):
    i, r, settings = task
    try:
        out = stage2.promote(r, settings, _WORKER["backend"])
    except Cancelled:
        raise
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        out = dict(r, stage2=dict(status="error", error=str(exc)))
    return i, out, os.getpid()


class ComputePool:
    """A job-scoped pool, with cancellation checked inside every objective."""

    def __init__(self, cube, energy, workers, backend, timer=None):
        self.workers = int(workers)
        self.backend = backend
        self.context = get_context("spawn")
        self.cancel = self.context.Event()
        self.ready = self.context.Event()
        self.messages = self.context.Queue()
        self.handles = []
        self.executor = None
        self.on_message = None
        self.cube, self.energy = cube, energy
        self.cleanup_errors = []
        self.timer = timer or PhaseTimer()
        self.worker_pids = []

    def __enter__(self):
        with self.timer.phase("pool_setup"):
            return self._start()

    def _start(self):
        specs = {}
        try:
            for key, source in (("cube", self.cube), ("energy", self.energy)):
                array = np.ascontiguousarray(source)
                handle = shared_memory.SharedMemory(create=True, size=array.nbytes)
                self.handles.append(handle)
                shared = np.ndarray(array.shape, dtype=array.dtype, buffer=handle.buf)
                shared[:] = array
                specs[key] = (handle.name, array.shape, array.dtype.str)
                del shared
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=self.context,
                initializer=_initialize, initargs=(specs, self.cancel, self.messages, self.ready),
            )
            # Start every process now, so spawn/import time is measured in
            # pool_setup rather than hidden in the first pixel's solve time.
            probes = [self.executor.submit(_ping) for _ in range(self.workers)]
            while len(self.worker_pids) < self.workers:
                self.backend.check()
                message = self._receive(timeout=0.1)
                if message and message["kind"] == "worker_ready":
                    self.worker_pids.append(message["pid"])
                for future in probes:
                    if future.done():
                        future.result()
            self.ready.set()
            for future in probes:
                future.result()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _receive(self, timeout=0):
        try:
            message = self.messages.get(timeout=timeout)
        except Empty:
            return None
        if self.on_message:
            self.on_message(message)
        return message

    def stripes(self, stripes, coords, settings, on_message):
        self.on_message = on_message
        iterator = iter(stripes)
        futures = set()

        def fill():
            # More stripes than processes: the shared task queue lets each
            # idle process claim the next chain. Bound queued payloads.
            while len(futures) < 2 * self.workers:
                try:
                    stripe = next(iterator)
                except StopIteration:
                    break
                futures.add(self.executor.submit(_fit_stripe, stripe, coords, settings))

        fill()
        finished = set()
        while len(finished) < len(stripes):
            self.backend.check()
            message = self._receive(timeout=0.1)
            if message and message["kind"] == "stripe_done":
                finished.add(message["stripe"])
            for future in list(futures):
                if future.done():
                    future.result()  # Surface crashed workers, not an endless wait.
                    futures.remove(future)
            fill()
        for future in futures:
            future.result()
        self.on_message = None

    def _mapped(self, function, tasks):
        # Bound pending payloads: old pixel results can be large. No cube is
        # copied per task; its shared-memory view stays alive for the whole job.
        iterator = iter(tasks)
        pending = set()

        def fill():
            while len(pending) < 2 * self.workers:
                try:
                    task = next(iterator)
                except StopIteration:
                    break
                pending.add(self.executor.submit(function, task))

        fill()
        while pending:
            self.backend.check()
            ready, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in ready:
                yield future.result()
            fill()

    def refits(self, tasks):
        return self._mapped(_refit, tasks)

    def stage2(self, tasks):
        return self._mapped(_stage2, tasks)

    def __exit__(self, exc_type, exc, tb):
        with self.timer.phase("pool_shutdown"):
            return self._shutdown(exc_type, exc, tb)

    def _shutdown(self, exc_type, exc, tb):
        self.cancel.set()
        if self.executor is not None:
            # multiprocessing.Queue feeder threads must be drained while
            # workers exit. Joining first can deadlock with large pixel arrays.
            def shutdown():
                try:
                    self.executor.shutdown(wait=True, cancel_futures=True)
                except BaseException as error:
                    self.cleanup_errors.append(error)

            joiner = Thread(target=shutdown, name="zest-pool-cleanup", daemon=True)
            joiner.start()
            while joiner.is_alive():
                try:
                    self._receive(timeout=0.05)
                except Exception as error:
                    self.cleanup_errors.append(error)
                    self.on_message = None
                joiner.join(timeout=0.01)
            while True:
                try:
                    if self._receive(timeout=0.01) is None:
                        break
                except Exception as error:
                    self.cleanup_errors.append(error)
                    self.on_message = None
        self.messages.close()
        self.messages.join_thread()
        for handle in self.handles:
            handle.close()
            handle.unlink()
        self.handles.clear()
        if exc_type is None and self.cleanup_errors:
            raise self.cleanup_errors[0]
        return False
