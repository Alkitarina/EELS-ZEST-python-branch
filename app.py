"""EELS-ZEST local web workbench. Run through start_web.ps1 or uvicorn."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from threading import Event, Lock
from typing import Literal
import copy
import gzip
import json
import os
import time
import traceback
import uuid
import zipfile

# Process-local BLAS settings keep interactive jobs from oversubscribing this
# workstation. They do not change the objective or parallelize pixels.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.io import savemat
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from threadpoolctl import threadpool_limits

from zest import data, pipeline, maps
from zest.backend import Backend, Cancelled, hardware
from zest.numeric import clean_json
from zest.settings import defaults
from zest.storage import save_json, save_pixel, load_pixel, restore
from zest.parallel import DEFAULT_WORKERS, MAX_WORKERS
from zest.timing import PhaseTimer

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("ZEST_PARALLEL_RUNTIME", ROOT / "runtime")).resolve()
LEGACY_RUNTIME = (ROOT.parent / "eels-zest-web" / "runtime").resolve()
if RUNTIME == LEGACY_RUNTIME or LEGACY_RUNTIME in RUNTIME.parents:
    raise RuntimeError("并行版必须使用独立 runtime，不能写入原版运行目录")
for folder in ("datasets", "jobs", "uploads"):
    (RUNTIME / folder).mkdir(parents=True, exist_ok=True)
# Optional local data; no private dataset is distributed with this repository.
REFERENCE = Path(
    os.environ.get("ZEST_REFERENCE_PATH", ROOT / "reference" / "example_data.mat")
).expanduser().resolve()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zest-compute")
DATASETS, JOBS = {}, {}
GUARD = Lock()
CAPABILITIES = {}


class ImportRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class PreprocessRequest(BaseModel):
    dataset_id: str
    device: Literal["cpu", "gpu"] = "cpu"
    align: bool = True
    rank: int = Field(default=12, ge=0, le=512)
    smooth_points: int = Field(default=1, ge=1, le=101)


class FitRequest(BaseModel):
    dataset_id: str
    device: Literal["cpu", "gpu"] = "cpu"
    execution_mode: Literal["serial", "stripes"] = "stripes"
    parallel_workers: int = Field(default=DEFAULT_WORKERS, ge=1, le=MAX_WORKERS, strict=True)
    stripe_count: int = Field(default=0, ge=0, le=10000, strict=True)
    scope: Literal["pixel", "roi", "all"] = "pixel"
    pixel: tuple[int, int] = (0, 0)
    roi: tuple[int, int, int, int] | None = None
    stage: Literal["stage1", "both", "stage2"] = "both"
    source_job_id: str | None = None
    settings: dict = Field(default_factory=dict)


def public_dataset(d):
    e = d["energy"]
    return clean_json(
        dict(
            id=d["id"],
            **d["info"],
            energy_min=e[0],
            energy_max=e[-1],
            energy_step=np.median(np.diff(e)),
            pixels=np.prod(d["cube"].shape[:2]),
        )
    )


def register_dataset(cube, energy, info):
    uid = uuid.uuid4().hex[:12]
    d = dict(id=uid, cube=cube, energy=energy, info=info)
    np.savez_compressed(RUNTIME / "datasets" / f"{uid}.npz", data=cube, ene=energy)
    save_json(RUNTIME / "datasets" / f"{uid}.json", dict(id=uid, info=info))
    DATASETS[uid] = d
    return d


def get_dataset(uid):
    if uid in DATASETS:
        return DATASETS[uid]
    if not uid.isalnum():
        raise HTTPException(404, "数据集不存在")
    path = RUNTIME / "datasets" / f"{uid}.json"
    if not path.exists():
        raise HTTPException(404, "数据集不存在，请重新导入")
    meta = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as f:
        DATASETS[uid] = dict(id=uid, cube=f["data"], energy=f["ene"], info=meta["info"])
    return DATASETS[uid]


def get_job(uid):
    if uid not in JOBS:
        raise HTTPException(404, "任务不存在")
    return JOBS[uid]


def job_path(job):
    return RUNTIME / "jobs" / job["id"]


def public_job(job):
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    elapsed = (
        time.time() - job["started_at"]
        if job["status"] == "running"
        else job.get("elapsed_seconds", 0)
    )
    out["elapsed_seconds"] = elapsed
    out["execution"] = job.get("_state", {}).get("execution", job.get("execution", {}))
    if job.get("_timer") is not None:
        out["timings"] = job["_timer"].snapshot()
    out["download_ready"] = (job_path(job) / "result.zip").exists()
    return clean_json(out)


def persist_job(job):
    save_json(job_path(job) / "job.json", public_job(job))


def update_job(job, phase, done, total, message):
    job.update(phase=phase, done=int(done), total=int(total), message=message)
    now = time.time()
    if now - job.get("_last_log", 0) > 1 or done == total:
        job["events"] = (job["events"] + [dict(time=now, message=message)])[-80:]
        job["_last_log"] = now
    if now - job.get("_last_save", 0) > 3:
        persist_job(job)
        job["_last_save"] = now


def job_phase(job, key):
    timer = job.get("_timer")
    return timer.phase(key) if timer is not None else nullcontext()


def checkpoint(job, i, r, timing_phase="checkpoint"):
    if r is None:
        return
    coords = job.get("coords", [])
    if i < len(coords):
        y, x = coords[i]
        r.setdefault("pixel_y", y + 1)
        r.setdefault("pixel_x", x + 1)
    with job_phase(job, timing_phase):
        save_pixel(job_path(job) / "pixels" / f"{i:06d}.json.gz", r)
    job.setdefault("_completed_indices", set()).add(i)
    job["completed_pixels"] = len(job["_completed_indices"])


def pixel_result(job, y, x):
    coords = [tuple(p) for p in job.get("coords", [])]
    try:
        i = coords.index((y, x))
    except ValueError:
        return None
    state = job.get("_state", {})
    if i < len(state.get("results", [])):
        r = state["results"][i]
        if r is not None:
            return r
    path = job_path(job) / "pixels" / f"{i:06d}.json.gz"
    return load_pixel(path) if path.exists() else None


def completed_entries(state):
    return [(i, state["coords"][i], r)
            for i, r in enumerate(state.get("results", [])) if r is not None]


def finalize(job, state):
    timer = job.get("_timer") or PhaseTimer()
    job["_timer"] = timer
    with timer.phase("export_metadata"):
        entries = completed_entries(state)
        if not entries:
            return
        coordinates = [p for _, p, _ in entries]
        results = [r for _, _, r in entries]
        folder = job_path(job)
    job["message"] = "正在生成结果图…"
    with timer.phase("maps"):
        state["maps"] = maps.build(
            state["shape"], coordinates, results, state["settings"], state.get("hierarchy")
        )
    mp = state["maps"]
    job["message"] = "正在导出 MAT / NPZ 图…"
    with timer.phase("export_maps"):
        np.savez_compressed(folder / "maps.npz", **mp, coords=np.asarray(coordinates),
                            pixel_indices=np.asarray([i for i, _, _ in entries]))
        savemat(folder / "maps.mat", mp, do_compression=True, long_field_names=True)
    with timer.phase("export_metadata"):
        save_json(folder / "settings.json", state["settings"], scientific=True)
        save_json(folder / "hierarchy.json", state.get("hierarchy", {}), scientific=True)
        execution = state.get("execution", {})
        for stripe in execution.get("stripes", []):
            if stripe["status"] != "completed":
                stripe["status"] = job.get("_final_status", "interrupted")
    for done, (i, _, r) in enumerate(entries, 1):
        job["message"] = f"最终保存像素结果 · {done}/{len(entries)}"
        checkpoint(job, i, r, timing_phase="export_pixels")
    with timer.phase("export_metadata"):
        failures = [
            dict(y=r.get("pixel_y"), x=r.get("pixel_x"),
                 error=r.get("error", r.get("stage2", {}).get("error")))
            for r in results if "error" in r or r.get("stage2", {}).get("status") == "error"
        ]
        job.update(
            failed_pixels=len(failures), gpu_operations=state.get("gpu_operations", 0),
            valid_temperatures=int(np.count_nonzero(mp["T_valid_mask"])),
            inferred_temperatures=int(np.count_nonzero(mp["T_inferred_mask"])),
            map_names=list(mp),
            hierarchy_status=state.get("hierarchy", {}).get("status", "disabled"),
        )
        if len(failures) == len(results) and job.get("_final_status") == "completed":
            job.update(_final_status="failed", _final_message="所有像素均有计算阶段失败，请查看错误记录")
        elif failures and job.get("_final_status") == "completed":
            job.update(_final_status="completed_with_errors",
                       _final_message=f"计算完成，{len(failures)} 个像素存在错误")
    # Measure all payload compression before writing the final timing snapshot.
    # The last small metadata append/atomic commit is outside the measured span.
    job["message"] = "正在压缩结果 ZIP…"
    temp = folder / "result.writing.zip"
    with timer.phase("archive"):
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("settings.json", "hierarchy.json", "maps.npz", "maps.mat"):
                z.write(folder / name, name)
            for path in sorted((folder / "pixels").glob("*.json.gz")):
                z.write(path, "pixels/" + path.name)
    job["timings"] = timer.finish()
    summary_job = public_job(job)
    summary_job.update(
        status=job.get("_final_status", job["status"]),
        phase=job.get("_final_status", job["status"]),
        message=job.get("_final_message", job["message"]), download_ready=True,
    )
    save_json(folder / "summary.json", dict(
        job=summary_job, failures=failures,
        numerical_note="Python 移植验证版：保留模型与目标函数，SciPy/CUDA 求解轨迹不同，温度结果尚不能视为与 MATLAB 等价。GPU 为混合执行，并非全流程 CUDA。",
    ))
    save_json(folder / "execution.json", execution)
    save_json(folder / "timings.json", job["timings"])
    with zipfile.ZipFile(temp, "a", zipfile.ZIP_DEFLATED) as z:
        for name in ("summary.json", "execution.json", "timings.json"):
            z.write(folder / name, name)
    temp.replace(folder / "result.zip")


def execute(job, request):
    job.update(status="running", started_at=time.time())
    timer = job["_timer"] = PhaseTimer()
    persist_job(job)
    state = job["_state"]
    b = None
    try:
        with timer.phase("prepare"):
            d = get_dataset(request.dataset_id)
            b = Backend(request.device, job["_cancel"].is_set)
        with threadpool_limits(limits=2):
            if job["kind"] == "preprocess":
                with timer.phase("preprocess"):
                    cube, history = data.preprocess(
                        d["cube"],
                        d["energy"],
                        b,
                        request.align,
                        request.rank,
                        request.smooth_points,
                        lambda msg: update_job(job, "preprocess", 0, 1, msg),
                        timer=timer,
                    )
                b.check()
                info = dict(d["info"])
                info["history"] = info.get("history", []) + [history]
                info["name"] = d["info"]["name"] + " · 已预处理"
                info["parent_id"] = d["id"]
                with timer.phase("dataset_save"):
                    output = register_dataset(cube, d["energy"], info)
                job.update(
                    output_dataset_id=output["id"], gpu_operations=b.gpu_operations
                )
            else:
                with timer.phase("prepare"):
                    if request.stage == "stage2":
                        old = get_job(request.source_job_id)
                        coords = [tuple(p) for p in old["coords"]]
                        state["results"] = [
                            copy.deepcopy(pixel_result(old, *p)) for p in coords
                        ]
                        if any(r is None for r in state["results"]):
                            raise ValueError("源任务尚有未完成的像素")
                        oldstate = old.get("_state", {})
                        saved = restore(
                            json.loads(
                                (job_path(old) / "settings.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                        )
                        s = defaults({**saved, **request.settings, "run_stage2": True})
                        hp = job_path(old) / "hierarchy.json"
                        state["hierarchy"] = copy.deepcopy(
                            oldstate.get("hierarchy")
                            or (
                                restore(json.loads(hp.read_text(encoding="utf-8")))
                                if hp.exists()
                                else {}
                            )
                        )
                    else:
                        coords = pipeline.coordinates(
                            d["cube"].shape, request.scope, request.pixel, request.roi
                        )
                        s = defaults(
                            {**request.settings, "run_stage2": request.stage == "both"}
                        )
                    job["coords"] = coords
                    job["total_pixels"] = len(coords)
                    persist_job(job)
                pipeline.run(
                    d["cube"],
                    d["energy"],
                    coords,
                    s,
                    b,
                    state,
                    lambda phase, done, total, msg: update_job(
                        job, phase, done, total, msg
                    ),
                    lambda i, r: checkpoint(job, i, r),
                    request.stage == "stage2",
                    execution_mode=request.execution_mode,
                    workers=request.parallel_workers,
                    stripe_count=request.stripe_count,
                    timer=timer,
                )
        job.update(
            status="completed",
            message="计算完成",
            done=job.get("total", 1),
            phase="completed",
        )
    except Cancelled:
        job.update(
            status="cancelled",
            message="任务已停止，已完成像素已保存",
            phase="cancelled",
        )
    except Exception as exc:
        job.update(status="failed", message=str(exc), error=str(exc), phase="failed")
        (job_path(job) / "error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    finally:
        job["elapsed_seconds"] = time.time() - job["started_at"]
        if b is not None:
            state["gpu_operations"] = b.gpu_operations
        outcome = job["status"]
        message = job["message"]
        job["_final_status"] = outcome
        job["_final_message"] = message
        for stripe in state.get("execution", {}).get("stripes", []):
            if stripe["status"] != "completed":
                stripe["status"] = outcome
        try:
            if job["kind"] == "fit":
                job.update(
                    status="running",
                    phase="export",
                    message="正在保存完整结果与生成下载文件…",
                )
                persist_job(job)
                finalize(job, state)
        except Exception as exc:
            job.update(
                _final_status="failed",
                _final_message="计算结果保留在 runtime，导出失败",
                error="导出失败: " + str(exc),
                message="计算结果保留在 runtime，导出失败",
            )
            (job_path(job) / "error.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        job["timings"] = timer.finish()
        save_json(job_path(job) / "timings.json", job["timings"])
        job["status"] = job.pop("_final_status")
        job["phase"] = job["status"]
        job["message"] = job.pop("_final_message", message)
        job["elapsed_seconds"] = time.time() - job["started_at"]
        persist_job(job)


def submit(kind, request):
    get_dataset(request.dataset_id)
    if request.device == "gpu" and not CAPABILITIES.get("gpu"):
        raise HTTPException(400, CAPABILITIES.get("gpu_error") or "CUDA GPU 不可用")
    with GUARD:
        if any(j["status"] in ("queued", "running") for j in JOBS.values()):
            raise HTTPException(409, "已有计算正在进行，请等待或停止当前任务")
        uid = uuid.uuid4().hex[:12]
        job = dict(
            id=uid,
            kind=kind,
            dataset_id=request.dataset_id,
            device=request.device,
            status="queued",
            phase="queued",
            message="准备计算",
            created_at=time.time(),
            started_at=time.time(),
            done=0,
            total=1,
            completed_pixels=0,
            events=[],
            request=request.model_dump(),
            _cancel=Event(),
            _state={},
        )
        JOBS[uid] = job
        job_path(job).mkdir(parents=True, exist_ok=True)
        persist_job(job)
        POOL.submit(execute, job, request)
    return public_job(job)


def restore_job(path):
    """Reconcile an unclean exit with checkpoints, never resume silently."""
    job = json.loads(path.read_text(encoding="utf-8"))
    if job["status"] in ("running", "queued"):
        job.update(
            interrupted_phase=job.get("phase"), status="interrupted",
            phase="interrupted", message="服务已重启，上次任务已中断；已保存像素可查看，未完成计算不会自动续跑",
        )
        if job["kind"] == "fit":
            job["completed_pixels"] = sum(
                1 for p in (path.parent / "pixels").glob("*.json.gz")
                if p.name.removesuffix(".json.gz").isdigit()
            )
        if job.get("timings"):
            job["timings"].update(active_phase=None, complete=False)
        for stripe in job.get("execution", {}).get("stripes", []):
            if stripe["status"] != "completed":
                stripe["status"] = "interrupted"
        # Persist once, so future launches do not report the task as running.
        save_json(path, job)
    job.update(_cancel=Event(), _state={})
    return job


@asynccontextmanager
async def lifespan(app):
    CAPABILITIES.update(hardware())
    for path in (RUNTIME / "jobs").glob("*/job.json"):
        try:
            job = restore_job(path)
            JOBS[job["id"]] = job
        except (ValueError, KeyError):
            continue
    yield
    for j in JOBS.values():
        j["_cancel"].set()
    POOL.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="EELS-ZEST Parallel Workbench", version="0.3.1", lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
)


@app.middleware("http")
async def local_origins(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
        from urllib.parse import urlsplit

        if urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "只允许本机工作台发起操作"}, status_code=403)
    return await call_next(request)


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.get("/api/health")
def health():
    return dict(app="eels-zest-parallel", status="ok", version="0.3.1", hardware=CAPABILITIES)


@app.get("/api/config")
def config():
    return clean_json(
        dict(
            hardware=CAPABILITIES,
            reference_available=REFERENCE.is_file(),
            reference_path=str(REFERENCE),
            defaults=defaults(),
            version="0.3.1",
            execution_modes=["stripes", "serial"],
            parallel_workers=DEFAULT_WORKERS,
            max_parallel_workers=MAX_WORKERS,
            logical_cpus=os.cpu_count() or 1,
            stripe_count=0,
            auto_stripes_per_worker=3,
            runtime=str(RUNTIME),
        )
    )


@app.get("/api/datasets")
def list_datasets():
    result = []
    for p in sorted(
        (RUNTIME / "datasets").glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        meta = json.loads(p.read_text(encoding="utf-8"))
        result.append(dict(id=meta["id"], **meta["info"]))
    return result


@app.post("/api/import")
def import_path(body: ImportRequest):
    cube, energy, info = data.load(body.path.strip().strip('"'))
    return public_dataset(register_dataset(cube, energy, info))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".mat", ".npz"):
        raise HTTPException(400, "请选择 MAT 或 NPZ 文件")
    dest = RUNTIME / "uploads" / (uuid.uuid4().hex + suffix)
    size = 0
    try:
        with dest.open("wb") as f:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > 2 * 1024**3:
                    raise HTTPException(413, "单文件上限为 2 GiB")
                f.write(block)
        # Heavy import runs outside the async event loop.
        from starlette.concurrency import run_in_threadpool

        cube, e, info = await run_in_threadpool(data.load, dest)
        info.update(
            name=Path(file.filename or "upload").name,
            source="浏览器上传: " + Path(file.filename or "").name,
        )
        d = await run_in_threadpool(register_dataset, cube, e, info)
        return public_dataset(d)
    finally:
        await file.close()
        if dest.exists():
            dest.unlink()


@app.get("/api/datasets/{uid}")
def dataset_info(uid: str):
    return public_dataset(get_dataset(uid))


@app.get("/api/datasets/{uid}/preview")
def preview(uid: str, y: int = 0, x: int = 0):
    d = get_dataset(uid)
    pipeline.coordinates(d["cube"].shape, "pixel", (y, x))
    intensity = np.nansum(d["cube"], axis=-1)
    return clean_json(
        dict(
            dataset=public_dataset(d),
            intensity=intensity,
            energy=d["energy"],
            spectrum=d["cube"][y, x],
            pixel=[y, x],
        )
    )


@app.post("/api/preprocess")
def preprocess(body: PreprocessRequest):
    d = get_dataset(body.dataset_id)
    if body.rank > min(d["cube"].shape[0] * d["cube"].shape[1], d["cube"].shape[2]):
        raise ValueError("SVD rank 不能超过矩阵最小维度")
    return submit("preprocess", body)


def validate_settings(changes, energy):
    unknown = set(changes) - set(defaults())
    if unknown:
        raise ValueError("未知参数: " + ", ".join(sorted(unknown)))
    # Expose only transcribed solver paths; unsupported MATLAB-only modes must
    # never silently masquerade as an implemented option.
    for key in ("nonlinear_inner_solver", "inner_solver", "stage2_scan_inner_solver"):
        if key in changes and changes[key] not in (
            "auto",
            "lsqlin",
            "lsqnonneg",
            "quadprog",
        ):
            raise ValueError(
                f"{key}: 本版支持 auto / lsqlin / lsqnonneg / quadprog 的对应凸问题"
            )
    s = defaults(changes)
    low, high = s["crop_range"]
    if (
        not np.isfinite([low, high]).all()
        or low >= high
        or np.count_nonzero((energy >= low) & (energy <= high)) < 12
    ):
        raise ValueError("能量裁剪范围无效，至少需要 12 个通道")
    if not (1 <= s["num_zlp_components"] <= 12):
        raise ValueError("ZLP 分量数应在 1–12 之间")
    grid = np.asarray(s["stage2_T_grid"], dtype=float).ravel()
    if (
        len(grid) < 3
        or len(grid) > 1000
        or np.any(~np.isfinite(grid))
        or np.any(grid <= 0)
    ):
        raise ValueError("温度网格需要 3–1000 个正数")
    return s


@app.post("/api/fit")
def fit(body: FitRequest):
    d = get_dataset(body.dataset_id)
    validate_settings(body.settings, d["energy"])
    pipeline.coordinates(d["cube"].shape, body.scope, body.pixel, body.roi)
    if body.execution_mode == "stripes" and body.device != "cpu":
        raise ValueError("竖条并行使用 CPU；GPU 混合计算请选择串行对照模式")
    if body.stage == "stage2":
        if not body.source_job_id:
            raise ValueError("请先选择一个已完成的 Stage 1 任务")
        old = get_job(body.source_job_id)
        if old["dataset_id"] != body.dataset_id:
            raise ValueError("Stage 1 任务来自其他数据集")
        if old["status"] not in ("completed", "completed_with_errors"):
            raise ValueError("请等待 Stage 1 任务完成")
        if old.get("request", {}).get("stage") != "stage1":
            raise ValueError("请选一个仅运行 Stage 1 的源任务")
    return submit("fit", body)


@app.get("/api/jobs")
def jobs():
    return [
        public_job(j)
        for j in sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)
    ]


@app.get("/api/jobs/{uid}")
def job_status(uid: str):
    return public_job(get_job(uid))


@app.post("/api/jobs/{uid}/cancel")
def cancel(uid: str):
    j = get_job(uid)
    if j["status"] in ("running", "queued"):
        j["_cancel"].set()
        j["message"] = "正在停止，将保存已完成像素…"
    return public_job(j)


@app.get("/api/jobs/{uid}/maps")
def job_maps(uid: str):
    j = get_job(uid)
    path = job_path(j) / "maps.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            return clean_json({k: f[k] for k in f.files if k not in ("coords", "pixel_indices")})
    state = j.get("_state", {})
    entries = completed_entries(state)
    if entries:
        return clean_json(
            maps.build(
                state["shape"],
                [p for _, p, _ in entries],
                [r for _, _, r in entries],
                state["settings"],
                state.get("hierarchy"),
            )
        )
    return {}


@app.get("/api/jobs/{uid}/pixel")
def job_pixel(uid: str, y: int, x: int):
    r = pixel_result(get_job(uid), y, x)
    if r is None:
        return dict(available=False)
    if "error" in r:
        return dict(available=True, error=r["error"])
    st = r.get("stage2", {})
    a = st.get("attention", {})
    return clean_json(
        dict(
            available=True,
            energy=r["ene_corrected"],
            raw=r["raw_data_norm"],
            model=r["fitted_spectrum_norm"],
            zlp=r["zlp_only"] / r["norm_val"],
            vibrational=r["vib_only"] / r["norm_val"],
            residual=r["residual_norm"],
            EM=r["EM_final"],
            PM_stokes=r["PM_stokes"],
            PM_antistokes=r["PM_antistokes"],
            metrics=r["metrics"],
            T=r.get("T_fit"),
            T_uncertainty=r.get("T_uncertainty"),
            reliability=st.get("reliability_score"),
            T_grid=st.get("T_grid", []),
            cost_grid=st.get("cost_grid", []),
            cost_smooth=st.get("cost_grid_smooth", []),
            attention=a.get("feature_score", []),
            alpha=st.get("fit_alpha", []),
            hard_idx=st.get("hard_idx", []),
            stage2_status=st.get("status", "not_run"),
            stage2_error=st.get("error"),
            iterations=r.get("iter_history", []),
            hierarchy=r.get("stage1_hierarchy", {}),
            execution=r.get("execution", {}),
        )
    )


@app.get("/api/jobs/{uid}/download")
def download(uid: str):
    path = job_path(get_job(uid)) / "result.zip"
    if not path.exists():
        raise HTTPException(404, "结果尚未生成")
    return FileResponse(
        path, filename=f"EELS-ZEST-{uid}.zip", media_type="application/zip"
    )


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
