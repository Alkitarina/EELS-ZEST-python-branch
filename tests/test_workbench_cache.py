import copy
import time
from threading import Event
from unittest.mock import Mock
import numpy as np
import pytest
from zest.cache import BoundedCache
from zest.settings import defaults


@pytest.fixture
def workbench(client, monkeypatch, tmp_path):
    import app

    monkeypatch.setattr(app, "RUNTIME", tmp_path)
    monkeypatch.setattr(app, "JOBS", {})
    monkeypatch.setattr(app, "DATASETS", BoundedCache(
        1, 4096, lambda d: d["cube"].nbytes + d["energy"].nbytes))
    monkeypatch.setattr(app, "INTENSITIES", BoundedCache(2, 256, lambda a: a.nbytes))
    for name in ("datasets", "jobs", "uploads"):
        (tmp_path / name).mkdir()
    return app


def dataset(app, value=1, parent=None):
    cube = np.full((1, 2, 20), value, dtype=float)
    energy = np.linspace(-10, 10, 20)
    info = dict(name="synthetic.npz", source="synthetic", shape=list(cube.shape), history=[])
    if parent:
        info.update(parent_id=parent, history=[dict(svd_rank=1)])
    return app.register_dataset(cube, energy, info)


def task(app, dataset_id, scope="pixel"):
    request = app.FitRequest(dataset_id=dataset_id, stage="stage1", scope=scope,
                             execution_mode="serial", settings=dict(crop_range=[-10, 10],
                                 datacube_stage1_auto_sparse_pm_enabled=False,
                                 stage1_hierarchical_enabled=False))
    job = dict(id="syntheticjob", dataset_id=dataset_id, kind="fit", device="cpu",
               status="queued", phase="queued", created_at=time.time(), request=request.model_dump(),
               done=0, total=1, events=[], _cancel=Event(), _state={})
    app.JOBS[job["id"]] = job
    app.job_path(job).mkdir()
    return job, request


@pytest.fixture
def solved_pixel(monkeypatch):
    from zest import stage1

    result = dict(theta=np.ones(1), metrics=dict(R2=1., NRMSE=0.),
                  settings=defaults(), stage2=dict(status="not_run"))
    monkeypatch.setattr(stage1, "solve", lambda *a, **k: copy.deepcopy(result))
    return result


def test_evicted_dataset_reloads_without_losing_original_or_preprocessing_history(workbench):
    app = workbench
    original = dataset(app)
    processed = dataset(app, value=2, parent=original["id"])
    assert len(app.DATASETS) == 1
    reloaded = app.get_dataset(original["id"])
    assert reloaded is not original
    np.testing.assert_array_equal(reloaded["cube"], original["cube"])
    assert len(app.DATASETS) == 1
    restored = app.get_dataset(processed["id"])
    assert restored["info"]["parent_id"] == original["id"]
    assert restored["info"]["history"] == [dict(svd_rank=1)]
    assert len(list((app.RUNTIME / "datasets").glob("*.npz"))) == 2


def test_intensity_is_reused_across_pixels_and_reimports_have_distinct_caches(workbench, monkeypatch):
    app = workbench
    # Even a cube too large for the data cache can retain its small intensity map.
    monkeypatch.setattr(app, "DATASETS", BoundedCache(1, 16, lambda d: d["cube"].nbytes))
    first = dataset(app)
    sum_cube = Mock(wraps=np.nansum)
    monkeypatch.setattr(app.np, "nansum", sum_cube)
    left = app.preview(first["id"], 0, 0)
    right = app.preview(first["id"], 0, 1)
    assert left["intensity"] == right["intensity"] == [[20., 20.]]
    assert sum_cube.call_count == 1 and len(app.DATASETS) == 0
    second = dataset(app, value=2, parent=first["id"])
    assert app.preview(second["id"], 0, 0)["intensity"] == [[40., 40.]]
    third = dataset(app, value=3)
    assert app.preview(third["id"], 0, 0)["intensity"] == [[60., 60.]]
    assert len(app.INTENSITIES) == 2
    app.preview(first["id"], 0, 0)  # evicted derived data is safely recomputed
    assert sum_cube.call_count == 4


def test_complete_job_builds_maps_once_releases_results_and_reloads_from_disk(workbench, solved_pixel, monkeypatch):
    app = workbench
    d = dataset(app)
    job, request = task(app, d["id"])
    build = Mock(wraps=app.maps.build)
    monkeypatch.setattr(app.maps, "build", build)
    app.execute(job, request)
    assert job["status"] == "completed"
    assert build.call_count == 1
    assert job["_state"] == {} and not job["_completed_indices"]
    assert app.public_job(job)["execution"]["mode"] == "serial"
    np.testing.assert_array_equal(app.pixel_result(job, 0, 0)["theta"], solved_pixel["theta"])
    assert app.job_maps(job["id"])["fit_status"] == [[1, 0]]
    assert app.public_job(job)["download_ready"]
    assert app.public_job(job)["timings"]["complete"]


def test_cancelled_partial_job_builds_missing_maps_and_releases_only_after_export(workbench, solved_pixel, monkeypatch):
    app = workbench
    d = dataset(app)
    job, request = task(app, d["id"], scope="all")
    checkpoint = app.checkpoint

    def save_then_cancel(*args, **kwargs):
        checkpoint(*args, **kwargs)
        job["_cancel"].set()

    monkeypatch.setattr(app, "checkpoint", save_then_cancel)
    build = Mock(wraps=app.maps.build)
    monkeypatch.setattr(app.maps, "build", build)
    app.execute(job, request)
    assert job["status"] == "cancelled" and job["completed_pixels"] == 1
    assert build.call_count == 1 and job["_state"] == {}
    assert app.pixel_result(job, 0, 0) is not None
    assert app.pixel_result(job, 0, 1) is None
    assert app.job_maps(job["id"])["fit_status"] == [[1, 0]]


def test_failed_export_keeps_results_in_memory(workbench, solved_pixel, monkeypatch):
    app = workbench
    d = dataset(app)
    job, request = task(app, d["id"])
    monkeypatch.setattr(app.np, "savez_compressed", Mock(side_effect=OSError("export unavailable")))
    app.execute(job, request)
    assert job["status"] == "failed"
    assert job["_state"]["results"]
    assert not app.public_job(job)["download_ready"]
    assert app.pixel_result(job, 0, 0) is job["_state"]["results"][0]


def test_maps_ready_in_memory_are_reused_before_export(workbench, monkeypatch):
    app = workbench
    app.JOBS["ready"] = dict(id="ready", _state=dict(maps=dict(R2=np.ones((1, 2)))))
    monkeypatch.setattr(app.maps, "build", Mock(side_effect=AssertionError("unexpected rebuild")))
    assert app.job_maps("ready") == dict(R2=[[1., 1.]])


def test_oversized_dataset_is_loaded_once_and_owned_by_queued_task(workbench, solved_pixel, monkeypatch):
    from types import SimpleNamespace

    app = workbench
    monkeypatch.setattr(app, "DATASETS", BoundedCache(1, 16, lambda d: d["cube"].nbytes))
    d = dataset(app)
    assert len(app.DATASETS) == 0
    load = Mock(wraps=app.get_dataset)
    monkeypatch.setattr(app, "get_dataset", load)
    pending = []
    monkeypatch.setattr(app, "POOL", SimpleNamespace(submit=lambda fn, *args: pending.append((fn, args))))
    response = app.fit(app.FitRequest(dataset_id=d["id"], stage="stage1",
                                     settings=dict(crop_range=[-10, 10])))
    assert response["status"] == "queued" and load.call_count == 1
    dataset(app, value=2)  # newer data does not displace the queued task's input
    fn, args = pending.pop()
    fn(*args)
    job = app.JOBS[response["id"]]
    assert load.call_count == 1
    assert job["status"] == "completed" and job["_state"] == {}
