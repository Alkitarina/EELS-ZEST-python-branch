"""Scheduling, numerical parity, global neighbours, and partial-result safety."""

import copy
import multiprocessing
from threading import Event
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from zest import hierarchy, pipeline, stage1
from zest.backend import Backend, Cancelled
from zest.parallel import ComputePool, vertical_stripes
from zest.partition import partition_blocks
from zest.settings import defaults
from zest.timing import PhaseTimer


@pytest.fixture
def small_cube(reference):
    # A small real-spectrum fixture, not a performance or temperature benchmark.
    e = reference["ene"][::8]
    y = reference["y"][::8]
    cube = np.tile(y, (2, 12, 1))
    for x in range(12):
        cube[:, x] *= 1 + x * 0.002
    s = defaults(dict(
        stage1_fixed_EM=[12, 24, 48, 72], num_zlp_components=2,
        stage1_hierarchical_enabled=False,
        datacube_stage1_auto_sparse_pm_enabled=False,
        stage1_strict_polish_enabled=False, run_stage2=False,
        stage1_rounds=[
            dict(name="fixture_coarse", em_step_meV=8,
                 lambda_be_soft=0.0015, max_nonlinear_iter=2),
            dict(name="fixture_fine", em_step_meV=8,
                 lambda_be_soft=0.0015, max_nonlinear_iter=2),
        ],
    ))
    return cube, e, s


def test_vertical_stripes_cover_whole_columns_and_adjacent_snake():
    coords = pipeline.coordinates((4, 17, 30), "roi", roi=(1, 2, 3, 16))
    stripes = vertical_stripes(coords)
    assert len(stripes) == 6
    indices = [i for strip in stripes for i in strip["indices"]]
    assert sorted(indices) == list(range(len(coords)))
    assert max(len(st["columns"]) for st in stripes) - min(
        len(st["columns"]) for st in stripes) <= 1
    for strip in stripes:
        path = [coords[i] for i in strip["indices"]]
        assert path[0][0] == 1
        for a, b in zip(path, path[1:]):
            assert sum(abs(v - w) for v, w in zip(a, b)) == 1
        for x in strip["columns"]:
            assert {y for y, xx in path if xx == x} == {1, 2, 3}
    assert len(vertical_stripes([(y, 8) for y in range(10)])) == 1
    with pytest.raises(ValueError):
        vertical_stripes([(0, 0), (0, 0)])


@pytest.mark.parametrize("mode,workers,target", [
    ("vertical", 6, 6), ("vertical", 2, 6), ("balanced", 2, 5),
    ("horizontal", 2, 2), ("grid", 2, 6),
])
def test_process_stripes_keep_solver_and_warm_start(small_cube, mode, workers, target):
    cube, e, s = small_cube
    coords = pipeline.coordinates(cube.shape, "all")
    state, saved = {}, {}
    pipeline.run(cube, e, coords, s, Backend(), state,
                 checkpoint=lambda i, r: saved.update({i: r}), execution_mode="stripes",
                 workers=workers, stripe_count=target,
                 partition_mode=mode, grid_rows=2, grid_columns=3)
    blocks = partition_blocks(coords, target, mode, 2, 3)
    assert state["execution"]["workers"] == workers
    assert len(state["execution"]["worker_pids"]) == workers
    assert state["execution"]["stripe_count"] == len(blocks)
    assert state["execution"]["partition_mode"] == mode
    assert state["execution"]["partition_boundaries"]
    assert state["execution"]["scheduling"] == "dynamic_queue"
    if workers < 6:
        assert len({st["pid"] for st in state["execution"]["stripes"]}) < 6
    assert state["timings"]["complete"] and state["timings"]["active_phase"] is None
    times = {p["key"]: p["seconds"] for p in state["timings"]["phases"]}
    assert all(times[key] > 0 for key in ("pool_setup", "stage1", "pool_shutdown", "maps"))
    assert set(saved) == set(range(len(coords)))
    assert all(st["done"] == st["total"] for st in state["execution"]["stripes"])
    for strip in blocks:
        previous = None
        for step, i in enumerate(strip["indices"]):
            r = state["results"][i]
            assert "error" not in r, r
            assert (r["pixel_y"] - 1, r["pixel_x"] - 1) == coords[i]
            assert r["execution"]["stripe_id"] == strip["id"] + 1
            assert r["execution"]["warm_start_from"] == previous
            assert len(r["iter_history"]) == (2 if step == 0 else 1)
            previous = [v + 1 for v in coords[i]]
    # Same sequence, same initialization, same one-thread solver: verify the
    # first entire snake chain against direct calls, including its upward leg.
    with threadpool_limits(1):
        previous = None
        for step, i in enumerate(blocks[0]["indices"]):
            ps = dict(s, stage1_warm_start_skip_coarse=step > 0)
            y, x = coords[i]
            expected = stage1.solve(e, cube[y, x], ps, Backend(), previous=previous)
            actual = state["results"][i]
            np.testing.assert_allclose(actual["theta"], expected["theta"], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(actual["PM_antistokes"], expected["PM_antistokes"], rtol=1e-9, atol=1e-10)
            previous = expected["theta"]


def test_cancel_parallel_keeps_completed_pixels_and_reaps_workers(small_cube):
    cube, e, s = small_cube
    cube = np.tile(cube, (3, 1, 1))
    coords = pipeline.coordinates(cube.shape, "all")
    stop = Event()
    state, saved = {}, {}
    children_before = {p.pid for p in multiprocessing.active_children()}

    def save(i, result):
        saved[i] = result
        stop.set()

    with pytest.raises(Cancelled):
        pipeline.run(cube, e, coords, s, Backend(cancel=stop.is_set), state,
                     checkpoint=save, execution_mode="stripes", workers=2, stripe_count=12)
    actual = {i for i, r in enumerate(state["results"]) if r is not None}
    assert actual == set(saved) and 0 < len(actual) < len(coords)
    for i, result in saved.items():
        assert (result["pixel_y"] - 1, result["pixel_x"] - 1) == coords[i]
    assert {p.pid for p in multiprocessing.active_children()} <= children_before


def test_parallel_hierarchy_matches_global_serial_with_cross_stripe_neighbours(reference):
    r = reference["r"]
    coords = pipeline.coordinates((5, 4, len(r["raw_data"])), "all")
    cube = np.tile(r["raw_data"], (5, 4, 1))
    s = defaults({**reference["s"], "stage1_hierarchical_template_max_iter": 1,
                  "stage1_hierarchical_spatial_max_iter": 1,
                  "stage1_spatial_zlp_iterations": 1})
    with threadpool_limits(1):
        expected, eh = hierarchy.run(cube, r["ene_raw"], coords,
                                      [copy.deepcopy(r) for _ in coords], s, Backend())
    recorded = []

    class Capture:
        def __init__(self, pool):
            self.pool = pool

        def refits(self, tasks):
            tasks = list(tasks)
            recorded.extend(tasks)
            return self.pool.refits(tasks)

    timer = PhaseTimer()
    with threadpool_limits(1), ComputePool(cube, r["ene_raw"], 2, Backend()) as pool, timer.phase("hierarchy"):
        actual, ah = hierarchy.run(cube, r["ene_raw"], coords,
                                    [copy.deepcopy(r) for _ in coords], s, Backend(),
                                    executor=Capture(pool), timer=timer)
    timing = timer.finish()
    assert {p["key"] for p in timing["phases"]} >= {"hierarchy", "template_refit", "spatial_refit"}
    assert timing["accounted_seconds"] <= timing["wall_seconds"]
    assert ah["status"] == eh["status"] == "ok"
    assert [(h["phase"], h.get("accepted")) for h in ah["history"]] == [
        (h["phase"], h.get("accepted")) for h in eh["history"]]
    for a, expected_r in zip(actual, expected):
        np.testing.assert_allclose(a["theta"], expected_r["theta"], rtol=1e-9, atol=1e-10)
    # With two stripes, X=2 (zero-based) is on a stripe boundary. Its full
    # eight-neighbour context must still include three pixels from X=1.
    contexts = [t[3] for t in recorded if t[1] == (2, 2) and t[-1].startswith("spatial_")]
    assert contexts and all(len(ctx["neighbor_phi"]) == 8 for ctx in contexts)


def test_partial_out_of_order_export_and_reload(client, reference):
    import app
    from pathlib import Path
    import gzip
    import json
    import uuid

    uid = uuid.uuid4().hex[:12]
    coords = pipeline.coordinates((2, 6, 10), "all")
    results = [None] * len(coords)
    state = dict(coords=coords, results=results, shape=(2, 6), settings=defaults(),
                 hierarchy={}, execution=dict(mode="stripes", workers=6, stripes=[]))
    job = dict(id=uid, status="cancelled", phase="cancelled", kind="fit",
               message="cancelled", coords=coords, completed_pixels=0,
               created_at=0, _cancel=Event(), _state=state, _final_status="cancelled")
    app.JOBS[uid] = job
    for i in (8, 3):
        r = copy.deepcopy(reference["r"])
        results[i] = r
        app.checkpoint(job, i, r)
    app.checkpoint(job, 8, results[8])
    assert job["completed_pixels"] == 2
    app.finalize(job, state)
    assert job["storage"]["unchanged_exports"] == 2
    path = app.job_path(job)
    with np.load(path / "maps.npz") as archive:
        np.testing.assert_array_equal(archive["pixel_indices"], [3, 8])
        np.testing.assert_array_equal(archive["coords"], [coords[3], coords[8]])
        assert np.count_nonzero(archive["fit_status"]) == 2
    assert {p.name for p in (path / "pixels").glob("*.json.gz")} == {
        "000003.json.gz", "000008.json.gz"}
    job["_state"] = {}
    assert app.pixel_result(job, *coords[0]) is None
    assert app.pixel_result(job, *coords[8])["pixel_x"] == coords[8][1] + 1
    with gzip.open(path / "pixels/000008.json.gz", "rt", encoding="utf-8") as stream:
        assert json.load(stream)["pixel_y"] == coords[8][0] + 1
    assert (Path(path) / "execution.json").exists()
    app.JOBS.pop(uid)


def test_parallel_api_stage2_reuse_matches_serial_and_rejects_gpu(client, small_cube, tmp_path):
    import io
    import json
    import time
    import zipfile
    import app

    def finished(uid):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            response = client.get(f"/api/jobs/{uid}")
            assert response.status_code == 200, response.text
            job = response.json()
            if job["status"] not in ("queued", "running"):
                assert job["status"] == "completed", job
                return job
            time.sleep(0.1)
        raise AssertionError(f"Parallel API task did not finish: {job}")

    cube, e, settings = small_cube
    source = tmp_path / "six-column-api-fixture.npz"
    np.savez(source, data=cube[:1, :6], ene=e)
    dataset = client.post("/api/import", json={"path": str(source)}).json()
    overrides = {k: settings[k] for k in (
        "stage1_fixed_EM", "num_zlp_components", "stage1_hierarchical_enabled",
        "datacube_stage1_auto_sparse_pm_enabled", "stage1_strict_polish_enabled", "stage1_rounds")}
    overrides.update(stage2_T_grid=[100, 300, 500, 700], stage2_temperature_refine_enabled=False)
    bad = client.post("/api/fit", json=dict(dataset_id=dataset["id"], device="gpu",
                                          execution_mode="stripes", settings=overrides))
    assert bad.status_code == 400 and "CPU" in bad.json()["detail"]
    response = client.post("/api/fit", json=dict(dataset_id=dataset["id"], scope="all",
                                                stage="stage1", settings=overrides,
                                                parallel_workers=2, stripe_count=0))
    assert response.status_code == 200, response.text
    initial = finished(response.json()["id"])
    assert initial["completed_pixels"] == initial["total_pixels"] == 6
    assert initial["execution"]["workers"] == 2
    assert initial["execution"]["stripe_count"] == 2
    assert initial["execution"]["requested_stripes"] == 0
    with zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{initial['id']}/download").content)) as archive:
        execution = json.loads(archive.read("execution.json"))
        assert execution["traversal"] == "balanced_vertical_snake"
        assert len(execution["stripes"]) == 2
        assert execution["partition_boundaries"]
        timing = json.loads(archive.read("timings.json"))
        assert timing == initial["timings"]
        assert timing["complete"]
        assert {p["key"] for p in timing["phases"]} >= {
            "prepare", "pool_setup", "stage1", "checkpoint", "pool_shutdown",
            "maps", "export_maps", "export_pixels", "archive"}
    # Re-open Stage 1 from persisted files, then use identical inputs for both
    # Stage 2 schedulers. This comparison does not change warm-start order.
    app.JOBS[initial["id"]]["_state"] = {}
    temperatures = []
    for mode in ("stripes", "serial"):
        response = client.post("/api/fit", json=dict(
            dataset_id=dataset["id"], scope="all", stage="stage2",
            source_job_id=initial["id"], execution_mode=mode, settings=overrides,
            parallel_workers=3, stripe_count=20))
        assert response.status_code == 200, response.text
        job = finished(response.json()["id"])
        assert job["completed_pixels"] == 6
        assert job["execution"]["workers"] == (3 if mode == "stripes" else 1)
        assert any(p["key"] == "stage2" and p["seconds"] > 0 for p in job["timings"]["phases"])
        values = []
        for x in range(6):
            pixel = client.get(f"/api/jobs/{job['id']}/pixel?y=0&x={x}").json()
            assert pixel["available"] and pixel["stage2_status"] == "ok", pixel
            values.append(pixel["T"])
        temperatures.append(values)
    np.testing.assert_array_equal(*temperatures)
