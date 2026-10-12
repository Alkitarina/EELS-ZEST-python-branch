import io
import time
import zipfile
import json
import pytest
import numpy as np
from scipy.io import loadmat


def wait_job(client, uid, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        j = client.get("/api/jobs/" + uid).json()
        if j["status"] not in ("queued", "running"):
            # final archive is generated after the compute phase changes.
            if (
                j["kind"] == "preprocess"
                or j["download_ready"]
                or j["status"] in ("failed", "cancelled")
            ):
                return j
        time.sleep(0.15)
    raise AssertionError("job did not finish: " + str(j))


def test_import_preprocess_fit_export_reload_and_stage2(client, reference, tmp_path):
    e = reference["ene"]
    y = reference["y"]
    source = tmp_path / "input.npz"
    np.savez(source, data=np.tile(y, (2, 2, 1)), ene=e)
    response = client.post("/api/import", json={"path": str(source)})
    assert response.status_code == 200, response.text
    d = response.json()
    assert d["pixels"] == 4
    bad = client.post("/api/fit", json={"dataset_id": d["id"], "pixel": [100, 0]})
    assert bad.status_code == 400
    prep = client.post(
        "/api/preprocess", json={"dataset_id": d["id"], "rank": 1, "align": True}
    ).json()
    finished = wait_job(client, prep["id"])
    assert finished["status"] == "completed", finished
    assert finished["timings"]["complete"]
    assert {p["key"] for p in finished["timings"]["phases"]} >= {"align", "svd", "dataset_save"}
    did = finished["output_dataset_id"]
    ds = client.get("/api/datasets/" + did).json()
    assert ds["history"][-1]["svd_rank"] == 1
    settings = {
        "stage1_fixed_EM": [12, 24, 48, 72, 96, 144, 200],
        "stage1_hierarchical_enabled": False,
        "datacube_stage1_auto_sparse_pm_enabled": False,
        "stage1_rounds": [
            {
                "name": "smoke_only",
                "em_step_meV": 8,
                "lambda_be_soft": 0.0015,
                "max_nonlinear_iter": 1,
            }
        ],
        "stage2_T_grid": [100, 300, 500, 700],
        "stage2_temperature_refine_enabled": False,
    }
    j = client.post(
        "/api/fit", json={"dataset_id": did, "stage": "stage1", "settings": settings}
    ).json()
    finished = wait_job(client, j["id"])
    assert finished["status"] == "completed", finished
    p = client.get("/api/jobs/" + j["id"] + "/pixel?y=0&x=0").json()
    assert p["available"] and p["metrics"]["R2"] > 0.99
    export = client.get("/api/jobs/" + j["id"] + "/download")
    assert export.status_code == 200
    with zipfile.ZipFile(io.BytesIO(export.content)) as z:
        assert {
            "maps.mat",
            "maps.npz",
            "settings.json",
            "pixels/000000.json.gz",
        } <= set(z.namelist())
        mat = loadmat(io.BytesIO(z.read("maps.mat")))
        assert mat["fit_status"][0, 0] == 1
        timing = json.loads(z.read("timings.json"))
        assert timing == finished["timings"]
        assert timing == json.loads(z.read("summary.json"))["job"]["timings"]
        assert timing["accounted_seconds"] + timing["other_seconds"] == pytest.approx(timing["wall_seconds"])
        assert {p["key"] for p in timing["phases"]} >= {"stage1", "checkpoint", "archive"}
    # Discard RAM copies: subsequent reads and Stage 2 must work from disk.
    import app

    app.DATASETS.clear()
    app.JOBS[j["id"]]["_state"] = {}
    reopened = client.get("/api/jobs/" + j["id"] + "/pixel?y=0&x=0").json()
    assert reopened["metrics"] == p["metrics"]
    st = client.post(
        "/api/fit",
        json={
            "dataset_id": did,
            "stage": "stage2",
            "source_job_id": j["id"],
            "settings": settings,
        },
    )
    assert st.status_code == 200, st.text
    last = wait_job(client, st.json()["id"])
    assert last["status"] == "completed", last
    pixel = client.get("/api/jobs/" + last["id"] + "/pixel?y=0&x=0").json()
    assert pixel["stage2_status"] == "ok", pixel
    assert pixel["T"] in settings["stage2_T_grid"]


def test_upload_and_local_origin_guard(client):
    f = io.BytesIO()
    np.savez(f, data=np.ones((1, 1, 20)), ene=np.arange(20.0))
    response = client.post(
        "/api/upload",
        files={"file": ("sample.npz", f.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["name"] == "sample.npz"
    blocked = client.post(
        "/api/import",
        json={"path": "not-real"},
        headers={"Origin": "https://example.org"},
    )
    assert blocked.status_code == 403


def test_cancel_task_preserves_server(client, reference, tmp_path):
    path = tmp_path / "cancel.npz"
    np.savez(path, data=np.tile(reference["y"], (3, 3, 1)), ene=reference["ene"])
    d = client.post("/api/import", json={"path": str(path)}).json()
    j = client.post("/api/fit", json={"dataset_id": d["id"], "scope": "all"}).json()
    stopped = client.post("/api/jobs/" + j["id"] + "/cancel").json()
    assert stopped["id"] == j["id"]
    out = wait_job(client, j["id"])
    assert out["status"] == "cancelled", out
    assert out["timings"]["complete"] and out["timings"]["active_phase"] is None
    assert client.get("/api/health").json()["status"] == "ok"


def test_scheduling_api_validation_and_config(client):
    from app import FitRequest

    config = client.get("/api/config").json()
    assert config["parallel_workers"] <= config["logical_cpus"]
    assert config["parallel_workers"] == max(1, min(61, config["logical_cpus"] - 2))
    assert config["stripe_count"] == 0
    assert config["auto_stripes_per_worker"] == 3
    assert config["auto_stripes_offset"] == 4
    assert config["partition_modes"] == ["balanced", "vertical", "horizontal", "grid"]
    for key, value in (("parallel_workers", 0), ("parallel_workers", 62),
                       ("parallel_workers", 1.5), ("parallel_workers", True),
                       ("stripe_count", -1), ("stripe_count", 10001),
                       ("stripe_count", 2.5), ("partition_mode", "spectral"),
                       ("grid_rows", 0), ("grid_rows", True),
                       ("grid_columns", 1.5), ("grid_columns", 10001)):
        response = client.post("/api/fit", json={"dataset_id": "missing", key: value})
        assert response.status_code == 422, response.text
    valid = FitRequest(dataset_id="example", parallel_workers=14, stripe_count=36)
    assert valid.parallel_workers == 14 and valid.stripe_count == 36


def test_grid_api_validation_and_forwarding(client, monkeypatch):
    import app
    monkeypatch.setattr(app, "get_dataset", lambda uid: {"cube": np.zeros((7, 17, 20)), "energy": np.arange(20)})
    monkeypatch.setattr(app, "validate_settings", lambda *_: {})
    monkeypatch.setattr(app, "submit", lambda kind, body, dataset: body.model_dump())
    request = dict(dataset_id="fixture", scope="all", partition_mode="grid", grid_rows=3, grid_columns=5)
    response = client.post("/api/fit", json=request)
    assert response.status_code == 200
    assert response.json()["grid_rows"] == 3 and response.json()["grid_columns"] == 5
    assert response.json()["partition_mode"] == "grid"
    response = client.post("/api/fit", json={**request, "grid_rows": 101, "grid_columns": 100})
    assert response.status_code == 400 and "10000" in response.json()["detail"]


def test_release_version_consistent_across_api_and_page(client):
    from zest import __version__

    assert __version__ == "1.1.0"
    assert client.get("/api/health").json()["version"] == __version__
    assert client.get("/api/config").json()["version"] == __version__
    assert client.get("/openapi.json").json()["info"]["version"] == __version__
    page = client.get("/").text
    assert f"CPU 并行版 · {__version__}" in page
    assert f"/static/style.css?v={__version__}" in page
    assert f"/static/app.js?v={__version__}" in page
