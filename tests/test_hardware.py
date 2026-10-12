import pytest


def test_detect_only_at_startup_and_manual_refresh(client, monkeypatch):
    import app
    original = dict(app.CAPABILITIES)
    monkeypatch.setattr(app, "CAPABILITIES", original)
    calls = []

    def detect():
        calls.append(1)
        return {**original, "logical_cpus": 8, "cpu_name": "Detected test CPU", "detected_at": "2026-10-12T00:00:00+00:00"}

    monkeypatch.setattr(app, "hardware", detect)
    for _ in range(3):
        client.get("/api/health")
        client.get("/api/config")
    assert not calls
    assert app.FitRequest(dataset_id="example").parallel_workers == max(1, min(61, original["logical_cpus"] - 2))
    response = client.post("/api/hardware/refresh", json={})
    assert response.status_code == 200
    assert calls == [1]
    assert response.json()["parallel_workers"] == 6
    assert response.json()["hardware"]["cpu_name"] == "Detected test CPU"
    assert app.FitRequest(dataset_id="example").parallel_workers == 6
    assert app.FitRequest(dataset_id="example", parallel_workers=3).parallel_workers == 3
    assert client.get("/api/config").json()["logical_cpus"] == 8
    assert client.post("/api/hardware/refresh", headers={"Origin": "https://untrusted.example"}).status_code == 403
    assert calls == [1]


def test_host_probe_failure_keeps_cpu_threads_usable(monkeypatch):
    from zest import host_info
    if host_info.os.name != "nt":
        pytest.skip("Windows CIM fallback")
    monkeypatch.setattr(host_info.os, "cpu_count", lambda: 4)

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] <= 10 and kwargs["creationflags"]
        raise host_info.subprocess.TimeoutExpired("CIM", 10)

    monkeypatch.setattr(host_info.subprocess, "run", timeout)
    info = host_info.cpu_info()
    assert info["logical_cpus"] == 4
    assert info["cpu_name"] and info["detection_warning"]
    assert info["memory_bytes"] is None
