import json
import pytest
from zest.storage import save_json, save_pixel, load_pixel


@pytest.mark.parametrize("status", ["running", "queued"])
def test_restore_interrupted_job_reconciles_checkpoints(client, tmp_path, status):
    from app import restore_job

    path = tmp_path / "job.json"
    save_json(path, dict(id="interrupted",kind="fit",status=status,phase="hierarchy",
        completed_pixels=1,timings=dict(active_phase="hierarchy",complete=False),
        execution=dict(stripes=[dict(status="completed"),dict(status="running")])) )
    for i in range(2):
        save_pixel(tmp_path / "pixels" / f"{i:06d}.json.gz", dict(pixel_index=i,value=1.5))
    (tmp_path / "pixels" / "000002.json.gz.writing").write_bytes(b"incomplete")
    restored=restore_job(path)
    assert restored["status"] == restored["phase"] == "interrupted"
    assert restored["interrupted_phase"] == "hierarchy"
    assert restored["completed_pixels"] == 2
    assert restored["timings"]["active_phase"] is None
    assert not restored["timings"]["complete"]
    assert [s["status"] for s in restored["execution"]["stripes"]] == ["completed","interrupted"]
    assert restored["_state"] == {}  # No silent restart of fitting.
    assert not restored["_cancel"].is_set()
    disk=json.loads(path.read_text(encoding="utf-8"))
    assert disk["status"] == "interrupted"
    assert "_cancel" not in disk
    assert load_pixel(tmp_path / "pixels" / "000001.json.gz")["value"] == 1.5
    assert restore_job(path)["interrupted_phase"] == "hierarchy"


def test_completed_job_metadata_is_preserved(client, tmp_path):
    from app import restore_job

    path=tmp_path / "job.json"
    expected=dict(id="finished",kind="fit",status="completed",phase="completed",
                  completed_pixels=12,timings=dict(complete=True,active_phase=None))
    save_json(path,expected)
    before=path.read_bytes()
    restored=restore_job(path)
    assert {k:v for k,v in restored.items() if not k.startswith("_")} == expected
    assert path.read_bytes() == before
