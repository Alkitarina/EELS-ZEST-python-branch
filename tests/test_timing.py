import pytest
from zest.timing import PhaseTimer
from zest.parallel import scheduling_options, vertical_stripes


def test_live_nested_time_is_exclusive_and_freezes():
    now = [10.0]
    timer = PhaseTimer(clock=lambda: now[0])
    with timer.phase("stage1"):
        now[0] += 2
        with timer.phase("checkpoint"):
            now[0] += 3
            live = timer.snapshot()
            assert live["active_phase"] == "checkpoint" and not live["complete"]
            assert {p["key"]: p["seconds"] for p in live["phases"]} == {
                "stage1": 2, "checkpoint": 3}
        now[0] += 4
    now[0] += 1  # Uninstrumented coordination remains visible as other time.
    final = timer.finish()
    assert final["wall_seconds"] == 10
    assert final["accounted_seconds"] == 9 and final["other_seconds"] == 1
    now[0] += 100
    assert timer.snapshot() == final


def test_interleaved_serial_stages_accumulate_and_failure_closes_phase():
    now = [0.0]
    timer = PhaseTimer(clock=lambda: now[0])
    for _ in range(2):
        with timer.phase("stage1"):
            now[0] += 2
        with timer.phase("stage2"):
            now[0] += 3
    with pytest.raises(RuntimeError, match="cancel"):
        with timer.phase("stage1"):
            now[0] += 1
            raise RuntimeError("cancel")
    final = timer.finish()
    assert final["active_phase"] is None and final["complete"]
    assert {p["key"]: p["seconds"] for p in final["phases"]} == {"stage1": 5, "stage2": 6}


def test_auto_and_manual_stripes_are_independent_of_worker_count():
    coords = [(y, x) for x in range(36) for y in range(2)]
    for workers, requested, expected in ((12, 0, 32), (2, 0, 2), (12, 6, 6), (2, 100, 36)):
        _, count = scheduling_options(workers, requested)
        stripes = vertical_stripes(coords, count)
        assert len(stripes) == expected
        assert sorted(i for st in stripes for i in st["indices"]) == list(range(len(coords)))
