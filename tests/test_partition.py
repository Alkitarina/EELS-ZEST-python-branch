import random
import numpy as np
import pytest
from zest.parallel import default_workers, scheduling_options
from zest.partition import partition_blocks, partition_boundaries


@pytest.mark.parametrize("threads,expected", [(1, 1), (2, 1), (3, 1), (4, 2), (20, 18), (128, 61)])
def test_threads_minus_two_with_pool_limits(threads, expected):
    assert default_workers(threads) == expected
    assert scheduling_options(expected, 0)[1] == max(1, 3 * expected - 4)


@pytest.mark.parametrize("mode", ["balanced", "vertical", "horizontal", "grid"])
@pytest.mark.parametrize("shape", [(7, 17), (1, 17), (17, 1), (1, 1)])
def test_prime_roi_coverage_adjacency_and_remainders(mode, shape):
    height, width = shape
    coords = [(y, x) for x in range(3, 3 + width) for y in range(2, 2 + height)]
    random.Random(10).shuffle(coords)  # Canonical index mapping cannot assume input order.
    blocks = partition_blocks(coords, 11, mode, 3, 5)
    assert sorted(i for block in blocks for i in block["indices"]) == list(range(len(coords)))
    assert all(block["indices"] for block in blocks)
    for block in blocks:
        path = [coords[i] for i in block["indices"]]
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
        if mode != "balanced":
            assert len(path) == len(block["rows"]) * len(block["columns"])
    if mode == "balanced":
        sizes = [len(b["indices"]) for b in blocks]
        assert len(blocks) == min(11, len(coords))
        assert max(sizes) - min(sizes) <= 1
    elif mode == "grid":
        assert len(blocks) == min(height, 3) * min(width, 5)


def test_boundaries_match_actual_partial_column_membership():
    coords = [(y, x) for x in range(3) for y in range(5)]
    blocks = partition_blocks(coords, 4)
    assert [len(b["indices"]) for b in blocks] == [4, 4, 4, 3]
    labels = {coords[i]: b["id"] for b in blocks for i in b["indices"]}
    expected = set()
    for (y, x), owner in labels.items():
        if labels.get((y, x - 1)) != owner:
            expected.add((x + .5, y + .5, x + .5, y + 1.5))
        if labels.get((y, x + 1)) != owner:
            expected.add((x + 1.5, y + .5, x + 1.5, y + 1.5))
        if labels.get((y - 1, x)) != owner:
            expected.add((x + .5, y + .5, x + 1.5, y + .5))
        if labels.get((y + 1, x)) != owner:
            expected.add((x + .5, y + 1.5, x + 1.5, y + 1.5))
    expanded = set()
    for x0, y0, x1, y1 in partition_boundaries(coords, blocks):
        for d in range(int(max(x1 - x0, y1 - y0))):
            expanded.add((x0, y0 + d, x1, y0 + d + 1) if x0 == x1 else
                         (x0 + d, y0, x0 + d + 1, y1))
    assert expanded == expected


def test_sparse_partition_is_complete_and_empty_selection_is_safe():
    coords = [(2, 2), (4, 2), (9, 8), (9, 9)]
    for mode in ("balanced", "vertical", "horizontal", "grid"):
        blocks = partition_blocks(coords, 50, mode, 50, 50)
        assert sorted(i for b in blocks for i in b["indices"]) == list(range(4))
    assert partition_blocks([], 3) == []
    assert partition_boundaries([], []) == []
    with pytest.raises(ValueError, match="重复"):
        partition_blocks([(0, 0), (0, 0)], 2)
    with pytest.raises(ValueError, match="10000"):
        partition_blocks(coords, 4, "grid", 101, 100)


def test_worker_reinitializes_at_sparse_gaps_and_after_failure(monkeypatch):
    from zest import parallel
    from zest.backend import Backend
    from zest.settings import defaults
    events, seeds = [], []
    coords = [(0, 0), (1, 0), (4, 0), (5, 0), (6, 0)]
    monkeypatch.setattr(parallel, "_WORKER", dict(backend=Backend(), cube=np.ones((7, 1, 4)), energy=np.arange(4)))
    monkeypatch.setattr(parallel, "_emit", lambda kind, **data: events.append((kind, data)))

    def solve(energy, spectrum, settings, backend, previous, progress):
        seeds.append(previous)
        if len(seeds) == 4:
            raise ValueError("fixture failure")
        if previous is None:
            assert not settings["stage1_warm_start_skip_coarse"]
        return {"theta": [len(seeds)]}

    monkeypatch.setattr(parallel.stage1, "solve", solve)
    block = partition_blocks(coords, 1)[0]
    parallel._fit_stripe(block, coords, defaults({"datacube_fit_retries": 0}))
    assert seeds == [None, [1], None, [3], None]
    assert [event[1]["result"]["execution"]["full_initialization"]
            for event in events if event[0] == "pixel"] == [True, False, True, True, True]
