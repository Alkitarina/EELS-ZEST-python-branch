from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import numpy as np
import pytest
from zest.cache import BoundedCache


def test_lru_count_and_byte_limits_preserve_in_use_references():
    cache = BoundedCache(2, 64, lambda a: a.nbytes)
    a, b, c = (np.full(4, x, dtype=float) for x in (1, 2, 3))
    cache.put("a", a)
    cache.put("b", b)
    assert cache.get_or_load("a", lambda: pytest.fail("unexpected reload")) is a
    cache.put("c", c)  # b, rather than recently used a, is evicted.
    assert cache.nbytes == 64 and len(cache) == 2
    assert cache.get_or_load("a", lambda: pytest.fail("a was not least-recent")) is a
    reloaded = b.copy()
    assert cache.get_or_load("b", lambda: reloaded) is reloaded
    # Count alone permits two items, but this larger one consumes the byte budget.
    cache.put("large", np.zeros(7))
    assert cache.nbytes == 56 and len(cache) == 1
    np.testing.assert_array_equal(a, np.ones(4))  # eviction is not invalidation


def test_oversized_values_are_usable_but_not_retained_and_failed_loads_can_retry():
    cache = BoundedCache(3, 16, lambda a: a.nbytes)
    oversized = np.arange(10.)
    assert cache.get_or_load("large", lambda: oversized) is oversized
    assert len(cache) == cache.nbytes == 0
    with pytest.raises(OSError):
        cache.get_or_load("retry", lambda: (_ for _ in ()).throw(OSError("read failed")))
    cache.put("retry", np.ones(2))
    cache.put("retry", np.ones(1))
    assert cache.nbytes == 8 and len(cache) == 1
    cache.clear()
    assert len(cache) == cache.nbytes == 0


def test_concurrent_readers_share_one_load():
    cache = BoundedCache(3, 64, lambda a: a.nbytes)
    ready = Barrier(4)
    loads = []

    def load():
        loads.append(1)
        return np.ones(4)

    def read(_):
        ready.wait(timeout=5)
        return cache.get_or_load("shared", load)

    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(read, range(4)))
    assert len(loads) == 1
    assert all(value is values[0] for value in values)
