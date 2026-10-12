"""Bounded process-local caches; eviction never invalidates a caller's reference."""

from collections import OrderedDict
from threading import RLock


class BoundedCache:
    def __init__(self, max_items, max_bytes, size_of):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.size_of = size_of
        self._items = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    def _store(self, key, value):
        size = self.size_of(value)
        old = self._items.pop(key, None)
        if old is not None:
            self._bytes -= old[1]
        # Oversized data can be used by this request without retaining it.
        if self.max_items <= 0 or size > self.max_bytes:
            return value
        self._items[key] = (value, size)
        self._bytes += size
        while len(self._items) > self.max_items or self._bytes > self.max_bytes:
            _, (_, removed) = self._items.popitem(last=False)
            self._bytes -= removed
        return value

    def put(self, key, value):
        with self._lock:
            return self._store(key, value)

    def get_or_load(self, key, load):
        # Coalesce concurrent requests for the same data/derived intensity.
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key][0]
            return self._store(key, load())

    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0

    @property
    def nbytes(self):
        with self._lock:
            return self._bytes

    def __len__(self):
        with self._lock:
            return len(self._items)
