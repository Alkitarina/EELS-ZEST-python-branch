"""Local, inspectable job storage. Never deserialize Python pickles."""

import gzip
import json
import math
import zlib
from pathlib import Path
import numpy as np
from .numeric import clean_json

try:
    import orjson
except ImportError:
    orjson = None

PIXEL_COMPRESSION_LEVEL = 1


def pack(value, native_arrays=False):
    if isinstance(value, dict):
        return {str(k): pack(v, native_arrays) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        # Scan finite numeric arrays once in NumPy, not once per Python scalar.
        if value.dtype.kind in "biuf" and np.isfinite(value).all():
            if native_arrays and value.dtype.itemsize <= 8 and value.ndim > 0:
                dtype = np.float64 if value.dtype.kind == "f" else value.dtype.newbyteorder("=")
                return np.ascontiguousarray(value, dtype=dtype)
            return value.tolist()
        return pack(value.tolist(), native_arrays)
    if isinstance(value, np.generic):
        return pack(value.item(), native_arrays)
    if isinstance(value, (list, tuple)):
        return [pack(v, native_arrays) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "__zest_float__": "nan"
            if np.isnan(value)
            else ("inf" if value > 0 else "-inf")
        }
    return value


def scientific_bytes(value):
    if orjson is not None:
        try:
            return orjson.dumps(pack(value, native_arrays=True), option=orjson.OPT_SERIALIZE_NUMPY)
        except orjson.JSONEncodeError:
            # Preserve the standard encoder's supported types (e.g. big ints).
            pass
    return json.dumps(pack(value), ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def save_json(path, value, scientific=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        pack(value) if scientific else clean_json(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    temp = path.with_name(path.name + ".writing")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def save_pixel(path, value, *, skip_unchanged=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    payload = scientific_bytes(value)
    if skip_unchanged and path.is_file():
        try:
            with gzip.open(path, "rb") as stream:
                if stream.read() == payload:
                    return False
        except (OSError, EOFError, zlib.error):
            # Replace an incomplete/damaged previous file from the in-memory result.
            pass
    temp.write_bytes(gzip.compress(payload, compresslevel=PIXEL_COMPRESSION_LEVEL, mtime=0))
    temp.replace(path)
    return True


def restore(value):
    if isinstance(value, dict):
        if set(value) == {"__zest_float__"}:
            return float(value["__zest_float__"])
        return {k: restore(v) for k, v in value.items()}
    if isinstance(value, list):
        value = [restore(v) for v in value]
        if value and all(
            isinstance(x, (int, float, bool, np.number)) or x is None for x in value
        ):
            return np.array([np.nan if x is None else x for x in value])
        if value and all(isinstance(x, (list, np.ndarray)) for x in value):
            try:
                return np.asarray(value, dtype=float)
            except (ValueError, TypeError):
                pass
        return value
    return value


def load_pixel(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return restore(json.load(f))
