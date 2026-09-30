"""Local, inspectable job storage. Never deserialize Python pickles."""

import gzip
import json
from pathlib import Path
import numpy as np
from .numeric import clean_json


def pack(value):
    if isinstance(value, dict):
        return {str(k): pack(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return pack(value.tolist())
    if isinstance(value, np.generic):
        return pack(value.item())
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {
            "__zest_float__": "nan"
            if np.isnan(value)
            else ("inf" if value > 0 else "-inf")
        }
    return value


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


def save_pixel(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    with gzip.open(temp, "wt", encoding="utf-8") as f:
        json.dump(
            pack(value), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    temp.replace(path)


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
