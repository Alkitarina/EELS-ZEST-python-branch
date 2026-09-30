"""Small MATLAB-compatible operations, including shrinking-window endpoints."""

import numpy as np
from scipy.special import expit
from scipy.interpolate import interp1d

EPS = np.finfo(np.float64).eps
KB = 0.08617333262


def col(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def mov(x, n, kind="mean"):
    x = col(x)
    n = max(1, int(np.floor(float(n) + 0.5)))
    if n == 1:
        return x.copy()
    left = n // 2
    right = (n - 1) // 2
    padded = np.pad(x, (left, right), constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(padded, n)
    return np.nanmedian(win, axis=1) if kind == "median" else np.nanmean(win, axis=1)


def quantile(x, q):
    a = np.sort(col(x)[np.isfinite(col(x))])
    return (
        float(a[np.clip(int(np.floor(q * (len(a) - 1) + 0.5)), 0, len(a) - 1)])
        if len(a)
        else np.nan
    )


def noise(y, mask):
    y = col(y)
    a = y[np.asarray(mask, dtype=bool) & np.isfinite(y)]
    if len(a) < 8:
        a = y[np.isfinite(y)]
    if len(a) < 8:
        return EPS
    v = 1.4826 * np.median(np.abs(a - np.median(a)))
    if not np.isfinite(v) or v <= 0:
        v = np.std(a, ddof=1)
    return float(v) if np.isfinite(v) and v > 0 else EPS


def unit_scale(x, mask):
    x = col(x)
    v = x[np.asarray(mask, dtype=bool) & np.isfinite(x)]
    if not len(v):
        return np.zeros_like(x)
    lo = np.median(v)
    hi = quantile(v, 0.98)
    if not np.isfinite(hi) or hi <= lo:
        hi = np.max(v)
    return np.nan_to_num(np.clip((x - lo) / max(hi - lo, EPS), 0, 1))


def interp(e, y, q, kind="linear", fill=0):
    return interp1d(
        e, y, axis=0, kind=kind, bounds_error=False, fill_value=fill, assume_sorted=True
    )(q)


def grid(a, step, b):
    if b < a:
        return np.empty(0)
    return a + step * np.arange(int(np.floor((b - a) / step + 1e-10)) + 1)


def smooth_mask(mask, e, taper):
    mask = np.asarray(mask, dtype=bool)
    e = col(e)
    alpha = np.zeros_like(e)
    if len(e) < 3 or taper <= 0:
        return mask.astype(float)
    taper = max(np.median(np.abs(np.diff(e))), taper / 4, EPS)
    for a, b in bands(e, mask):
        alpha = np.maximum(
            alpha, expit((e - a) / max(taper, EPS)) * expit((b - e) / max(taper, EPS))
        )
    return alpha


def bands(e, mask):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return np.empty((0, 2))
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return np.array([[e[g[0]], e[g[-1]]] for g in groups])


def clean_json(x):
    if isinstance(x, dict):
        return {
            str(k): clean_json(v) for k, v in x.items() if not str(k).startswith("_")
        }
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x
