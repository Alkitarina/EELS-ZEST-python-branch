"""Direct transcription of the v2.1 Pearson sparse Stage 1 objective.

Model/regularization formulas follow EELS_PearsonSparse_FullFit.m. The SciPy
optimizer replaces MATLAB's proprietary solver; scientific equivalence must
be assessed at objective and output level, not by identical iteration paths.
"""

import time
import numpy as np
from scipy.optimize import least_squares
from .numeric import EPS, KB, col, mov, noise, unit_scale, expit, interp, grid
from .settings import defaults
from .backend import Backend


def prepare(e, y, s):
    e = col(e)
    y = col(y)
    if e.shape != y.shape:
        raise ValueError("能量轴与谱长不一致")
    mask = (
        np.isfinite(e)
        & np.isfinite(y)
        & (e >= s["crop_range"][0])
        & (e <= s["crop_range"][1])
    )
    e = e[mask]
    y = y[mask].copy()
    if len(e) < 12 or np.any(np.diff(e) <= 0):
        raise ValueError("需要至少 12 个严格递增的有效能量通道")
    y[(y < 0) & (abs(y) < 100 * np.spacing(np.max(abs(y))))] = 0
    norm = float(np.max(abs(y))) or 1.0
    return e, y, y / norm, norm


def align(e, y, s):
    e = col(e)
    y = col(y)
    info = dict(
        enabled=bool(s["stage1_zlp_align_enabled"]),
        status="disabled",
        applied_shift_meV=0.0,
        estimated_peak_meV=np.nan,
    )
    if not info["enabled"]:
        return e, info
    limit = abs(s["stage1_zlp_align_max_shift_meV"])
    mask = np.isfinite(y) & (
        abs(e) <= max(limit, abs(s["stage1_zlp_align_search_meV"]))
    )
    if len(e) < 3 or np.count_nonzero(mask) < 3 or limit <= 0:
        info["status"] = "insufficient_data"
        return e, info
    ys = mov(y, s["stage1_zlp_align_smooth_points"])
    idx = np.flatnonzero(mask)
    i = idx[np.argmax(ys[idx])]
    if ys[i] < s["stage1_zlp_align_min_peak_fraction"] * np.max(ys) or np.max(ys) <= 0:
        info["status"] = "weak_peak_near_zero"
        return e, info
    peak = e[i]
    if 0 < i < len(e) - 1:
        p = np.polyfit(e[i - 1 : i + 2], y[i - 1 : i + 2], 2)
        candidate = -p[1] / (2 * p[0]) if abs(2 * p[0]) > EPS else np.nan
        if p[0] < 0 and e[i - 1] <= candidate <= e[i + 1]:
            peak = candidate
    shift = float(np.clip(peak, -limit, limit))
    info.update(status="ok", estimated_peak_meV=float(peak), applied_shift_meV=shift)
    return e - shift, info


def make_em(e, s):
    maximum = np.max(abs(e))
    fixed = col(s.get("stage1_fixed_EM", []))
    if len(fixed):
        fixed = np.unique(
            fixed[np.isfinite(fixed) & (fixed >= s["em_min_meV"]) & (fixed <= maximum)]
        )
        if np.size(s["em_max_meV"]):
            fixed = fixed[fixed <= s["em_max_meV"]]
        if len(fixed):
            return fixed
    end = (
        min(s["em_max_meV"], maximum)
        if np.size(s["em_max_meV"])
        else max(s["em_min_meV"], maximum - s["em_exclude_core_meV"])
    )
    em = grid(s["em_min_meV"], s["em_step_meV"], end)
    if s["em_lowfreq_dense_enabled"]:
        low, high = sorted(abs(col(s["em_lowfreq_dense_range_meV"])))[:2]
        a = max(s["em_min_meV"], low)
        b = min(end, high)
        step = min(
            s["em_lowfreq_dense_max_step_meV"],
            max(0.75, s["em_step_meV"] * s["em_lowfreq_dense_step_factor"]),
        )
        if b > a + step:
            em = np.r_[em, grid(a, step, b)]
    return np.unique(em[(em >= s["em_min_meV"]) & (em <= end)])


def expand(x, n, fallback):
    a = col(x)
    if not len(a):
        return np.full(n, fallback, dtype=float)
    if len(a) < n:
        a = np.r_[a, np.repeat(a[-1], n - len(a))]
    return a[:n]


def is_pvoigt(s):
    return s["zlp_model_type"].lower() in (
        "pvoigt",
        "pseudo_voigt",
        "asym_pvoigt",
        "asym_pseudo_voigt",
    )


def bounds(s):
    n = int(s["num_zlp_components"])
    wmin = np.maximum(
        expand(s["zlp_component_width_min_meV"], n, s["zlp_width_bounds"][0]),
        s["zlp_width_bounds"][0],
    )
    wmax = np.minimum(
        np.maximum(
            expand(s["zlp_component_width_max_meV"], n, s["zlp_width_bounds"][1]),
            wmin + EPS,
        ),
        s["zlp_width_bounds"][1],
    )
    shape = s["pvoigt_eta_bounds"] if is_pvoigt(s) else s["pearson_m_bounds"]
    lb = np.r_[
        s["eshift_range"][0],
        np.full(n, -s["zlp_center_limit"]),
        wmin,
        wmin,
        np.full(n, shape[0]),
        np.full(n, shape[0]),
        np.log(s["T_soft_bounds"][0]),
    ]
    ub = np.r_[
        s["eshift_range"][1],
        np.full(n, s["zlp_center_limit"]),
        wmax,
        wmax,
        np.full(n, shape[1]),
        np.full(n, shape[1]),
        np.log(s["T_soft_bounds"][1]),
    ]
    ctx = s.get("stage1_hierarchy_context", {})
    if ctx and ctx.get("enabled") and len(ctx.get("theta_lb", [])) == len(lb):
        l = col(ctx["theta_lb"])
        u = col(ctx["theta_ub"])
        good = np.isfinite(l) & np.isfinite(u) & (u > l)
        lb[good] = np.maximum(lb[good], l[good])
        ub[good] = np.minimum(ub[good], u[good])
        bad = ub < lb
        mid = (ub[bad] + lb[bad]) / 2
        lb[bad] = mid - 10 * np.spacing(np.maximum(abs(mid), 1))
        ub[bad] = mid + 10 * np.spacing(np.maximum(abs(mid), 1))
    return lb, ub


def initial(e, y, s, previous=None):
    n = int(s["num_zlp_components"])
    lb, ub = bounds(s)
    if previous is not None and np.size(previous) == 2 + 5 * n:
        return np.clip(col(previous), lb, ub)
    widths = np.minimum(
        np.linspace(0.8, 26, n),
        0.65 * expand(s["zlp_component_width_max_meV"], n, s["zlp_width_bounds"][1]),
    )
    heavy = s["stage1_zlp_gain_heavy_init_enabled"]
    sl = (
        widths * np.linspace(1.02, max(s["stage1_zlp_gain_width_boost"], 1.02), n)
        if heavy
        else widths
    )
    sr = (
        widths
        * np.linspace(1, min(max(s["stage1_zlp_loss_width_scale"], 0.55), 1.25), n)
        if heavy
        else widths * np.linspace(1.05, 1.8, n)
    )
    if is_pvoigt(s):
        m = np.linspace(0.12, 0.36, n)
        ml = (
            np.minimum(m * np.linspace(1.05, 1.35, n), s["pvoigt_eta_bounds"][1])
            if heavy
            else m
        )
        mr = (
            np.maximum(m * np.linspace(1, 0.82, n), s["pvoigt_eta_bounds"][0])
            if heavy
            else np.minimum(m * np.linspace(1, 1.35, n), s["pvoigt_eta_bounds"][1])
        )
    elif heavy:
        m = np.linspace(4.4, 1.8, n)
        ml = m * np.linspace(
            1, min(max(s["stage1_zlp_gain_shape_heavy_factor"], 0.55), 1.05), n
        )
        mr = m * np.linspace(
            1.04, max(s["stage1_zlp_loss_shape_light_factor"], 1.04), n
        )
    else:
        ml = np.linspace(4.5, 1.8, n)
        mr = np.linspace(3.8, 1.6, n)
    return np.clip(
        np.r_[e[np.argmax(y)], np.zeros(n), sl, sr, ml, mr, np.log(s["T_soft_init"])],
        lb,
        ub,
    )


def decode_theta(theta, s):
    n = int(s["num_zlp_components"])
    p = {
        k: col(theta)[1 + i * n : 1 + (i + 1) * n]
        for i, k in enumerate(("center", "sL", "sR", "mL", "mR"))
    }
    return dict(p, E_shift=float(theta[0]), T_soft=float(np.exp(theta[-1])))


def shape(x, sl, sr, ml, mr, s, b):
    xp = b.xp
    x = b.array(x)
    width = xp.maximum(xp.where(x < 0, sl, sr), EPS)
    m = xp.where(x < 0, ml, mr)
    if is_pvoigt(s):
        m = xp.clip(m, 0, 1)
        out = m / (1 + (x / width) ** 2) + (1 - m) * xp.exp(-0.5 * (x / width) ** 2)
    else:
        m = xp.maximum(m, 0.51)
        out = (1 + (x / width) ** 2 / m) ** (-m)
    return xp.where(xp.isfinite(out), out, 0)


def normalize(k, b):
    xp = b.xp
    k = b.array(k)
    k = xp.where(xp.isfinite(k) & (k >= 0), k, 0)
    total = float(k.sum())
    if total <= 0:
        k = xp.zeros_like(k)
        k[(k.size - 1) // 2] = 1
        return k
    return k / total


def shift_matrix(e, k, em, b):
    xp = b.xp
    e = b.array(e)
    k = b.array(k)
    em = b.array(em)
    n = e.size
    b.gpu_operations += int(b.device == "gpu")
    if not em.size:
        return xp.zeros((n, 0))
    de = xp.diff(e)
    step = float(xp.median(de))
    if (
        not np.isfinite(step)
        or step == 0
        or float(xp.max(xp.abs(de - step))) > max(1e-9, 1e-7 * max(abs(step), EPS))
    ):
        M = xp.column_stack([xp.interp(e - d, e, k, left=0, right=0) for d in em])
    else:
        # Preserve MATLAB's one-based arithmetic before converting indices.
        f = (e[:, None] - em[None, :] - e[0]) / step + 1
        lo = xp.floor(f)
        alpha = f - lo
        valid = (lo >= 1) & (lo < n)
        lo = xp.where(valid, lo, 1).astype(xp.int64) - 1
        M = xp.where(valid, (1 - alpha) * k[lo] + alpha * k[lo + 1], 0)
    M = xp.where(xp.isfinite(M) & (M >= 0), M, 0)
    s = M.sum(axis=0)
    return M / xp.where(s > 0, s, 1)[None, :]


def design(e, em, p, s, b, kernel=None):
    xp = b.xp
    n = int(s["num_zlp_components"])
    x = b.array(e) - p["E_shift"]
    z = xp.column_stack(
        [
            shape(
                x - p["center"][i], p["sL"][i], p["sR"][i], p["mL"][i], p["mR"][i], s, b
            )
            for i in range(n)
        ]
    )
    if kernel is None:
        k = (
            z.sum(axis=1)
            if s["kernel_mode"] in ("full_zlp", "fitted_zlp")
            else z[:, int(np.argmin((p["sL"] + p["sR"]) / 2))]
        )
    else:
        k = kernel
    k = normalize(k, b)
    ks = shift_matrix(e, k, em, b)
    ka = shift_matrix(e, k, -col(em), b)
    layout = dict(
        zlp=np.arange(n),
        pm_s=np.arange(n, n + len(em)),
        pm_a=np.arange(n + len(em), n + 2 * len(em)),
        bg=np.empty(0, dtype=int),
    )
    return xp.column_stack([z, ks, ka]), layout, z, dict(stokes=ks, anti=ka, base=k)


def adaptive(e, y, s):
    s = s.copy()
    search = abs(e) <= s["adaptive_zlp_region_search_meV"]
    if not s["adaptive_zlp_region_enabled"] or sum(search) < 7:
        return s
    ys = mov(mov(np.maximum(y, 0), 5, "median"), 5)
    ids = np.flatnonzero(search)
    ip = ids[np.argmax(ys[ids])]
    peak = ys[ip]
    core = search & (ys >= s["adaptive_zlp_core_fraction"] * peak)
    if peak <= 0 or sum(core) < 3:
        return s
    half = np.max(abs(e[core] - e[ip]))
    if half <= 0:
        return s
    start = np.clip(
        half + s["adaptive_zlp_guard_margin_meV"],
        s["adaptive_zlp_guard_min_start_meV"],
        s["adaptive_zlp_guard_max_start_meV"],
    )
    end = min(
        s["zlp_signal_guard_end_meV"],
        s["adaptive_zlp_guard_end_fraction"] * max(abs(e)),
    )
    if end <= start + 5:
        end = s["zlp_signal_guard_end_meV"]
    if end <= start + 5:
        return s
    s["zlp_signal_guard_start_meV"] = float(start)
    s["zlp_signal_guard_end_meV"] = float(end)
    if s["adaptive_zlp_region_adjust_weights"]:
        s["core_downweight_width"] = float(
            np.clip(
                half + 0.5 * s["adaptive_zlp_guard_margin_meV"],
                s["adaptive_zlp_core_min_meV"],
                s["adaptive_zlp_core_max_meV"],
            )
        )
    return s


def tail_profile(e, s):
    start = max(s["stage1_zlp_fartail_start_meV"], 0)
    end = s["stage1_zlp_fartail_end_meV"]
    end = end if np.size(end) and np.isfinite(end) else np.inf
    w = max(s["stage1_zlp_fartail_taper_meV"], EPS)
    a = abs(e)
    mask = (a >= start) & (a <= end)
    return expit((a - start) / w) * expit((end - a) / w) * np.where(
        e < 0, max(s["stage1_zlp_fartail_gain_boost"], 1), 1
    ) * mask, mask


def weights(e, y, s):
    a = abs(e)
    core = expit((s["core_downweight_width"] - a) / 4)
    shoulder = expit((a - s["core_downweight_width"]) / 4) * expit(
        (s["highfreq_start_meV"] - a) / 12
    )
    high = expit((a - s["highfreq_start_meV"]) / 12)
    start = max(s["highfreq_start_meV"], s["tail_start_fraction"] * max(a))
    tail = expit((a - start) / 12)
    w = (
        s["core_weight"] * core
        + s["shoulder_weight"] * shoulder
        + s["signal_weight"] * (1 - core)
        + s["highfreq_weight"] * high
        + s["tail_weight"] * tail
    )
    if s["stage1_zlp_core_fartail_enabled"]:
        cw = max(s["stage1_zlp_core_anchor_width_meV"], EPS)
        tp, tm = tail_profile(e, s)
        w += (
            s["stage1_zlp_core_data_weight"] * expit((cw - a) / max(0.25 * cw, 1))
            + s["stage1_zlp_fartail_data_weight"] * tp * tm
        )
    local = mov(np.maximum(y, 0), 15, "median")
    ref = (
        np.median(local[a > s["core_downweight_width"]])
        if np.any(a > s["core_downweight_width"])
        else np.nan
    )
    if not np.isfinite(ref) or ref <= 0:
        ref = max(local)
    floor = max(s["dynamic_weight_noise_sigma"] * noise(y, a > start), 0.01 * ref)
    balance = np.clip(
        (ref / np.maximum(local, max(floor, EPS))) ** 0.45,
        1,
        s["dynamic_weight_max_gain"],
    )
    w *= 1 + s["dynamic_weight_strength"] * (1 - core) * (balance - 1)
    return np.sqrt(np.where(np.isfinite(w) & (w > 0), w, 1))


def guard_profile(e, y, s):
    if not s["zlp_signal_guard_feature_weighted"]:
        return np.ones_like(e)
    mask = (abs(e) >= s["zlp_signal_guard_start_meV"]) & (
        abs(e) <= s["zlp_signal_guard_end_meV"]
    )
    if sum(mask) < 8:
        return np.ones_like(e)
    ys = mov(mov(np.maximum(y, 0), 7, "median"), 7)
    no = noise(ys, mask)
    base = mov(ys, 35, "median")
    contrast = np.maximum(ys - base, 0) / np.maximum(base + 4 * no, EPS)
    log = np.log(np.maximum(ys, max(no, EPS)))
    slope = np.r_[0, abs(np.diff(mov(log, 7)))]
    curvature = np.r_[0, abs(np.diff(mov(slope, 7)))]
    feature = mov(
        0.58 * unit_scale(contrast, mask)
        + 0.27 * unit_scale(slope, mask)
        + 0.15 * unit_scale(curvature, mask),
        max(3, s["zlp_signal_guard_feature_smooth_points"]),
    )
    feature[~mask | ~np.isfinite(feature)] = 0
    floor = np.clip(s["zlp_signal_guard_min_profile"], 0, 1)
    return floor + (1 - floor) * np.clip(feature, 0, 1)


def regularization(em, l, T, n, s, A, e, y, b):
    xp = b.xp
    ns = len(em)
    rows = []
    rhs = []
    z = l["zlp"]
    A = b.array(A)

    def add(R, w, target=None):
        rows.append(np.sqrt(w) * R)
        rhs.append(
            xp.zeros(R.shape[0]) if target is None else np.sqrt(w) * b.array(target)
        )

    def blank(nrow):
        return xp.zeros((nrow, n), dtype=xp.float64)

    if s["lambda_be_soft"] > 0 and ns:
        R = blank(ns)
        R[xp.arange(ns), b.xp.asarray(l["pm_a"])] = 1
        R[xp.arange(ns), b.xp.asarray(l["pm_s"])] = -xp.exp(
            -b.array(em) / max(KB * T, EPS)
        )
        add(R, s["lambda_be_soft"])
    if s["lambda_pm_smooth"] > 0 and ns >= 3:
        D = xp.diff(xp.eye(ns), n=2, axis=0)
        for key in ("pm_s", "pm_a"):
            R = blank(ns - 2)
            R[:, l[key]] = D
            add(R, s["lambda_pm_smooth"])
    if s["lambda_pm_l2"] > 0 and ns:
        R = blank(2 * ns)
        R[:ns, l["pm_s"]] = xp.eye(ns)
        R[ns:, l["pm_a"]] = xp.eye(ns)
        add(R, s["lambda_pm_l2"])
    if s["zlp_wide_component_l2"] > 0:
        p = z[max(0, min(len(z) - 1, round(s["zlp_wide_component_start_index"]) - 1)) :]
        R = blank(len(p))
        R[:, p] = xp.eye(len(p))
        add(R, s["zlp_wide_component_l2"])
    a = abs(e)
    for prefix in ("zlp_signal_guard", "zlp_lowfreq_guard"):
        if not s[prefix + "_enabled"] or s[prefix + "_weight"] <= 0:
            continue
        start = s[prefix + "_start_meV"]
        end = s[prefix + "_end_meV"]
        mask = (a >= start) & (a <= end)
        if end <= start or sum(mask) < 3:
            continue
        tw = max(s[prefix + "_taper_meV"], EPS)
        t = expit((a[mask] - start) / tw) * expit((end - a[mask]) / tw)
        gp = guard_profile(e, y, s)[mask]
        if prefix == "zlp_signal_guard":
            floor = np.clip(s["zlp_signal_guard_tail_floor"], 0, 1)
            t *= (
                floor
                + (1 - floor)
                * expit(
                    (s["zlp_signal_guard_tail_release_start_meV"] - a[mask])
                    / max(s["zlp_signal_guard_tail_release_width_meV"], EPS)
                )
            ) * gp
        else:
            t *= 0.35 + 0.65 * gp
        R = blank(sum(mask))
        R[:, z] = A[mask, :][:, z] * b.array(t)[:, None]
        add(R, s[prefix + "_weight"])
    if s["stage1_zlp_core_fartail_enabled"]:
        cw = max(s["stage1_zlp_core_anchor_width_meV"], EPS)
        tw = max(0.25 * cw, 1)
        mask = a <= cw + 3 * tw
        if s["stage1_zlp_core_anchor_weight"] > 0 and sum(mask) >= 3:
            t = expit((cw - a[mask]) / tw)
            R = blank(sum(mask))
            R[:, z] = A[mask, :][:, z] * b.array(t)[:, None]
            add(R, s["stage1_zlp_core_anchor_weight"], y[mask] * t)
        tp, tm = tail_profile(e, s)
        side = s["stage1_zlp_fartail_anchor_side"]
        if side in ("gain", "left", "negative"):
            tm &= e < 0
        elif side in ("loss", "right", "positive"):
            tm &= e > 0
        if s["stage1_zlp_fartail_anchor_weight"] > 0 and sum(tm) >= 4:
            R = blank(sum(tm))
            R[:, z] = A[tm, :][:, z] * b.array(tp[tm])[:, None]
            add(R, s["stage1_zlp_fartail_anchor_weight"], y[tm] * tp[tm])
        if (
            s["stage1_zlp_gain_tail_prior_enabled"]
            and s["stage1_zlp_gain_tail_prior_weight"] > 0
        ):
            start = s["stage1_zlp_fartail_start_meV"]
            end = s["stage1_zlp_fartail_end_meV"]
            end = float(end) if np.size(end) and np.isfinite(end) else max(a)
            pos = (e > 0) & (e >= start) & (e <= end)
            neg = (e < 0) & (-e >= start) & (-e <= end)
            if sum(pos) >= 3 and sum(neg) >= 3:
                end = min(max(e[pos]), max(-e[neg]), end)
                step = max(s["stage1_zlp_gain_tail_pair_step_meV"], EPS)
                pairs = grid(start, step, end)
                if len(pairs) < 2:
                    pairs = np.linspace(start, end, 3)
                R = blank(len(pairs))
                tol = max(step, 2 * np.median(abs(np.diff(e))))
                keep = []
                for i, energy in enumerate(pairs):
                    ip = np.argmin(abs(e - energy))
                    im = np.argmin(abs(e + energy))
                    if abs(e[ip] - energy) <= tol and abs(e[im] + energy) <= tol:
                        R[i, z] = (
                            A[im, z]
                            - max(s["stage1_zlp_gain_tail_prior_ratio"], 1) * A[ip, z]
                        )
                        keep.append(i)
                if keep:
                    add(R[keep], s["stage1_zlp_gain_tail_prior_weight"])
    return (
        (xp.concatenate(rows), xp.concatenate(rhs)) if rows else (blank(0), xp.zeros(0))
    )


def tail_constraints(A, y, e, s):
    mask = (abs(e) >= s["tail_overshoot_start_meV"]) & np.isfinite(y)
    if not s["tail_overshoot_constraint"] or sum(mask) < 5:
        return None, None
    margin = max(
        s["tail_overshoot_abs_margin"],
        s["tail_overshoot_noise_sigma"] * noise(y, mask),
        s["tail_overshoot_rel_margin"] * max(np.max(abs(y[mask])), EPS),
    )
    return A[mask], np.maximum(y[mask] + margin, margin)


def inner(A, y, w, R, r0, e, s, b, layout=None, bound=None, l1=True):
    xp = b.xp
    C = xp.concatenate([A * b.array(w)[:, None], R])
    d = xp.concatenate([b.array(y * w), r0])
    G, h = tail_constraints(b.host(A), y, e, s)
    penalty = np.zeros(A.shape[1])
    if l1 and layout is not None:
        penalty[np.r_[layout["pm_s"], layout["pm_a"]]] = s["lambda_pm_l1"]
    bound = bound or {}
    requested = s["inner_solver"]
    orders = {
        "quadprog": ["quadprog", "lsqlin", "lsqnonneg"],
        "lsqlin": ["lsqlin", "quadprog", "lsqnonneg"],
        "lsqnonneg": ["lsqnonneg"],
    }
    methods = orders.get(requested, ["lsqlin", "lsqnonneg", "quadprog"])
    failures = []
    for method in methods:
        try:
            if method == "lsqnonneg":
                coef, info = b.solve(C, d, max_iter=s["inner_solver_max_iter"])
                # This is the original explicit lsqnonneg fallback, which
                # clips finite bounds and does not enforce tail inequalities.
                coef = np.clip(coef, bound.get("lb", 0), bound.get("ub", np.inf))
            else:
                coef, info = b.solve(
                    C,
                    d,
                    G=G,
                    h=h,
                    l1=penalty if method == "quadprog" else None,
                    max_iter=s["inner_solver_max_iter"],
                    ridge=1e-12 if method == "quadprog" else 0,
                    **bound,
                )
            info.update(
                original_method=method,
                fallback_failures=failures,
                max_tail_overshoot=float(np.max(G @ coef - h))
                if G is not None
                else np.nan,
            )
            return coef, info
        except RuntimeError as exc:
            failures.append(f"{method}: {exc}")
    raise RuntimeError("内层求解全部失败: " + "; ".join(failures))


def decode(theta, e, y, em, s, b):
    p = decode_theta(theta, s)
    fs = adaptive(e, y, s)
    A, l, z, k = design(e, em, p, fs, b)
    w = weights(e, y, fs)
    R, r0 = regularization(em, l, p["T_soft"], A.shape[1], fs, A, e, y, b)
    coef, info = inner(A, y, w, R, r0, e, fs, b, l)
    if fs["kernel_mode"] in ("full_zlp", "fitted_zlp"):
        seed = z @ b.array(coef[l["zlp"]])
        ks = shift_matrix(e, normalize(seed, b), em, b)
        ka = shift_matrix(e, normalize(seed, b), -em, b)
        A[:, l["pm_s"]] = ks
        A[:, l["pm_a"]] = ka
        k = dict(stokes=ks, anti=ka, base=normalize(seed, b))
        coef, info2 = inner(A, y, w, R, r0, e, fs, b, l)
        info["fitted_zlp_kernel_refit"] = info2
    c = b.array(coef)
    model = b.host(A @ c)
    zlp = b.host(z @ c[l["zlp"]])
    vib = b.host(k["stokes"] @ c[l["pm_s"]] + k["anti"] @ c[l["pm_a"]])
    res = np.r_[(model - y) * w, b.host(R @ c - r0)]
    return dict(
        params=p,
        coef=coef,
        model_norm=model,
        zlp_norm=zlp,
        zlp_component_norm=b.host(z * c[l["zlp"]][None, :]),
        vib_norm=vib,
        bg_norm=np.zeros_like(y),
        PM_stokes=coef[l["pm_s"]],
        PM_antistokes=coef[l["pm_a"]],
        optimizer_residual=res,
        weights=w * w,
        layout=l,
        EM=em,
        T_soft=p["T_soft"],
        inner_solver=info,
        effective_settings=fs,
    )


def phi(theta, s, center_scale=None):
    n = int(s["num_zlp_components"])
    scale = center_scale or s.get("stage1_hierarchy_context", {}).get(
        "center_scale_meV", 1
    )
    return np.r_[
        col(theta)[1 : 1 + n] / max(scale, EPS),
        np.log(np.maximum(col(theta)[1 + n : 1 + 5 * n], EPS)),
    ]


def extra_residual(theta, s):
    rows = []
    p = decode_theta(theta, s)
    n = int(s["num_zlp_components"])
    if (
        s["stage1_zlp_gain_asym_shape_prior_enabled"]
        and s["stage1_zlp_gain_asym_shape_prior_weight"] > 0
    ):
        i = max(0, min(n - 1, round(s["stage1_zlp_gain_asym_prior_start_index"]) - 1))
        wr = max(s["stage1_zlp_gain_asym_width_ratio"], 1)
        sr = max(s["stage1_zlp_gain_asym_shape_ratio"], 1)
        w = np.maximum(0, np.log(wr * p["sR"][i:] / np.maximum(p["sL"][i:], EPS)))
        ratio = p["mR"][i:] / p["mL"][i:] if is_pvoigt(s) else p["mL"][i:] / p["mR"][i:]
        rows.append(
            np.sqrt(s["stage1_zlp_gain_asym_shape_prior_weight"])
            * np.r_[w, np.maximum(0, np.log(sr * ratio))]
        )
    ctx = s.get("stage1_hierarchy_context", {})
    if ctx and ctx.get("enabled"):
        ph = phi(theta, s)
        cost = max(ctx.get("data_cost", 1), 1e-10)
        nphi = len(ph)
        if len(ctx.get("template_phi", [])) == nphi:
            frac = ctx.get(
                "template_cost_fraction", s["stage1_cluster_template_cost_fraction"]
            ) * np.clip(ctx.get("membership", 1), 0, 1)
            if frac > 0:
                rows.append(
                    np.sqrt(frac * cost / nphi)
                    * (ph - col(ctx["template_phi"]))
                    / np.maximum(ctx.get("template_scale", np.ones(nphi)), EPS)
                )
        neighbors = np.asarray(ctx.get("neighbor_phi", []))
        ws = col(ctx.get("neighbor_weight", np.ones(len(neighbors))))
        if s["stage1_spatial_zlp_enabled"] and neighbors.size:
            scale = np.maximum(ctx.get("spatial_scale", np.ones(nphi)), EPS)
            delta = max(s["stage1_spatial_zlp_huber_delta"], EPS)
            frac = max(s["stage1_spatial_zlp_weight"], 0)
            for nb, w in zip(neighbors, ws):
                u = (ph - nb) / scale
                r = np.sign(u) * np.sqrt(
                    np.where(abs(u) <= delta, u * u, 2 * delta * abs(u) - delta * delta)
                )
                rows.append(np.sqrt(frac * cost * w / max(nphi * sum(ws), EPS)) * r)
    return np.concatenate(rows) if rows else np.empty(0)


def metrics(y, model, e, s):
    r = y - model
    w = weights(e, y, s)
    hi = abs(e) >= s["highfreq_start_meV"]
    return dict(
        R2=1 - sum(r * r) / max(sum((y - np.mean(y)) ** 2), EPS),
        NRMSE=np.sqrt(np.mean(r * r)) / max(np.ptp(y), EPS),
        weighted_NRMSE=np.sqrt(np.mean((w * r) ** 2)) / max(np.ptp(w * y), EPS),
        highfreq_NRMSE=np.sqrt(np.mean(r[hi] ** 2)) / max(np.ptp(y[hi]), EPS)
        if sum(hi) >= 5
        else np.nan,
        residual_RMS=np.sqrt(np.mean(r * r)),
        max_abs_residual=max(abs(r)),
    )


def candidate(fast, cand, e, y, s):
    fm = metrics(y, fast["model_norm"], e, s)
    cm = metrics(y, cand["model_norm"], e, s)
    mask = abs(e) >= s["tail_overshoot_start_meV"]

    def over(f):
        if sum(mask) < 5:
            return np.nan
        margin = max(
            s["tail_overshoot_abs_margin"],
            s["tail_overshoot_noise_sigma"] * noise(y, mask),
            s["tail_overshoot_rel_margin"] * max(np.max(abs(y[mask])), EPS),
        )
        return np.max(f["model_norm"][mask] - y[mask] - margin)

    fover = over(fast)
    cover = over(cand)
    accept = cand["inner_solver"].get("original_method") == "lsqnonneg" or (
        cm["weighted_NRMSE"] <= 1.03 * max(fm["weighted_NRMSE"], EPS)
        and (
            (fover > 0 and cover < 0.55 * fover)
            or cm["highfreq_NRMSE"] < 0.95 * fm["highfreq_NRMSE"]
        )
    )
    result = cand if accept else fast
    result["inner_solver"]["accepted"] = bool(accept)
    return result


def solve(e, y, settings=None, backend=None, previous=None, progress=None):
    s = defaults(settings)
    b = backend or Backend()
    start = time.perf_counter()
    e, y, yn, norm = prepare(e, y, s)
    e, alignment = align(e, yn, s)
    previous = previous if previous is not None else s.get("initial_theta", [])
    rounds = s["stage1_rounds"]
    rounds = rounds if isinstance(rounds, list) else [rounds]
    if np.size(previous) and s["stage1_warm_start_skip_coarse"] and len(rounds) > 1:
        rounds = rounds[1:]
    best = None
    history = []
    for ri, round_options in enumerate(rounds):
        rs = defaults(dict(s, **round_options))
        em = make_em(e, rs)
        lb, ub = bounds(rs)
        theta = initial(e, yn, rs, previous)
        free = ub > lb
        theta[~free] = lb[~free]
        obj_s = dict(
            rs,
            inner_solver=rs["nonlinear_inner_solver"],
            tail_overshoot_constraint=rs["nonlinear_tail_overshoot_constraint"],
            lambda_pm_l1=0,
        )
        count = [0]
        tick = time.perf_counter()

        def expand_theta(v):
            t = theta.copy()
            t[free] = v
            return t

        def objective(v):
            b.check()
            t = expand_theta(v)
            fit = decode(t, e, yn, em, obj_s, b)
            count[0] += 1
            if progress and count[0] % 25 == 0:
                progress(f"Stage 1 · {rs['name']} · 目标评估 {count[0]}")
            return np.r_[fit["optimizer_residual"], extra_residual(t, rs)]

        def jac(v):
            r = objective(v)
            J = np.empty((len(r), len(v)))
            for j in range(len(v)):
                d = 1e-4 * (1 if v[j] >= 0 else -1) * max(abs(v[j]), 1)
                d = d if lb[free][j] <= v[j] + d <= ub[free][j] else -d
                vv = v.copy()
                vv[j] = np.clip(vv[j] + d, lb[free][j], ub[free][j])
                J[:, j] = (objective(vv) - r) / (vv[j] - v[j])
            return J

        opt = least_squares(
            objective,
            theta[free],
            jac=jac,
            bounds=(lb[free], ub[free]),
            method="trf",
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=int(rs["max_nonlinear_iter"]) + 1,
        )
        th = expand_theta(opt.x)
        fast = decode(th, e, yn, em, obj_s, b)
        try:
            cand = decode(th, e, yn, em, rs, b)
            fit = candidate(fast, cand, e, yn, rs)
        except RuntimeError as exc:
            fit = fast
            fit["inner_solver"]["candidate_failure"] = str(exc)
        m = metrics(yn, fit["model_norm"], e, rs)
        fit.update(
            metrics=m,
            theta=th,
            EM=em,
            settings=rs,
            round_name=rs["name"],
            resnorm=2 * opt.cost,
        )
        history.append(
            dict(
                m,
                round_name=rs["name"],
                num_EM=len(em),
                elapsed_seconds=time.perf_counter() - tick,
                funcCount=count[0],
                iterations=opt.nfev,
                exitflag=int(opt.status),
            )
        )
        if best is None or m["NRMSE"] < best["metrics"]["NRMSE"]:
            best = fit
        previous = th
        if (
            s["stage1_early_stop_enabled"]
            and ri + 1 >= s["stage1_early_stop_min_rounds"]
            and ri + 1 < len(rounds)
            and m["R2"] >= s["stage1_early_stop_R2"]
            and m["weighted_NRMSE"] <= s["stage1_early_stop_weighted_NRMSE"]
            and (
                not np.isfinite(m["highfreq_NRMSE"])
                or m["highfreq_NRMSE"] <= s["stage1_early_stop_highfreq_NRMSE"]
            )
        ):
            break
    b.sync()
    p = best["params"]
    eff = best["effective_settings"].copy()
    eff["stage1_hierarchy_context"] = {}
    result = dict(
        analysis_mode="pearson_sparse_stage1",
        ene_raw=e,
        ene_corrected=e - p["E_shift"],
        raw_data=y,
        raw_data_norm=yn,
        norm_val=norm,
        fitted_spectrum_norm=best["model_norm"],
        fitted_spectrum=best["model_norm"] * norm,
        zlp_only=best["zlp_norm"] * norm,
        vib_only=best["vib_norm"] * norm,
        residual_norm=yn - best["model_norm"],
        residual=(yn - best["model_norm"]) * norm,
        metrics=best["metrics"],
        EM_final=best["EM"],
        PM_stokes=best["PM_stokes"],
        PM_antistokes=best["PM_antistokes"],
        theta=best["theta"],
        T_soft=p["T_soft"],
        stage1=best,
        settings=eff,
        original_settings=s,
        stage1_zlp_alignment=alignment,
        iter_history=history,
        stage1_elapsed_seconds=time.perf_counter() - start,
        T_fit=np.nan,
        T_uncertainty=np.nan,
    )
    result["total_energy_shift_meV"] = alignment["applied_shift_meV"] + p["E_shift"]
    result["weights_used"] = best["weights"]
    return result
