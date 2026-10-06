"""Attention-paired hard-BE thermometry, independent soft modes, original scan."""

import time
import numpy as np
from .numeric import (
    EPS,
    KB,
    col,
    mov,
    noise,
    unit_scale,
    expit,
    interp,
    grid,
    smooth_mask,
    bands,
)
from .settings import defaults
from .backend import Backend
from . import stage1
from .acceleration import LinearWorkspace, update_hard_columns


def attention(r, s):
    e = col(r["ene_corrected"])
    a = abs(e)
    hard = sorted(abs(col(s["stage2_window_meV"])))
    win = sorted(abs(col(s["stage2_attention_window_meV"])))
    win = [min(win[0], hard[0]), max(win[1], hard[1])]
    broad = (a >= win[0]) & (a <= win[1])
    sig = mov(abs(r["raw_data_norm"] - r["stage1"]["zlp_norm"]), 7, "median")
    grad = np.r_[0, abs(np.diff(mov(sig, 9)))]
    res = abs(r["residual_norm"])
    pm = np.zeros_like(e)
    em = col(r["EM_final"])
    ps = col(r["PM_stokes"])
    pa = col(r["PM_antistokes"])
    sigma = max(s["stage2_attention_peak_sigma_meV"], EPS)
    scale = max(np.max(np.r_[ps, pa, EPS]), EPS)
    for energy, vs, va in zip(em, ps, pa):
        pm += vs / scale * np.exp(
            -0.5 * ((e - energy) / sigma) ** 2
        ) + va / scale * np.exp(-0.5 * ((e + energy) / sigma) ** 2)
    feature = (
        0.48 * unit_scale(sig, broad)
        + 0.20 * unit_scale(grad, broad)
        + 0.22 * unit_scale(pm, broad)
        + 0.10 * unit_scale(res, broad)
    )
    anti = broad & (e < -hard[0])
    if np.any(anti):
        af = (
            0.52 * unit_scale(sig, anti)
            + 0.20 * unit_scale(grad, anti)
            + 0.22 * unit_scale(pm, anti)
            + 0.06 * unit_scale(res, anti)
        )
        gate = anti & (
            (sig >= s["stage2_antistokes_noise_sigma"] * noise(sig, anti))
            | (pm >= 0.04 * max(pm[anti]))
        )
        feature[gate] = np.maximum(
            feature[gate],
            np.minimum(1, s["stage2_antistokes_attention_boost"] * af[gate]),
        )
    feature = np.maximum(feature, 0.65 * interp(e, feature, -e))
    feature[~broad | ~np.isfinite(feature)] = 0
    feature = np.clip(mov(feature, 5), 0, 1)
    active = broad & (feature >= s["stage2_attention_threshold"])
    active = broad & (active | (interp(e, active.astype(float), -e, "nearest") > 0))
    w = np.ones_like(e)
    w[broad] = s["stage2_attention_floor"] + s["stage2_attention_gain"] * feature[broad]
    w = np.maximum(w, s["stage2_attention_floor"])
    att = dict(
        energy=e,
        feature_score=feature,
        weight=w,
        valid_mask=broad,
        window_meV=win,
        hard_be_window_meV=hard,
        active_mask=active,
        selected_fraction=sum(active) / max(sum(broad), 1),
        bands_meV=bands(e, active),
    )
    return tail_masks(att, e, s)


def tail_masks(att, e, s):
    score = np.clip(att["feature_score"], 0, 1).copy()
    score[att["active_mask"]] = 1
    if "fit_candidate_mask" in att:
        score[att["fit_candidate_mask"]] = np.maximum(
            score[att["fit_candidate_mask"]],
            s["stage2_attention_tail_signal_threshold"],
        )
    high = abs(e) >= max(
        s["stage2_attention_tail_min_abs_meV"], s["highfreq_start_meV"]
    )
    signal = (
        att["valid_mask"]
        & high
        & ((score >= s["stage2_attention_tail_signal_threshold"]) | att["active_mask"])
    )
    if "fit_candidate_mask" in att:
        signal |= att["valid_mask"] & high & att["fit_candidate_mask"]
    att.update(signal_score=score, signal_mask=signal, tail_mask=high & ~signal)
    return att


def att_score(att):
    score = np.maximum(att["feature_score"], att.get("signal_score", 0)).copy()
    score[att["active_mask"]] = 1
    return np.clip(score, 0, 1)


def pairs(em, att):
    score = att_score(att)
    pos = np.maximum(interp(att["energy"], score, em), 0)
    neg = np.maximum(interp(att["energy"], score, -em), 0)
    return np.sqrt(pos * neg), np.maximum(pos, neg)


def dense_grid(att, s, kind, hard=None):
    p = f"stage2_{kind}_dense"
    if not s[p + "_em_enabled"]:
        return np.empty(0)
    lo = max(min(abs(col(s["stage2_window_meV"]))), max(s["em_min_meV"], 0))
    hi = max(
        max(abs(col(s["stage2_attention_window_meV"]))),
        max(abs(col(s["stage2_window_meV"]))),
    )
    if np.size(s[p + "_min_meV"]):
        lo = max(0, float(s[p + "_min_meV"]))
    if not np.isfinite(hi) or hi <= lo:
        hi = 220
    hi += max(s[p + "_extra_meV"], 0)
    if np.size(s[p + "_max_meV"]):
        hi = min(hi, float(s[p + "_max_meV"]))
    hi = min(hi, max(abs(att["energy"])))
    em = grid(lo, max(0.5, s[p + "_step_meV"]), hi)
    em = em[em > 0]
    score, _ = pairs(em, att)
    if kind == "hard":
        return em[score >= s["stage2_hard_be_pair_threshold"]]
    keep = score < s["stage2_soft_dense_pair_score_max"]
    gap = max(s["stage2_soft_dense_hard_exclusion_meV"], 0)
    if hard is not None and len(hard) and gap > 0:
        keep &= np.min(abs(em[:, None] - hard[None, :]), axis=1) > gap
    return em[keep]


def make_basis(r, att, s):
    old = col(r["EM_final"])
    score, feature = pairs(old, att)
    threshold = np.clip(s["stage2_hard_be_pair_threshold"], 0, 1)
    floor = np.clip(s["stage2_hard_be_pair_floor"], 0, threshold)
    minimum = int(s["stage2_ratio_min_pairs"])
    hard = np.ones(len(old), bool)
    status = "legacy_all_hard_be"
    if s["stage2_attention_hardbe_only_enabled"]:
        hard = score >= threshold
        status = "threshold_ok"
        if sum(hard) < minimum:
            cand = np.flatnonzero(~hard & (score >= floor))
            order = np.argsort(-score[cand], kind="stable")
            add = cand[order[: max(0, minimum - sum(hard))]]
            if len(add):
                hard[add] = True
                status = "floor_backfill"
        if sum(hard) < minimum:
            status = "insufficient_hard_be_attention_pairs"
    eh = np.unique(np.r_[old[hard], dense_grid(att, s, "hard")])
    es = np.unique(np.r_[old[~hard], dense_grid(att, s, "soft", eh)])
    em = np.unique(np.r_[eh, es])
    hard2 = np.any(abs(em[:, None] - eh[None, :]) <= 1e-8, axis=1)
    soft = ~hard2
    ps = np.maximum(interp(old, col(r["PM_stokes"]), em, fill="extrapolate"), 0)
    pa = np.maximum(interp(old, col(r["PM_antistokes"]), em, fill="extrapolate"), 0)
    sc, feat = pairs(em, att)
    return dict(
        energy_axis=col(r["ene_corrected"]),
        EM_all=em,
        EM_stage1=old,
        EM_hard=em[hard2],
        EM_soft=em[soft],
        hard_idx=hard2,
        soft_idx=soft,
        hard_fraction=sum(hard2) / max(len(em), 1),
        hard_selection_status=status,
        hard_pair_score=sc,
        hard_pair_score_hard=sc[hard2],
        soft_pair_score=sc[soft],
        hard_feature_score=feat,
        stage1_PM_stokes_soft=ps[soft],
        stage1_PM_antistokes_soft=pa[soft],
        num_hard_pairs=int(sum(hard2)),
        num_soft_pairs=int(sum(soft)),
        min_hard_pairs=minimum,
        hard_pair_threshold=threshold,
        hard_pair_floor=floor,
    )


def local_diff(x, max_gap):
    x = col(x)
    n = len(x)
    if n < 3:
        return np.empty((0, n))
    order = np.argsort(x)
    gaps = np.diff(x[order])
    keep = np.flatnonzero(
        (gaps[:-1] > 0)
        & (gaps[1:] > 0)
        & (gaps[:-1] <= max_gap)
        & (gaps[1:] <= max_gap)
    )
    D = np.zeros((len(keep), n))
    ref = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 1
    for j, i in enumerate(keep):
        D[j, order[i : i + 3]] = (
            ref / max(0.5 * (gaps[i] + gaps[i + 1]), EPS)
        ) ** 2 * np.array([1, -2, 1])
    return D


def regularization(basis, s, q):
    nh = len(basis["EM_hard"])
    ns = len(basis["EM_soft"])
    n = nh + 2 * ns
    rows = []
    rhs = []

    def add(a, w, target=None):
        rows.append(np.sqrt(w) * a)
        rhs.append(np.zeros(len(a)) if target is None else np.sqrt(w) * target)

    D = local_diff(basis["EM_hard"], s["stage2_pm_smooth_max_gap_meV"])
    if s["stage2_pm_smooth_weight"] > 0 and len(D):
        R = np.zeros((len(D), n))
        R[:, :nh] = D
        add(R, s["stage2_pm_smooth_weight"])
    D = local_diff(basis["EM_soft"], s["stage2_pm_smooth_max_gap_meV"])
    if s["stage2_soft_pm_smooth_weight"] > 0 and len(D):
        for j in range(2):
            R = np.zeros((len(D), n))
            R[:, nh + j * ns : nh + (j + 1) * ns] = D
            add(R, s["stage2_soft_pm_smooth_weight"])
    if ns:
        for key in ("stage2_soft_pm_l2_weight", "stage2_soft_pm_stage1_prior_weight"):
            if s[key] > 0:
                R = np.zeros((2 * ns, n))
                R[:ns, nh : nh + ns] = np.eye(ns)
                R[ns:, nh + ns :] = np.eye(ns)
                target = (
                    np.r_[
                        basis["stage1_PM_stokes_soft"],
                        basis["stage1_PM_antistokes_soft"],
                    ]
                    if key.endswith("prior_weight")
                    else None
                )
                add(R, s[key], target)
        if s["stage2_soft_be_weight"] > 0:
            R = np.zeros((ns, n))
            R[np.arange(ns), nh + np.arange(ns)] = -q
            R[np.arange(ns), nh + ns + np.arange(ns)] = 1
            add(R, s["stage2_soft_be_weight"])
    return (
        (np.concatenate(rows), np.concatenate(rhs))
        if rows
        else (np.empty((0, n)), np.empty(0))
    )


def zlp_basis(r, mask, s):
    e = col(r["ene_corrected"])
    z = col(r["stage1"]["zlp_norm"])
    columns = []
    names = []
    limits = []
    penalties = []
    if s["stage2_zlp_refine_enabled"] and sum(mask) >= 8:
        derivative = np.gradient(z) / np.maximum(abs(np.gradient(e)), EPS)
        tg = np.log1p((abs(e) / max(s["stage2_zlp_refine_tail_scale_meV"], EPS)) ** 2)
        tg /= max(np.max(tg[mask]), EPS)
        tilt = np.tanh(e / max(s["stage2_zlp_refine_tilt_scale_meV"], EPS))
        for key, v in [
            ("scale", z),
            ("tilt", z * tilt),
            ("shift", -derivative),
            ("width", -e * derivative),
            ("tail", z * tg),
            ("tail_asym", z * tg * tilt),
        ]:
            p = "stage2_zlp_refine_" + key
            if s[p + "_enabled"] and np.any(abs(v[mask]) > 0):
                columns.append(v)
                names.append(key)
                limits.append(
                    abs(s[p + ("_bound_meV" if key == "shift" else "_bound")])
                )
                penalties.append(max(s[p + "_l2"], 0))
    return dict(
        B=np.column_stack(columns) if columns else np.empty((len(e), 0)),
        names=names,
        lb=-col(limits),
        ub=col(limits),
        l2=col(penalties),
    )


class ScanWorkspace:
    """Fixed scan blocks and bounds; update only temperature-dependent entries."""
    def __init__(self, basis, settings, zr, ks, ka, y, w, backend):
        b, xp = backend, backend.xp
        self.b = b
        self.nh, self.ns = len(basis["EM_hard"]), len(basis["EM_soft"])
        nh, ns, nz = self.nh, self.ns, zr["B"].shape[1]
        nb = nh + 2 * ns
        self.ks_h, self.ka_h = ks[:, basis["hard_idx"]], ka[:, basis["hard_idx"]]
        self.ks_s, self.ka_s = ks[:, basis["soft_idx"]], ka[:, basis["soft_idx"]]
        self.A = xp.column_stack([self.ks_h + self.ka_h,
                                  self.ks_s, self.ka_s, b.array(zr["B"])])
        self.hard_fortran = self.ks_h.flags.f_contiguous
        self.ks_h_native = xp.ascontiguousarray(self.ks_h)
        self.ka_h_native = xp.ascontiguousarray(self.ka_h)
        R, rhs = regularization(basis, settings, np.zeros(ns))
        self.be_weight = settings["stage2_soft_be_weight"]
        self.be_rows = np.arange(len(R) - ns, len(R)) if ns and self.be_weight > 0 else None
        if nz:
            R = np.pad(R, ((0, 0), (0, nz)))
            Z = np.zeros((nz, nb + nz))
            Z[:, nb:] = np.diag(np.sqrt(zr["l2"]))
            R = np.r_[R, Z]
            rhs = np.r_[rhs, np.zeros(nz)]
        self.R, self.rhs = b.array(R), b.array(rhs)
        self.bounds = dict(lb=np.r_[np.zeros(nb), zr["lb"]],
                           ub=np.r_[np.full(nb, np.inf), zr["ub"]])
        self.linear = LinearWorkspace(y, w, rhs, nb + nz, b)

    def update(self, qh, qs):
        b, xp = self.b, self.b.xp
        if self.A.flags.c_contiguous:
            update_hard_columns(self.ks_h_native, self.ka_h_native, qh, self.A, b)
            # Retain the original matrix layout for BLAS diagnostics as well.
            hard = xp.array(self.A[:, :self.nh], order="F" if self.hard_fortran else "C")
        else:
            hard = self.ks_h + self.ka_h * b.array(qh)[None, :]
            self.A[:, :self.nh] = hard
        if self.be_rows is not None:
            self.R[self.be_rows, self.nh + np.arange(self.ns)] = b.array(np.sqrt(self.be_weight) * -qs)
        return hard


def smooth_cost(T, c, s):
    T = col(T)
    c = col(c)
    cs = c.copy()
    good = np.flatnonzero(np.isfinite(T) & np.isfinite(c))
    points = max(3, int(round(s["stage2_cost_smooth_points"])))
    points += int(points % 2 == 0)
    points = min(points, len(good))
    if points < 3:
        return cs
    for i in good:
        distance = abs(T[good] - T[i])
        order = np.argsort(distance, kind="stable")
        band = distance[order[min(points, len(good) - 1)]]
        if not np.isfinite(band) or band <= 0:
            continue
        ids = good[order[:points]]
        x = T[ids] - T[i]
        w = (1 - np.minimum(abs(x) / band, 1) ** 3) ** 3
        valid = w > 0
        x = x[valid]
        y = c[ids][valid]
        w = w[valid]
        if len(x) < 3:
            continue
        X = np.column_stack([np.ones(len(x)), x, x * x])
        normal = X.T @ (w[:, None] * X)
        if 1 / np.linalg.cond(normal, 1) < 1e-12:
            X = X[:, :2]
            normal = X.T @ (w[:, None] * X)
        if 1 / np.linalg.cond(normal, 1) >= 1e-12:
            cs[i] = np.clip(np.linalg.solve(normal, X.T @ (w * y))[0], min(y), max(y))
    return cs


def cost_valley(T, c):
    good = np.isfinite(T) & np.isfinite(c)
    T = T[good]
    c = c[good]
    if len(T) < 3:
        return dict(
            available=False,
            uncertainty_K=np.nan,
            lower_K=np.nan,
            upper_K=np.nan,
            best_T=np.nan,
        )
    order = np.argsort(T)
    T = T[order]
    c = c[order]
    i = int(np.argmin(c))
    near = c <= c[i] + 0.1 * max(abs(c[i]), EPS)
    boundary = i in (0, len(T) - 1)
    if sum(near) >= 2:
        lo = min(T[near])
        hi = max(T[near])
        unc = (hi - lo) / 2
    else:
        step = np.median(np.diff(T))
        lo = max(min(T), T[i] - step)
        hi = min(max(T), T[i] + step)
        unc = step
    if boundary:
        unc = max(unc, 0.25 * (max(T) - min(T)))
    return dict(
        available=True,
        best_T=float(T[i]),
        best_cost=float(c[i]),
        lower_K=float(lo),
        upper_K=float(hi),
        uncertainty_K=float(unc),
        boundary=boundary,
        in_valley=near,
    )


def solve(r, settings=None, backend=None, progress=None):
    s = defaults(settings or r.get("settings"))
    b = backend or Backend()
    xp = b.xp
    tick = time.perf_counter()
    e = col(r["ene_corrected"])
    y = col(r["raw_data_norm"])
    z = col(r["stage1"]["zlp_norm"])
    att = attention(r, s)
    hard_win = sorted(abs(col(s["stage2_window_meV"])))
    hw = (abs(e) >= hard_win[0]) & (abs(e) <= hard_win[1])
    mirrored = interp(e, att["feature_score"], -e)
    miractive = interp(e, att["active_mask"].astype(float), -e, "nearest") > 0
    pairfeat = np.maximum(att["feature_score"], np.clip(mirrored, 0, 1))
    seed = (
        hw
        & att["valid_mask"]
        & (
            (pairfeat >= s["stage2_attention_fit_threshold"])
            | att["active_mask"]
            | miractive
        )
    )
    if sum(seed) < 12:
        seed = hw & att["valid_mask"]
    support = smooth_mask(hw & att["valid_mask"], e, s["stage2_fit_sigmoid_taper_meV"])
    alpha = np.clip(
        support
        * np.maximum(smooth_mask(seed, e, s["stage2_fit_sigmoid_taper_meV"]), pairfeat),
        0,
        1,
    )
    mask = att["valid_mask"] & (alpha >= np.clip(s["stage2_fit_alpha_min"], 1e-5, 0.25))
    if sum(mask) < 12:
        alpha = support
        mask = att["valid_mask"] & (alpha > 1e-4)
    if sum(mask) < 12:
        return dict(
            status="insufficient_window_points",
            T_fit=np.nan,
            T_uncertainty=np.nan,
            enabled=True,
        )
    att.update(fit_candidate_mask=seed, fit_alpha=alpha, pair_feature=pairfeat)
    att["weight"] = np.clip(
        np.maximum(
            att["weight"],
            s["stage2_attention_floor"] + (1 - s["stage2_attention_floor"]) * pairfeat,
        ),
        0,
        1,
    )
    bas = make_basis(r, att, s)
    em = bas["EM_all"]
    hard = bas["hard_idx"]
    soft = bas["soft_idx"]
    nh = sum(hard)
    ns = sum(soft)
    A, l, _, kernels = stage1.design(col(r["ene_raw"]), em, r["stage1"]["params"], s, b)
    if s["kernel_mode"] in ("full_zlp", "fitted_zlp"):
        kernels = dict(
            stokes=stage1.shift_matrix(e, stage1.normalize(z, b), em, b),
            anti=stage1.shift_matrix(e, stage1.normalize(z, b), -em, b),
        )
    floor = np.clip(s["stage2_fit_sigmoid_min_weight"], 0, 0.95)
    local = (floor + (1 - floor) * alpha) * att["weight"]
    joint = np.maximum(
        np.clip(s["stage2_global_joint_weight_floor"], 0.01, 1), np.clip(local, 0, 1)
    )
    w = stage1.weights(e, y, s) * np.sqrt(joint)
    anti = (e < -hard_win[0]) & (e >= -hard_win[1])
    w[anti] *= np.sqrt(max(s["stage2_antistokes_fit_weight_boost"], 1))
    if s["stage2_far_tail_downweight_enabled"]:
        scale = np.clip(s["stage2_far_tail_weight_scale"], 0.02, 1)
        if s["stage2_attention_tail_enabled"]:
            release = expit(
                (att["signal_score"] - s["stage2_attention_tail_signal_threshold"])
                / max(s["stage2_attention_tail_soft_width"], EPS)
            )
            tm = (
                abs(e)
                >= max(s["stage2_attention_tail_min_abs_meV"], s["highfreq_start_meV"])
            ) | att["tail_mask"]
            profile = np.ones_like(e)
            profile[tm] = scale + (1 - scale) * release[tm]
        else:
            profile = scale + (1 - scale) * expit(
                (s["stage2_far_tail_downweight_start_meV"] - abs(e))
                / max(s["stage2_far_tail_downweight_width_meV"], EPS)
            )
        w *= np.sqrt(np.clip(profile, scale, 1))
    zr = zlp_basis(r, mask, s)
    nb = nh + 2 * ns
    nz = zr["B"].shape[1]
    ks = kernels["stokes"]
    ka = kernels["anti"]
    scanset = {}
    scan_s = dict(
        s,
        inner_solver="lsqlin",
        tail_overshoot_constraint=s["stage2_scan_tail_overshoot_constraint"],
    )
    target = y - z
    work = ScanWorkspace(bas, s, zr, ks, ka, target, w, b)
    A2, Reg, rhs, bounds = work.A, work.R, work.rhs, work.bounds
    # Cost reporting stays on NumPy, as before, including in GPU hybrid mode.
    reg_host = b.host(Reg).copy() if b.device == "gpu" else Reg
    rhs_host = b.host(rhs)
    pair_weights = np.clip(
        bas["hard_pair_score_hard"],
        np.clip(s["stage2_effective_pair_weight_floor"], 0, 1), 1)
    weight_mean = max(np.mean(w * w), EPS)
    attention_weight_mean = max(np.mean(w[mask] ** 2), EPS)

    def evaluate(T):
        b.check()
        T = float(T)
        if T in scanset:
            return scanset[T]
        qh = np.exp(-bas["EM_hard"] / max(KB * T, EPS))
        qs = np.exp(-bas["EM_soft"] / max(KB * T, EPS))
        hardsig = work.update(qh, qs)
        if b.device == "gpu" and work.be_rows is not None:
            reg_host[work.be_rows, nh + np.arange(ns)] = np.sqrt(work.be_weight) * -qs
        coef, info = stage1.inner(
            A2,
            target,
            w,
            Reg,
            rhs,
            e,
            scan_s,
            b,
            bound=bounds,
            l1=False,
            workspace=work.linear,
        )
        c = b.array(coef)
        delta = zr["B"] @ coef[nb:]
        model = z + b.host(A2 @ c)
        ps = np.zeros(len(em))
        pa = np.zeros(len(em))
        ps[hard] = coef[:nh]
        pa[hard] = ps[hard] * qh
        ps[soft] = coef[nh : nh + ns]
        pa[soft] = coef[nh + ns : nb]
        vh = b.host(hardsig @ c[:nh])
        vs = b.host(work.ks_s @ c[nh : nh + ns] + work.ka_s @ c[nh + ns : nb])
        res = model - y
        data = np.mean((res * w) ** 2) / weight_mean
        attcost = np.mean((res[mask] * w[mask]) ** 2) / attention_weight_mean
        rr = reg_host @ coef - rhs_host
        regcost = np.mean(rr * rr) if len(rr) else 0
        cost = (
            s["stage2_profile_cost_weight"]
            * (
                s["stage2_joint_rmse_cost_weight"] * data
                + s["stage2_attention_cost_weight"] * attcost
            )
            + s["stage2_regularization_cost_weight"] * regcost
        )
        log_ratio = np.log(np.maximum(ps[hard], EPS) / np.maximum(pa[hard], EPS))
        prediction = bas["EM_hard"] / max(KB * T, EPS)
        good = (
            np.isfinite(log_ratio)
            & np.isfinite(prediction)
            & (pair_weights > 0)
            & (ps[hard] > 0)
            & (pa[hard] > 0)
        )
        paircount = int(sum(good))
        linearity = np.nan
        if paircount >= 2:
            ww = pair_weights[good] / sum(pair_weights[good])
            observed = log_ratio[good]
            total = sum(ww * (observed - sum(ww * observed)) ** 2)
            error = sum(ww * (observed - prediction[good]) ** 2)
            linearity = 1 - error / total if total > EPS else float(error <= 1e-10)
        out = dict(
            T_fit=T,
            cost=float(cost),
            cost_raw=float(cost),
            data_cost=float(data),
            attention_cost=float(attcost),
            regularization_cost=float(regcost),
            coef=coef,
            model_norm=model,
            zlp_norm=z + delta,
            zlp_delta_norm=delta,
            vib_norm=vh + vs,
            vib_hard_norm=vh,
            vib_soft_norm=vs,
            PM_stokes=ps,
            PM_antistokes=pa,
            PM_stokes_hard=ps[hard],
            PM_antistokes_hard=pa[hard],
            PM_stokes_soft=ps[soft],
            PM_antistokes_soft=pa[soft],
            inner_solver=info,
            linearity_R2=linearity,
            effective_pair_count=paircount,
            zlp_refine=dict(
                mode_names=zr["names"], coef=coef[nb:], lb=zr["lb"], ub=zr["ub"]
            ),
        )
        scanset[T] = out
        if progress:
            progress(f"Stage 2 · {T:g} K · 已扫描 {len(scanset)} 点")
        return out

    def select():
        ts = np.array(sorted(scanset))
        costs = np.array([scanset[t]["cost_raw"] for t in ts])
        smooth = smooth_cost(ts, costs, s)
        return scanset[float(ts[np.nanargmin(smooth)])]

    requested = np.unique(col(s["stage2_T_grid"]))
    requested = requested[np.isfinite(requested) & (requested > 0)]
    if not len(requested):
        raise ValueError("温度扫描网格为空")
    if s["stage2_adaptive_grid_scan_enabled"] and len(requested) > max(
        3, s["stage2_adaptive_grid_min_full_points"]
    ):
        step = np.median(np.diff(requested))
        stride = max(
            1,
            int(
                np.floor(
                    max(s["stage2_adaptive_grid_coarse_step_K"], step) / step + 0.5
                )
            ),
        )
        coarse = np.unique(np.r_[requested[::stride], requested[-1]])
        for T in coarse:
            evaluate(T)
        best = select()
        fine = requested[
            abs(requested - best["T_fit"])
            <= max(s["stage2_adaptive_grid_fine_half_width_K"], 0)
        ]
        for T in fine:
            evaluate(T)
    else:
        for T in requested:
            evaluate(T)
    best = select()
    refined = []
    if s["stage2_temperature_refine_enabled"] and len(requested) >= 2:
        half = min(
            max(s["stage2_temperature_refine_half_width_K"], 0),
            np.median(np.diff(requested)),
        )
        tol = max(s["stage2_temperature_refine_tol_K"], 0.05)
        step = min(max(s["stage2_temperature_refine_initial_step_K"], tol), half)
        lo = max(min(requested), best["T_fit"] - half)
        hi = min(max(requested), best["T_fit"] + half)
        maximum = max(0, int(s["stage2_temperature_refine_max_evals"]))
        while len(refined) < maximum and step >= tol:
            trials = np.unique([best["T_fit"] - step, best["T_fit"] + step])
            improved = False
            for T in trials:
                if len(refined) >= maximum:
                    break
                if (
                    T < lo
                    or T > hi
                    or min(abs(np.array(list(scanset)) - T)) <= 0.25 * tol
                ):
                    continue
                cand = evaluate(T)
                refined.append(T)
                if cand["cost_raw"] < best["cost_raw"]:
                    best = cand
                    improved = True
            if not improved:
                step *= 0.5
    ts = np.array(sorted(scanset))
    costs = np.array([scanset[t]["cost_raw"] for t in ts])
    cs = smooth_cost(ts, costs, s)
    valley = cost_valley(ts, cs)
    unc = valley["uncertainty_K"]
    lin = best["linearity_R2"] if np.isfinite(best["linearity_R2"]) else 0.5
    reliability = np.clip(
        0.25 * min(1, att["selected_fraction"] / 0.18)
        + 0.25 * min(1, max(0, r["metrics"]["R2"] - 0.999) / 0.001)
        + 0.25 * np.exp(-unc / 220)
        + 0.25 * np.clip(lin, 0, 1),
        0,
        1,
    )
    if (
        best["T_fit"] <= min(s["T_soft_bounds"]) + s["stage2_boundary_margin_K"]
        or best["T_fit"] >= max(s["T_soft_bounds"]) - s["stage2_boundary_margin_K"]
    ):
        reliability *= 0.25
    b.sync()
    out = dict(
        best,
        enabled=True,
        status="ok",
        T_uncertainty=unc,
        T_grid=ts,
        cost_grid=costs,
        cost_grid_smooth=cs,
        cost_valley=valley,
        attention=att,
        basis=bas,
        EM_final=em,
        EM_hard=bas["EM_hard"],
        EM_soft=bas["EM_soft"],
        hard_idx=hard,
        soft_idx=soft,
        hard_pair_score=bas["hard_pair_score"],
        num_hard_pairs=nh,
        num_soft_pairs=ns,
        hard_be_attention_status=bas["hard_selection_status"],
        reliability_score=float(reliability),
        metrics=stage1.metrics(y, best["model_norm"], e, s),
        fit_mask=mask,
        fit_alpha=alpha,
        elapsed_seconds=time.perf_counter() - tick,
        temperature_profile=dict(
            T_grid=ts,
            cost_raw=costs,
            cost_smooth=cs,
            linearity_R2=np.array([scanset[t]["linearity_R2"] for t in ts]),
            effective_pair_count=np.array(
                [scanset[t]["effective_pair_count"] for t in ts]
            ),
        ),
    )
    out.update(
        T_cost_valley_lower_K=valley["lower_K"],
        T_cost_valley_upper_K=valley["upper_K"],
        min_hard_pairs=int(s["stage2_ratio_min_pairs"]),
    )
    return out


def promote(r, s=None, backend=None, progress=None):
    r = r.copy()
    s = defaults(s or r["settings"])
    s["run_stage2"] = True
    out = solve(r, s, backend, progress)
    r["stage2"] = out
    r["T_fit"] = out.get("T_fit", np.nan)
    r["T_uncertainty"] = out.get("T_uncertainty", np.nan)
    if out["status"] == "ok":
        r["reliability_score"] = out["reliability_score"]
        r["stage1_fitted_spectrum_norm"] = r["fitted_spectrum_norm"]
        r["stage1_EM_final"] = r["EM_final"]
        r["stage1_PM_stokes"] = r["PM_stokes"]
        r["stage1_PM_antistokes"] = r["PM_antistokes"]
        r["fitted_spectrum_norm"] = out["model_norm"]
        r["fitted_spectrum"] = out["model_norm"] * r["norm_val"]
        r["zlp_only"] = out["zlp_norm"] * r["norm_val"]
        r["vib_only"] = out["vib_norm"] * r["norm_val"]
        r["residual_norm"] = r["raw_data_norm"] - out["model_norm"]
        r["residual"] = r["residual_norm"] * r["norm_val"]
        r["metrics"] = out["metrics"]
        r["EM_final"] = out["EM_final"]
        r["PM_stokes"] = out["PM_stokes"]
        r["PM_antistokes"] = out["PM_antistokes"]
    return r
