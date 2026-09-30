"""Existing whole-cube sparse mode-grid calibration; no new pruning policy."""

import numpy as np
from . import stage1
from .numeric import EPS, col, mov, unit_scale, interp, grid


def spaced(em, score, count, spacing):
    selected = []
    for i in np.argsort(-col(score), kind="stable"):
        if not selected or min(abs(em[i] - np.asarray(selected))) >= spacing:
            selected.append(float(em[i]))
            if len(selected) >= max(1, int(round(count))):
                break
    return np.unique(selected)


def estimate(r, s, b):
    em = np.unique(col(r["EM_final"]))
    em = em[np.isfinite(em) & (em > 0)]
    if len(em) <= s["stage1_auto_sparse_pm_min_count"]:
        return em, dict(
            status="already_sparse", reference_count=len(em), selected_count=len(em)
        )
    _, _, _, k = stage1.design(r["ene_raw"], em, r["stage1"]["params"], s, b)
    target = r["stage1"]["vib_norm"]
    mass = np.maximum(abs(r["PM_stokes"]), abs(r["PM_antistokes"]))
    signal = mov(abs(r["raw_data_norm"] - r["stage1"]["zlp_norm"]), 7, "median")
    feature = np.maximum(
        interp(r["ene_corrected"], signal, em), interp(r["ene_corrected"], signal, -em)
    )
    feature = unit_scale(feature, np.ones(len(em), bool))
    score = unit_scale(mass, np.ones(len(em), bool)) + 0.45 * feature
    protect = (score >= s["stage1_auto_sparse_pm_feature_threshold"]) | (
        mass >= 0.02 * max(mass)
    )
    protected = (
        spaced(
            em[protect],
            score[protect] + 0.25 * feature[protect],
            min(s["stage1_auto_sparse_pm_protect_max_centers"], sum(protect)),
            max(0.75, 0.75 * s["stage1_auto_sparse_pm_min_spacing_meV"]),
        )
        if np.any(protect)
        else np.empty(0)
    )
    w = stage1.weights(r["ene_corrected"], r["raw_data_norm"], s)
    w[abs(r["ene_corrected"]) < max(4, s["em_min_meV"])] = 0
    good = np.isfinite(target) & (w > 0)

    def fit_score(keep):
        X = b.xp.column_stack([k["stokes"][:, keep], k["anti"][:, keep]])
        C = X[good] * b.array(w[good])[:, None]
        coef, _ = b.solve(C, target[good] * w[good])
        model = b.host(X[good] @ b.array(coef))
        y = target[good]
        ww = w[good] ** 2
        ybar = sum(ww * y) / sum(ww)
        total = sum(ww * (y - ybar) ** 2)
        error = sum(ww * (model - y) ** 2)
        return 1 - error / total if total > EPS else float(error <= 1e-12)

    ref = fit_score(np.ones(len(em), bool))
    maximum = min(len(em), s["stage1_auto_sparse_pm_max_count"])
    minimum = min(max(4, s["stage1_auto_sparse_pm_min_count"]), maximum)
    counts = np.unique(
        np.r_[grid(minimum, s["stage1_auto_sparse_pm_step"], maximum), maximum]
    )
    trials = []
    for count in counts:
        cand = np.unique(
            np.r_[
                spaced(em, score, count, s["stage1_auto_sparse_pm_min_spacing_meV"]),
                protected,
            ]
        )
        keep = np.min(abs(em[:, None] - cand[None, :]), axis=1) <= 1e-8 * max(abs(em))
        trials.append((em[keep], fit_score(keep)))
    threshold = max(
        s["stage1_auto_sparse_pm_min_R2"], ref - s["stage1_auto_sparse_pm_R2_tolerance"]
    )
    eligible = [i for i, (_, v) in enumerate(trials) if v >= threshold]
    idx = (
        min(eligible, key=lambda i: len(trials[i][0]))
        if eligible
        else int(np.argmax([v for _, v in trials]))
    )
    selected = trials[idx][0]
    return selected, dict(
        status="sparsest_within_R2_tolerance" if eligible else "best_available",
        reference_count=len(em),
        selected_count=len(selected),
        reference_R2=ref,
        selected_R2=trials[idx][1],
        candidate_counts=[len(x) for x, _ in trials],
        candidate_R2=[v for _, v in trials],
        R2_threshold=threshold,
    )


def calibrate(cube, e, coords, s, b, progress=None):
    if (
        not s["datacube_stage1_auto_sparse_pm_enabled"]
        or np.size(s["stage1_fixed_EM"])
        or not len(coords)
    ):
        return s
    ny, nx, _ = cube.shape
    indices = np.asarray(coords)
    d = np.hypot(indices[:, 0] - (ny - 1) / 2, indices[:, 1] - (nx - 1) / 2)
    pick = int(np.argmin(d))
    y, x = coords[pick]
    if np.size(s.get("stage1_auto_sparse_pm_probe_pixel", [])):
        target = int(s["stage1_auto_sparse_pm_probe_pixel"]) - 1
        coord = (target % ny, target // ny)
        if coord in coords:
            y, x = coord
    if progress:
        progress(f"模态网格标定 · 像素 ({y + 1}, {x + 1})")
    ps = dict(
        s,
        datacube_stage1_auto_sparse_pm_enabled=False,
        stage1_fixed_EM=[],
        run_stage2=False,
    )
    r = stage1.solve(e, cube[y, x], ps, b, progress=progress)
    em, policy = estimate(r, s, b)
    s = s.copy()
    s["stage1_fixed_EM"] = em
    policy.update(probe_y=y + 1, probe_x=x + 1)
    s["stage1_sparse_pm_policy"] = policy
    return s
