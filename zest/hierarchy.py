"""v2.1 ZLP clustering, soft templates and four-colour spatial refitting."""

import time
import numpy as np
from scipy.signal import convolve2d
from .numeric import EPS, col, interp
from . import stage1
from .backend import Cancelled


def neighbors(coord, lookup):
    y, x = coord
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            point = (y + dy, x + dx)
            if point in lookup:
                yield lookup[point], point, dy != 0 and dx != 0


def standardize(X):
    X = np.asarray(X)
    med = np.nanmedian(X, axis=0)
    scale = 1.4826 * np.nanmedian(abs(X - med), axis=0)
    fallback = np.nanstd(X, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1)
    return np.nan_to_num((X - med) / scale)


def features(results, coords, shape, s):
    n = len(results)
    valid = np.array(["error" not in r for r in results])
    guard = [
        r["settings"]["zlp_signal_guard_start_meV"] for r in results if "error" not in r
    ]
    lo, hi = sorted(s["stage1_cluster_core_window_bounds_meV"])
    half = min(hi, max(lo, np.median(guard) - 4)) if guard else hi
    axis = np.linspace(-half, half, 41)
    profiles = np.full((n, 41), np.nan)
    stats = np.full((n, 4), np.nan)
    area = np.full((n, 1), np.nan)
    quality = np.zeros(n)
    lb, ub = stage1.bounds(s)
    for i in np.flatnonzero(valid):
        r = results[i]
        e = r["ene_corrected"]
        y = r["raw_data_norm"]
        p = interp(e, np.maximum(y, 0), axis)
        total = sum(np.maximum(p, 0))
        if not np.isfinite(total) or total <= 0:
            valid[i] = False
            continue
        profiles[i] = p / total
        area[i] = np.log(max(np.trapezoid(np.maximum(p, 0), axis), EPS))
        for j, side in enumerate((-1, 1)):
            keep = (side * axis >= 0) & (p >= 0.5 * max(p))
            stats[i, j] = (
                max(abs(axis[keep]))
                if np.any(keep)
                else max(abs(axis[side * axis >= 0]))
            )
        mu = sum(axis * p) / max(sum(p), EPS)
        sig = np.sqrt(sum((axis - mu) ** 2 * p) / max(sum(p), EPS))
        stats[i, 2] = sum((axis - mu) ** 3 * p) / max(sum(p) * max(sig, EPS) ** 3, EPS)
        outer = abs(axis) >= 0.7 * half
        stats[i, 3] = np.log(
            (sum(p[outer & (axis < 0)]) + EPS) / (sum(p[outer & (axis > 0)]) + EPS)
        )
        core = abs(e) <= 0.55 * half
        shoulder = (abs(e) > 0.55 * half) & (abs(e) <= half)
        scale = max(np.max(y[core]), EPS)
        z = r["stage1"]["zlp_norm"]
        core_rmse = np.sqrt(np.mean((y[core] - z[core]) ** 2)) / scale
        shoulder_rmse = np.sqrt(np.mean((y[shoulder] - z[shoulder]) ** 2)) / scale
        de = np.median(abs(np.diff(e)))
        shift = abs(r.get("total_energy_shift_meV", 0))
        th = r["theta"]
        span = np.maximum(ub - lb, EPS)
        hit = np.mean(((th - lb) / span < 0.01) | ((ub - th) / span < 0.01))
        quality[i] = np.clip(
            0.4 * np.exp(-core_rmse / 0.025)
            + 0.25 * np.exp(-shoulder_rmse / 0.020)
            + 0.15 * np.exp(-shift / max(2 * de, EPS))
            + 0.20 * max(0, 1 - hit),
            0,
            1,
        )
    valid &= np.all(np.isfinite(profiles), axis=1)
    X = profiles[valid]
    center = np.mean(X, axis=0)
    X = X - center
    k = int(min(s["stage1_cluster_max_pca_components"], len(X) - 1, X.shape[1]))
    scores = np.empty((len(X), 0))
    if k >= 1:
        U, D, V = np.linalg.svd(X, full_matrices=False)
        scores = U[:, :k] * D[:k]
    lookup = {p: i for i, p in enumerate(coords)}
    consistency = np.full(n, np.nan)
    for i in np.flatnonzero(valid):
        vals = [
            np.linalg.norm(profiles[i] - profiles[j])
            for j, _, _ in neighbors(coords[i], lookup)
            if valid[j]
        ]
        if vals:
            consistency[i] = np.mean(vals)
    fill = np.nanmedian(consistency) if np.any(np.isfinite(consistency)) else 0
    consistency[~np.isfinite(consistency)] = fill
    positive = consistency[consistency > EPS]
    edge_scale = np.median(positive) if len(positive) else 1
    F = np.full((n, scores.shape[1] + 6), np.nan)
    F[valid] = np.column_stack(
        [
            np.sqrt(0.6 / max(scores.shape[1], 1)) * standardize(scores),
            np.sqrt(0.2 / 4) * standardize(stats[valid]),
            np.sqrt(0.1) * standardize(area[valid]),
            np.sqrt(0.1) * standardize(consistency[valid, None]),
        ]
    )
    return F, dict(
        valid=valid,
        profiles=profiles,
        axis=axis,
        quality=quality,
        consistency=consistency,
        edge_scale=edge_scale,
        shape=shape,
        lookup=lookup,
    )


def distances(X, C):
    return np.maximum(
        np.sum(X * X, axis=1)[:, None] + np.sum(C * C, axis=1)[None, :] - 2 * X @ C.T, 0
    )


def kmeans(X, k, seed):
    n, d = X.shape
    best = None
    for rep in range(5):
        rng = np.random.RandomState(int(seed) + 7919 * rep)
        first = int(np.floor(rng.random_sample() * n))
        C = np.zeros((k, d))
        C[0] = X[first]
        closest = np.sum((X - C[0]) ** 2, axis=1)
        for c in range(1, k):
            weights = np.maximum(closest, 0)
            pick = (
                (first + c) % n
                if sum(weights) <= EPS
                else min(
                    int(
                        np.searchsorted(
                            np.cumsum(weights) / sum(weights), rng.random_sample()
                        )
                    ),
                    n - 1,
                )
            )
            C[c] = X[pick]
            closest = np.minimum(closest, np.sum((X - C[c]) ** 2, axis=1))
        labels = np.zeros(n, dtype=int)
        for it in range(100):
            D = distances(X, C)
            new = np.argmin(D, axis=1)
            if it and np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                C[c] = (
                    np.median(X[labels == c], axis=0)
                    if np.any(labels == c)
                    else X[np.argmax(np.min(D, axis=1))]
                )
        D = distances(X, C)
        sse = sum(D[np.arange(n), labels])
        if best is None or sse < best[0]:
            best = (sse, labels.copy(), C.copy())
    return best


def cluster(F, info, coords, s):
    valid = info["valid"]
    ids = np.flatnonzero(valid)
    X = F[valid]
    n = len(X)
    minimum = max(
        s["stage1_cluster_min_pixels"], np.ceil(s["stage1_cluster_min_fraction"] * n)
    )
    kr = sorted(s["stage1_cluster_k_range"])
    ks = (
        [int(min(max(1, s["stage1_cluster_count"]), n))]
        if s["stage1_cluster_count"] > 0
        else range(int(kr[0]), int(min(kr[-1], n)) + 1)
    )
    best = None
    for k in ks:
        sse, labels, C = kmeans(X, k, s["stage1_cluster_random_seed"])
        counts = np.bincount(labels, minlength=k)
        if k > 1 and min(counts) < minimum:
            continue
        bic = n * np.log(max(sse / max(n, 1), EPS)) + k * X.shape[1] * np.log(max(n, 2))
        if best is None or bic < best[0]:
            best = (bic, sse, labels, C)
    if best is None:
        sse, lab, C = kmeans(X, 1, s["stage1_cluster_random_seed"])
        best = (np.nan, sse, lab, C)
    bic, sse, lab, C = best
    k = len(C)
    D = distances(X, C)
    if k == 1:
        p = np.ones((n, 1))
    else:
        nearest = np.min(D, axis=1)
        nz = nearest[nearest > EPS]
        sigma = (
            np.median(nz)
            if len(nz)
            else np.median(D[D > EPS])
            if np.any(D > EPS)
            else 1
        )
        p = np.exp(-0.5 * D / max(sigma, EPS))
        p /= np.maximum(p.sum(axis=1), EPS)[:, None]
    valid_coords = [coords[i] for i in ids]
    lookup = {q: i for i, q in enumerate(valid_coords)}
    profiles = info["profiles"][valid]
    ds = []
    for i, (y, x) in enumerate(valid_coords):
        for pt in ((y, x + 1), (y + 1, x)):
            if pt in lookup:
                ds.append(np.linalg.norm(profiles[i] - profiles[lookup[pt]]))
    positive = col(ds)
    positive = positive[positive > EPS]
    edge_scale = np.median(positive) if len(positive) else 1
    for _ in range(2):
        for i, point in enumerate(valid_coords):
            energy = -np.log(np.maximum(p[i], EPS))
            for j, _, diagonal in neighbors(point, lookup):
                energy += (
                    0.35
                    * np.exp(-np.linalg.norm(profiles[i] - profiles[j]) / edge_scale)
                    / (np.sqrt(2) if diagonal else 1)
                    * (np.arange(k) != lab[j])
                )
            lab[i] = np.argmin(energy)
    mem = np.max(p, axis=1)
    labels = np.zeros(len(coords), int)
    membership = np.zeros(len(coords))
    probs = np.zeros((len(coords), k))
    labels[ids] = lab + 1
    labels[ids[mem < s["stage1_cluster_membership_min"]]] = 0
    membership[ids] = mem
    probs[ids] = p
    return labels, membership, probs, dict(k=k, bic=bic, sse=sse, centers=C)


def interfaces(F, info, coords, probs, s):
    ny, nx = info["shape"]
    out = dict(
        edge_x_mask=np.zeros((ny, max(0, nx - 1)), bool),
        edge_y_mask=np.zeros((max(0, ny - 1), nx), bool),
        edge_x_score=np.full((ny, max(0, nx - 1)), np.nan),
        edge_y_score=np.full((max(0, ny - 1), nx), np.nan),
        score_map=np.full((ny, nx), np.nan),
        mask=np.zeros((ny, nx), bool),
    )
    records = []
    if not s["stage1_zlp_interface_enabled"]:
        return out
    lookup = info["lookup"]
    valid = info["valid"]
    F = F[:, :-1]
    for i, (y, x) in enumerate(coords):
        if not valid[i]:
            continue
        for kind, pt in (("x", (y, x + 1)), ("y", (y + 1, x))):
            if pt in lookup and valid[lookup[pt]]:
                j = lookup[pt]
                p = probs[i] / max(sum(probs[i]), EPS)
                q = probs[j] / max(sum(probs[j]), EPS)
                sep = np.clip(1 - sum(p * q), 0, 1)
                records.append((kind, y, x, np.sqrt(np.mean((F[i] - F[j]) ** 2)), sep))
    if len(records) < 2:
        return out
    d = col([r[3] for r in records])
    med = np.median(d)
    fallback = 0.25 * np.median(d[d > 0]) if np.any(d > 0) else np.std(d, ddof=1)
    fallback = fallback if np.isfinite(fallback) and fallback > EPS else 1
    scale = max(1.4826 * np.median(abs(d - med)), fallback)
    score = np.maximum((d - med) / scale, 0) + s[
        "stage1_zlp_interface_probability_weight"
    ] * col([r[4] for r in records])
    ratio = d / max(med, scale, EPS)
    cx = np.zeros_like(out["edge_x_mask"])
    cy = np.zeros_like(out["edge_y_mask"])
    for i, (kind, y, x, _, _) in enumerate(records):
        out["edge_" + kind + "_score"][y, x] = score[i]
        out["edge_" + kind + "_mask"][y, x] = (
            score[i] >= s["stage1_zlp_interface_score_threshold"]
            and ratio[i] >= s["stage1_zlp_interface_distance_ratio_min"]
        )
        (cx if kind == "x" else cy)[y, x] = (
            score[i] >= s["stage1_zlp_interface_coherent_score_threshold"]
            and ratio[i] >= s["stage1_zlp_interface_distance_ratio_min"]
        )
    support = max(1, int(s["stage1_zlp_interface_min_coherent_neighbors"]))
    if cx.size:
        out["edge_x_mask"] |= cx & (
            convolve2d(cx, np.ones((3, 1)), mode="same") >= support
        )
    if cy.size:
        out["edge_y_mask"] |= cy & (
            convolve2d(cy, np.ones((1, 3)), mode="same") >= support
        )
    for y, x in np.ndindex((ny, nx)):
        vals = []
        sharp = False
        for kind, yy, xx in [
            ("x", y, x - 1),
            ("x", y, x),
            ("y", y - 1, x),
            ("y", y, x),
        ]:
            a = out["edge_" + kind + "_score"]
            mask = out["edge_" + kind + "_mask"]
            if 0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]:
                vals.append(a[yy, xx])
                sharp |= bool(mask[yy, xx])
        vals = col(vals)
        vals = vals[np.isfinite(vals)]
        out["score_map"][y, x] = max(vals) if len(vals) else np.nan
        out["mask"][y, x] = sharp
    return out


def crosses(edge, p, q):
    y, x = p
    yy, xx = q
    ex = edge["edge_x_mask"]
    ey = edge["edge_y_mask"]
    if y == yy:
        return bool(ex[y, min(x, xx)])
    if x == xx:
        return bool(ey[min(y, yy), x])
    return bool(
        ex[y, min(x, xx)]
        or ex[yy, min(x, xx)]
        or ey[min(y, yy), x]
        or ey[min(y, yy), xx]
    )


def templates(e, results, labels, mem, info, s):
    cons = info["consistency"]
    positive = cons[cons > EPS]
    scale = np.median(positive) if len(positive) else 1
    score = 0.85 * info["quality"] + 0.15 * np.exp(-cons / scale)
    de = np.median(abs(np.diff(e)))
    n = int(s["num_zlp_components"])
    out = []
    for c in range(1, int(max(labels)) + 1):
        idx = np.flatnonzero(labels == c)
        if not len(idx):
            out.append(None)
            continue
        eligible = idx[score[idx] >= s["stage1_cluster_template_score_min"]]
        minimum = min(len(idx), max(8, int(np.ceil(0.5 * len(idx)))))
        if len(eligible) < minimum:
            eligible = idx[np.argsort(-score[idx], kind="stable")[:minimum]]
        th = np.array([results[i]["theta"] for i in eligible])
        template = np.median(th, axis=0)
        phi = stage1.phi(template, s, de)
        allphi = np.array([stage1.phi(t, s, de) for t in th])
        mad = 1.4826 * np.median(abs(allphi - np.median(allphi, axis=0)), axis=0)
        phiscale = np.r_[
            np.clip(mad[:n], 0.5, s["stage1_cluster_center_channels"]),
            np.clip(
                mad[n : 3 * n],
                np.log1p(s["stage1_cluster_width_rel_floor"]),
                np.log1p(s["stage1_cluster_width_rel_cap"]),
            ),
            np.clip(
                mad[3 * n :],
                np.log1p(s["stage1_cluster_shape_rel_floor"]),
                np.log1p(s["stage1_cluster_shape_rel_cap"]),
            ),
        ]
        lb, ub = stage1.bounds(s)
        delta = s["stage1_cluster_center_channels"] * de
        lb[1 : 1 + n] = np.maximum(lb[1 : 1 + n], template[1 : 1 + n] - delta)
        ub[1 : 1 + n] = np.minimum(ub[1 : 1 + n], template[1 : 1 + n] + delta)
        for kind, indices in [
            ("width", range(1 + n, 1 + 3 * n)),
            ("shape", range(1 + 3 * n, 1 + 5 * n)),
        ]:
            for i in indices:
                vals = th[:, i]
                mad = (
                    1.4826
                    * np.median(abs(vals - np.median(vals)))
                    / max(abs(template[i]), EPS)
                )
                rel = np.clip(
                    s["stage1_cluster_mad_multiplier"] * mad,
                    s[f"stage1_cluster_{kind}_rel_floor"],
                    s[f"stage1_cluster_{kind}_rel_cap"],
                )
                lb[i] = max(lb[i], template[i] * (1 - rel))
                ub[i] = min(ub[i], template[i] * (1 + rel))
        out.append(
            dict(
                label=c,
                theta=template,
                phi=phi,
                phi_scale=phiscale,
                theta_lb=lb,
                theta_ub=ub,
                seed_count=len(eligible),
            )
        )
    return out, score


def soft_template(prob, ts, s):
    ids = [i for i, t in enumerate(ts) if t is not None]
    lb, ub = stage1.bounds(s)
    if not ids:
        return None
    p = col([max(prob[i], 0) for i in ids])
    total = sum(p)
    if total <= 0:
        return None
    p /= total
    ph = np.array([ts[i]["phi"] for i in ids])
    sc = np.array([ts[i]["phi_scale"] for i in ids])
    mu = p @ ph
    scale = np.sqrt(
        np.maximum(np.sum(p[:, None] * (sc * sc + (ph - mu) ** 2), axis=0), EPS)
    )
    confidence = max(p)
    blend = (
        1
        if len(p) == 1
        else np.clip((confidence - 1 / len(p)) / (1 - 1 / len(p)), 0, 1)
    )
    low = lb + blend * (p @ np.array([ts[i]["theta_lb"] for i in ids]) - lb)
    high = ub + blend * (p @ np.array([ts[i]["theta_ub"] for i in ids]) - ub)
    bad = ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
    low[bad] = lb[bad]
    high[bad] = ub[bad]
    return dict(
        phi=mu,
        phi_scale=scale,
        theta_lb=low,
        theta_ub=high,
        confidence=confidence,
        weights=p,
        bound_blend=blend,
    )


def context(
    i, e, results, coords, labels, mem, probs, ts, info, edge, s, include_neighbors
):
    if "error" in results[i]:
        return dict(enabled=False)
    de = np.median(abs(np.diff(e)))
    lab = int(labels[i])
    soft = (
        soft_template(probs[i], ts, s)
        if s["stage1_cluster_soft_constraints_enabled"]
        else None
    )
    ctx = dict(
        enabled=True,
        center_scale_meV=de,
        membership=mem[i],
        data_cost=max(results[i]["stage1"]["resnorm"], 1e-10),
        cluster_label=lab,
    )
    if soft:
        ctx.update(
            template_phi=soft["phi"],
            template_scale=soft["phi_scale"],
            theta_lb=soft["theta_lb"],
            theta_ub=soft["theta_ub"],
            membership=soft["confidence"],
        )
    elif lab > 0 and lab <= len(ts) and ts[lab - 1]:
        t = ts[lab - 1]
        ctx.update(
            template_phi=t["phi"],
            template_scale=t["phi_scale"],
            theta_lb=t["theta_lb"],
            theta_ub=t["theta_ub"],
        )
    else:
        lb, ub = stage1.bounds(s)
        ctx.update(theta_lb=lb, theta_ub=ub, membership=0, cluster_label=0)
    if include_neighbors:
        ph = []
        ws = []
        for j, pt, diagonal in neighbors(coords[i], info["lookup"]):
            if "error" in results[j] or crosses(edge, coords[i], pt):
                continue
            delta = np.linalg.norm(info["profiles"][i] - info["profiles"][j])
            w = (
                np.exp(-delta / max(info["edge_scale"], EPS))
                * sum(probs[i] * probs[j])
                / (np.sqrt(2) if diagonal else 1)
            )
            if w > 0:
                ph.append(stage1.phi(results[j]["theta"], s, de))
                ws.append(w)
        ctx.update(neighbor_phi=np.asarray(ph), neighbor_weight=col(ws))
        if lab > 0 and lab <= len(ts) and ts[lab - 1]:
            ctx["spatial_scale"] = ts[lab - 1]["phi_scale"]
        elif ph:
            arr = np.asarray(ph)
            ctx["spatial_scale"] = np.maximum(
                1.4826 * np.median(abs(arr - np.median(arr, axis=0)), axis=0), 0.05
            )
    return ctx


def spatial_loss(e, results, coords, probs, info, edge, s):
    de = np.median(abs(np.diff(e)))
    ph = [stage1.phi(r["theta"], s, de) if "error" not in r else None for r in results]
    loss = np.full(len(coords), np.nan)
    counts = np.zeros(len(coords), int)
    for i in range(len(coords)):
        if ph[i] is None:
            continue
        vals = []
        for j, pt, diagonal in neighbors(coords[i], info["lookup"]):
            if ph[j] is None or crosses(edge, coords[i], pt):
                continue
            w = (
                np.exp(
                    -np.linalg.norm(info["profiles"][i] - info["profiles"][j])
                    / max(info["edge_scale"], EPS)
                )
                * sum(probs[i] * probs[j])
                / (np.sqrt(2) if diagonal else 1)
            )
            if w > 0:
                vals.append(w * np.mean(abs(ph[i] - ph[j])))
        counts[i] = len(vals)
        if vals:
            loss[i] = np.mean(vals)
    return loss, counts


def refit_one(e, spectrum, old, ctx, s, iterations, phase, b):
    """Original per-pixel refit and acceptance rule, usable in either scheduler."""
    b.check()
    if not ctx.get("enabled"):
        return None, None
    ps = dict(
        s,
        run_stage2=False,
        stage1_hierarchy_context=ctx,
        stage1_fixed_EM=old["EM_final"],
        stage1_strict_polish_enabled=False,
        stage1_zlp_gain_heavy_init_enabled=False,
        stage1_zlp_gain_tail_prior_enabled=False,
        stage1_zlp_gain_asym_shape_prior_enabled=False,
        stage1_rounds=[dict(name=phase, em_step_meV=s["em_step_meV"],
                           lambda_be_soft=s["lambda_be_soft"],
                           max_nonlinear_iter=max(1, int(iterations)))],
    )
    if abs(np.diff(ps["eshift_range"])[0]) < 1e-9:
        ps["eshift_range"] = [-1e-3, 1e-3]
    try:
        candidate = stage1.solve(e, spectrum, ps, b, previous=old["theta"])
        cm, om = candidate["metrics"], old["metrics"]
        if (np.isfinite(cm["R2"]) and cm["R2"] >= om["R2"] - 2e-4
                and cm["NRMSE"] <= om["NRMSE"] + 0.002):
            return candidate, None
    except Cancelled:
        raise
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return None, str(exc)
    return None, None


def run(cube, e, coords, results, s, b, progress=None, executor=None, on_result=None,
        timer=None):
    minimum = max(8, min(s["stage1_cluster_min_pixels"], len(coords)))
    if sum("error" not in r for r in results) < minimum:
        return results, dict(enabled=False, status="insufficient_valid_zlp_features")
    F, info = features(results, coords, cube.shape[:2], s)
    valid = info["valid"]
    minimum = max(8, min(s["stage1_cluster_min_pixels"], len(coords)))
    if sum(valid) < minimum:
        return results, dict(enabled=False, status="insufficient_valid_zlp_features")
    labels, mem, probs, model = cluster(F, info, coords, s)
    edge = interfaces(F, info, coords, probs, s)
    ts, score = templates(e, results, labels, mem, info, s)
    if not any(t is not None for t in ts):
        return results, dict(enabled=False, status="no_valid_cluster_templates")
    history = []
    refitted = np.zeros(cube.shape[:2], bool)

    def refit(indices, contexts, iterations, phase):
        if timer is None:
            return refit_batch(indices, contexts, iterations, phase)
        key = "spatial_refit" if phase.startswith("spatial_") else "template_refit"
        with timer.phase(key):
            return refit_batch(indices, contexts, iterations, phase)

    def refit_batch(indices, contexts, iterations, phase):
        accepted = 0
        start = time.perf_counter()

        def commit(i, candidate, error):
            nonlocal accepted
            old = results[i]
            if candidate is not None:
                if executor is not None:
                    for key in ("pixel_y", "pixel_x", "pixel_index", "execution"):
                        if key in old:
                            candidate[key] = old[key]
                results[i] = candidate
                accepted += 1
                refitted[coords[i]] = True
            elif error is not None:
                old.setdefault("refit_failures", []).append(error)
            if on_result and (candidate is not None or error is not None):
                on_result(i, results[i])

        if executor is None:
            for k, i in enumerate(indices):
                b.check()
                if not contexts[i].get("enabled"):
                    continue
                y, x = coords[i]
                if progress:
                    progress(f"{phase} · {k + 1}/{len(indices)} · ({y + 1},{x + 1})")
                candidate, error = refit_one(
                    e, cube[y, x], results[i], contexts[i], s, iterations, phase, b
                )
                commit(i, candidate, error)
        else:
            # All contexts are frozen before submitting this phase. No task
            # reads another task's in-flight result. Return is a phase barrier.
            tasks = ((int(i), coords[i], results[i], contexts[i], s, iterations, phase)
                     for i in indices if contexts[i].get("enabled"))
            for k, (i, candidate, error, pid) in enumerate(executor.refits(tasks)):
                commit(i, candidate, error)
                if progress:
                    progress(f"{phase} · 并行完成 {k + 1}/{len(indices)} · PID {pid}")
        history.append(
            dict(
                phase=phase,
                attempted=len(indices),
                accepted=accepted,
                elapsed_seconds=time.perf_counter() - start,
            )
        )

    ids = np.flatnonzero(valid)
    contexts = {
        i: context(i, e, results, coords, labels, mem, probs, ts, info, edge, s, False)
        for i in ids
    }
    refit(ids, contexts, s["stage1_hierarchical_template_max_iter"], "template_polish")
    changed = 0
    if sum(t is not None for t in ts) >= 2:
        for i, r in enumerate(results):
            old = labels[i]
            if old <= 0 or "error" in r:
                continue
            ph = stage1.phi(r["theta"], s, np.median(abs(np.diff(r["ene_corrected"]))))
            scores = np.full(len(ts), -np.inf)
            for c, t in enumerate(ts):
                if t:
                    scores[c] = (
                        0.7
                        * np.exp(
                            -0.5
                            * np.mean(
                                ((ph - t["phi"]) / np.maximum(t["phi_scale"], 0.05))
                                ** 2
                            )
                        )
                        + 0.3 * probs[i, c]
                    )
            new = int(np.argmax(scores)) + 1
            if (
                new != old
                and scores[new - 1]
                >= scores[old - 1] + s["stage1_cluster_label_switch_margin"]
            ):
                labels[i] = new
                mem[i] = max(mem[i], probs[i, new - 1])
                changed += 1
        if changed:
            ts, score = templates(e, results, labels, mem, info, s)
            history.append(dict(phase="single_label_update", accepted=changed))
    full = len(coords) == cube.shape[0] * cube.shape[1]
    sweeps = (
        int(s["stage1_spatial_zlp_iterations"])
        if full and s["stage1_spatial_zlp_enabled"]
        else 0
    )
    for sweep in range(sweeps):
        loss, counts = spatial_loss(e, results, coords, probs, info, edge, s)
        ids = np.flatnonzero(valid & (counts >= 2))
        if sweep and len(ids):
            ids = ids[
                np.argsort(-loss[ids], kind="stable")[
                    : max(
                        1,
                        int(
                            np.ceil(
                                s["stage1_spatial_zlp_second_pass_fraction"] * len(ids)
                            )
                        ),
                    )
                ]
            ]
        for color in range(4):
            selected = [
                i for i in ids if 2 * (coords[i][0] % 2) + coords[i][1] % 2 == color
            ]
            contexts = {
                i: context(
                    i, e, results, coords, labels, mem, probs, ts, info, edge, s, True
                )
                for i in selected
            }
            refit(
                selected,
                contexts,
                s["stage1_hierarchical_spatial_max_iter"],
                f"spatial_{sweep + 1}_color_{color + 1}",
            )
    loss, _ = spatial_loss(e, results, coords, probs, info, edge, s)
    deviation = np.full(len(coords), np.nan)
    for i, r in enumerate(results):
        if "error" in r:
            continue
        soft = soft_template(probs[i], ts, s)
        if soft:
            ph = stage1.phi(r["theta"], s, np.median(abs(np.diff(e))))
            deviation[i] = (
                np.sqrt(
                    np.mean(
                        ((ph - soft["phi"]) / np.maximum(soft["phi_scale"], EPS)) ** 2
                    )
                )
                * soft["confidence"]
            )
        r["stage1_hierarchy"] = dict(
            cluster_label=int(labels[i]),
            membership=float(mem[i]),
            template_deviation=deviation[i],
            spatial_loss=loss[i],
        )
    shape = cube.shape[:2]

    def map_values(vals):
        out = np.full(shape, np.nan)
        for point, v in zip(coords, vals):
            out[point] = v
        return out

    return results, dict(
        enabled=True,
        status="ok" if full else "cluster_templates_only_sparse_sampling",
        cluster_label_map=map_values(labels),
        cluster_membership_map=map_values(mem),
        template_deviation_map=map_values(deviation),
        zlp_spatial_loss_map=map_values(loss),
        membership_probabilities=probs,
        zlp_interface_score_map=edge["score_map"],
        zlp_interface_mask=edge["mask"],
        edge_x_mask=edge["edge_x_mask"],
        edge_y_mask=edge["edge_y_mask"],
        hierarchical_refit_mask=refitted,
        history=history,
        cluster_model=model,
    )
