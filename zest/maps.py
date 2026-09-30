"""Original v2.1 fixed-anchor, interface-gated temperature maps."""

import numpy as np
from .numeric import EPS
from .hierarchy import neighbors, crosses


def binned_profile(energy, values, count):
    edges = np.linspace(-max(abs(energy)), max(abs(energy)), count + 1)
    out = np.full(count, np.nan)
    for i in range(count):
        mask = (
            (energy >= edges[i])
            & ((energy <= edges[i + 1]) if i == count - 1 else (energy < edges[i + 1]))
            & np.isfinite(values)
        )
        if np.any(mask):
            out[i] = np.mean(values[mask])
    if not np.any(np.isfinite(out)):
        return np.zeros(count)
    out[~np.isfinite(out)] = np.nanmedian(out)
    return out / max(np.max(abs(out)), EPS)


def weighted_median(values, weights):
    v, w = np.asarray(values), np.asarray(weights)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(keep):
        return np.nan
    v, w = v[keep], w[keep]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    return float(v[min(np.searchsorted(np.cumsum(w) / sum(w), 0.5), len(v) - 1)])


def build(shape, coords, results, settings, hierarchy=None):
    s = settings
    h = hierarchy or {}
    nbins = int(s["spatial_attention_profile_bins"])
    mapping = dict(R2="R2", NRMSE="NRMSE")
    m = {
        k: np.full(shape, np.nan)
        for k in (
            "R2",
            "NRMSE",
            "T_raw",
            "T_uncertainty",
            "reliability",
            "T_valid",
            "T_joint",
            "T_joint_uncertainty",
            "T_joint_shift",
            "attention",
        )
    }
    m.update(
        fit_status=np.zeros(shape, int),
        T_valid_mask=np.zeros(shape, bool),
        T_inferred_mask=np.zeros(shape, bool),
    )
    att, signal = np.zeros((*shape, nbins)), np.zeros((*shape, nbins))
    low, high = min(s["stage2_T_grid"]), max(s["stage2_T_grid"])
    for point, r in zip(coords, results):
        if "error" in r:
            m["fit_status"][point] = -1
            continue
        m["fit_status"][point] = 1
        for key in mapping:
            m[key][point] = r["metrics"].get(key, np.nan)
        st = r.get("stage2", {})
        if st.get("status") != "ok":
            continue
        m["fit_status"][point] = 2
        m["T_raw"][point] = st["T_fit"]
        m["T_uncertainty"][point] = st["T_uncertainty"]
        m["reliability"][point] = st["reliability_score"]
        e = r["ene_corrected"]
        a = st["attention"]
        att[point] = binned_profile(e, a["feature_score"], nbins)
        signal[point] = binned_profile(
            e, np.maximum(r["raw_data_norm"] - r["stage1"]["zlp_norm"], 0), nbins
        )
        m["attention"][point] = a["selected_fraction"]
        t, c = np.asarray(st["T_grid"]), np.asarray(st["cost_grid"])
        enough = len(np.unique(t[np.isfinite(t) & np.isfinite(c)])) >= 3
        valley = st["cost_valley"]
        m["T_valid_mask"][point] = bool(
            enough
            and low + s["stage2_boundary_margin_K"]
            < st["T_fit"]
            < high - s["stage2_boundary_margin_K"]
            and low < valley["lower_K"]
            and valley["upper_K"] < high
            and st["reliability_score"] >= s["stage2_spatial_map_valid_reliability"]
            and st["T_uncertainty"] <= s["stage2_spatial_map_max_uncertainty_K"]
            and st["num_hard_pairs"]
            >= st.get("min_hard_pairs", s["stage2_ratio_min_pairs"])
        )
    for name, target in [
        ("cluster_label_map", "cluster"),
        ("cluster_membership_map", "membership"),
        ("zlp_interface_score_map", "interface"),
        ("template_deviation_map", "template_deviation"),
    ]:
        if name in h:
            m[target] = h[name]
    valid = m["T_valid_mask"]
    m["T_valid"][valid] = m["T_raw"][valid]
    m["T_joint"][valid] = m["T_raw"][valid]
    m["T_joint_uncertainty"][valid] = m["T_uncertainty"][valid]
    if not s["stage2_spatial_map_enabled"]:
        m["T_joint"] = m["T_raw"].copy()
        return m
    if np.any(valid) and s["stage2_zlp_guided_inference_enabled"]:
        lookup = {p: i for i, p in enumerate(coords)}
        edge = dict(
            edge_x_mask=h.get(
                "edge_x_mask", np.zeros((shape[0], max(0, shape[1] - 1)), bool)
            ),
            edge_y_mask=h.get(
                "edge_y_mask", np.zeros((max(0, shape[0] - 1), shape[1]), bool)
            ),
        )
        probabilities = h.get("membership_probabilities")
        for i, point in enumerate(coords):
            if valid[point]:
                continue
            values, weights, uncertainties = [], [], []
            for j, other, diagonal in neighbors(point, lookup):
                if not valid[other] or crosses(edge, point, other):
                    continue
                delta = max(
                    np.mean(abs(att[point] - att[other])),
                    0.5 * np.mean(abs(signal[point] - signal[other])),
                    0,
                )
                overlap = (
                    1
                    if probabilities is None
                    else s["stage2_spatial_map_cluster_floor"]
                    + (1 - s["stage2_spatial_map_cluster_floor"])
                    * sum(probabilities[i] * probabilities[j])
                )
                weight = (
                    np.exp(-delta / 0.18) * overlap / (np.sqrt(2) if diagonal else 1)
                )
                if np.isfinite(weight) and weight > 0:
                    values.append(m["T_raw"][other])
                    weights.append(weight)
                    uncertainties.append(m["T_uncertainty"][other])
            if len(values) < s["stage2_spatial_map_min_neighbor_support"]:
                continue
            t = weighted_median(values, weights)
            u = weighted_median(uncertainties, weights)
            spread = 1.4826 * weighted_median(abs(np.asarray(values) - t), weights)
            m["T_joint"][point] = t
            m["T_inferred_mask"][point] = np.isfinite(t)
            m["T_joint_uncertainty"][point] = (
                np.hypot(u, spread) if np.isfinite(u) else spread
            )
    m["T_joint_shift"] = m["T_joint"] - m["T_raw"]
    return m
