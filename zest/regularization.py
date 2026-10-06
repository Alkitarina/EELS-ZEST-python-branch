"""Stage 1 regularization structure, prepared once per pixel and fit round."""
import numpy as np
from .numeric import EPS, KB, expit, grid


class RegularizationPlan:
    def __init__(self, em, layout, n, s, e, y, backend, guard):
        b, xp = backend, backend.xp
        self.b = b
        self.em = b.array(em)
        self.z = layout["zlp"]
        self.blocks = []
        self.be = None
        self.pairs = None
        ns = len(em)
        rows, rhs = [], []
        cursor = 0

        def blank(count):
            return xp.zeros((count, n), dtype=xp.float64)

        def add(R, w, target=None):
            nonlocal cursor
            start = cursor
            rows.append(np.sqrt(w) * R)
            rhs.append(xp.zeros(R.shape[0]) if target is None else np.sqrt(w) * b.array(target))
            cursor += len(R)
            return slice(start, cursor)

        def zlp_rows(mask, taper, weight, target=None):
            span = add(blank(int(sum(mask))), weight, target)
            self.blocks.append((span, xp.asarray(np.flatnonzero(mask)), b.array(taper), np.sqrt(weight)))

        if s["lambda_be_soft"] > 0 and ns:
            R = blank(ns)
            index = xp.arange(ns)
            R[index, xp.asarray(layout["pm_a"])] = 1
            add(R, s["lambda_be_soft"])
            self.be = (index, xp.asarray(layout["pm_s"]), np.sqrt(s["lambda_be_soft"]))
        if s["lambda_pm_smooth"] > 0 and ns >= 3:
            D = xp.diff(xp.eye(ns), n=2, axis=0)
            for key in ("pm_s", "pm_a"):
                R = blank(ns - 2)
                R[:, layout[key]] = D
                add(R, s["lambda_pm_smooth"])
        if s["lambda_pm_l2"] > 0 and ns:
            R = blank(2 * ns)
            R[:ns, layout["pm_s"]] = xp.eye(ns)
            R[ns:, layout["pm_a"]] = xp.eye(ns)
            add(R, s["lambda_pm_l2"])
        if s["zlp_wide_component_l2"] > 0:
            z = self.z
            p = z[max(0, min(len(z) - 1, round(s["zlp_wide_component_start_index"]) - 1)):]
            R = blank(len(p))
            R[:, p] = xp.eye(len(p))
            add(R, s["zlp_wide_component_l2"])
        a = abs(e)
        gp = None
        for prefix in ("zlp_signal_guard", "zlp_lowfreq_guard"):
            if not s[prefix + "_enabled"] or s[prefix + "_weight"] <= 0:
                continue
            start, end = s[prefix + "_start_meV"], s[prefix + "_end_meV"]
            mask = (a >= start) & (a <= end)
            if end <= start or sum(mask) < 3:
                continue
            tw = max(s[prefix + "_taper_meV"], EPS)
            t = expit((a[mask] - start) / tw) * expit((end - a[mask]) / tw)
            if gp is None:
                gp = guard(e, y, s)
            if prefix == "zlp_signal_guard":
                floor = np.clip(s["zlp_signal_guard_tail_floor"], 0, 1)
                t *= (floor + (1 - floor) * expit(
                    (s["zlp_signal_guard_tail_release_start_meV"] - a[mask]) /
                    max(s["zlp_signal_guard_tail_release_width_meV"], EPS))) * gp[mask]
            else:
                t *= 0.35 + 0.65 * gp[mask]
            zlp_rows(mask, t, s[prefix + "_weight"])
        if s["stage1_zlp_core_fartail_enabled"]:
            cw = max(s["stage1_zlp_core_anchor_width_meV"], EPS)
            tw = max(0.25 * cw, 1)
            mask = a <= cw + 3 * tw
            if s["stage1_zlp_core_anchor_weight"] > 0 and sum(mask) >= 3:
                t = expit((cw - a[mask]) / tw)
                zlp_rows(mask, t, s["stage1_zlp_core_anchor_weight"], y[mask] * t)
            # Supplied by the caller to share the original tail formula.
            tp, tm = s["_tail_profile"]
            tm = tm.copy()
            side = s["stage1_zlp_fartail_anchor_side"]
            if side in ("gain", "left", "negative"):
                tm &= e < 0
            elif side in ("loss", "right", "positive"):
                tm &= e > 0
            if s["stage1_zlp_fartail_anchor_weight"] > 0 and sum(tm) >= 4:
                zlp_rows(tm, tp[tm], s["stage1_zlp_fartail_anchor_weight"], y[tm] * tp[tm])
            if s["stage1_zlp_gain_tail_prior_enabled"] and s["stage1_zlp_gain_tail_prior_weight"] > 0:
                start, end = s["stage1_zlp_fartail_start_meV"], s["stage1_zlp_fartail_end_meV"]
                end = float(end) if np.size(end) and np.isfinite(end) else max(a)
                pos, neg = (e > 0) & (e >= start) & (e <= end), (e < 0) & (-e >= start) & (-e <= end)
                if sum(pos) >= 3 and sum(neg) >= 3:
                    end = min(max(e[pos]), max(-e[neg]), end)
                    step = max(s["stage1_zlp_gain_tail_pair_step_meV"], EPS)
                    pairs = grid(start, step, end)
                    if len(pairs) < 2:
                        pairs = np.linspace(start, end, 3)
                    tol = max(step, 2 * np.median(abs(np.diff(e))))
                    ip, im = [], []
                    for energy in pairs:
                        p, m = np.argmin(abs(e - energy)), np.argmin(abs(e + energy))
                        if abs(e[p] - energy) <= tol and abs(e[m] + energy) <= tol:
                            ip.append(p)
                            im.append(m)
                    if ip:
                        span = add(blank(len(ip)), s["stage1_zlp_gain_tail_prior_weight"])
                        self.pairs = (span, xp.asarray(ip), xp.asarray(im),
                                      max(s["stage1_zlp_gain_tail_prior_ratio"], 1),
                                      np.sqrt(s["stage1_zlp_gain_tail_prior_weight"]))
        self.matrix, self.rhs = ((xp.concatenate(rows), xp.concatenate(rhs)) if rows else
                                 (blank(0), xp.zeros(0)))

    def evaluate(self, temperature, A):
        xp = self.b.xp
        R = self.matrix
        if self.be is not None:
            rows, columns, weight = self.be
            R[rows, columns] = weight * -xp.exp(-self.em / max(KB * temperature, EPS))
        for span, indices, taper, weight in self.blocks:
            R[span, self.z] = weight * (A[indices, :][:, self.z] * taper[:, None])
        if self.pairs is not None:
            span, ip, im, ratio, weight = self.pairs
            R[span, self.z] = weight * (A[im, :][:, self.z] - ratio * A[ip, :][:, self.z])
        return R, self.rhs
