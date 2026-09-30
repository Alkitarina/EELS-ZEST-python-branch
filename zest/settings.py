"""Defaults exported from EELS_PearsonSparse_FullFit.defaults (v2.1.0)."""

import copy
import json
from pathlib import Path

_DEFAULTS = json.loads(
    Path(__file__).with_name("defaults.json").read_text(encoding="utf-8")
)


def defaults(overrides=None):
    s = copy.deepcopy(_DEFAULTS)
    if overrides:
        s.update(copy.deepcopy(overrides))
    s["use_bg"] = False
    s["bg_order"] = 0
    s["lambda_bg_l2"] = 0
    s["stage2_temperature_algorithm"] = "grid_scan"
    s["stage2_pm_smooth_weight"] = max(s["stage2_pm_smooth_weight"], 1.2e-3)
    if not s["stage2_allow_aggressive_antistokes"]:
        s["stage2_antistokes_attention_boost"] = min(
            s["stage2_antistokes_attention_boost"], 1.35
        )
        s["stage2_antistokes_fit_weight_boost"] = min(
            s["stage2_antistokes_fit_weight_boost"], 1.45
        )
        s["stage2_antistokes_noise_sigma"] = max(
            s["stage2_antistokes_noise_sigma"], 1.8
        )
    for key in ("num_zlp_components", "inner_solver_max_iter", "max_nonlinear_iter"):
        s[key] = int(s[key])
    if (
        overrides
        and "stage1_speed_profile" in overrides
        and "stage1_rounds" not in overrides
    ):
        if s["stage1_speed_profile"] in ("full", "quality", "v19_full", "legacy_full"):
            s["stage1_rounds"] = [
                dict(name=n, em_step_meV=e, lambda_be_soft=l, max_nonlinear_iter=i)
                for n, e, l, i in [
                    ("round1_coarse", 10, 0.002, 35),
                    ("round2_dense", 6, 0.001, 40),
                    ("round3_dense_weakBE", 4, 0.0005, 45),
                ]
            ]
    return s
