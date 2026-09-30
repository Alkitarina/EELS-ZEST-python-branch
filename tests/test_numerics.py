import copy
import numpy as np
import pytest
from zest import stage1, stage2, data, maps, hierarchy
from zest.backend import Backend, Cancelled
from zest.settings import defaults
from zest.storage import save_pixel, load_pixel


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_matlab_design_regularization_and_temperature(reference, device):
    try:
        b = Backend(device)
    except (ImportError, RuntimeError) as e:
        if device == "gpu":
            pytest.skip(str(e))
        raise
    r = reference
    p = stage1.decode_theta(r["theta"], r["s"])
    A, layout, _, _ = stage1.design(r["ene"], r["EM"], p, r["s"], b)
    R, rhs = stage1.regularization(
        r["EM"], layout, p["T_soft"], A.shape[1], r["s"], A, r["ene"], r["y"], b
    )
    np.testing.assert_allclose(b.host(A), r["A"], atol=2e-14, rtol=1e-11)
    np.testing.assert_allclose(b.host(R), r["R"], atol=2e-13, rtol=1e-10)
    np.testing.assert_allclose(b.host(rhs), r["r0"], atol=1e-13)
    np.testing.assert_allclose(
        stage1.weights(r["ene"], r["y"], r["s"]), r["weights"], atol=2e-13
    )
    f = stage1.decode(r["theta"], r["ene"], r["y"], r["EM"], r["obj_settings"], b)
    # Different bounded LS implementations: document a measured tolerance,
    # not bitwise equality or equivalence of end-to-end temperatures.
    np.testing.assert_allclose(
        f["model_norm"], r["fit"]["model_norm"], atol=1e-5, rtol=0
    )
    st = stage2.solve(r["r"], r["s"], b)
    np.testing.assert_array_equal(st["EM_final"], r["stage2"]["EM_final"])
    np.testing.assert_array_equal(st["hard_idx"], r["stage2"]["hard_idx"])
    np.testing.assert_allclose(
        st["attention"]["feature_score"],
        r["stage2"]["attention"]["feature_score"],
        atol=1e-12,
    )
    assert st["T_fit"] == r["stage2"]["T_fit"]
    assert st["T_uncertainty"] == r["stage2"]["T_uncertainty"]
    if device == "gpu":
        assert b.gpu_operations > 0


def test_gpu_centered_svd_matches_cpu():
    rng = np.random.default_rng(21)
    e = np.linspace(-100, 100, 160)
    cube = (rng.normal(size=(24, 4)) @ rng.normal(size=(4, 160))).reshape(
        6, 4, 160
    ) + 10
    cpu, _ = data.preprocess(cube, e, Backend(), False, 3)
    try:
        b = Backend("gpu")
    except (ImportError, RuntimeError) as exc:
        pytest.skip(str(exc))
    gpu, _ = data.preprocess(cube, e, b, False, 3)
    np.testing.assert_allclose(gpu, cpu, atol=1e-8, rtol=1e-9)
    assert b.gpu_operations > 0


def test_storage_roundtrip_retains_bounds_and_diagnostics(reference, tmp_path):
    r = copy.deepcopy(reference["r"])
    r["probe_bounds"] = np.array([[np.nan, np.inf], [-np.inf, 4.0]])
    path = tmp_path / "pixel.json.gz"
    save_pixel(path, r)
    saved = load_pixel(path)
    np.testing.assert_equal(saved["probe_bounds"], r["probe_bounds"])
    np.testing.assert_allclose(saved["theta"], r["theta"])
    st = stage2.solve(saved, reference["s"])
    assert st["T_fit"] == reference["stage2"]["T_fit"]


def test_nonpropagating_temperature_inference(reference):
    r = reference["r"]
    st = reference["stage2"]
    s = defaults(reference["s"])
    good = copy.deepcopy(r)
    good["stage2"] = copy.deepcopy(st)
    good["stage2"].update(
        T_fit=340.0,
        T_uncertainty=20.0,
        reliability_score=0.9,
        num_hard_pairs=10,
        cost_valley={"lower_K": 320.0, "upper_K": 360.0},
    )
    invalid = copy.deepcopy(good)
    invalid["stage2"]["T_uncertainty"] = 9999
    # Only two original anchors. Column 2 can be inferred, but column 3 must
    # not acquire inferred neighbours as new anchors.
    coords = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    results = [copy.deepcopy(good if x == 0 else invalid) for y, x in coords]
    m = maps.build((2, 3), coords, results, s)
    assert np.all(m["T_valid_mask"][:, 0]) and not np.any(m["T_valid_mask"][:, 1:])
    assert np.all(m["T_inferred_mask"][:, 1])
    assert np.isnan(m["T_joint"][:, 2]).all()
    assert np.all(m["T_raw"] == 340)
    gate = dict(edge_x_mask=np.ones((2, 2), bool), edge_y_mask=np.zeros((1, 3), bool))
    gated = maps.build((2, 3), coords, results, s, gate)
    assert not np.any(gated["T_inferred_mask"])


def test_cancel_is_checked_inside_objective(reference):
    b = Backend(cancel=lambda: True)
    with pytest.raises(Cancelled):
        stage1.solve(reference["ene"], reference["y"], reference["s"], b)


def test_original_auto_order_and_explicit_l1():
    e = np.arange(20.0) - 10
    A = np.ones((20, 1))
    y = np.full(20, 2.0)
    w = np.ones(20)
    R = np.zeros((0, 1))
    rhs = np.zeros(0)
    s = defaults(
        {
            "inner_solver": "auto",
            "tail_overshoot_constraint": False,
            "lambda_pm_l1": 10.0,
        }
    )
    layout = {"pm_s": np.array([0]), "pm_a": np.array([], int)}
    x, info = stage1.inner(A, y, w, R, rhs, e, s, Backend(), layout)
    assert info["original_method"] == "lsqlin" and abs(x[0] - 2) < 1e-10
    x, info = stage1.inner(
        A, y, w, R, rhs, e, dict(s, inner_solver="quadprog"), Backend(), layout
    )
    assert info["original_method"] == "quadprog" and abs(x[0] - 1.5) < 1e-8


def test_hierarchy_full_grid_smoke(reference):
    # Reduced iteration budget ONLY in this smoke fixture; product defaults
    # retain the original iteration counts. Use real spectra / initial fits.
    r = reference["r"]
    coords = [(y, x) for x in range(4) for y in range(5)]
    cube = np.tile(r["raw_data"], (5, 4, 1))
    results = [copy.deepcopy(r) for _ in coords]
    s = defaults(
        {
            **reference["s"],
            "stage1_hierarchical_template_max_iter": 1,
            "stage1_hierarchical_spatial_max_iter": 1,
            "stage1_spatial_zlp_iterations": 1,
        }
    )
    output, h = hierarchy.run(cube, r["ene_raw"], coords, results, s, Backend())
    assert h["enabled"] and h["status"] == "ok"
    assert h["cluster_label_map"].shape == (5, 4)
    assert len(output) == 20 and all("error" not in item for item in output)
    assert any(item["phase"].startswith("spatial_") for item in h["history"])
    assert sum(item.get("accepted", 0) for item in h["history"]) > 0
    assert all("refit_failures" not in item for item in output)
