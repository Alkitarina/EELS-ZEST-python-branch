"""Regression checks for immutable inputs, reusable buffers and durable storage."""
import copy
import gzip
import json
import numpy as np
import pytest
from zest import acceleration, stage1, stage2, storage
from zest.backend import Backend
from zest.settings import defaults


@pytest.mark.parametrize("grid", ["uniform", "irregular", "empty_modes"])
def test_shift_native_and_numpy_match_on_edges_and_invalid_values(monkeypatch, grid):
    e = np.linspace(-30, 30, 101)
    if grid == "irregular":
        e[50] += 0.03
    em = np.array([]) if grid == "empty_modes" else np.array([-60, -8.3, 0, 9.7, 60])
    k = np.exp(-e * e / 20)
    k[1:4] = [np.nan, np.inf, -1]
    monkeypatch.setenv("ZEST_CPU_KERNELS", "numpy")
    expected = stage1.shift_matrix(e, k, em, Backend())
    if acceleration._kernels is None:
        pytest.skip("optional C++ extension not built")
    monkeypatch.setenv("ZEST_CPU_KERNELS", "native")
    plan = acceleration.ShiftPlan(e, em, Backend())
    np.testing.assert_array_equal(plan.apply(k), expected)
    # The geometry is reusable with a different kernel; old outputs stay intact.
    before = expected.copy()
    plan.apply(np.ones_like(k))
    np.testing.assert_array_equal(expected, before)


@pytest.mark.parametrize("mode", ["numpy", "native"])
def test_linear_and_residual_buffers_preserve_float64_arithmetic(monkeypatch, mode):
    if mode == "native" and acceleration._kernels is None:
        pytest.skip("optional C++ extension not built")
    monkeypatch.setenv("ZEST_CPU_KERNELS", mode)
    rng = np.random.default_rng(86)
    a, r = rng.normal(size=(19, 6)), rng.normal(size=(7, 6))
    y, w, rhs = rng.normal(size=19), rng.random(19), rng.normal(size=7)
    b = Backend()
    work = acceleration.LinearWorkspace(y, w, rhs, 6, b)
    for matrix in (a, a * 1.5, np.asfortranarray(a)):
        C, d = work.assemble(matrix, r)
        np.testing.assert_array_equal(C, np.concatenate([matrix * w[:, None], r]))
        np.testing.assert_array_equal(d, np.r_[y * w, rhs])
    model, rc = rng.normal(size=19), rng.normal(size=7)
    np.testing.assert_array_equal(acceleration.residual(model, y, w, rc, rhs, b),
                                  np.r_[(model - y) * w, rc - rhs])


def test_decode_context_does_not_recompute_spectrum_features_or_alias_results(reference, monkeypatch):
    ref = reference
    settings = defaults(ref["obj_settings"])
    b = Backend()
    calls = {key: 0 for key in ("adaptive", "weights", "guard_profile")}
    for name in calls:
        original = getattr(stage1, name)
        def wrapper(*args, _name=name, _function=original, **kwargs):
            calls[_name] += 1
            return _function(*args, **kwargs)
        monkeypatch.setattr(stage1, name, wrapper)
    e, y, em = ref["ene"], ref["r"]["raw_data_norm"], ref["EM"]
    ctx = stage1.DecodeContext(e, y, em, settings, b)
    prepared_calls = calls.copy()
    first = stage1.decode(ref["theta"], e, y, em, settings, b, context=ctx)
    before = copy.deepcopy(first)
    theta = ref["theta"].copy()
    theta[-1] += 0.1
    second = stage1.decode(theta, e, y, em, settings, b, context=ctx)
    residual = stage1.decode(theta, e, y, em, settings, b, context=ctx, residual_only=True)
    assert calls == prepared_calls
    assert all(count <= 1 for count in prepared_calls.values())
    for key in ("model_norm", "zlp_norm", "vib_norm", "coef", "optimizer_residual"):
        np.testing.assert_array_equal(first[key], before[key])
    np.testing.assert_array_equal(residual, second["optimizer_residual"])
    standalone = stage1.decode(theta, e, y, em, settings, b)
    np.testing.assert_array_equal(residual, standalone["optimizer_residual"])


@pytest.mark.parametrize("mode", ["numpy", "native"])
@pytest.mark.parametrize("counts", [(3, 4), (0, 4), (4, 0)])
def test_scan_workspace_keeps_all_regularizers_and_updates_temperature(monkeypatch, mode, counts):
    if mode == "native" and acceleration._kernels is None:
        pytest.skip("optional C++ extension not built")
    monkeypatch.setenv("ZEST_CPU_KERNELS", mode)
    nh, ns = counts
    rng = np.random.default_rng(82)
    n = nh + ns
    hard = np.arange(n) < nh
    basis = dict(EM_hard=np.arange(nh) * 4 + 12., EM_soft=np.arange(ns) * 4 + 28.,
                 hard_idx=hard, soft_idx=~hard, stage1_PM_stokes_soft=rng.random(ns),
                 stage1_PM_antistokes_soft=rng.random(ns))
    s = defaults(dict(stage2_soft_be_weight=0.002, stage2_soft_pm_smooth_weight=0.01,
                      stage2_soft_pm_stage1_prior_weight=0.02))
    zr = dict(B=rng.normal(size=(25, 2)), l2=np.array([0.2, 0.6]),
              lb=np.array([-0.1, -0.2]), ub=np.array([0.1, 0.2]))
    ks, ka = rng.random((25, n)), rng.random((25, n))
    work = stage2.ScanWorkspace(basis, s, zr, ks, ka, np.ones(25), np.ones(25), Backend())
    for temperature in (100., 800., 350.):
        qh = np.exp(-basis["EM_hard"] / (0.08617333262 * temperature))
        qs = np.exp(-basis["EM_soft"] / (0.08617333262 * temperature))
        h = work.update(qh, qs)
        expected_h = ks[:, hard] + ka[:, hard] * qh[None, :]
        np.testing.assert_array_equal(h, expected_h)
        np.testing.assert_array_equal(work.A, np.column_stack([expected_h, ks[:, ~hard], ka[:, ~hard], zr["B"]]))
        r, rhs = stage2.regularization(basis, s, qs)
        expected_r = np.pad(r, ((0, 2), (0, 2)))
        expected_r[-2:, -2:] = np.diag(np.sqrt(zr["l2"]))
        np.testing.assert_array_equal(work.R, expected_r)
        np.testing.assert_array_equal(work.rhs, np.r_[rhs, np.zeros(2)])


@pytest.mark.parametrize("native", [False, True])
def test_storage_preserves_nonfinite_tags_float32_and_noncontiguous_arrays(tmp_path, monkeypatch, native):
    if not native:
        monkeypatch.setattr(storage, "orjson", None)
    value = {"中文": np.array([[np.nan, np.inf], [-np.inf, -0.0]]),
             "strided": np.arange(24.).reshape(4, 6)[:, ::2],
             "float32": np.array([0.1, np.pi], dtype=np.float32),
             "endian": np.arange(3., dtype=">f8"), "scalar": np.float64(np.inf),
             "nested": [True, {"x": np.array(0.4)}]}
    path = tmp_path / "pixel.json.gz"
    assert storage.save_pixel(path, value)
    loaded = storage.load_pixel(path)
    for key in ("中文", "strided", "float32", "endian"):
        np.testing.assert_equal(loaded[key], value[key])
    assert np.isposinf(loaded["scalar"])
    assert np.signbit(loaded["中文"][1, 1])
    assert loaded["nested"][1]["x"] == 0.4
    # Native encoding and the old pack/standard-json path have identical values.
    assert json.loads(storage.scientific_bytes(value)) == json.loads(json.dumps(storage.pack(value)))


def test_unchanged_export_detects_in_place_mutations_and_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "pixel.json.gz"
    value = {"array": np.array([1., 2.]), "stage": 1}
    storage.save_pixel(path, value)
    original = path.read_bytes()
    assert not storage.save_pixel(path, value, skip_unchanged=True)
    assert path.read_bytes() == original
    value["array"][1] = 3.
    assert storage.save_pixel(path, value, skip_unchanged=True)
    np.testing.assert_array_equal(storage.load_pixel(path)["array"], [1, 3])
    before_failure = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated compression/write failure")
    monkeypatch.setattr(storage.gzip, "compress", fail)
    with pytest.raises(OSError):
        storage.save_pixel(path, {"stage": 2})
    assert path.read_bytes() == before_failure


def test_native_unavailable_is_reported_and_numpy_fallback_works(monkeypatch):
    monkeypatch.setattr(acceleration, "_kernels", None)
    monkeypatch.setenv("ZEST_CPU_KERNELS", "auto")
    assert not acceleration.native_enabled()
    monkeypatch.setenv("ZEST_CPU_KERNELS", "native")
    with pytest.raises(RuntimeError, match="unavailable"):
        acceleration.native_enabled()


def test_native_rejects_invalid_indices_and_read_only_outputs():
    native = acceleration._kernels
    if native is None:
        pytest.skip("optional C++ extension not built")
    k = np.ones(4)
    lo = np.array([[np.iinfo(np.int64).max]], dtype=np.int64)
    alpha, valid, out = np.ones((1, 1)), np.ones((1, 1), dtype=bool), np.empty((1, 1))
    with pytest.raises(ValueError, match="index"):
        native.shift(k, lo, alpha, valid, out)
    lo[:] = 0
    out.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        native.shift(k, lo, alpha, valid, out)


def test_corrupt_checkpoint_can_be_replaced_from_complete_result(tmp_path):
    path = tmp_path / "pixel.json.gz"
    path.write_bytes(b"damaged gzip")
    assert storage.save_pixel(path, {"complete": True}, skip_unchanged=True)
    assert storage.load_pixel(path) == {"complete": True}
