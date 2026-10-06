// Strict Float64 array kernels. Solvers and BLAS operations remain in SciPy.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;
template <typename T> using Array = py::array_t<T, py::array::c_style>;

static void writable(const py::array& a) {
    if (!a.writeable()) throw std::invalid_argument("output must be writable");
}

static void shift(const Array<double>& kernel, const Array<py::ssize_t>& lo,
                  const Array<double>& alpha, const Array<bool>& valid,
                  Array<double>& out) {
    if (kernel.ndim() != 1 || lo.ndim() != 2 || alpha.ndim() != 2 ||
        valid.ndim() != 2 || out.ndim() != 2 || kernel.size() < 2 ||
        lo.shape(0) != out.shape(0) || lo.shape(1) != out.shape(1) ||
        alpha.shape(0) != out.shape(0) || alpha.shape(1) != out.shape(1) ||
        valid.shape(0) != out.shape(0) || valid.shape(1) != out.shape(1))
        throw std::invalid_argument("incompatible shift buffers");
    writable(out);
    const auto* k = kernel.data();
    const auto* index = lo.data();
    const auto* a = alpha.data();
    const auto* ok = valid.data();
    auto* dest = out.mutable_data();
    // Validate indices even for direct callers of this private extension.
    for (py::ssize_t i = 0; i < lo.size(); ++i)
        if (ok[i] && (index[i] < 0 || index[i] >= kernel.size() - 1))
            throw std::invalid_argument("shift index outside kernel");
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < out.size(); ++i) {
        double value = 0.0;
        if (ok[i]) value = (1.0 - a[i]) * k[index[i]] + a[i] * k[index[i] + 1];
        dest[i] = std::isfinite(value) && value >= 0.0 ? value : 0.0;
    }
    // Column normalization deliberately stays in NumPy to preserve its
    // reduction order exactly, including SIMD-dependent summation behavior.
}

static void weighted_matrix(const Array<double>& a, const Array<double>& weights,
                            const Array<double>& reg, Array<double>& out) {
    if (a.ndim() != 2 || reg.ndim() != 2 || out.ndim() != 2 || weights.ndim() != 1 ||
        a.shape(0) != weights.size() || a.shape(1) != reg.shape(1) ||
        out.shape(0) != a.shape(0) + reg.shape(0) || out.shape(1) != a.shape(1))
        throw std::invalid_argument("incompatible least-squares buffers");
    writable(out);
    const auto* src = a.data();
    const auto* w = weights.data();
    const auto* r = reg.data();
    auto* dest = out.mutable_data();
    const auto n = a.shape(1);
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < a.shape(0); ++i)
        for (py::ssize_t j = 0; j < n; ++j) dest[i * n + j] = src[i * n + j] * w[i];
    if (reg.size()) std::memcpy(dest + a.size(), r, sizeof(double) * reg.size());
}

static Array<double> residual(const Array<double>& model, const Array<double>& y,
                              const Array<double>& weights, const Array<double>& reg,
                              const Array<double>& rhs) {
    if (model.ndim() != 1 || y.ndim() != 1 || weights.ndim() != 1 ||
        reg.ndim() != 1 || rhs.ndim() != 1 || model.size() != y.size() ||
        model.size() != weights.size() || reg.size() != rhs.size())
        throw std::invalid_argument("incompatible residual buffers");
    Array<double> out(model.size() + reg.size());
    auto* dest = out.mutable_data();
    const auto* m = model.data(); const auto* obs = y.data(); const auto* w = weights.data();
    const auto* r = reg.data(); const auto* target = rhs.data();
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < model.size(); ++i) dest[i] = (m[i] - obs[i]) * w[i];
    for (py::ssize_t i = 0; i < reg.size(); ++i) dest[model.size() + i] = r[i] - target[i];
    return out;
}

static void hard_columns(const Array<double>& stokes, const Array<double>& anti,
                         const Array<double>& q, Array<double>& out) {
    if (stokes.ndim() != 2 || anti.ndim() != 2 || q.ndim() != 1 || out.ndim() != 2 ||
        stokes.shape(0) != anti.shape(0) || stokes.shape(1) != anti.shape(1) ||
        stokes.shape(1) != q.size() || out.shape(0) != stokes.shape(0) ||
        out.shape(1) < q.size()) throw std::invalid_argument("incompatible scan buffers");
    writable(out);
    const auto* s = stokes.data(); const auto* a = anti.data(); const auto* ratio = q.data();
    auto* dest = out.mutable_data();
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < stokes.shape(0); ++i)
        for (py::ssize_t j = 0; j < q.size(); ++j)
            dest[i * out.shape(1) + j] = s[i * q.size() + j] + a[i * q.size() + j] * ratio[j];
}

PYBIND11_MODULE(_kernels, m) {
    m.attr("precision") = "float64-strict";
    m.def("shift", &shift, py::arg("kernel").noconvert(), py::arg("lo").noconvert(),
          py::arg("alpha").noconvert(), py::arg("valid").noconvert(), py::arg("out").noconvert());
    m.def("weighted_matrix", &weighted_matrix, py::arg("a").noconvert(),
          py::arg("weights").noconvert(), py::arg("reg").noconvert(), py::arg("out").noconvert());
    m.def("residual", &residual, py::arg("model").noconvert(), py::arg("y").noconvert(),
          py::arg("weights").noconvert(), py::arg("reg").noconvert(), py::arg("rhs").noconvert());
    m.def("hard_columns", &hard_columns, py::arg("stokes").noconvert(),
          py::arg("anti").noconvert(), py::arg("q").noconvert(), py::arg("out").noconvert());
}
