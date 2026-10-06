"""Build the optional strict Float64 C++ kernels into zest/ (no solver changes)."""
import os
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
flags = ["/O2", "/fp:strict", "/utf-8"] if os.name == "nt" else ["-O3", "-fno-fast-math", "-ffp-contract=off"]
setup(
    name="zest-native-kernels", version="0.1.0",
    ext_modules=[Pybind11Extension("zest._kernels", ["native/kernels.cpp"],
                                  cxx_std=17, extra_compile_args=flags)],
    cmdclass={"build_ext": build_ext},
    script_args=["build_ext", "--inplace", "--build-temp", ".tmp_native_build"],
)
