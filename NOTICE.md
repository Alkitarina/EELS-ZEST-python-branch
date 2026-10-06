# Attribution and adaptation notice

This project is a Python/Web rewrite and adaptation of selected workflows from
EELS-ZEST v2.1.0, especially `EELS_PearsonSparse_FullFit.m` and the preprocessing
interface. It adds a browser workbench, CPU multiprocessing, configurable stripe
scheduling, timing, persistence and interrupted-job status handling.

本项目来源于对 EELS-ZEST 发布版本的改写和改编。

- Original project: https://github.com/Iridium-191/EELS-ZEST
- Original releases: https://github.com/Iridium-191/EELS-ZEST/releases
- Baseline release: https://github.com/Iridium-191/EELS-ZEST/releases/tag/v2.1.0
- Adaptation repository: https://github.com/Alkitarina/EELS-ZEST-python-branch
- Public adaptation preparation date: 2026-09-30
- Adaptation update (0.4.0): 2026-10-06; optional C++/pybind11 kernels,
  per-solve caches and buffer reuse, and optimized JSON/gzip persistence.

## Upstream copyright and license

EELS-ZEST: Zero-loss Estimation and Sparse Thermometry

Copyright (C) 2026 Ruilin Mao

International Center for Quantum Materials (ICQM), Peking University

The original work and this derivative are distributed under the GNU General
Public License version 3 only (`GPL-3.0-only`). See LICENSE for the complete
terms. Original authorship is retained; the adaptation is independently
maintained and is not an official release or endorsement by the upstream author.
Python/SciPy solvers differ from MATLAB; end-to-end numerical equivalence is not
established. See VALIDATION.md for limitations.

## Third-party components

The web design references GradePrism's general visual style. The HTML, CSS and
JavaScript in this application were written for this application.

Plotly.js, bundled at `static/vendor/plotly.min.js`, is Copyright Plotly, Inc.,
MIT licensed; its license notice is included in the distribution header.
NumPy, SciPy, FastAPI, CuPy, mat-io, pybind11, orjson and other dependencies retain their respective
licenses in the installed packages. mat-io reads MATLAB MCOS objects;
MATLAB Engine is not required by the application.

## Public distribution

Private reference spectra, MAT fixtures, validation output values, screenshots,
runtime datasets, fit results, logs, virtual environments and temporary files
are excluded. The original local working directory is not part of this
repository's Git history. Reference-dependent tests explicitly skip when the
private fixture is absent; these skips are not validation passes.
