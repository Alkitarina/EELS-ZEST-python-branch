# Optional local reference data

No scientific dataset or private MATLAB fixture is distributed in this repository.
The application works by importing your own MAT/NPZ data. To enable the optional
reference-data button, set `ZEST_REFERENCE_PATH` before starting the server or
place a permitted input file at `reference/example_data.mat`.

Tests requiring `reference/pixel_reference.mat` explicitly skip if it is absent.
Other tests can still run. A skipped comparison does not establish MATLAB/Python
numerical equivalence. Install optional GPU dependencies separately if needed.

To export your own permitted reference, obtain the upstream MATLAB v2.1.0 source
from https://github.com/Iridium-191/EELS-ZEST/releases and prepare a MAT file with
`results` and `s`. The exporter expects `results{1,1}` to be a Stage 1 result and
`results{1,3}` to contain the corresponding Stage 2 result. In MATLAB, add this
repository's `tools` directory to the path, then call:

```matlab
export_reference('C:/path/to/EELS-ZEST', 'C:/data/representative_results.mat')
```

Run the optional CPU comparison from the repository root:

```powershell
.venv\Scripts\python.exe -X utf8 tools\validate_native.py
# With a compatible NVIDIA GPU and requirements-gpu.txt installed:
.venv\Scripts\python.exe -X utf8 tools\validate_native.py --devices cpu gpu
```

`tools/read_source.py` can inspect upstream MATLAB functions when
`ZEST_UPSTREAM_SOURCE` points to `EELS_PearsonSparse_FullFit.m`:

```powershell
$env:ZEST_UPSTREAM_SOURCE = 'C:\path\to\EELS_PearsonSparse_FullFit.m'
python tools\read_source.py defaults
```

MATLAB is only needed to produce the optional baseline. It is not used by the web
application. Reference files and generated results remain ignored by Git.
