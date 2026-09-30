"""Run the native CPU and CUDA paths against the saved MATLAB reference."""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zest import stage1, stage2
from zest.backend import Backend
from zest.numeric import clean_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "gpu"), default=["cpu"])
    args = parser.parse_args()
    path = ROOT / "reference/pixel_reference.mat"
    if not path.is_file():
        parser.error("Local MATLAB reference fixture is absent; see reference/README.md")
    ref = loadmat(path, simplify_cells=True)
    rows = []
    with threadpool_limits(2):
        for device in args.devices:
            b = Backend(device)
            print(f"{device}: fixed-theta and Stage 2 reference", flush=True)
            f = stage1.decode(
                ref["theta"],
                ref["ene"],
                ref["r"]["raw_data_norm"],
                ref["EM"],
                ref["obj_settings"],
                b,
            )
            s2 = stage2.solve(ref["r"], ref["s"], b)
            row = dict(
                device=device,
                fixed_model_max_error=float(
                    np.max(abs(f["model_norm"] - ref["fit"]["model_norm"]))
                ),
                matlab_stage2_temperature=ref["stage2"]["T_fit"],
                native_stage2_temperature=s2["T_fit"],
                native_stage2_uncertainty=s2["T_uncertainty"],
            )
            print(row, flush=True)
            print(f"{device}: full native optimizer (original two rounds)", flush=True)
            t = time.perf_counter()
            r = stage1.solve(
                ref["r"]["ene_raw"],
                ref["r"]["raw_data"],
                ref["s"],
                b,
                progress=lambda msg: print(msg, flush=True),
            )
            r = stage2.promote(r, ref["s"], b)
            row.update(
                elapsed_seconds=time.perf_counter() - t,
                R2=r["metrics"]["R2"],
                T=r["T_fit"],
                T_uncertainty=r["T_uncertainty"],
                gpu_operations=b.gpu_operations,
                stage1_history=r["iter_history"],
            )
            rows.append(row)
            np.savez_compressed(
                ROOT / f"reference/native_{device}.npz",
                ene=r["ene_corrected"],
                model=r["fitted_spectrum_norm"],
                raw=r["raw_data_norm"],
                theta=r["theta"],
            )
            (ROOT / "reference/native_validation.json").write_text(
                json.dumps(clean_json(rows), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(row, flush=True)


if __name__ == "__main__":
    main()
