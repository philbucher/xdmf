"""Same benchmark as `cfd_benchmark.py` (identical mesh, identical write/zip/report harness --
see `bench_common.py`), but with the point data replaced by pseudo-random, reproducible values
instead of `mesh_gen.build_case`'s smooth, spatially-correlated fields.

Why this matters: `cfd_benchmark.py`'s duct-flow fields (parabolic velocity profile, linear
pressure drop) are unusually easy to compress -- neighboring points have nearly identical
values, which is exactly what the shuffle+deflate filters added to `src/hdf5_writer.rs` are
good at exploiting. Independent per-point noise removes that spatial correlation, which is a
more representative stand-in for turbulent/measured/noisy field data. The RNG is seeded
(`SEED` below), so re-running this script always writes byte-identical field values.

Values still sit at realistic magnitudes/units (velocity fluctuations around 0 m/s, pressure
around atmospheric), not raw `standard_normal()` noise -- the point is to remove spatial
correlation, not to also change the numeric range in a way that would make comparisons to
`cfd_benchmark.py` harder to interpret.
"""

import dataclasses
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from bench_common import Result, print_summary, run_pyvista, run_xdmf
from mesh_gen import build_case

OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./random_benchmark_output")

SEED = 42
VELOCITY_SCALE = 1.0  # m/s, matches mesh_gen.U_CENTERLINE's order of magnitude
PRESSURE_MEAN = 101325.0  # Pa, matches mesh_gen.P_INLET
PRESSURE_SCALE = 500.0  # Pa

CASES = [
    ("1e3", 10, 10, 10),
    ("1e5", 10, 10, 1000),
    ("1e7", 100, 100, 1000),
]

RESULTS: list[Result] = []


def randomize_fields(case, rng: np.random.Generator):
    """Returns a copy of `case` with `velocity`/`pressure` replaced by independent per-point
    noise at realistic magnitudes -- mesh/connectivity/blocks are untouched."""
    velocity = rng.normal(loc=0.0, scale=VELOCITY_SCALE, size=case.num_points * 3)
    pressure = rng.normal(loc=PRESSURE_MEAN, scale=PRESSURE_SCALE, size=case.num_points)
    return dataclasses.replace(case, velocity=velocity, pressure=pressure)


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    rng = np.random.default_rng(SEED)

    for case_label, nx, ny, nz in CASES:
        print(f"\n=== case {case_label} ({nx}x{ny}x{nz} hex = {nx * ny * nz:,} elements) ===")
        out_dir = OUTPUT_DIR / case_label
        out_dir.mkdir(parents=True)

        t0 = time.perf_counter()
        case = randomize_fields(build_case(nx, ny, nz), rng)
        print(
            f"  [setup] mesh gen + random fields: {time.perf_counter() - t0:.3f}s "
            f"({case.num_points:,} points, {case.num_cells:,} cells)"
        )

        run_xdmf("random", case_label, case, "Binary", out_dir, RESULTS)
        run_xdmf("random", case_label, case, "Hdf5SingleFile", out_dir, RESULTS)
        run_xdmf("random", case_label, case, "Hdf5MultipleFiles", out_dir, RESULTS)

        if case_label == "1e7":
            print("  xdmf-ascii     skipped (impractical at 10M elements)")
            RESULTS.append(Result(case_label, "xdmf-ascii", 0, 0, 0, 0, skipped="skipped at 10M"))
        else:
            run_xdmf("random", case_label, case, "Ascii", out_dir, RESULTS)

        run_pyvista("random", case_label, case, out_dir, RESULTS)

    print_summary(RESULTS)


if __name__ == "__main__":
    main()
