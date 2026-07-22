"""Benchmark: writing a structured-but-unstructured CFD case (hex volume domain + inlet/outlet/
sides quad boundary patches, velocity vector + pressure scalar point data) via xdmf's Python
bindings vs pyvista/VTK.

xdmf: ONE file per case (per storage mode), using write_mesh_with_blocks to combine the domain
and the 3 boundary patches as named blocks sharing one point array. Compared storage modes:
Ascii (separate .txt data files), Binary (separate .bin data files), and both HDF5 modes.

pyvista: 4 separate .vtu files per case (domain, inlet, outlet, sides), each a self-contained
UnstructuredGrid, written with pyvista's default binary+zlib-compressed appended data.

For each case, all produced files are zipped together and the archive size + write/zip times
are reported. Fields here are smooth/physically-derived (parabolic velocity profile, linear
pressure drop) -- see `random_benchmark.py` for the same mesh with pseudo-random field values,
which stresses compression very differently.
"""

import shutil
import sys
import time
from pathlib import Path

from bench_common import Result, print_summary, run_pyvista, run_xdmf
from mesh_gen import build_case

OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./benchmark_output")

CASES = [
    ("1e3", 10, 10, 10),
    ("1e5", 10, 10, 1000),
    ("1e7", 100, 100, 1000),
]

RESULTS: list[Result] = []


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    for case_label, nx, ny, nz in CASES:
        print(f"\n=== case {case_label} ({nx}x{ny}x{nz} hex = {nx * ny * nz:,} elements) ===")
        out_dir = OUTPUT_DIR / case_label
        out_dir.mkdir(parents=True)

        t0 = time.perf_counter()
        case = build_case(nx, ny, nz)
        print(
            f"  [setup] mesh gen: {time.perf_counter() - t0:.3f}s "
            f"({case.num_points:,} points, {case.num_cells:,} cells)"
        )

        run_xdmf("cfd", case_label, case, "Binary", out_dir, RESULTS)
        run_xdmf("cfd", case_label, case, "Hdf5SingleFile", out_dir, RESULTS)
        run_xdmf("cfd", case_label, case, "Hdf5MultipleFiles", out_dir, RESULTS)

        if case_label == "1e7":
            print("  xdmf-ascii     skipped (impractical at 10M elements)")
            RESULTS.append(Result(case_label, "xdmf-ascii", 0, 0, 0, 0, skipped="skipped at 10M"))
        else:
            run_xdmf("cfd", case_label, case, "Ascii", out_dir, RESULTS)

        run_pyvista("cfd", case_label, case, out_dir, RESULTS)

    print_summary(RESULTS)


if __name__ == "__main__":
    main()
