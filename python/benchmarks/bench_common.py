"""Shared write/zip/report harness for the CFD-shaped benchmarks (`cfd_benchmark.py`,
`random_benchmark.py`). Both write the same mesh shape (hex domain + inlet/outlet/sides quad
patches, one point array shared via xdmf's block API / pyvista's per-zone files) -- they only
differ in what field values (`case.velocity`/`case.pressure`) they put on top of that mesh, so
everything about writing, zipping, and reporting lives here once.
"""

import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyvista as pv

import xdmf
from mesh_gen import extract_submesh

CELL_TYPE_MAP = {
    "Hexahedron": xdmf.CellType.Hexahedron,
    "Quadrilateral": xdmf.CellType.Quadrilateral,
}

DATA_SUFFIX = {
    "Binary": ".bin",
    "Ascii": ".txt",
    "Hdf5SingleFile": ".h5",
    "Hdf5MultipleFiles": ".h5",
}


@dataclass
class Result:
    case: str
    method: str
    write_s: float
    zip_s: float
    raw_bytes: int
    zip_bytes: int
    skipped: str = ""


def _dir_size(*paths: Path) -> int:
    total = 0
    for p in paths:
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        elif p.is_file():
            total += p.stat().st_size
    return total


def _zip_files(zip_path: Path, files: list[Path], arc_prefix: str = "") -> float:
    start = time.perf_counter()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.is_dir():
                for sub in sorted(f.rglob("*")):
                    if sub.is_file():
                        zf.write(sub, arcname=str(Path(arc_prefix) / sub.relative_to(f.parent)))
            else:
                zf.write(f, arcname=str(Path(arc_prefix) / f.name))
    return time.perf_counter() - start


def run_xdmf(
    tag: str, case_label: str, case, storage_name: str, out_dir: Path, results: list[Result]
):
    storage = getattr(xdmf.DataStorage, storage_name)
    file_prefix = out_dir / f"{tag}_{case_label}_{storage_name.lower()}"

    cell_types = [CELL_TYPE_MAP[name] for name in case.cell_type_names]

    start = time.perf_counter()
    writer = xdmf.TimeSeriesWriter(str(file_prefix), storage)
    data_writer = writer.write_mesh_with_blocks(
        case.points, case.connectivity, cell_types, case.blocks
    )
    data_writer.write_data(
        "0",
        [
            ("velocity", xdmf.DataAttribute.VECTOR, case.velocity),
            ("pressure", xdmf.DataAttribute.SCALAR, case.pressure),
        ],
        [],
    )
    write_s = time.perf_counter() - start
    # Hdf5* writers keep the underlying .h5 file(s) open (for further writes); drop the
    # writer to close them before reading file sizes / zipping, same as the Rust tests do.
    del data_writer, writer

    xdmf_file = file_prefix.with_suffix(".xdmf2")
    data_dir = file_prefix.with_suffix(DATA_SUFFIX[storage_name])
    raw_bytes = _dir_size(xdmf_file, data_dir)

    zip_path = out_dir / f"{tag}_{case_label}_xdmf_{storage_name.lower()}.zip"
    zip_s = _zip_files(zip_path, [xdmf_file, data_dir])
    zip_bytes = zip_path.stat().st_size

    results.append(
        Result(case_label, f"xdmf-{storage_name.lower()}", write_s, zip_s, raw_bytes, zip_bytes)
    )
    print(
        f"  xdmf-{storage_name.lower():8s} write={write_s:8.3f}s zip={zip_s:7.3f}s "
        f"raw={raw_bytes / 1e6:9.2f}MB zip={zip_bytes / 1e6:9.2f}MB"
    )


def run_pyvista(tag: str, case_label: str, case, out_dir: Path, results: list[Result]):
    start = time.perf_counter()

    zones = {}
    points_3d = case.points.reshape(-1, 3)
    velocity_3d = case.velocity.reshape(-1, 3)

    domain_conn = case.connectivity[: case.num_hex * 8].reshape(-1, 8)
    grid = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: domain_conn}, points_3d)
    grid.point_data["velocity"] = velocity_3d
    grid.point_data["pressure"] = case.pressure
    zones["domain"] = grid

    flat_offset = case.num_hex * 8
    for name, count in (("inlet", case.num_inlet), ("outlet", case.num_outlet), ("sides", case.num_sides)):
        flat_start = flat_offset
        flat_end = flat_offset + count * 4
        local_points, local_conn, local_velocity, local_pressure = extract_submesh(
            case, flat_start, flat_end, 4
        )
        g = pv.UnstructuredGrid({pv.CellType.QUAD: local_conn}, local_points)
        g.point_data["velocity"] = local_velocity
        g.point_data["pressure"] = local_pressure
        zones[name] = g
        flat_offset = flat_end

    files = []
    for name, g in zones.items():
        path = out_dir / f"{tag}_{case_label}_pyvista_{name}.vtu"
        g.save(path, binary=True)
        files.append(path)

    write_s = time.perf_counter() - start
    raw_bytes = _dir_size(*files)

    zip_path = out_dir / f"{tag}_{case_label}_pyvista.zip"
    zip_s = _zip_files(zip_path, files)
    zip_bytes = zip_path.stat().st_size

    results.append(Result(case_label, "pyvista-vtu", write_s, zip_s, raw_bytes, zip_bytes))
    print(
        f"  pyvista-vtu      write={write_s:8.3f}s zip={zip_s:7.3f}s "
        f"raw={raw_bytes / 1e6:9.2f}MB zip={zip_bytes / 1e6:9.2f}MB"
    )


def print_summary(results: list[Result]):
    print("\n\n=== summary ===")
    header = f"{'case':6s} {'method':14s} {'write_s':>10s} {'zip_s':>8s} {'total_s':>9s} {'raw_MB':>10s} {'zip_MB':>10s} {'ratio':>7s}"
    print(header)
    for r in results:
        if r.skipped:
            print(f"{r.case:6s} {r.method:14s} {r.skipped}")
            continue
        total = r.write_s + r.zip_s
        ratio = r.zip_bytes / r.raw_bytes if r.raw_bytes else 0.0
        print(
            f"{r.case:6s} {r.method:14s} {r.write_s:10.3f} {r.zip_s:8.3f} {total:9.3f} "
            f"{r.raw_bytes / 1e6:10.2f} {r.zip_bytes / 1e6:10.2f} {ratio:7.3f}"
        )
