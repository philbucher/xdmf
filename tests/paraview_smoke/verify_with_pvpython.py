"""Run under `pvpython` (ParaView's bundled interpreter, not a regular Python) to check that the
xdmf time series -- written beforehand by `cargo run --example paraview_smoke` -- actually open
and read back correctly in ParaView. Every fixture listed in `expected.json` is checked; that is
one per float precision (f64 and f32 coordinates/attributes) for the storage backend under test.
Also checks that vector/tensor fields come back with the right number of components, not just the
right numeric values, since XDMF's `AttributeType` (Scalar/Vector/.../Matrix) is what ParaView uses
to shape each array.

The `stress` field (AttributeType="Matrix", used for Tensor6/Matrix/Generic data) is only
checked on VTK >= 9.6 (ParaView >= 6.1): https://github.com/Kitware/VTK/commit/7199be5854
changed how VTK's XDMF2 reader computes a Matrix attribute's component count, and the xdmf
crate's writers target that newer behavior. On older VTK, Matrix-shaped attributes are known
to read back incorrectly -- see `Values::dimensions` in the crate for the writer-side details.

Usage: pvpython verify_with_pvpython.py <expected.json>
"""

import json
import sys
from pathlib import Path

from paraview import servermanager
from paraview.simple import UpdatePipeline, XDMFReader
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkVersion

SUPPORTS_MATRIX_ATTRIBUTE = (vtkVersion.GetVTKMajorVersion(), vtkVersion.GetVTKMinorVersion()) >= (
    9,
    6,
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def check_array(
    field_data, name: str, expected_values: list, num_components: int, fixture: str
) -> None:
    array = field_data.GetArray(name)
    if array is None:
        fail(f"{fixture}: {name}: array not found")

    if array.GetNumberOfComponents() != num_components:
        fail(
            f"{fixture}: {name}: expected {num_components} component(s), "
            f"got {array.GetNumberOfComponents()}"
        )

    got = vtk_to_numpy(array).tolist()
    if got != expected_values:
        fail(f"{fixture}: {name}: value mismatch: got {got}, expected {expected_values}")


def check_fixture(fixture: dict, directory: Path) -> None:
    xdmf_file = directory / fixture["xdmf_file"]

    reader = XDMFReader(FileNames=[str(xdmf_file)])
    UpdatePipeline(time=fixture["timesteps"][0]["time"], proxy=reader)

    data = servermanager.Fetch(reader)
    points = vtk_to_numpy(data.GetPoints().GetData())
    # the f32 fixture's expectations are recorded after the same narrowing the writer applies, so
    # both fixtures compare exactly
    if points.tolist() != fixture["points"]:
        fail(f"{xdmf_file}: points mismatch: got {points.tolist()}, expected {fixture['points']}")

    for step in fixture["timesteps"]:
        UpdatePipeline(time=step["time"], proxy=reader)
        data = servermanager.Fetch(reader)

        name = fixture["xdmf_file"]
        check_array(data.GetPointData(), "temperature", step["temperature"], 1, name)
        check_array(data.GetPointData(), "displacement", step["displacement"], 3, name)
        check_array(data.GetPointData(), "velocity_gradient", step["velocity_gradient"], 9, name)
        check_array(data.GetCellData(), "region_id", step["region_id"], 1, name)
        if SUPPORTS_MATRIX_ATTRIBUTE:
            check_array(data.GetCellData(), "stress", step["stress"], 6, name)

    skip_note = "" if SUPPORTS_MATRIX_ATTRIBUTE else " (stress field skipped on VTK < 9.6)"
    print(f"OK: {len(fixture['timesteps'])} timestep(s) verified against {xdmf_file}{skip_note}")


def main(expected_path: Path) -> None:
    expected = json.loads(expected_path.read_text())

    for fixture in expected["fixtures"]:
        check_fixture(fixture, expected_path.parent)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
