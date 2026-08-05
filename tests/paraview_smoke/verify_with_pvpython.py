"""Run under `pvpython` (ParaView's bundled interpreter, not a regular Python) to check that a
xdmf time series -- written beforehand by `cargo run --example paraview_smoke` -- actually opens
and reads back correctly in ParaView. Also checks that vector/tensor fields come back with the
right number of components, not just the right numeric values, since XDMF's `AttributeType`
(Scalar/Vector/.../Matrix) is what ParaView uses to shape each array.

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


def check_array(field_data, name: str, expected_values: list, num_components: int) -> None:
    array = field_data.GetArray(name)
    if array is None:
        fail(f"{name}: array not found")

    if array.GetNumberOfComponents() != num_components:
        fail(
            f"{name}: expected {num_components} component(s), "
            f"got {array.GetNumberOfComponents()}"
        )

    got = vtk_to_numpy(array).tolist()
    if got != expected_values:
        fail(f"{name}: value mismatch: got {got}, expected {expected_values}")


def main(expected_path: Path) -> None:
    expected = json.loads(expected_path.read_text())
    xdmf_file = expected_path.parent / expected["xdmf_file"]

    reader = XDMFReader(FileNames=[str(xdmf_file)])
    UpdatePipeline(time=expected["timesteps"][0]["time"], proxy=reader)

    data = servermanager.Fetch(reader)
    points = vtk_to_numpy(data.GetPoints().GetData())
    if points.tolist() != expected["points"]:
        fail(f"points mismatch: got {points.tolist()}, expected {expected['points']}")

    for step in expected["timesteps"]:
        UpdatePipeline(time=step["time"], proxy=reader)
        data = servermanager.Fetch(reader)

        check_array(data.GetPointData(), "temperature", step["temperature"], 1)
        check_array(data.GetPointData(), "displacement", step["displacement"], 3)
        check_array(data.GetPointData(), "velocity_gradient", step["velocity_gradient"], 9)
        check_array(data.GetCellData(), "region_id", step["region_id"], 1)
        if SUPPORTS_MATRIX_ATTRIBUTE:
            check_array(data.GetCellData(), "stress", step["stress"], 6)

    skip_note = "" if SUPPORTS_MATRIX_ATTRIBUTE else " (stress field skipped on VTK < 9.6)"
    print(f"OK: {len(expected['timesteps'])} timestep(s) verified against {xdmf_file}{skip_note}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
