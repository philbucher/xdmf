"""Run under `pvpython` (ParaView's bundled interpreter, not a regular Python) to check that the
xdmf time series -- written beforehand by `cargo run --example paraview_smoke` -- actually opens
and reads back correctly in ParaView. Every fixture listed in `expected.json` is checked; that is
one per (float precision, connectivity index type) pair for the storage backend under test, so both
f64/f32 coordinates and all four connectivity types (u32/u64/i32/i64) are covered. The cells are
checked by VTK class and point ids, since the connectivity type is what decides the mesh size limit
and a misread one shows up as mangled topology rather than as a wrong number.
Also checks that vector/tensor fields come back with the right number of components, not just the
right numeric values, since XDMF's `AttributeType` (Scalar/Vector/.../Matrix) is what ParaView uses
to shape each array.

The `integers` list of each timestep carries one field per integer element type the storage
supports, so the `NumberType`/`Precision` pair written into the light data is checked against what
ParaView decodes. This is what pins down the 64-bit handling in particular: every element type is
written at its own width, and the `level_i64_wide` field is deliberately out of 32-bit range, so a
storage that quietly narrowed it would be caught here. `DataStorage::Binary` carries the 32-bit
fields only -- ParaView reads 64-bit integers in `Format="Binary"` at the wrong stride, so the
writer refuses them rather than narrowing behind the caller's back.

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

# How many fixtures `paraview_smoke` writes per run: two float precisions times the connectivity
# index types the storage can carry. `Binary` has no 64-bit integer types -- ParaView reads them at
# the wrong stride -- so it writes half as many. Checked rather than just iterated over, so that an
# `expected.json` which lists fewer fixtures than expected -- or none at all -- fails loudly
# instead of passing this script vacuously.
NUM_FIXTURES_PER_STORAGE = {"binary": 4}
DEFAULT_NUM_FIXTURES = 8

# One field per integer element type the storage carries: the two 32-bit ones everywhere, plus the
# two 64-bit ones and the out-of-32-bit-range field where 64-bit integers are supported. Checked at
# all so a fixture that stopped emitting them fails instead of passing this script vacuously.
NUM_INTEGER_FIELDS_PER_STORAGE = {"binary": 2}
DEFAULT_NUM_INTEGER_FIELDS = 5


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


def check_cells(data, expected_cells: list, fixture: str) -> None:
    """Compare the topology ParaView built against what the fixture wrote.

    Both the cell class and its point ids are compared: a connectivity read at the wrong width
    typically still yields *some* cells, just not these ones.
    """
    if data.GetNumberOfCells() != len(expected_cells):
        fail(
            f"{fixture}: expected {len(expected_cells)} cell(s), got {data.GetNumberOfCells()}"
        )

    for index, expected in enumerate(expected_cells):
        cell = data.GetCell(index)
        class_name = cell.GetClassName()
        if class_name != expected["type"]:
            fail(
                f"{fixture}: cell {index}: expected {expected['type']}, got {class_name}"
            )

        point_ids = cell.GetPointIds()
        got = [point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds())]
        if got != expected["points"]:
            fail(
                f"{fixture}: cell {index}: point ids mismatch: got {got}, "
                f"expected {expected['points']}"
            )


def check_fixture(fixture: dict, directory: Path, num_integer_fields: int) -> None:
    xdmf_file = directory / fixture["xdmf_file"]

    reader = XDMFReader(FileNames=[str(xdmf_file)])
    UpdatePipeline(time=fixture["timesteps"][0]["time"], proxy=reader)

    data = servermanager.Fetch(reader)
    points = vtk_to_numpy(data.GetPoints().GetData())
    # the f32 fixture's expectations are recorded after the same narrowing the writer applies, so
    # both fixtures compare exactly
    if points.tolist() != fixture["points"]:
        fail(f"{xdmf_file}: points mismatch: got {points.tolist()}, expected {fixture['points']}")

    check_cells(data, fixture["cells"], str(xdmf_file))

    for step in fixture["timesteps"]:
        UpdatePipeline(time=step["time"], proxy=reader)
        data = servermanager.Fetch(reader)

        name = fixture["xdmf_file"]
        check_array(data.GetPointData(), "temperature", step["temperature"], 1, name)
        check_array(data.GetPointData(), "displacement", step["displacement"], 3, name)
        check_array(data.GetPointData(), "velocity_gradient", step["velocity_gradient"], 9, name)

        if len(step["integers"]) != num_integer_fields:
            fail(
                f"{name}: expected {num_integer_fields} integer field(s), "
                f"got {len(step['integers'])}"
            )
        for field in step["integers"]:
            # compared as Python ints, so a value that only fits in 64 bits stays exact -- going
            # through a float here would hide exactly the truncation this is looking for
            check_array(data.GetCellData(), field["name"], field["values"], 1, name)

        if SUPPORTS_MATRIX_ATTRIBUTE:
            check_array(data.GetCellData(), "stress", step["stress"], 6, name)

    skip_note = "" if SUPPORTS_MATRIX_ATTRIBUTE else " (stress field skipped on VTK < 9.6)"
    print(f"OK: {len(fixture['timesteps'])} timestep(s) verified against {xdmf_file}{skip_note}")


def main(expected_path: Path) -> None:
    expected = json.loads(expected_path.read_text())

    fixtures = expected.get("fixtures", [])
    storage = expected.get("storage", "")
    num_expected = NUM_FIXTURES_PER_STORAGE.get(storage, DEFAULT_NUM_FIXTURES)
    if len(fixtures) != num_expected:
        fail(
            f"{expected_path}: expected {num_expected} fixture(s) for storage "
            f"'{storage}', got {len(fixtures)}"
        )

    num_integer_fields = NUM_INTEGER_FIELDS_PER_STORAGE.get(storage, DEFAULT_NUM_INTEGER_FIELDS)
    for fixture in fixtures:
        check_fixture(fixture, expected_path.parent, num_integer_fields)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
