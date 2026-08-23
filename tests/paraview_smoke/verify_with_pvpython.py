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

The `submesh_fixtures` list is checked separately, by `check_submesh_fixture`: a mesh written with
`write_mesh_with_submeshes` is a `Spatial` collection of one `Temporal` collection per submesh,
which ParaView reads back as a multi-block dataset rather than as a single grid, so it needs a
traversal of its own. Each block is matched by name and checked for the points its own cells use,
for those cells in the block's own point numbering, and for its share of every field -- point data
included, since a block carries only its own points and therefore only their values. For the HDF5
storages that share is a selection out of the one array the field was written to, rather than a
copy, so this is also what checks that every block reads the part of it that block holds: the
`stress` field is there because a field of several values per point is selected differently from a
scalar one.

Matching on the name is what pins the nesting down. The other way round -- one `Spatial` collection
per step, inside one `Temporal` -- reads back with the right data, but ParaView makes a grid name
unique across the whole document, so a submesh named once per step comes back as `quad` at the
first step and `quad[1]`, `quad[2]`, ... after it, and the block loses whatever the user set for it
in the Multi-block Inspector as the animation runs. Checking every step, by name, is what catches
that.

The `stress` field (AttributeType="Matrix", used for Tensor6/Matrix/Generic data) is only
checked on VTK >= 9.6 (ParaView >= 6.1): https://github.com/Kitware/VTK/commit/7199be5854
changed how VTK's XDMF2 reader computes a Matrix attribute's component count, and the xdmf
crate's writers target that newer behavior. On older VTK, Matrix-shaped attributes are known
to read back incorrectly -- see `Values::dimensions` in the crate for the writer-side details.

Usage: pvpython verify_with_pvpython.py <expected.json>
"""

import json
import re
import sys
from pathlib import Path

from paraview import servermanager
from paraview.simple import UpdatePipeline, XDMFReader
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkVersion
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet

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

# How many multi-block fixtures `paraview_smoke` writes per run. One per storage: what it covers is
# the grid *structure*, which does not vary with the element types the fixtures above sweep. Checked
# rather than just iterated over, so an `expected.json` that stopped carrying it fails loudly
# instead of passing this script vacuously.
NUM_SUBMESH_FIXTURES = 1

# ParaView 6.1's client-side gather folds the outer collection's generated name into each leaf's,
# so the block `quad` arrives at `servermanager.Fetch` as `Block 0_quad`; 5.13 hands it over
# unchanged. Neither the file nor the reader differs between the two -- `vtkXdmfReader` emits
# `quad` in both -- and neither does the block hierarchy ParaView selects by, which
# `check_block_hierarchy` pins down exactly, so this only undoes what the gather did to the name.
GATHERED_BLOCK_PREFIX = re.compile(r"^Block \d+_")


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
        # a name carrying spaces, parentheses, apostrophes, a dot and a comma -- it is only ever an
        # XML attribute value, so finding the array by this exact name is what shows the escaping
        # of it round-tripped
        check_array(
            data.GetPointData(),
            step["solver_style_name"],
            step["solver_style_values"],
            1,
            name,
        )

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


def collect_blocks(data, blocks: dict = None, name: str = None) -> dict:
    """Flatten whatever composite dataset ParaView built into `{block name: leaf dataset}`.

    Written as a descent rather than reading a fixed nesting depth, because how many levels the
    XDMF2 reader wraps a spatial collection in is its business, not something the writer controls.
    """
    if blocks is None:
        blocks = {}

    if hasattr(data, "GetNumberOfBlocks"):
        for index in range(data.GetNumberOfBlocks()):
            meta = data.GetMetaData(index)
            block_name = None
            if meta is not None and meta.Has(vtkCompositeDataSet.NAME()):
                block_name = meta.Get(vtkCompositeDataSet.NAME())
            collect_blocks(data.GetBlock(index), blocks, block_name)
    elif data is not None:
        blocks[GATHERED_BLOCK_PREFIX.sub("", name) if name is not None else name] = data

    return blocks


def check_block_hierarchy(reader, expected_names: list, fixture: str, time) -> None:
    """Check the block paths ParaView selects by, against the submesh names, at this time step.

    This is what the Multi-block Inspector lists and what `BlockSelectors` takes, i.e. the selection
    a user makes to isolate one submesh -- so it has to be the submesh's own name, and it has to
    still be that name at every step: a layout ParaView had to uniquify per step would show up here
    as `/Root/quad[1]`. Unlike the block names `Fetch` reports, these are unaffected by how the
    client-side gather renames things, so they are also what makes the normalization above safe.
    """
    hierarchy = reader.GetDataInformation().DataInformation.GetHierarchy()
    if hierarchy is None:
        fail(f"{fixture}: t={time}: no block hierarchy")

    root = hierarchy.GetRootNode()
    got = sorted(
        hierarchy.GetNodePath(node) for node in hierarchy.GetChildNodes(root, False)
    )
    expected = sorted(f"/Root/{name}" for name in expected_names)
    if got != expected:
        fail(f"{fixture}: t={time}: expected block paths {expected}, got {got}")


def check_submesh_fixture(fixture: dict, directory: Path) -> None:
    """Check the multi-block fixture: one block per submesh, each with its own points and cells.

    The cell data is compared per block and in the order the submesh named its cells -- the
    "reversed" submesh lists them out of order on purpose, so a writer that quietly sorted them, or
    a reader that did, cannot pass. The points are compared per block too: each block holds only
    the points its own cells touch, so a block reading back the whole mesh's points -- what the
    writer produced before the geometry was compacted -- fails here.
    """
    xdmf_file = directory / fixture["xdmf_file"]
    name = fixture["xdmf_file"]

    reader = XDMFReader(FileNames=[str(xdmf_file)])

    for step in fixture["timesteps"]:
        UpdatePipeline(time=step["time"], proxy=reader)
        blocks = collect_blocks(servermanager.Fetch(reader))

        expected_names = sorted(block["name"] for block in step["blocks"])
        check_block_hierarchy(reader, expected_names, name, step["time"])
        if sorted(blocks) != expected_names:
            fail(
                f"{name}: t={step['time']}: expected blocks {expected_names}, "
                f"got {sorted(blocks)}"
            )

        for expected_block in step["blocks"]:
            block = blocks[expected_block["name"]]
            label = f"{name}[{expected_block['name']}]"

            # each block carries the points its own cells use, and nothing else
            points = vtk_to_numpy(block.GetPoints().GetData()).tolist()
            if points != expected_block["points"]:
                fail(
                    f"{label}: points mismatch: got {points}, "
                    f"expected {expected_block['points']}"
                )

            check_cells(block, expected_block["cells"], label)
            check_array(
                block.GetPointData(), "temperature", expected_block["temperature"], 1, label
            )
            # a point field of six values per point: the shape a submesh selects with an index
            # array of its own, and the one VTK reads back correctly only since 9.6
            if SUPPORTS_MATRIX_ATTRIBUTE:
                check_array(block.GetPointData(), "stress", expected_block["stress"], 6, label)
            check_array(block.GetCellData(), "level_i32", expected_block["level_i32"], 1, label)
            check_array(
                block.GetCellData(), "cell_velocity", expected_block["cell_velocity"], 3, label
            )

    num_blocks = len(fixture["timesteps"][0]["blocks"])
    print(
        f"OK: {len(fixture['timesteps'])} timestep(s) x {num_blocks} block(s) verified "
        f"against {xdmf_file}"
    )


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

    submesh_fixtures = expected.get("submesh_fixtures", [])
    if len(submesh_fixtures) != NUM_SUBMESH_FIXTURES:
        fail(
            f"{expected_path}: expected {NUM_SUBMESH_FIXTURES} submesh fixture(s) for storage "
            f"'{storage}', got {len(submesh_fixtures)}"
        )
    for fixture in submesh_fixtures:
        check_submesh_fixture(fixture, expected_path.parent)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
