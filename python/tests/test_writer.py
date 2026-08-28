"""End-to-end checks for the xdmf python bindings: write a small mesh and a couple of time steps
of point/cell data from numpy arrays, and verify what lands in the produced files.

The dtype-specific cases exist because nothing in this path casts: an array is stored as the type
it is passed in, and a dtype a storage cannot carry back is rejected instead (see `src/paraview.rs`
in the core crate).
"""

import ast
import inspect
import re
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

import xdmf

# a unit square as two triangles, the mesh most tests here write
SQUARE_COORDS = np.array(
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    dtype=np.float64,
)
SQUARE_CONNECTIVITY = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
SQUARE_CELL_TYPES = [xdmf.CellType.Triangle, xdmf.CellType.Triangle]

TEMPERATURE = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
REGION_ID = np.array([100, 200], dtype=np.uint32)

requires_hdf5 = pytest.mark.skipif(
    not xdmf.is_hdf5_enabled(), reason="this build has no HDF5 storage"
)


def write_square(tmp_path, data_storage, name="test_output", connectivity=SQUARE_CONNECTIVITY):
    """Writes the square mesh, returning its path (without suffix) and the data writer."""
    file_path = tmp_path / name
    writer = xdmf.TimeSeriesWriter(str(file_path), data_storage)
    return file_path, writer.write_mesh(SQUARE_COORDS, connectivity, SQUARE_CELL_TYPES)


def write_points(tmp_path, data_storage, name="points"):
    """Writes a single-point mesh, for cases where the mesh itself is not what is being tested."""
    file_path = tmp_path / name
    writer = xdmf.TimeSeriesWriter(str(file_path), data_storage)
    coords = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    connectivity = np.array([0], dtype=np.uint32)
    return file_path, writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex])


def test_write_mesh_and_data_binary(tmp_path):
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Binary)

    data_writer.write_time_step(
        "0",
        [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)],
        [("region_id", xdmf.DataAttribute.SCALAR, REGION_ID)],
    )

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert 'Format="Binary"' in xml
    assert 'Endian="Little"' in xml
    assert 'NumberType="UInt" Format="Binary" Precision="4"' in xml
    assert 'NumberType="Float" Format="Binary" Precision="8"' in xml

    bin_dir = file_path.with_suffix(".bin")
    points_bytes = (bin_dir / "points.bin").read_bytes()
    assert struct.unpack("<12d", points_bytes) == tuple(SQUARE_COORDS)

    # "temperature" (point data) is the step's first array, "region_id" (cell data) its second
    temperature_bytes = (bin_dir / "data_t_0_0.bin").read_bytes()
    assert struct.unpack("<4d", temperature_bytes) == tuple(TEMPERATURE)

    region_bytes = (bin_dir / "data_t_0_1.bin").read_bytes()
    assert struct.unpack("<2I", region_bytes) == tuple(REGION_ID)


@requires_hdf5
def test_write_mesh_and_data_hdf5_single_file(tmp_path):
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Hdf5SingleFile)

    with data_writer:
        data_writer.write_time_step(
            "0",
            [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)],
            [("region_id", xdmf.DataAttribute.SCALAR, REGION_ID)],
        )

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert 'Format="HDF"' in xml
    assert "test_output.h5:data/t_0/0" in xml
    assert "test_output.h5:data/t_0/1" in xml

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5")) as h5_file:
        np.testing.assert_array_equal(h5_file["data/t_0/0"][:], TEMPERATURE)
        np.testing.assert_array_equal(h5_file["data/t_0/1"][:], REGION_ID)


@requires_hdf5
def test_write_mesh_and_data_hdf5_multiple_files(tmp_path):
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Hdf5MultipleFiles)

    with data_writer:
        data_writer.write_time_step(
            "0", [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)]
        )

    assert 'Format="HDF"' in file_path.with_suffix(".xdmf2").read_text()

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5") / "data_t_0.h5") as h5_file:
        np.testing.assert_array_equal(h5_file["0"][:], TEMPERATURE)


@requires_hdf5
def test_hdf5_custom_deflate_level(tmp_path):
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.hdf5_single_file(3))

    with data_writer:
        data_writer.write_time_step(
            "0", [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)]
        )

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5")) as h5_file:
        dataset = h5_file["data/t_0/0"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 3
        np.testing.assert_array_equal(dataset[:], TEMPERATURE)


@pytest.mark.parametrize("storage", [xdmf.DataStorage.Ascii, xdmf.DataStorage.AsciiInline])
def test_write_mesh_and_data_ascii(tmp_path, storage):
    file_path, data_writer = write_square(tmp_path, storage)

    data_writer.write_time_step(
        "0.0", [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)]
    )

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert 'Format="XML"' in xml
    assert 'NumberType="Float" Format="XML" Precision="8"' in xml


def test_write_several_time_steps(tmp_path):
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Ascii)

    for step in range(3):
        data_writer.write_time_step(
            str(step),
            [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE + step)],
            [("region_id", xdmf.DataAttribute.SCALAR, REGION_ID)],
        )

    xml = file_path.with_suffix(".xdmf2").read_text()
    for step in range(3):
        assert f'<Time Value="{step}"/>' in xml


def test_data_can_be_float32(tmp_path):
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.AsciiInline)

    temperature = TEMPERATURE.astype(np.float32)
    data_writer.write_time_step(
        "0.0", [("temperature", xdmf.DataAttribute.SCALAR, temperature)]
    )

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert 'NumberType="Float" Format="XML" Precision="4"' in xml


def test_points_can_be_float32(tmp_path):
    # float32 points are stored at that precision instead of being widened to float64
    file_path = tmp_path / "f32_points"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.AsciiInline)
    writer.write_mesh(
        SQUARE_COORDS.astype(np.float32), SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES
    )

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert 'NumberType="Float" Format="XML" Precision="4"' in xml


@pytest.mark.parametrize(
    ("dtype", "precision", "number_type"),
    [
        (np.uint32, "4", "UInt"),
        (np.int32, "4", "Int"),
        (np.uint64, "8", "UInt"),
        (np.int64, "8", "Int"),
    ],
)
def test_connectivity_is_stored_as_the_dtype_it_is_passed_in(
    tmp_path, dtype, precision, number_type
):
    # numpy's default integer dtype is signed int64, so all four must work without a cast -- and
    # the connectivity dtype is what caps how large a mesh can be written.
    file_path = tmp_path / f"conn_{np.dtype(dtype).name}"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.AsciiInline)
    writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY.astype(dtype), SQUARE_CELL_TYPES)

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert f'NumberType="{number_type}" Format="XML" Precision="{precision}"' in xml


def test_accepts_2d_point_and_vector_shapes(tmp_path):
    # (N, 3) points/vectors are the natural numpy layout and, being C-contiguous, have exactly the
    # flat memory layout the underlying Rust API wants -- no `reshape(-1)` needed.
    file_path = tmp_path / "shapes_2d"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    data_writer = writer.write_mesh(
        SQUARE_COORDS.reshape(-1, 3), SQUARE_CONNECTIVITY.reshape(-1, 3), SQUARE_CELL_TYPES
    )

    velocity = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
    assert velocity.shape == (4, 3)
    data_writer.write_time_step("0.0", [("velocity", xdmf.DataAttribute.VECTOR, velocity)])

    assert 'AttributeType="Vector"' in file_path.with_suffix(".xdmf2").read_text()


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]
)
def test_cell_types_as_numpy_codes(tmp_path, dtype):
    # the CellType values are the XDMF topology type codes, so an array of codes is an equivalent,
    # cheaper-to-produce alternative to a list of CellType
    assert int(xdmf.CellType.Triangle) == 4

    file_path = tmp_path / f"codes_{np.dtype(dtype).name}"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, np.array([4, 4], dtype=dtype))

    # both cells decode to the same CellType (Triangle), so they're written as uniform topology
    assert 'TopologyType="Triangle" NumberOfElements="2"' in file_path.with_suffix(".xdmf2").read_text()


def test_unknown_cell_type_code_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "unknown_code"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, np.array([4, 3], dtype=np.uint8))
    assert str(exc_info.value) == "unknown cell type code 3"


def test_negative_cell_type_code_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "negative_code"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, np.array([4, -4], dtype=np.int64))
    assert str(exc_info.value) == "cell type code -4 is negative"


def test_cell_types_of_wrong_type_are_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "cell_types"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, ["Triangle", "Triangle"])
    assert str(exc_info.value) == (
        "cell_types must be a sequence of xdmf.CellType values, or a numpy array of dtype "
        "uint8, uint16, uint32, uint64, int8, int16, int32, or int64"
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_negative_connectivity_is_rejected(tmp_path, dtype):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / f"neg_{np.dtype(dtype).name}"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(
            SQUARE_COORDS, np.array([0, 1, -1], dtype=dtype), [xdmf.CellType.Triangle]
        )
    assert str(exc_info.value) == "invalid mesh: connectivity index -1 is negative"


def test_points_of_a_non_float_dtype_are_rejected(tmp_path):
    # uint64 is a dtype this crate understands, just not for points -- the rejection names which
    # parameter was wrong, not only which dtypes exist
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "int_points"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(
            SQUARE_COORDS.astype(np.uint64), SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES
        )
    assert str(exc_info.value) == (
        "expected points as a numpy array with dtype float64 or float32, "
        "got a numpy array with dtype uint64"
    )


def test_connectivity_of_a_float_dtype_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "float_conn"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(
            SQUARE_COORDS, SQUARE_CONNECTIVITY.astype(np.float64), SQUARE_CELL_TYPES
        )
    assert str(exc_info.value) == (
        "expected connectivity as a numpy array with dtype uint64, uint32, int64, or int32, "
        "got a numpy array with dtype float64"
    )


def test_data_of_an_unsupported_dtype_is_rejected(tmp_path):
    _file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        data_writer.write_time_step(
            "0.0",
            [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE.astype(np.int16))],
        )
    assert str(exc_info.value) == (
        "expected data of 'temperature' as a numpy array with dtype float64, float32, uint64, "
        "uint32, int64, or int32, got a numpy array with dtype int16"
    )


def test_a_list_instead_of_an_array_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "list_points"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh([0.0, 0.0, 0.0], SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)
    assert str(exc_info.value) == (
        "expected points as a numpy array with dtype float64 or float32, got list"
    )


def test_non_contiguous_arrays_are_rejected(tmp_path):
    # a strided view is rejected rather than silently copied into a contiguous one
    non_contiguous = np.arange(24, dtype=np.float64)[::2]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    assert non_contiguous.ndim == 1  # so it is the contiguity that is rejected, not the shape

    writer = xdmf.TimeSeriesWriter(str(tmp_path / "non_contiguous"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(non_contiguous, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)
    assert str(exc_info.value) == (
        "array must be C-contiguous; call `numpy.ascontiguousarray()` on it first"
    )


def test_transposed_points_are_rejected(tmp_path):
    # a (3, N) array of separate x/y/z rows is C-contiguous, so without a shape check it would be
    # accepted and read as interleaved xyz -- a valid file holding a mesh that was never passed
    rows = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    assert rows.flags["C_CONTIGUOUS"] and rows.size == SQUARE_COORDS.size

    writer = xdmf.TimeSeriesWriter(str(tmp_path / "transposed"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(rows, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)
    assert "whose last dimension is 4" in str(exc_info.value)

    # and the documented fix works
    writer.write_mesh(np.ascontiguousarray(rows.T), SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)


@pytest.mark.parametrize("shape", [(12,), (4, 3), (2, 2, 3)])
def test_point_shapes_with_three_trailing_components_are_accepted(tmp_path, shape):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / f"shape_{len(shape)}"), xdmf.DataStorage.Ascii)
    writer.write_mesh(SQUARE_COORDS.reshape(shape), SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)


def test_a_rejected_write_mesh_leaves_the_writer_usable(tmp_path):
    # only the core crate's `write_mesh(self)` consumes the writer; a dtype/shape/cell-type the
    # caller can fix must not also cost them the writer, or the RuntimeError they get on the retry
    # would claim a call succeeded that never happened
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "retry"), xdmf.DataStorage.Ascii)
    rejected = [
        (SQUARE_COORDS.astype(np.uint64), SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES),
        (SQUARE_COORDS.reshape(3, 4), SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES),
        (np.arange(24, dtype=np.float64)[::2], SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES),
        (SQUARE_COORDS, SQUARE_CONNECTIVITY.astype(np.float64), SQUARE_CELL_TYPES),
        (SQUARE_COORDS, SQUARE_CONNECTIVITY, np.array([99, 99], dtype=np.uint8)),
    ]
    for points, connectivity, cell_types in rejected:
        with pytest.raises(ValueError):
            writer.write_mesh(points, connectivity, cell_types)

    writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)


def test_write_mesh_with_submeshes(tmp_path):
    # each submesh becomes its own <Grid>, holding only the points its own cells use; point and
    # cell data are passed for the whole mesh and the writer gives each submesh its own share.
    # Once a step is written, a submesh's grid is a Temporal collection of one grid per step.
    file_path = tmp_path / "submeshes"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.AsciiInline)
    data_writer = writer.write_mesh_with_submeshes(
        SQUARE_COORDS,
        SQUARE_CONNECTIVITY,
        SQUARE_CELL_TYPES,
        [("tri0", [0]), ("tri1", [1])],
    )
    data_writer.write_time_step(
        "0.0",
        point_data=[("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)],
        cell_data=[("material", xdmf.DataAttribute.SCALAR, np.array([10.0, 20.0]))],
    )

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert 'GridType="Collection" CollectionType="Spatial"' in xml
    assert '<Grid Name="tri0" GridType="Collection" CollectionType="Temporal">' in xml
    assert '<Grid Name="tri1" GridType="Collection" CollectionType="Temporal">' in xml
    assert '<Grid Name="tri0-t0.0" GridType="Uniform">' in xml
    assert '<Grid Name="tri1-t0.0" GridType="Uniform">' in xml

    tri0_xml = xml.split('<Grid Name="tri0-t0.0"')[1].split("</Grid>")[0]
    tri1_xml = xml.split('<Grid Name="tri1-t0.0"')[1].split("</Grid>")[0]
    assert attribute_value(tri0_xml, "material", "Cell") == "1e1"
    assert attribute_value(tri1_xml, "material", "Cell") == "2e1"
    # tri0 holds points 0, 1, 2 and tri1 points 0, 2, 3, so each gets that share of TEMPERATURE
    assert attribute_value(tri0_xml, "temperature", "Node") == "1e1 1.1e1 1.2e1"
    assert attribute_value(tri1_xml, "temperature", "Node") == "1e1 1.2e1 1.3e1"


def attribute_value(grid_xml, name, center):
    """The inline text of one Scalar Attribute's DataItem, within one <Grid>'s XML slice."""
    match = re.search(
        rf'<Attribute Name="{name}" AttributeType="Scalar" Center="{center}">\s*'
        r"<DataItem[^>]*>([^<]+)</DataItem>",
        grid_xml,
    )
    assert match, grid_xml
    return match.group(1)


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]
)
def test_submesh_cells_as_numpy_array(tmp_path, dtype):
    file_path = tmp_path / f"submesh_codes_{np.dtype(dtype).name}"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    writer.write_mesh_with_submeshes(
        SQUARE_COORDS,
        SQUARE_CONNECTIVITY,
        SQUARE_CELL_TYPES,
        [("all", np.array([0, 1], dtype=dtype))],
    )
    assert '<Grid Name="all" GridType="Uniform">' in file_path.with_suffix(".xdmf2").read_text()


def test_submesh_cells_as_a_plain_list(tmp_path):
    file_path = tmp_path / "submesh_list"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    writer.write_mesh_with_submeshes(
        SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("all", [0, 1])]
    )
    assert '<Grid Name="all" GridType="Uniform">' in file_path.with_suffix(".xdmf2").read_text()


def test_submesh_cells_as_a_range(tmp_path):
    # a range of consecutive cells is taken as the block it names, without its indices ever being
    # built -- the file records it as the "<start>:<len>" pair such a submesh is stored as
    file_path = tmp_path / "submesh_range"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    writer.write_mesh_with_submeshes(
        SQUARE_COORDS,
        SQUARE_CONNECTIVITY,
        SQUARE_CELL_TYPES,
        [("lower", range(1)), ("upper", range(1, 2))],
    )

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert '<Grid Name="lower" GridType="Uniform">' in xml
    assert '<Grid Name="upper" GridType="Uniform">' in xml
    assert '<Information Name="submesh_cells" Value="0:1 1:1"/>' in xml


def test_submesh_cells_as_a_strided_range(tmp_path):
    # a range that is not a block of consecutive cells is read as the plain sequence it also is,
    # so it lands in the scattered form, with an index array per submesh
    file_path = tmp_path / "submesh_strided_range"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    writer.write_mesh_with_submeshes(
        SQUARE_COORDS,
        np.array([0, 1, 2, 3], dtype=np.uint32),
        [xdmf.CellType.Vertex] * 4,
        [("even", range(0, 4, 2)), ("odd", range(1, 4, 2))],
    )

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert (
        '<Information Name="submesh_cells" Value="submesh_cells_0 submesh_cells_1"/>' in xml
    )


def test_empty_range_submesh_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "empty_range"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("empty", range(1, 1))]
        )
    assert "must contain at least one cell" in str(exc_info.value)


def test_out_of_range_submesh_range_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "oob_range"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("part", range(0, 3))]
        )
    assert "references cell 2" in str(exc_info.value)


def test_negative_range_start_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "negative_range"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("part", range(-1, 2))]
        )
    assert "is negative" in str(exc_info.value)


def test_overlapping_submeshes_are_allowed(tmp_path):
    file_path = tmp_path / "overlap"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    writer.write_mesh_with_submeshes(
        SQUARE_COORDS,
        SQUARE_CONNECTIVITY,
        SQUARE_CELL_TYPES,
        [("a", [0, 1]), ("b", [1])],
    )
    xml = file_path.with_suffix(".xdmf2").read_text()
    assert '<Grid Name="a" GridType="Uniform">' in xml
    assert '<Grid Name="b" GridType="Uniform">' in xml


def test_empty_submesh_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "empty_submesh"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("empty", [])]
        )
    assert "must contain at least one cell" in str(exc_info.value)


def test_duplicate_submesh_name_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "dup_submesh"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS,
            SQUARE_CONNECTIVITY,
            SQUARE_CELL_TYPES,
            [("part", [0]), ("part", [1])],
        )
    assert "used more than once" in str(exc_info.value)


def test_out_of_range_submesh_cell_is_rejected(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "oob_submesh"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("part", [0, 2])]
        )
    assert "but the mesh only has" in str(exc_info.value)


def test_a_cell_in_no_submesh_is_rejected(tmp_path):
    # a cell in none of the submeshes would silently vanish from the visualization
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "uncovered"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("part", [0])]
        )
    assert "belong to no submesh" in str(exc_info.value)


@pytest.mark.parametrize("as_numpy", [False, True])
def test_negative_submesh_cell_index_is_rejected(tmp_path, as_numpy):
    cells = np.array([-1], dtype=np.int64) if as_numpy else [-1]
    writer = xdmf.TimeSeriesWriter(str(tmp_path / f"neg_submesh_{as_numpy}"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("part", cells)]
        )
    assert str(exc_info.value) == "submesh cell index -1 is negative"


def test_a_rejected_write_mesh_with_submeshes_leaves_the_writer_usable(tmp_path):
    # only python-layer-detectable rejections (dtype, shape, cell type) leave the writer usable, the
    # same asymmetry as plain write_mesh: a submesh validation failure (empty, duplicate name,
    # out-of-range, uncovered cell) happens inside the core crate's own self-consuming
    # write_mesh_with_submeshes, so retrying after one of those raises RuntimeError instead -- the
    # writer is genuinely gone, just as it is when write_mesh's own mesh validation rejects a call
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "submesh_retry"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError):
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS.astype(np.uint64),
            SQUARE_CONNECTIVITY,
            SQUARE_CELL_TYPES,
            [("all", [0, 1])],
        )
    writer.write_mesh_with_submeshes(
        SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("all", [0, 1])]
    )


def test_write_mesh_with_submeshes_twice_raises(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "submesh_twice"), xdmf.DataStorage.Ascii)
    writer.write_mesh_with_submeshes(
        SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("all", [0, 1])]
    )
    with pytest.raises(RuntimeError) as exc_info:
        writer.write_mesh_with_submeshes(
            SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES, [("all", [0, 1])]
        )
    assert str(exc_info.value) == "write_mesh was already called on this TimeSeriesWriter"


def test_a_numpy_scalar_is_not_described_as_an_array(tmp_path):
    # `arr[0]` yields one of these, and calling it "a numpy array with dtype float64" in a message
    # that expects exactly that dtype leaves the caller nothing to act on
    _file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        data_writer.write_time_step(
            "0.0", [("temperature", xdmf.DataAttribute.SCALAR, np.float64(1.0))]
        )
    assert str(exc_info.value).endswith("got numpy.float64")


def test_a_deflate_level_outside_the_range_is_rejected(tmp_path):
    # the range is the core crate's (`validate_deflate_level`), and is deliberately not restated
    # here: the level is handed over as passed, so the message naming 0-9 comes from Rust, when the
    # storage is used to build a writer. A level outside a u8 does not survive the argument
    # conversion to get there, and is pyo3's own OverflowError.
    for factory in (xdmf.DataStorage.hdf5_single_file, xdmf.DataStorage.hdf5_multiple_files):
        with pytest.raises(ValueError) as exc_info:
            xdmf.TimeSeriesWriter(str(tmp_path / "out"), factory(10))
        assert str(exc_info.value) == (
            "invalid configuration: deflate level 10 is out of range, must be between 0 and 9"
        )

        for level in (-1, 300):
            with pytest.raises(OverflowError):
                factory(level)


def test_storages_and_attributes_are_usable_as_dict_keys(tmp_path):
    # they are frozen value types; a Python caller naturally puts one in a dict or a set
    suffixes = {xdmf.DataStorage.Ascii: ".txt", xdmf.DataStorage.Hdf5SingleFile: ".h5"}
    assert suffixes[xdmf.DataStorage.Ascii] == ".txt"
    assert len({xdmf.DataAttribute.SCALAR, xdmf.DataAttribute.VECTOR, xdmf.DataAttribute.SCALAR}) == 2
    assert len({xdmf.CellType.Triangle, xdmf.CellType.Triangle}) == 1
    assert xdmf.DataStorage.hdf5_single_file(4) != xdmf.DataStorage.hdf5_single_file(5)


def test_classes_report_their_module(tmp_path):
    # without `module = "xdmf"` every repr and TypeError says "builtins.DataStorage"
    for cls in (xdmf.DataStorage, xdmf.CellType, xdmf.DataAttribute):
        assert cls.__module__ == "xdmf"


def test_write_mesh_twice_raises(tmp_path):
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "twice"), xdmf.DataStorage.Ascii)
    writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)
    with pytest.raises(RuntimeError) as exc_info:
        writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)
    assert str(exc_info.value) == "write_mesh was already called on this TimeSeriesWriter"


def test_invalid_mesh_raises_value_error(tmp_path):
    # empty points -> xdmf::Error::InvalidMesh, which must map to ValueError. The `reason` text is
    # core-crate content with its own tests there, so only the mapping is checked here.
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "empty"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError):
        writer.write_mesh(
            np.array([], dtype=np.float64), np.array([], dtype=np.uint32), []
        )


def test_time_step_without_data_raises_value_error(tmp_path):
    _file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        data_writer.write_time_step("0.0")
    assert "no data written" in str(exc_info.value)


def test_duplicated_time_raises_value_error(tmp_path):
    _file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Ascii)
    point_data = [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)]
    data_writer.write_time_step("0.1", point_data)
    with pytest.raises(ValueError) as exc_info:
        data_writer.write_time_step("0.10", point_data)
    assert "already written" in str(exc_info.value)


def test_data_of_the_wrong_length_raises_value_error(tmp_path):
    _file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        data_writer.write_time_step(
            "0.0", [("velocity", xdmf.DataAttribute.VECTOR, TEMPERATURE)]
        )
    assert "size of point_data 'velocity' must be 12, but is 4" in str(exc_info.value)


def test_rejected_attribute_discards_the_whole_step(tmp_path):
    # the step is all-or-nothing: the attribute written before the failing one leaves no <Grid>,
    # and the time stays available
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError):
        data_writer.write_time_step(
            "0.0",
            [
                ("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE),
                ("velocity", xdmf.DataAttribute.VECTOR, TEMPERATURE),
            ],
        )

    data_writer.write_time_step("0.0", [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)])
    xml = file_path.with_suffix(".xdmf2").read_text()
    assert xml.count('<Time Value="0.0"/>') == 1
    assert "velocity" not in xml


@pytest.mark.parametrize(
    "storage",
    [
        xdmf.DataStorage.Ascii,
        xdmf.DataStorage.AsciiInline,
        xdmf.DataStorage.Binary,
        pytest.param(xdmf.DataStorage.Hdf5SingleFile, marks=requires_hdf5),
    ],
)
def test_uint64_beyond_32_bits_raises_overflow_error(tmp_path, storage):
    # ParaView decodes UInt data at 32 bits whatever precision is declared, so this value cannot be
    # read back from any storage -- an OverflowError rather than a ValueError, since it is the one
    # failure a caller may want to catch to react (e.g. pass the data as int64 instead)
    _file_path, data_writer = write_points(tmp_path, storage)
    too_large = np.array([2**32], dtype=np.uint64)
    with pytest.raises(OverflowError) as exc_info:
        data_writer.write_time_step("0.0", [("big", xdmf.DataAttribute.SCALAR, too_large)])
    assert "no DataStorage avoids this" in str(exc_info.value)


def test_int64_beyond_the_ascii_range_raises_overflow_error(tmp_path):
    # the ascii storages are read back through a double, so an i64 past 2^53 would be shown rounded
    _file_path, data_writer = write_points(tmp_path, xdmf.DataStorage.Ascii)
    too_large = np.array([2**53 + 1], dtype=np.int64)
    with pytest.raises(OverflowError) as exc_info:
        data_writer.write_time_step("0.0", [("big", xdmf.DataAttribute.SCALAR, too_large)])
    assert "read back through a double" in str(exc_info.value)


@requires_hdf5
def test_int64_beyond_the_ascii_range_is_fine_in_hdf5(tmp_path):
    file_path, data_writer = write_points(tmp_path, xdmf.DataStorage.Hdf5SingleFile)
    large = np.array([2**53 + 1], dtype=np.int64)
    with data_writer:
        data_writer.write_time_step("0.0", [("big", xdmf.DataAttribute.SCALAR, large)])

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5")) as h5_file:
        np.testing.assert_array_equal(h5_file["data/t_0.0/0"][:], large)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_binary_rejects_64_bit_integers(tmp_path, dtype):
    # ParaView reads 64-bit integers in Format="Binary" at the wrong stride, so the type itself is
    # refused rather than narrowed behind the caller's back
    _file_path, data_writer = write_points(tmp_path, xdmf.DataStorage.Binary)
    with pytest.raises(ValueError) as exc_info:
        data_writer.write_time_step(
            "0.0", [("ids", xdmf.DataAttribute.SCALAR, np.array([1], dtype=dtype))]
        )
    assert "the Binary storage cannot hold" in str(exc_info.value)


@requires_hdf5
def test_close_releases_the_hdf5_file(tmp_path):
    file_path, data_writer = write_points(tmp_path, xdmf.DataStorage.Hdf5SingleFile)
    with data_writer:
        data_writer.write_time_step(
            "0", [("temperature", xdmf.DataAttribute.SCALAR, np.array([1.0]))]
        )

    # using the writer after __exit__ raises instead of silently doing nothing
    with pytest.raises(RuntimeError) as exc_info:
        data_writer.write_time_step(
            "1", [("temperature", xdmf.DataAttribute.SCALAR, np.array([2.0]))]
        )
    assert str(exc_info.value) == "this TimeSeriesDataWriter has already been closed"

    data_writer.close()  # idempotent

    h5py = pytest.importorskip("h5py")
    # if close() released the HDF5 file handle, another handle can open the file
    with h5py.File(file_path.with_suffix(".h5")) as h5_file:
        np.testing.assert_array_equal(h5_file["data/t_0/0"][:], [1.0])


@requires_hdf5
def test_write_mesh_result_is_usable_as_a_context_manager(tmp_path):
    file_path = tmp_path / "ctx"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Hdf5SingleFile)
    with writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES) as data_writer:
        data_writer.write_time_step(
            "0", [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)]
        )
    assert file_path.with_suffix(".xdmf2").exists()


def test_writes_run_concurrently_without_the_gil(tmp_path):
    # the writes release the GIL, so several threads can write their own file at the same time --
    # this is what the `Send + Sync` bound on the core crate's `DataWriter` is for
    def write(index):
        file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Ascii, f"thread_{index}")
        for step in range(5):
            data_writer.write_time_step(
                str(step),
                [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE + index)],
            )
        assert file_path.with_suffix(".xdmf2").exists()

    threads = [threading.Thread(target=write, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for index in range(4):
        xml = (tmp_path / f"thread_{index}").with_suffix(".xdmf2").read_text()
        assert xml.count("<Grid Name=") == 6  # the collection plus one grid per time step


def test_is_hdf5_enabled_matches_the_hdf5_storages_working(tmp_path):
    if xdmf.is_hdf5_enabled():
        write_points(tmp_path, xdmf.DataStorage.Hdf5SingleFile)
    else:
        with pytest.raises(ValueError) as exc_info:
            write_points(tmp_path, xdmf.DataStorage.Hdf5SingleFile)
        assert "requires the 'hdf5' feature" in str(exc_info.value)


def stub_names(body):
    """The names one stub body declares: classes, functions and annotated attributes."""
    names = set()
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_type_stubs_cover_the_module_surface():
    # a stub file is only useful while it is complete; this catches anything added to the module
    # without a matching entry in `xdmf.pyi`. Only that direction is checked -- the stub also holds
    # type aliases, which have nothing to correspond to at runtime.
    stub = ast.parse((Path(__file__).parent.parent / "xdmf.pyi").read_text())

    exported = {
        name
        for name, value in vars(xdmf).items()
        # the extension module itself is bound in the package namespace as a side effect of the
        # generated `__init__.py` importing from it, and is not part of the API
        if not name.startswith("_") and not inspect.ismodule(value)
    }
    assert exported - stub_names(stub.body) == set()

    for node in stub.body:
        if isinstance(node, ast.ClassDef):
            members = {
                name for name in dir(getattr(xdmf, node.name)) if not name.startswith("_")
            }
            assert members - stub_names(node.body) == set(), node.name


@pytest.mark.parametrize(
    ("given", "base"),
    [
        ("plain", "plain"),
        # an XDMF extension the caller spelled out is replaced rather than doubled
        ("spelled.xdmf2", "spelled"),
    ],
)
@requires_hdf5
def test_file_name_reports_what_is_written(tmp_path, given, base):
    # a pathlib.Path both ways: `os.fspath` on the way in, so a plain str works too (every other
    # test here passes one)
    writer = xdmf.TimeSeriesWriter(tmp_path / given, xdmf.DataStorage.Hdf5SingleFile)
    assert writer.file_name == tmp_path / f"{base}.xdmf2"

    data_writer = writer.write_mesh(SQUARE_COORDS, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)
    assert data_writer.file_name == writer.file_name

    # the name survives close(), where a caller tends to want it for a log line
    data_writer.close()
    assert data_writer.file_name == writer.file_name
    assert (tmp_path / f"{base}.xdmf2").exists()
    assert (tmp_path / f"{base}.h5").exists()
