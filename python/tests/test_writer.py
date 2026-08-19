"""End-to-end checks for the xdmf python bindings: write a small mesh and a couple of time steps
of point/cell data from numpy arrays, and verify what lands in the produced files.

The dtype-specific cases exist because nothing in this path casts: an array is stored as the type
it is passed in, and a dtype a storage cannot carry back is rejected instead (see `src/paraview.rs`
in the core crate).
"""

import ast
import inspect
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

    temperature_bytes = (bin_dir / "data_t_0_point_data_temperature.bin").read_bytes()
    assert struct.unpack("<4d", temperature_bytes) == tuple(TEMPERATURE)

    region_bytes = (bin_dir / "data_t_0_cell_data_region_id.bin").read_bytes()
    assert struct.unpack("<2I", region_bytes) == tuple(REGION_ID)


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
    assert "test_output.h5:data/t_0/point_data/temperature" in xml
    assert "test_output.h5:data/t_0/cell_data/region_id" in xml

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5")) as h5_file:
        np.testing.assert_array_equal(h5_file["data/t_0/point_data/temperature"][:], TEMPERATURE)
        np.testing.assert_array_equal(h5_file["data/t_0/cell_data/region_id"][:], REGION_ID)


def test_write_mesh_and_data_hdf5_multiple_files(tmp_path):
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.Hdf5MultipleFiles)

    with data_writer:
        data_writer.write_time_step(
            "0", [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)]
        )

    assert 'Format="HDF"' in file_path.with_suffix(".xdmf2").read_text()

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5") / "data_t_0.h5") as h5_file:
        np.testing.assert_array_equal(h5_file["point_data/temperature"][:], TEMPERATURE)


def test_hdf5_custom_deflate_level(tmp_path):
    file_path, data_writer = write_square(tmp_path, xdmf.DataStorage.hdf5_single_file(3))

    with data_writer:
        data_writer.write_time_step(
            "0", [("temperature", xdmf.DataAttribute.SCALAR, TEMPERATURE)]
        )

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5")) as h5_file:
        dataset = h5_file["data/t_0/point_data/temperature"]
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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint64, np.int64])
def test_cell_types_as_numpy_codes(tmp_path, dtype):
    # the CellType values are the raw VTK codes (Triangle == 4), so an array of codes is an
    # equivalent, cheaper-to-produce alternative to a list of CellType
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
        "uint8, uint64, or int64"
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

    writer = xdmf.TimeSeriesWriter(str(tmp_path / "non_contiguous"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(non_contiguous, SQUARE_CONNECTIVITY, SQUARE_CELL_TYPES)
    assert str(exc_info.value) == (
        "array must be C-contiguous; call `numpy.ascontiguousarray()` on it first"
    )


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
        xdmf.DataStorage.Hdf5SingleFile,
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


def test_int64_beyond_the_ascii_range_is_fine_in_hdf5(tmp_path):
    file_path, data_writer = write_points(tmp_path, xdmf.DataStorage.Hdf5SingleFile)
    large = np.array([2**53 + 1], dtype=np.int64)
    with data_writer:
        data_writer.write_time_step("0.0", [("big", xdmf.DataAttribute.SCALAR, large)])

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5")) as h5_file:
        np.testing.assert_array_equal(h5_file["data/t_0.0/point_data/big"][:], large)


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
        np.testing.assert_array_equal(h5_file["data/t_0/point_data/temperature"][:], [1.0])


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
