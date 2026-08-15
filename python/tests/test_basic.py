"""End-to-end checks for the xdmf python bindings: write a small mesh + a couple of time steps
of point/cell data using numpy arrays, and verify the produced files (and, for the reader, that
they round-trip back through the bindings).
"""

import struct

import numpy as np
import pytest

import xdmf


def _write_small_mesh(tmp_path, data_storage, connectivity_dtype=np.uint64):
    coords = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    connectivity = np.array([0, 1, 2, 0, 2, 3], dtype=connectivity_dtype)
    cell_types = [xdmf.CellType.Triangle, xdmf.CellType.Triangle]

    file_path = tmp_path / "test_output"
    writer = xdmf.TimeSeriesWriter(str(file_path), data_storage)
    data_writer = writer.write_mesh(coords, connectivity, cell_types)
    return file_path, data_writer


def test_write_mesh_and_data_binary(tmp_path):
    file_path, data_writer = _write_small_mesh(tmp_path, xdmf.DataStorage.Binary)

    temperature = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    region_id = np.array([100, 200], dtype=np.uint64)

    data_writer.write_data(
        "0",
        [("temperature", xdmf.DataAttribute.SCALAR, temperature)],
        [("region_id", xdmf.DataAttribute.SCALAR, region_id)],
    )

    xdmf_file = file_path.with_suffix(".xdmf2")
    xml = xdmf_file.read_text()
    assert 'Format="Binary"' in xml
    assert 'Endian="Little"' in xml
    # UInt data narrowed to 4 bytes (Paraview's legacy Xdmf2 reader can't read 8-byte ints)
    assert 'NumberType="UInt" Format="Binary" Precision="4"' in xml
    assert 'NumberType="Float" Format="Binary" Precision="8"' in xml

    bin_dir = file_path.with_suffix(".bin")
    points_bytes = (bin_dir / "points.bin").read_bytes()
    assert struct.unpack("<12d", points_bytes) == (
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    )

    temp_bytes = (bin_dir / "data_t_0_point_data_temperature.bin").read_bytes()
    assert struct.unpack("<4d", temp_bytes) == (10.0, 11.0, 12.0, 13.0)

    region_bytes = (bin_dir / "data_t_0_cell_data_region_id.bin").read_bytes()
    assert struct.unpack("<2I", region_bytes) == (100, 200)


def test_write_mesh_and_data_hdf5_single_file(tmp_path):
    file_path, data_writer = _write_small_mesh(tmp_path, xdmf.DataStorage.Hdf5SingleFile)

    temperature = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    region_id = np.array([100, 200], dtype=np.uint64)

    data_writer.write_data(
        "0",
        [("temperature", xdmf.DataAttribute.SCALAR, temperature)],
        [("region_id", xdmf.DataAttribute.SCALAR, region_id)],
    )
    data_writer.close()

    xdmf_file = file_path.with_suffix(".xdmf2")
    xml = xdmf_file.read_text()
    assert 'Format="HDF"' in xml
    assert "test_output.h5:data/t_0/point_data/temperature" in xml
    assert "test_output.h5:data/t_0/cell_data/region_id" in xml

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5")) as f:
        np.testing.assert_array_equal(f["data/t_0/point_data/temperature"][:], temperature)
        np.testing.assert_array_equal(f["data/t_0/cell_data/region_id"][:], region_id)


def test_write_mesh_and_data_hdf5_multiple_files(tmp_path):
    file_path, data_writer = _write_small_mesh(tmp_path, xdmf.DataStorage.Hdf5MultipleFiles)

    temperature = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    data_writer.write_data("0", [("temperature", xdmf.DataAttribute.SCALAR, temperature)], [])
    data_writer.close()

    xdmf_file = file_path.with_suffix(".xdmf2")
    xml = xdmf_file.read_text()
    assert 'Format="HDF"' in xml

    h5py = pytest.importorskip("h5py")
    h5_dir = file_path.with_suffix(".h5")
    with h5py.File(h5_dir / "data_t_0.h5") as f:
        np.testing.assert_array_equal(f["point_data/temperature"][:], temperature)


def test_write_mesh_and_data_hdf5_custom_deflate_level(tmp_path):
    file_path, data_writer = _write_small_mesh(tmp_path, xdmf.DataStorage.hdf5_single_file(3))

    temperature = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    data_writer.write_data("0", [("temperature", xdmf.DataAttribute.SCALAR, temperature)], [])
    data_writer.close()

    xdmf_file = file_path.with_suffix(".xdmf2")
    assert "deflate_level: Some(3)" in xdmf_file.read_text()

    h5py = pytest.importorskip("h5py")
    with h5py.File(file_path.with_suffix(".h5")) as f:
        dataset = f["data/t_0/point_data/temperature"]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 3
        np.testing.assert_array_equal(dataset[:], temperature)


def test_write_mesh_and_data_ascii(tmp_path):
    file_path, data_writer = _write_small_mesh(tmp_path, xdmf.DataStorage.Ascii)

    temperature = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    data_writer.write_data("0.0", [("temperature", xdmf.DataAttribute.SCALAR, temperature)], [])

    xdmf_file = file_path.with_suffix(".xdmf2")
    assert xdmf_file.exists()
    assert 'Format="XML"' in xdmf_file.read_text()


def test_write_data_accepts_float32(tmp_path):
    file_path, data_writer = _write_small_mesh(tmp_path, xdmf.DataStorage.Ascii)

    temperature = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
    data_writer.write_data("0.0", [("temperature", xdmf.DataAttribute.SCALAR, temperature)], [])

    xml = file_path.with_suffix(".xdmf2").read_text()
    assert 'NumberType="Float" Format="XML" Precision="4"' in xml


def test_write_mesh_accepts_2d_point_and_vector_shapes(tmp_path):
    # (N, 3) points/vectors are a natural numpy layout and, being C-contiguous, have exactly the
    # flat memory layout the underlying Rust API wants -- no `reshape(-1)` needed.
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    connectivity = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint64)
    cell_types = [xdmf.CellType.Triangle, xdmf.CellType.Triangle]

    file_path = tmp_path / "shapes_2d"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    data_writer = writer.write_mesh(coords, connectivity, cell_types)

    velocity = np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float64)
    data_writer.write_data("0.0", [("velocity", xdmf.DataAttribute.VECTOR, velocity)], [])

    assert file_path.with_suffix(".xdmf2").exists()


def test_write_mesh_accepts_cell_types_as_numpy_array(tmp_path):
    coords = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    connectivity = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint64)
    # raw VTK cell type codes: Triangle == 4
    cell_types = np.array([4, 4], dtype=np.uint8)

    file_path = tmp_path / "codes"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    writer.write_mesh(coords, connectivity, cell_types)
    assert file_path.with_suffix(".xdmf2").exists()


def test_connectivity_accepts_int64(tmp_path):
    # numpy's default integer dtype is signed int64 -- must work without a copy/cast.
    _write_small_mesh(tmp_path, xdmf.DataStorage.Binary, connectivity_dtype=np.int64)


def test_connectivity_accepts_uint32(tmp_path):
    # a 32-bit connectivity array is stored at that width instead of being widened to 64-bit.
    _write_small_mesh(tmp_path, xdmf.DataStorage.Ascii, connectivity_dtype=np.uint32)


def test_connectivity_accepts_int32(tmp_path):
    _write_small_mesh(tmp_path, xdmf.DataStorage.Ascii, connectivity_dtype=np.int32)


def test_connectivity_rejects_negative_int64(tmp_path):
    coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    connectivity = np.array([0, 1, -1], dtype=np.int64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "neg"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Triangle])
    assert str(exc_info.value) == (
        "invalid mesh: value -1 is negative, but indices/counts must be non-negative"
    )


def test_connectivity_rejects_negative_int32(tmp_path):
    coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    connectivity = np.array([0, 1, -1], dtype=np.int32)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "neg32"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Triangle])
    assert str(exc_info.value) == (
        "invalid mesh: value -1 is negative, but indices/counts must be non-negative"
    )


def test_rejects_wrong_dtype(tmp_path):
    coords = np.array([0.0, 0.0, 0.0], dtype=np.int16)  # not one of the six recognized dtypes
    connectivity = np.array([0], dtype=np.uint64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "wrong_dtype"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex])
    assert str(exc_info.value) == (
        "expected a numpy array with dtype float64, float32, uint64, uint32, int64, or int32, "
        "got a numpy array with dtype int16"
    )


def test_rejects_integer_points(tmp_path):
    # uint64 is a recognized dtype (valid for connectivity/attribute data), just not for points;
    # that rejection happens in the core crate (xdmf::Error::InvalidMesh), not the binding, so only
    # the type mapping is checked here -- see test_invalid_mesh_raises_value_error.
    coords = np.array([0, 0, 0], dtype=np.uint64)
    connectivity = np.array([0], dtype=np.uint64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "int_points"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError):
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex])


def test_write_mesh_accepts_float32_points(tmp_path):
    # float32 points are stored at that precision on disk instead of being widened to float64.
    coords = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        dtype=np.float32,
    )
    connectivity = np.array([0, 1, 2], dtype=np.uint64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "f32_points"), xdmf.DataStorage.AsciiInline)
    writer.write_mesh(coords, connectivity, [xdmf.CellType.Triangle])

    xdmf_file = (tmp_path / "f32_points").with_suffix(".xdmf2")
    xml = xdmf_file.read_text()
    assert 'NumberType="Float" Format="XML" Precision="4"' in xml


def test_write_mesh_accepts_uint32_connectivity(tmp_path):
    # a uint32 connectivity array is stored at that precision on disk instead of being widened to
    # uint64 -- and signed int32 connectivity is normalized to unsigned on disk (indices have no
    # sign), so it round-trips through the same Precision="4" NumberType="UInt" representation.
    coords = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        dtype=np.float64,
    )
    for connectivity_dtype in (np.uint32, np.int32):
        connectivity = np.array([0, 1, 2], dtype=connectivity_dtype)
        file_path = tmp_path / f"u32_conn_{connectivity_dtype.__name__}"
        writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.AsciiInline)
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Triangle])

        xml = file_path.with_suffix(".xdmf2").read_text()
        assert 'NumberType="UInt" Format="XML" Precision="4"' in xml


def test_rejects_non_contiguous_array(tmp_path):
    # 3 points (9 floats) but strided (stride-2 view), so genuinely non-contiguous.
    non_contig = np.arange(18, dtype=np.float64)[::2]
    assert not non_contig.flags["C_CONTIGUOUS"]
    connectivity = np.array([0], dtype=np.uint64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "non_contig"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError) as exc_info:
        writer.write_mesh(non_contig, connectivity, [xdmf.CellType.Vertex])
    assert str(exc_info.value) == (
        "array must be C-contiguous; call `numpy.ascontiguousarray()` on it first"
    )


def test_write_mesh_twice_raises(tmp_path):
    coords = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    connectivity = np.array([0], dtype=np.uint64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "twice"), xdmf.DataStorage.Ascii)
    writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex])
    with pytest.raises(RuntimeError) as exc_info:
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex])
    assert str(exc_info.value) == "write_mesh was already called on this TimeSeriesWriter"


def test_invalid_mesh_raises_value_error(tmp_path):
    # empty points -> xdmf::Error::InvalidMesh, which must map to ValueError, not some other type.
    # The `reason` text itself is core-crate content, covered by its own error_messages tests, so
    # only the type mapping is checked here.
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "empty"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError):
        writer.write_mesh(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.uint64),
            [],
        )


def test_data_writer_context_manager_closes_hdf5_file(tmp_path):
    file_path = tmp_path / "ctx"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Hdf5SingleFile)
    coords = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    connectivity = np.array([0], dtype=np.uint64)

    with writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex]) as data_writer:
        temperature = np.array([1.0], dtype=np.float64)
        data_writer.write_data("0", [("temperature", xdmf.DataAttribute.SCALAR, temperature)], [])

    # after __exit__ the writer is closed; using it again raises rather than silently no-op'ing
    with pytest.raises(RuntimeError) as exc_info:
        data_writer.write_data("1", [], [])
    assert str(exc_info.value) == "this TimeSeriesDataWriter has already been closed"

    h5py = pytest.importorskip("h5py")
    # if close() actually released the HDF5 file handle, a second process/handle can open it
    with h5py.File(file_path.with_suffix(".h5")) as f:
        np.testing.assert_array_equal(f["data/t_0/point_data/temperature"][:], [1.0])


@pytest.mark.parametrize(
    "storage",
    [
        xdmf.DataStorage.Ascii,
        xdmf.DataStorage.AsciiInline,
        xdmf.DataStorage.Binary,
        xdmf.DataStorage.Hdf5SingleFile,
        xdmf.DataStorage.Hdf5MultipleFiles,
    ],
)
def test_reader_round_trips_writer_output(tmp_path, storage):
    coords = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0],
        dtype=np.float64,
    )
    connectivity = np.array([0, 1, 2, 3, 0, 1, 2, 4], dtype=np.uint64)
    cell_types = [xdmf.CellType.Quadrilateral, xdmf.CellType.Tetrahedron]

    file_path = tmp_path / "roundtrip"
    writer = xdmf.TimeSeriesWriter(str(file_path), storage)
    with writer.write_mesh(coords, connectivity, cell_types) as data_writer:
        temperature = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        cell_id = np.array([10, 20], dtype=np.uint64)
        data_writer.write_data(
            "0.0",
            [("temperature", xdmf.DataAttribute.SCALAR, temperature)],
            [("cell_id", xdmf.DataAttribute.SCALAR, cell_id)],
        )

    reader = xdmf.TimeSeriesReader(str(file_path.with_suffix(".xdmf2")))
    assert reader.num_points() == 5
    assert reader.num_cells() == 2

    read_points, read_connectivity, read_cell_types, data_reader = reader.read_mesh()
    np.testing.assert_allclose(read_points, coords)
    np.testing.assert_array_equal(read_connectivity, connectivity)
    assert read_cell_types == cell_types

    assert data_reader.num_steps() == 1
    assert data_reader.times() == ["0.0"]
    assert data_reader.num_point_data(0) == 1
    assert data_reader.num_cell_data(0) == 1

    info = data_reader.point_data_info(0, 0)
    assert info.name == "temperature"
    assert info.dtype == "float64"
    assert info.len == 5

    read_temperature = data_reader.read_point_data(0, 0)
    np.testing.assert_allclose(read_temperature, temperature)
    assert read_temperature.dtype == np.float64

    read_cell_id = data_reader.read_cell_data(0, 0)
    np.testing.assert_array_equal(read_cell_id, cell_id)

    [(name, attr, array)] = data_reader.read_point_step(0)
    assert name == "temperature"
    assert attr == xdmf.DataAttribute.SCALAR
    np.testing.assert_allclose(array, temperature)

    [(name, _attr, array)] = data_reader.read_cell_step(0)
    assert name == "cell_id"
    np.testing.assert_array_equal(array, cell_id)


def test_reader_f32_widens_into_f64_but_not_the_reverse(tmp_path):
    coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    connectivity = np.array([0, 1, 2], dtype=np.uint64)

    file_path = tmp_path / "f32"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    data_writer = writer.write_mesh(coords, connectivity, [xdmf.CellType.Triangle])
    pressure = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data_writer.write_data("0.0", [("pressure", xdmf.DataAttribute.SCALAR, pressure)], [])

    reader = xdmf.TimeSeriesReader(str(file_path.with_suffix(".xdmf2")))
    _points, _connectivity, _cell_types, data_reader = reader.read_mesh()

    info = data_reader.point_data_info(0, 0)
    assert info.dtype == "float32"

    read_back = data_reader.read_point_data(0, 0)
    assert read_back.dtype == np.float32
    np.testing.assert_allclose(read_back, pressure)


def test_unsupported_feature_raises_not_implemented_error(tmp_path):
    # This crate's own writer never emits NumberType="Int" (only "Float"/"UInt"), so simulate a
    # foreign file using it by patching a real, valid file this crate wrote -- that is simpler and
    # less brittle than hand-writing a full light-data document from scratch.
    import re

    coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    connectivity = np.array([0, 1, 2], dtype=np.uint64)
    file_path = tmp_path / "unsupported"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    data_writer = writer.write_mesh(coords, connectivity, [xdmf.CellType.Triangle])
    region_id = np.array([7], dtype=np.uint64)
    data_writer.write_data("0.0", [], [("region_id", xdmf.DataAttribute.SCALAR, region_id)])

    xdmf_file = file_path.with_suffix(".xdmf2")
    xml = xdmf_file.read_text()
    attr_block = re.search(r'<Attribute Name="region_id".*?</Attribute>', xml, re.DOTALL).group(0)
    patched_block = attr_block.replace('NumberType="UInt"', 'NumberType="Int"')
    assert patched_block != attr_block
    xdmf_file.write_text(xml.replace(attr_block, patched_block))

    reader = xdmf.TimeSeriesReader(str(xdmf_file))
    _points, _connectivity, _cell_types, data_reader = reader.read_mesh()
    # the `reason` text is core-crate content (xdmf::Error::Unsupported), covered by its own
    # error_messages tests, so only the type mapping is checked here.
    with pytest.raises(NotImplementedError):
        data_reader.cell_data_info(0, 0)
