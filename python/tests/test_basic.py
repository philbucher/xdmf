"""End-to-end checks for the xdmf python bindings: write a small mesh + a couple of time steps
of point/cell data using numpy arrays, and verify the produced files.
"""

import struct
from pathlib import Path

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


def test_write_mesh_and_data_ascii(tmp_path):
    file_path, data_writer = _write_small_mesh(tmp_path, xdmf.DataStorage.Ascii)

    temperature = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    data_writer.write_data(
        "0.0", [("temperature", xdmf.DataAttribute.SCALAR, temperature)], []
    )

    xdmf_file = file_path.with_suffix(".xdmf2")
    assert xdmf_file.exists()
    assert 'Format="XML"' in xdmf_file.read_text()


def test_connectivity_accepts_int64(tmp_path):
    # numpy's default integer dtype is signed int64 -- must work without a copy/cast.
    _write_small_mesh(tmp_path, xdmf.DataStorage.Binary, connectivity_dtype=np.int64)


def test_connectivity_rejects_negative_int64(tmp_path):
    coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    connectivity = np.array([0, 1, -1], dtype=np.int64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "neg"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError, match="negative"):
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Triangle])


def test_rejects_wrong_dtype(tmp_path):
    coords = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # wrong dtype: float32
    connectivity = np.array([0], dtype=np.uint64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "wrong_dtype"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError, match="float64"):
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex])


def test_rejects_non_contiguous_array(tmp_path):
    # 3 points (9 floats) but strided (stride-2 view), so genuinely non-contiguous.
    non_contig = np.arange(18, dtype=np.float64)[::2]
    assert not non_contig.flags["C_CONTIGUOUS"]
    connectivity = np.array([0], dtype=np.uint64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "non_contig"), xdmf.DataStorage.Ascii)
    with pytest.raises(ValueError, match="contiguous"):
        writer.write_mesh(non_contig, connectivity, [xdmf.CellType.Vertex])


def test_write_mesh_twice_raises(tmp_path):
    coords = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    connectivity = np.array([0], dtype=np.uint64)
    writer = xdmf.TimeSeriesWriter(str(tmp_path / "twice"), xdmf.DataStorage.Ascii)
    writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex])
    with pytest.raises(RuntimeError, match="already called"):
        writer.write_mesh(coords, connectivity, [xdmf.CellType.Vertex])


def test_write_mesh_with_blocks(tmp_path):
    coords = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    connectivity = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint64)
    cell_types = [xdmf.CellType.Triangle, xdmf.CellType.Triangle]

    file_path = tmp_path / "blocks"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Ascii)
    writer.write_mesh_with_blocks(
        coords, connectivity, cell_types, [("a", [0]), ("b", [1])]
    )

    xdmf_file = file_path.with_suffix(".xdmf2")
    xml = xdmf_file.read_text()
    assert 'Name="a"' in xml
    assert 'Name="b"' in xml
