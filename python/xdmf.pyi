"""Type stubs for the `xdmf` extension module (pyo3/maturin), hand-written since the API surface
is small. Keep in sync with `src/{enums,writer,reader}.rs`.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

CellTypeCodes = Sequence["CellType"] | npt.NDArray[np.uint8] | npt.NDArray[np.uint64] | npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64] | npt.NDArray[np.float32]
IntArray = (
    npt.NDArray[np.uint64] | npt.NDArray[np.uint32] | npt.NDArray[np.int64] | npt.NDArray[np.int32]
)
ValueArray = FloatArray | IntArray

class DataStorage:
    """Heavy-data storage format."""

    Ascii: "DataStorage"
    AsciiInline: "DataStorage"
    Hdf5SingleFile: "DataStorage"
    Hdf5MultipleFiles: "DataStorage"
    Binary: "DataStorage"

    @staticmethod
    def hdf5_single_file(deflate_level: int) -> "DataStorage": ...
    @staticmethod
    def hdf5_multiple_files(deflate_level: int) -> "DataStorage": ...

class CellType:
    """Cell types as defined in the VTK file format."""

    Vertex: "CellType"
    Edge: "CellType"
    Triangle: "CellType"
    Quadrilateral: "CellType"
    Tetrahedron: "CellType"
    Pyramid: "CellType"
    Wedge: "CellType"
    Hexahedron: "CellType"
    Edge3: "CellType"
    Quadrilateral9: "CellType"
    Triangle6: "CellType"
    Quadrilateral8: "CellType"
    Tetrahedron10: "CellType"
    Pyramid13: "CellType"
    Wedge15: "CellType"
    Wedge18: "CellType"
    Hexahedron20: "CellType"
    Hexahedron24: "CellType"
    Hexahedron27: "CellType"

class DataAttribute:
    """Type of the data (scalar, vector, tensor, etc.)."""

    SCALAR: "DataAttribute"
    VECTOR: "DataAttribute"
    TENSOR: "DataAttribute"
    TENSOR6: "DataAttribute"

    @staticmethod
    def matrix(rows: int, cols: int) -> "DataAttribute": ...
    @staticmethod
    def generic(size: int) -> "DataAttribute": ...

class DataInfo:
    """Metadata about one point/cell data field, returned by `point_data_info`/`cell_data_info`."""

    name: str
    attribute: DataAttribute
    dtype: str
    len: int

class TimeSeriesDataWriter:
    """Writer for per-step point/cell attribute data, obtained from `TimeSeriesWriter.write_mesh`.

    Supports the context manager protocol; `close()`/`__exit__` release any open file handles
    (relevant for the HDF5 backends, whose file otherwise stays open until this object is
    garbage-collected).
    """

    def write_data(
        self,
        time: str,
        point_data: Sequence[tuple[str, DataAttribute, ValueArray]],
        cell_data: Sequence[tuple[str, DataAttribute, ValueArray]],
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "TimeSeriesDataWriter": ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...

class TimeSeriesWriter:
    """Writer for time series data in XDMF format."""

    def __init__(self, file_name: str, data_storage: DataStorage) -> None: ...
    def write_mesh(
        self,
        points: FloatArray,
        connectivity: IntArray,
        cell_types: CellTypeCodes,
    ) -> TimeSeriesDataWriter: ...

class TimeSeriesDataReader:
    """Reader for per-step point/cell attribute data, obtained from `TimeSeriesReader.read_mesh`."""

    def num_steps(self) -> int: ...
    def times(self) -> list[str]: ...
    def num_point_data(self, step: int) -> int: ...
    def num_cell_data(self, step: int) -> int: ...
    def point_data_info(self, step: int, index: int) -> DataInfo: ...
    def cell_data_info(self, step: int, index: int) -> DataInfo: ...
    def point_data_index(self, step: int, name: str) -> int: ...
    def cell_data_index(self, step: int, name: str) -> int: ...
    def read_point_step(self, step: int) -> list[tuple[str, DataAttribute, ValueArray]]: ...
    def read_cell_step(self, step: int) -> list[tuple[str, DataAttribute, ValueArray]]: ...
    def read_point_data(self, step: int, index: int) -> ValueArray: ...
    def read_cell_data(self, step: int, index: int) -> ValueArray: ...

class TimeSeriesReader:
    """Parses an `.xdmf2` file's light data (XML metadata)."""

    def __init__(self, file_name: str) -> None: ...
    def num_points(self) -> int: ...
    def num_cells(self) -> int: ...
    def times(self) -> list[str]: ...
    def read_mesh(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.uint64],
        list[CellType],
        TimeSeriesDataReader,
    ]: ...
