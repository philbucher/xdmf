"""Type stubs for the `xdmf` extension module (pyo3/maturin), hand-written since the API surface
is small. Keep in sync with `src/{enums,writer}.rs`.

These bindings expose the crate's writing interface only; its `TimeSeriesReader` is deliberately
not bound, so reading an XDMF file back is Rust-only for now.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

CellTypes = Sequence["CellType"] | npt.NDArray[np.integer]
PointArray = npt.NDArray[np.float64] | npt.NDArray[np.float32]
IndexArray = (
    npt.NDArray[np.uint64] | npt.NDArray[np.uint32] | npt.NDArray[np.int64] | npt.NDArray[np.int32]
)
ValueArray = PointArray | IndexArray
NamedData = Sequence[tuple[str, "DataAttribute", ValueArray]]
# a `range` of step 1 is passed straight through as the block it names, without its indices being
# built; any other range reads as the plain sequence of ints it also is
SubmeshCells = range | Sequence[int] | npt.NDArray[np.integer]
NamedSubmesh = Sequence[tuple[str, SubmeshCells]]

def is_hdf5_enabled() -> bool:
    """Whether this build can write the HDF5 storages."""

class DataStorage:
    """Heavy-data storage format."""

    Ascii: "DataStorage"
    AsciiInline: "DataStorage"
    Hdf5SingleFile: "DataStorage"
    Hdf5MultipleFiles: "DataStorage"
    Binary: "DataStorage"

    @staticmethod
    def hdf5_single_file(deflate_level: int) -> "DataStorage":
        """HDF5, all data in a single file, at the given deflate level.

        Raises `ValueError` if the level is outside 0-9.
        """

    @staticmethod
    def hdf5_multiple_files(deflate_level: int) -> "DataStorage":
        """HDF5, one file per time step, at the given deflate level.

        Raises `ValueError` if the level is outside 0-9.
        """

class CellType:
    """Cell types, mirroring the XDMF topology types.

    The values are the raw XDMF topology type codes, *not* the VTK cell codes -- a hexahedron is
    9 here and 12 in VTK. What follows VTK is the node ordering within a cell.
    """

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
    def matrix(rows: int, cols: int) -> "DataAttribute":
        """Matrix with the given number of rows and columns."""

    @staticmethod
    def generic(size: int) -> "DataAttribute":
        """Generic data with the given size."""

class TimeSeriesDataWriter:
    """Writer for the per-step data, obtained from `TimeSeriesWriter.write_mesh`.

    Supports the context manager protocol; `close()`/`__exit__` release any open file handles
    (relevant for the HDF5 backends, whose file otherwise stays open until this object is
    garbage-collected).
    """

    def write_time_step(
        self,
        time: str,
        point_data: NamedData | None = None,
        cell_data: NamedData | None = None,
    ) -> None:
        """Write the point and cell data of one time step.

        Raises `ValueError` for a time or an attribute the mesh cannot take (a duplicated time, a
        wrong array length, a step with no data at all), `OverflowError` for an integer the chosen
        storage cannot represent, and `OSError` for a failing write.

        The arrays are borrowed rather than copied and the write releases the GIL, so another thread
        must not modify an array while a write of it is running.
        """

    @property
    def file_name(self) -> Path:
        """The XDMF file this writer writes"""

    def close(self) -> None: ...
    def __enter__(self) -> "TimeSeriesDataWriter": ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...

class TimeSeriesWriter:
    """Writer for time series data in XDMF format."""

    def __init__(self, file_name: str | os.PathLike[str], data_storage: DataStorage) -> None:
        """Every file of the series takes its name from `file_name`, swapping the extension it
        carries for the one that file needs. See `file_name`.
        """

    @property
    def file_name(self) -> Path:
        """The XDMF file this writer writes: the name it was given with the `.xdmf2` extension.

        The heavy data takes the same base and its own storage's extension.
        """

    def write_mesh(
        self,
        points: PointArray,
        connectivity: IndexArray,
        cell_types: CellTypes,
    ) -> TimeSeriesDataWriter:
        """Write the mesh, returning the writer for the time step data.

        `points` is flat x/y/z coordinates or shaped `(..., 3)`; a last dimension that is not 3 is
        rejected, so a `(3, N)` array of separate x/y/z rows raises `ValueError` instead of being
        read as interleaved coordinates.

        Consumes this writer; calling it a second time raises `RuntimeError`. A *rejected* call
        leaves the writer usable, so a dtype or shape that can be fixed can simply be retried.
        """

    def write_mesh_with_submeshes(
        self,
        points: PointArray,
        connectivity: IndexArray,
        cell_types: CellTypes,
        submeshes: NamedSubmesh,
    ) -> TimeSeriesDataWriter:
        """Write the mesh split into named submeshes, returning the writer for the time step data.

        `submeshes` is a sequence of `(name, cells)` pairs, `cells` being a `range`, a numpy integer
        array or a sequence of `int` naming which cells (indices into `cell_types`) belong to that
        submesh. A submesh that is one block of consecutive cells is best given as
        `range(start, stop)`, which is taken as the two numbers it is stored as without its indices
        being built at all. Each submesh becomes its own selectable block in ParaView's Multi-block
        Inspector (`View -> Multi-block Inspector`); every cell must belong to at least one
        submesh, but submeshes may overlap. Each submesh is written with the points its own cells
        use, so a viewer holds the mesh about once however many submeshes it is split into. Point
        and cell data are still written over the whole mesh in
        `TimeSeriesDataWriter.write_time_step`, exactly as for `write_mesh`.

        Raises `ValueError` for a bad submesh (empty, a duplicate or out-of-range cell index, a name
        used twice, or a cell in no submesh at all).

        Consumes this writer; calling it a second time (or after `write_mesh`) raises `RuntimeError`.
        A call rejected over a dtype, a shape or a cell type leaves the writer usable, but the
        submesh and mesh `ValueError`s above are raised from inside the consuming call, so retrying
        after one of those raises `RuntimeError` instead.
        """
