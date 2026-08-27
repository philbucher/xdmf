# xdmf - python interface

Python bindings for the [xdmf](https://github.com/philbucher/xdmf) crate, built with
[pyo3](https://pyo3.rs)/[maturin](https://www.maturin.rs): write meshes with time-series data as
XDMF files, for ParaView or VisIt to read.

No wheels on PyPI yet, so build from the repository:

~~~sh
pip install ./python
~~~

## Example

Pass the mesh and the data as numpy arrays. The bindings borrow them rather than copy them, and
write each one at the dtype you pass (`float64`, `float32`, `uint64`, `uint32`, `int64`, `int32`),
as in Rust:

~~~py
import numpy as np
import xdmf

writer = xdmf.TimeSeriesWriter("xdmf_writing", xdmf.DataStorage.Hdf5SingleFile)

# define 3 points and 2 cells (a line and a triangle)
coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
connectivity = np.array([0, 1, 0, 2, 1], dtype=np.uint32)  # line (0,1) and triangle (0,2,1)
cell_types = [xdmf.CellType.Edge, xdmf.CellType.Triangle]

# write the mesh, then the data for 10 time steps
with writer.write_mesh(coords, connectivity, cell_types) as data_writer:
    for step in range(10):
        point_values = np.full(9, float(step))
        data_writer.write_time_step(
            str(step),
            point_data=[("point_data", xdmf.DataAttribute.VECTOR, point_values)],
            cell_data=[("cell_data", xdmf.DataAttribute.SCALAR, np.array([0.0, 1.0]))],
        )
~~~

## Submeshes

`write_mesh_with_submeshes` splits the mesh into named blocks, which ParaView lists in its
Multi-block Inspector (`View -> Multi-block Inspector`) and shows or hides one at a time. Each
`(name, cells)` pair names the cells belonging to one block, as a `range`, a numpy integer array or
a sequence of `int`:

~~~py
writer = xdmf.TimeSeriesWriter("xdmf_blocks", xdmf.DataStorage.Hdf5SingleFile)

with writer.write_mesh_with_submeshes(
    coords, connectivity, cell_types, [("line", range(0, 1)), ("triangle", [1])]
) as data_writer:
    # the data still covers the whole mesh, and each block gets its share
    data_writer.write_time_step(
        "0.0",
        point_data=[("point_data", xdmf.DataAttribute.VECTOR, np.zeros(9))],
        cell_data=[("material", xdmf.DataAttribute.SCALAR, np.array([10.0, 20.0]))],
    )
~~~

Give a block of consecutive cells as `range(start, stop)` to save storage space.

Blocks may overlap, every cell has to belong to at least one, and no two may share a name.

`tests/test_writer.py` has many more examples, including every failure case.

## Good to know

- Points and vector data may also be shaped `(N, 3)`: a C-contiguous `(N, 3)` array is the same
  memory as the flat one, so it needs no `reshape`. Points are the one array whose shape is
  checked, so the transposed `(3, N)` layout raises instead of passing as interleaved coordinates.
- `cell_types` also takes a numpy array of cell type codes, in any integer dtype. Those codes are
  the `CellType` values, i.e. the XDMF topology types and *not* the VTK cell codes: a hexahedron is
  9 here and 12 in VTK. (The node ordering *within* a cell does follow VTK.)
- The file name is a `str` or any `os.PathLike`, and `writer.file_name` hands back the
  `pathlib.Path` it writes.
- A time step is all-or-nothing, as in Rust: one rejected attribute drops the whole step, heavy
  data and all, and leaves the time free again.
- The data writer is a context manager, so the `with` block closes the HDF5 file instead of the
  garbage collector. `close()` does the same.
- Writes release the GIL, so other threads keep running and threads writing their own file write in
  parallel. The arrays are borrowed, so no thread may modify one mid-write.
- A rejected `write_mesh` leaves the writer usable, so fix the dtype or shape and retry. Only a
  successful call consumes it, as in Rust.
- `DataStorage`, `DataAttribute` and `CellType` are frozen, comparable and hashable, so they work
  as `dict` keys and in sets.
- Type stubs (`xdmf.pyi`) ship with the package.
