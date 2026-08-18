# xdmf — python interface

Python bindings for the [xdmf](https://github.com/philbucher/xdmf) crate, built with
[pyo3](https://pyo3.rs)/[maturin](https://www.maturin.rs): write meshes with time-series data as XDMF
files, to be read and visualized by ParaView or VisIt.

There are no wheels on PyPI yet, so it is built from the repository:

~~~sh
pip install ./python
~~~

## Example

The mesh and the data are passed as numpy arrays, which are borrowed directly — nothing is copied on
the way into the file, and each array is written at the dtype it is passed in (`float64`, `float32`,
`uint64`, `uint32`, `int64`, `int32`), just like in Rust:

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

`tests/test_writer.py` has many more examples, including every failure case.

## Good to know

- Points and vector data may also be shaped `(N, 3)`, the natural numpy layout — a C-contiguous
  `(N, 3)` array is the same memory as the flat one, so no `reshape` is needed. `cell_types` may be a
  numpy array of the raw VTK cell type codes instead of a list.
- A time step is all-or-nothing, as in Rust: it is written when every attribute of it was accepted,
  and discarded — leaving no heavy data behind and the time still available — as soon as one is not.
- The data writer is a context manager, so the HDF5 file is closed at the end of the `with` block
  instead of whenever the object happens to be garbage-collected. `close()` does the same explicitly.
- Writes release the GIL, so other Python threads keep running — and several threads writing their own
  file do so in parallel. The flip side of that, combined with the arrays being borrowed rather than
  copied, is that another thread must not modify an array while a write of it is running.
  Single-threaded code cannot hit this, since `write_mesh`/`write_time_step` return before the next
  statement runs.
- Failures are the Python exception matching what went wrong: `ValueError` for anything the mesh or
  the step does not accept, `OverflowError` for an integer the chosen storage cannot represent (see
  the [integer data](https://github.com/philbucher/xdmf#can-integer-data-be-written) section,
  whose limits apply here too),
  `OSError` for a failing write, and `RuntimeError` for using a consumed writer.
- Type stubs (`xdmf.pyi`) ship with the package, so editors and type checkers see the full interface.
