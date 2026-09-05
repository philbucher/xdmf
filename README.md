# xdmf

Write meshes with data as [xdmf](https://www.xdmf.org) files, for ParaView or VisIt to read.

An xdmf file splits into light and heavy data. The light data is xml metadata saying where and how the heavy data is stored; the heavy data holds the numbers, in one of several formats. HDF is the format to pick, for space and for write speed.

The advantage over the VTK based formats is that xdmf can reference data. Write the mesh once, then reference it from every time step instead of repeating it per step: fewer bytes on disk, less to write.

<!--
xdmf readers: <https://discourse.paraview.org/t/xmdf-reader-names-xdmf2-reader/4756> => using "xdmf2" file extension to use this reader

 -->

## Example

You can compose an xdmf file out of the individual xdmf elements (see [here](./tests/xdmf_elements.rs)), but for most cases reach for `TimeSeriesWriter` instead. [This file](./tests/time_series_writer.rs) has elaborate examples.

Its interface writes a mesh, then adds time-step data to it:

~~~rust,no_run
use xdmf::TimeSeriesWriter;

// construct the writer (using HDF5 for heavy data storage)
let xdmf_writer = TimeSeriesWriter::new(
    "xdmf_writing",
    xdmf::DataStorage::Hdf5SingleFile { deflate_level: None }
).expect("failed to create XDMF writer");

// define 3 points and 2 cells (a line and a triangle)
let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
let connectivity = [0_u32, 1, 0, 2, 1]; // line (0,1) and triangle (0,2,1)
let cell_types = [xdmf::CellType::Edge, xdmf::CellType::Triangle];

// write the mesh
let mut time_series_writer = xdmf_writer.write_mesh(&coords, &connectivity, &cell_types).expect("failed to write mesh");

// each attribute is written as it is passed, so these buffers can be refilled and reused
// for every field of every time step
let mut point_values = vec![0.0; 9];
let cell_values = vec![0.0, 1.0];

// write the data for 10 time steps
for i in 0..10 {
    time_series_writer
        .write_time_step(&i.to_string(), |step| {
            step.point_data("point_data", xdmf::DataAttribute::Vector, &point_values)?;

            point_values.fill(i as f64); // the same buffer, refilled for the next attribute
            step.point_data("more_point_data", xdmf::DataAttribute::Vector, &point_values)?;

            step.cell_data("cell_data", xdmf::DataAttribute::Scalar, &cell_values)
        })
        .expect("failed to write time step");
}
~~~

The closure scopes the step. Return `Ok` and the writer adds the step to the XDMF file; return an
error and it drops the step, removing the heavy data already written for it. The error can be one of
your own: any type that `xdmf::Error` converts into works, so `?` covers both inside the closure,
and `write_time_step` hands your error back unchanged.

### Can I show parts of the mesh separately?

Yes. `write_mesh_with_submeshes` takes named subsets of the mesh's cells alongside the mesh itself,
and each one becomes a separately selectable block in ParaView's Multi-block Inspector
(`View -> Multi-block Inspector`). Submeshes may overlap: a cell can belong to any number of them.

Give a submesh's cells as an index list, or as a range for one block of consecutive cells:
`("fluid", 0..1_000_000)` rather than a million-entry `Vec`. See `SubmeshCells` for every shape it
accepts.

You still write time step data over the whole mesh, exactly as above, point data over all points
and cell data over all cells. The writer gives each submesh its share.

What the split costs on disk depends on the storage, see [Data storage](#data-storage).

To look at one submesh on its own, use the Multi-block Inspector or an `Extract Block` filter, which
both select a block by name and hold it for the whole animation. Do not use the `Grids` list in the
reader's Properties panel: it holds one entry per (submesh, time step), so unchecking entries there
hides a submesh at some steps and not at others.

See [`examples/submeshes.rs`](examples/submeshes.rs) for a complete example.

### Which precision should I use for the floating point data?

Both `f32` and `f64` work, for the mesh coordinates as well as for the point and cell data, and each
goes into the file at the width you pass it.

`f32` halves the size of the written data. For the mesh coordinates it comes with a caveat: on a
domain far from the origin, `f32` coordinates jitter visibly in ParaView.

### Can I write integer data?

Yes. Point and cell data also take `i32`, `i64`, `u32` and `u64`, for a partition rank, a material
id or a flag. Mesh coordinates stay floating point.

Each type goes in at its own width: the file holds the type you passed, narrowed, widened or cast
nowhere along the way. Where ParaView cannot read a type back as it was written, the writer refuses
the write instead. Which types those are depends on the storage, and on the value in one case: see
[what ParaView reads back](#what-paraview-reads-back).

If file size matters to you more than keeping your own type, pass the narrow type in the first
place.

### Which integer type should the connectivity be?

All of `u32`, `u64`, `i32` and `i64` work, and the connectivity goes into the file as the type you
pass. That choice sets the largest mesh you can write, since it decides the `NumberType`/`Precision`
pair in the light data:

| connectivity type | light data | largest mesh | storages |
| --- | --- | --- | --- |
| `u32` | `NumberType="UInt" Precision="4"` | 2^32 points | all |
| `i32` | `NumberType="Int" Precision="4"` | 2^31 points | all |
| `u64` | `NumberType="UInt" Precision="8"` | 2^32 points | all but `Binary` |
| `i64` | `NumberType="Int" Precision="8"` | beyond any mesh | `Hdf5SingleFile`, `Hdf5MultipleFiles` |

`u32` is the one to reach for: it indexes any mesh whose connectivity ParaView can read, in half the
bytes of `u64`. `u64` writes the same indices at twice the width without raising the cap. `i64` is
the one type that lifts it, and only the HDF5 storages take it.

A mesh too large for the type it is written with is rejected up front, rather than wrapping around.

## Reading

`TimeSeriesReader::new` parses the whole file up front, so every read call after it is a plain,
independent, repeatable query. The reader handles the two HDF5 storages
(`Hdf5SingleFile`/`Hdf5MultipleFiles`) so far, and opening a file written with
`Ascii`/`AsciiInline`/`Binary` fails right there.

~~~rust,no_run
use xdmf::TimeSeriesReader;

// open the file the writer example above produced
let reader = TimeSeriesReader::new("xdmf_writing.xdmf2").expect("failed to open XDMF file");

// points and topology (connectivity + cell types) are independent reads, each filling a buffer
// of whichever element type the caller wants it at
let mut points: Vec<f64> = Vec::new();
reader.read_points(&mut points).expect("failed to read points");

let mut connectivity: Vec<u64> = Vec::new();
let mut cell_types = Vec::new();
reader
    .read_topology(&mut connectivity, &mut cell_types)
    .expect("failed to read topology");

// if the mesh was written with submeshes, each one's own cells (and points) can be recovered,
// as indices into the buffers above -- empty for a mesh with no submeshes
for (index, name) in reader.submesh_names().iter().enumerate() {
    let cells = reader.submesh_cells(index).expect("failed to read submesh cells");
    println!("{name}: {} cells", cells.len());
}

// then read each step's data, reusing the same buffers
let mut point_data = Vec::new();
let mut cell_data = Vec::new();
for step in 0..reader.num_steps() {
    reader
        .read_point_data::<f64>(step, "point_data", &mut point_data)
        .expect("failed to read point data");
    reader
        .read_cell_data::<f64>(step, "cell_data", &mut cell_data)
        .expect("failed to read cell data");
}
~~~

`point_data_info`/`cell_data_info` report a field's shape and element type before you read it.
Reading a field into a wider type than it was written as works (`f32` file data into a `Vec<f64>`),
narrowing fails. `read_points` follows the same rule (`f32`/`f64`, see `Coordinate`).
`read_topology` takes any of `u32`/`u64`/`i32`/`i64` (see `ConnectivityIndex`) and checks the
*values* instead, so any type that holds every index works whatever the file was written as.

A mesh written with `write_mesh_with_submeshes` reads back as the single, whole mesh it started
from, and `submesh_names()`/`submesh_cells()`/`submesh_points()` recover which mesh cells and points
each submesh holds.

See [`tests/reader.rs`](./tests/reader.rs) for more examples.

## Python interface

`pip install xdmf` gets bindings that expose the same `TimeSeriesWriter` interface to Python, taking
the mesh and the data as numpy arrays it borrows rather than copies. They live in
[`python/`](./python); see [`python/README.md`](./python/README.md) for what they look like.

## Comparison with vtk/vtu

First comparisons show smaller files and faster writes. The numbers still need summarizing here;
until then, [this file](./tests/vtk_comparison.rs) holds the comparison.

## Data storage

xdmf keeps light and heavy data apart, and this crate offers five ways to store the heavy half:

- `Hdf5SingleFile`: one hdf5 file for all the heavy data. **Use this one** unless you have a reason
  not to.
- `Hdf5MultipleFiles`: one hdf5 file per time step, plus one for the mesh. More files to handle,
  worth it when something reads the data while you are still writing it.
- `Ascii`: text files, one per array.
- `AsciiInline`: the heavy data inline in the light data's xml. Neither fast nor space efficient, so
  keep it for testing and small meshes. It is the one method that puts everything in a single file.
- `Binary`: raw binary files, one per array, uncompressed.

Only the two HDF5 storages can be read back (see [Reading](#reading)).

### What a submesh costs

The two HDF5 storages copy nothing per block: the mesh's arrays are written once, and each block's
`<Geometry>` and `<Attribute>`s name its share of them. The extra light data that takes makes the
file slower to step through in ParaView.

The ascii and binary storages keep a copy per block instead: of its points, and of its share of
every field. A point on a block boundary is written once per block that touches it.

### What ParaView reads back

Three types come back from ParaView's legacy Xdmf2 reader as something other than what the file
holds, so the writer refuses them (measured against ParaView 5.13 and 6.1, see
`examples/paraview_smoke.rs` and `tests/paraview_smoke/`):

- **`Binary` does not accept `i64` or `u64` at all.** The reader walks them at the wrong stride:
  attribute values come back with every second one replaced by zero, and 64-bit connectivity makes
  the reader give up (`vtkXdmfReader: Failed to read data`). Pass `i32`/`u32`, or use another
  storage.
- **Every storage rejects `u64` above `u32::MAX`.** ParaView builds a 32-bit array for
  `NumberType="UInt"` whatever `Precision` the light data declares, so a larger value comes back
  truncated (ascii) or clamped to `u32::MAX` (HDF5), with no reader error to show for it. Use `i64`
  for integer data that has to exceed 32 bits. The cap covers `u32`/`u64` connectivity too, and so
  the mesh size (see [the connectivity table](#which-integer-type-should-the-connectivity-be)).
- **The ascii storages cap `i64` at ±2^53.** ParaView parses their integers through a `double`, so a
  larger value comes back rounded, and `i64::MAX` comes back as `i64::MIN` with the sign flipped.
  `Hdf5SingleFile`/`Hdf5MultipleFiles` have no such limit.

## General information

- The node ordering follows [vtk](https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf).
- The focus is data that ParaView can visualize, so the writer checks what you hand it.
- Nobody seems to develop the xdmf format any more, and [hdf-based vtk files](https://www.kitware.com/vtk-hdf-reader/) will likely supersede it. ParaView should keep reading xdmf for a while yet.
- `DataAttribute::Tensor6`, `DataAttribute::Matrix`, and `DataAttribute::Generic` data (written as XDMF's `AttributeType="Matrix"`) needs **ParaView >= 6.1 / VTK >= 9.6** ([Kitware/VTK@7199be5](https://github.com/Kitware/VTK/commit/7199be5854)); older versions merge every node's or cell's values into one tuple. `Scalar`, `Vector`, and `Tensor` read back on every ParaView version.

<!-- <https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/>
<https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format>  -->

## Roadmap / planned features

- MPI support <!-- (writing to one file => writing separate independent files can already work if file names passed have ranks) -->
- Reading `Ascii`/`AsciiInline`/`Binary` files (HDF5 already supported, see [Reading](#reading)).

<!-- ## TODOs

- check h5 file flushing
- test with bigger example -->
