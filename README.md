# xdmf

This crate implements the [xdmf](https://www.xdmf.org) file format for writing meshes with data, to be read and visualized by ParaView or VisIt.

The data storage is split into light and heavy data. The light data is metadata in xml-format, describing where and how the heavy data is stored. The heavy data can be stored in different formats. HDF is the preferred format, for space and time efficient data storage.

A large advantage over VTK based formats is that data can be referenced. The mesh can be written only once, and then referenced for the visualization of time step data. This reduces the storage requirements and write times significantly.

<!--
xdmf readers: <https://discourse.paraview.org/t/xmdf-reader-names-xdmf2-reader/4756> => using "xdmf2" file extension to use this reader

 -->

## Example

While this crate allows to construct the individual xdmf elements to compose an xdmf file (see [here](./tests/xdmf_elements.rs)), it is recommended for most cases to use the `TimeSeriesWriter`. Check [this file](./tests/time_series_writer.rs) for elaborate examples.

It has a simple interface that allows to write a mesh and add time-step data to it:

~~~rs
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

The step is scoped to the closure: when it returns `Ok`, the step is added to the XDMF file; when it
returns an error, the step is discarded and the heavy data already written for it is removed again.
So a step can neither be left half-written by forgetting to complete it, nor leave data behind that
nothing references. The error may be one of your own — any type that `xdmf::Error` converts into
works, so `?` can be used on both inside the closure, and `write_time_step` hands your error back
unchanged.

### Can parts of the mesh be shown separately?

Yes — `write_mesh_with_submeshes` takes named subsets of the mesh's cells alongside the mesh itself,
and each becomes a separately selectable block in ParaView's Multi-block Inspector
(`View -> Multi-block Inspector`). Submeshes may overlap: a cell can belong to any number of them.

A submesh's cells are given either as an index list or, when it is one block of consecutive cells
(as element blocks and material zones usually are), as a range — `("fluid", 0..1_000_000)` rather
than a million-entry `Vec`. Both are stored as the same two numbers; the range just skips building
the list. See `SubmeshCells` for every shape accepted.

Time step data is still written over the whole mesh, exactly as above — point data over all points,
cell data over all cells — and the writer gives each submesh its share, so submeshes can be added to
existing code without changing how it produces its field data.

Each block holds only the points its own cells use, with its connectivity numbered against them,
so ParaView holds the mesh roughly once however many blocks it is split into. Measured on a
40,401-point mesh in ParaView 5.13.3 and 6.1: 2.8 MB as one block, 4.8 MB as 256 — where a layout
that gave every block the whole point set took 317.8 MB.

For the two HDF5 storages nothing is copied per block: the mesh's coordinates are written once, and
each block's `<Geometry>` and `<Attribute>`s say which part of them and of every field that block
holds. A point on a block boundary is therefore stored once however many blocks touch it, and a
step costs the same however many blocks there are and however much they overlap. Measured on the
same mesh over 10 steps at 64 blocks: 27.2 MB of heavy data before, 17.2 MB now, and 65.1 MB
against 20.2 MB when the blocks are strided rather than contiguous. It is paid for in light data
(+78%) and in read time (about 40% longer to step through that animation, roughly double when the
blocks are strided).

The ascii and binary storages keep a copy per block instead — of its points, and of its share of
every field — because ParaView reads a selection out of those formats from the start of the array
instead of where it points, silently, so the writer does not emit one there. Their cost is the
duplication: points on a block boundary are written once per block that touches them (+73% of the
mesh's heavy data at 256 blocks, nothing measurable at a handful).

To look at one submesh on its own, use the Multi-block Inspector or an `Extract Block` filter,
which both select a block by name and hold it for the whole animation. ParaView's *other* selection
list — `Grids` in the reader's Properties panel — does not: it lists one entry per grid in the file,
which for a time series is one per (submesh, time step), so unchecking entries there hides a submesh
at some steps and not at others. That list is per-step for any time series this crate writes,
submeshes or not.

See [`examples/submeshes.rs`](examples/submeshes.rs) for a complete example
(`cargo run --example submeshes`).

### Which precision should be used for the floating point data?

Both `f32` and `f64` are accepted, for the mesh coordinates as well as for the point and cell data,
and are written at the precision they are passed in — whichever the calling code already holds. No
conversion happens in either direction.

`f32` halves the size of the written data. For attribute data that is usually all there is to it,
but for the mesh coordinates it comes with a caveat: on a domain far away from the origin, `f32`
coordinates produce visible geometric jitter in ParaView, because the absolute coordinate eats up
the mantissa.

### Can integer data be written?

Yes — point and cell data also accept `i32`, `i64`, `u32` and `u64` (for example a partition rank,
a material id or a flag). Mesh coordinates stay floating point.

Every type is written at its own width: the file holds the type that was passed in, and nothing is
narrowed, widened or cast on the way out. Where ParaView cannot read a type back as it was written,
the write is refused instead. There are three such cases, all measured against ParaView 5.13 and
6.1 (see `examples/paraview_smoke.rs` and `tests/paraview_smoke/`):

- **`DataStorage::Binary` does not accept `i64` or `u64` at all.** ParaView's legacy Xdmf2 reader
  walks 64-bit integers in `Format="Binary"` at the wrong stride: attribute values come back with
  every second one replaced by zero, and 64-bit connectivity makes the reader give up outright
  (`vtkXdmfReader: Failed to read data`). The damage does not depend on how large the numbers are,
  so the *type* is rejected rather than a range — pass `i32`/`u32`, or use another storage. Earlier
  versions narrowed to 32 bits instead; that produced a loadable file, but one holding a different
  type than the caller handed over.
- **`u64` above `u32::MAX` is rejected, by every storage method.** ParaView builds a 32-bit array
  for `NumberType="UInt"` no matter what `Precision` the light data declares, so a larger value
  comes back truncated (ascii) or clamped to `u32::MAX` (HDF5) — silently, without a reader error.
  Values *within* that range read back exactly at the full 8 bytes, so this caps the value and not
  the width. Use `i64` for integer data that has to exceed 32 bits; `NumberType="Int"` really is
  decoded at 64 bits. The same cap applies to `u32`/`u64` connectivity, and so to the mesh size —
  see the connectivity section below.
- **In the ascii storage methods, `i64` is limited to ±2^53.** ParaView parses their integers
  through a `double`, so a larger value comes back rounded — and `i64::MAX` comes back as
  `i64::MIN`, sign flipped. Values past that are rejected rather than written, so nothing lands in
  the output that ParaView would display as a different number. `Hdf5SingleFile`/`Hdf5MultipleFiles`
  have no such limit and read `i64` exactly at both extremes, which is what the error points at.

If file size matters more than keeping the caller's type, pass the narrow type in the first place:
`u32` data is half the bytes of `u64` data, and on a 205k-hexahedron mesh choosing `u32` over `u64`
for the connectivity halves that dataset (14.8 MB → 7.4 MB before compression) and shrinks the whole
`Hdf5SingleFile` output by 8%. That is the caller's trade-off to make, and it is visible in the
light data either way.

Floating point data is not affected by any of this: the ascii storage methods write each value with
the fewest digits that read back as the exact same `f32`/`f64`, so nothing is lost there and round
values stay short (`1.05e1`, not `1.05000000e1`).

### Which integer type should the connectivity be?

`u32`, `u64`, `i32` and `i64` are all accepted, and the connectivity is written as the type it is
passed in. That choice is what sets the largest mesh that can be written, since it decides the
`NumberType`/`Precision` pair in the light data:

| connectivity type | light data | largest mesh | storages |
| --- | --- | --- | --- |
| `u32` | `NumberType="UInt" Precision="4"` | 2^32 points | all |
| `i32` | `NumberType="Int" Precision="4"` | 2^31 points | all |
| `u64` | `NumberType="UInt" Precision="8"` | 2^32 points | all but `Binary` |
| `i64` | `NumberType="Int" Precision="8"` | beyond any mesh | `Hdf5SingleFile`, `Hdf5MultipleFiles` |

`u32` is the one to reach for: it indexes any mesh that ParaView can read the connectivity of, in
half the bytes of `u64`. `u64` writes the same indices at twice the width without raising the cap,
since that cap is the reader's and not the type's.

`i64` is the one type that lifts the 2^32 cap, because `NumberType="Int"` is the one ParaView
decodes at the declared width. What that means in practice is only partly measured: the reader
handles `Int`/8 connectivity correctly (verified on ParaView 5.13 and 6.1, whose `vtkIdType` is
64 bits in both builds), but a mesh with an index actually beyond 2^32 needs over 100 GB of
coordinates and has not been tested here. Only the HDF5 storages could carry such a mesh anyway:
`Binary` refuses 64-bit integers, and the ascii storages cap `i64` at 2^53.

A mesh too large for the type it is written with is rejected up front, rather than silently wrapping
around.

### Which data storage should be used for the heavy data?

The xdmf format allows to separate the storing of light and heavy data. Different data storage methods are implemented for the latter:

- `Ascii`: This format stores the heavy data in ascii text files.
- `AsciiInline`: This format stores the heavy data together with the light data in the xml file. This is only recommended for testing or little data, since its neither fast nor space efficient. It however is the only method that stores everything in one single file
- `XdmfH5Single`: The heavy data is stored in a single hdf5 file. This is the **recommended format** unless special requirements exist.
- `XdmfH5Multiple`: The heavy data is stored in a multiple hdf5 files, one for each time step (and mesh). This creates more files and usually only makes sense when the data is accessed concurrently while its being written.

## Reading

`TimeSeriesReader::new` parses the whole file up front, so every read call after it is a plain,
independent, repeatable query -- there is no phase to pass through first, unlike the writer (which
writes the mesh once and irreversibly before any time step). Only the two HDF5 storages
(`Hdf5SingleFile`/`Hdf5MultipleFiles`) can be read so far; opening a file written with
`Ascii`/`AsciiInline`/`Binary` fails right there, rather than at the first call that reaches the
heavy data.

~~~rs
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

`point_data_info`/`cell_data_info` report a field's shape and element type before it is read, so a
caller can size a buffer and pick a type without guessing. Reading a field into a wider type than it
was written as is allowed (e.g. `f32` file data read into a `Vec<f64>`); narrowing is rejected
instead of silently losing precision. `read_points` follows the same rule (`f32`/`f64`, see
`Coordinate`). `read_topology` takes any of `u32`/`u64`/`i32`/`i64` (see `ConnectivityIndex`) and
checks the *values* instead: what it hands back are positions in the mesh it reassembled, not the
file's own array, so any type that holds every index works whatever the file was written as.

A mesh written with `write_mesh_with_submeshes` reads back as the single, whole mesh it started from
— `read_points`/`read_topology` put it back together from the submeshes' own points, cells and
connectivity, and `points`/`cell_types`/`connectivity` come back exactly as they would from a mesh
written without submeshes at all. `submesh_names()`/`submesh_cells()`/`submesh_points()` recover
which mesh cells and points each submesh holds, in case a caller wants that split back too.

See [`tests/reader.rs`](./tests/reader.rs) for more examples.

## Python interface

The [`python/`](./python) directory holds bindings that expose the same `TimeSeriesWriter` interface
to Python, with the mesh and the data passed as numpy arrays that are borrowed rather than copied.
See [`python/README.md`](./python/README.md) for how to build them and what they look like.

## Comparison with vtk/vtu

Initial comparisons show smaller storage sizes as well as faster write times. The conclusions still have to be summarized here. In the meantime check [this file](./tests/vtk_comparison.rs) for a comparison.

## General information

- The node ordering is same as for [vtk](https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf).
- The focus is writing data that can be visualized with ParaView. Therefore, consistency checks were added to ensure that the data is correctly written.
- The xdmf format seems does not seem to be actively developed any more. It will probably be superseded by [hdf-based vtk files](https://www.kitware.com/vtk-hdf-reader/). However, it can be assumed that xdmf will still be supported for a while by ParaView
- `DataAttribute::Tensor6`, `DataAttribute::Matrix`, and `DataAttribute::Generic` data (written as XDMF's `AttributeType="Matrix"`) requires **ParaView >= 6.1 / VTK >= 9.6** to be read back correctly. Older versions misread the shape and merge every node's/cell's values into a single tuple, due to a change in VTK's XDMF2 reader ([Kitware/VTK@7199be5](https://github.com/Kitware/VTK/commit/7199be5854)). `Scalar`, `Vector`, and `Tensor` are unaffected and work on all ParaView versions.

<!-- <https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/>
<https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format>  -->

## Roadmap / planned features

- MPI support <!-- (writing to one file => writing separate independent files can already work if file names passed have ranks) -->
- Reading `Ascii`/`AsciiInline`/`Binary` files (HDF5 already supported, see [Reading](#reading)).

<!-- ## TODOs

- check h5 file flushing
- test with bigger example -->
