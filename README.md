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
let connectivity = [0, 1, 0, 2, 1]; // line (0,1) and triangle (0,2,1)
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
a material id or a flag), again at the width they are passed in. Mesh coordinates stay floating
point.

The one exception is `DataStorage::Binary`: ParaView's legacy Xdmf2 reader misreads 64-bit integers
there, so `i64`/`u64` data is narrowed to 32 bits on the way out, and a value that does not fit is
reported as an error rather than silently truncated. The other storage methods write 64-bit
integers as they are.

### Which data storage should be used for the heavy data?

The xdmf format allows to separate the storing of light and heavy data. Different data storage methods are implemented for the latter:

- `Ascii`: This format stores the heavy data in ascii text files.
- `AsciiInline`: This format stores the heavy data together with the light data in the xml file. This is only recommended for testing or little data, since its neither fast nor space efficient. It however is the only method that stores everything in one single file
- `XdmfH5Single`: The heavy data is stored in a single hdf5 file. This is the **recommended format** unless special requirements exist.
- `XdmfH5Multiple`: The heavy data is stored in a multiple hdf5 files, one for each time step (and mesh). This creates more files and usually only makes sense when the data is accessed concurrently while its being written.

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
- SubMesh support, so that parts of the mesh can be visualized with the MultiBlock inspector
- Reading files. Hopefully even concurrently, perhaps consuming to safe space.

<!-- ## TODOs

- check h5 file flushing
- test with bigger example -->
