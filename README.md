# xdmf

This crate implements the [xdmf](https://www.xdmf.org) file format for writing meshes with data, to be read and visualized by ParaView or VisIt.

The data storage is split into light and heavy data. The light data is metadata in xml-format, describing where and how the heavy data is stored. The heavy data can be stored in different formats. HDF is the preferred format, for space and time efficient data storage.

A large advantage over VTK based formats is that data can be referenced. The mesh can be written only once, and then referenced for the visualization of time step data. This reduces the storage requirements and write times significatly.

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
    xdmf::DataStorage::Hdf5SingleFile
).expect("failed to create XDMF writer");

// define 3 points and 2 cells (a line and a triangle)
let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
let connectivity = [0, 1, 0, 2, 1]; // line (0,1) and triangle (0,2,1)
let cell_types = [xdmf::CellType::Edge, xdmf::CellType::Triangle];

// write the mesh
let mut time_series_writer = xdmf_writer.write_mesh(&coords, (&connectivity, &cell_types)).expect("failed to write mesh");

// define some point and cell data for time step 0.0
let point_data = vec![(
       "point_data".to_string(),
       (xdmf::DataAttribute::Vector, vec![0.0; 9].into()),
   )]
   .into_iter()
   .collect();

let cell_data = vec![(
       "cell_data".to_string(),
       (xdmf::DataAttribute::Scalar, vec![0.0, 1.0].into()),
   )]
   .into_iter()
   .collect();

// write the data for 10 time steps
for i in 0..10 {
    time_series_writer
        .write_data(&i.to_string(), Some(&point_data), Some(&cell_data))
        .expect("failed to write time step data");
}
~~~

### Which data storage should be used for the heavy data?

The xdmf format allows to separate the storing of light and heavy data. Different data storage methods are implemented for the latter:

- `Ascii`: This format stores the heavy data in ascii text files.
- `AsciiInline`: This format stores the heavy data together with the light data in the xml file. This is only recommended for testing or little data, since its neither fast nor space efficient. It however is the only method that stores everything in one single file
- `XdmfH5Single`: The heavy data is stored in a single hdf5 file. This is the **recommended format** unless special requirements exist.
- `XdmfH5Multiple`: The heavy data is stored in a multiple hdf5 files, one for each time step (and mesh). This creates more files and usually only makes sense when the data is accessed concurrently while its being written.

## Comparison with vtk/vtu

Initial comparisons show smaller storage sizes as well as faster write times. The conclusions still have to be summarized here. In the meantime check [this file](./tests/vtk_comparison.rs) for a comparion.

## General information

- The node ordering is same as for [vtk](https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf).
- The focus is writing data that can be visualized with ParaView. Therefore, consistency checks were added to ensure that the data is correctly written.
- The xdmf format seems does not seem to be actively developed any more. It will probably be superseded by [hdf-based vtk files](https://www.kitware.com/vtk-hdf-reader/). However, it can be assumed that xdmf will still be supported for a while by ParaView

<!-- <https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/>
<https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format>  -->

## Roadmap / planned features

- MPI suport <!-- (writing to one file => writing separate independent files can already work if file names passed have ranks) -->
- SubMesh support, so that parts of the mesh can be visualized with the MultiBlock inspector
- Reading files. Hopefully even concurrently, perhaps consuming to safe space.
- Maybe binary support (could be nice for platforms that dont have HDF installed)

<!-- ## TODOs

- check h5 file flushing
- test with bigger example -->
