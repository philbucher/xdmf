# xdmf

This crate does ...
For unstructured meshes (although might also work for others)

xdmf readers: <https://discourse.paraview.org/t/xmdf-reader-names-xdmf2-reader/4756>

## Example

### TimeSeriesWriter: Which data storage should be used?
The xdmf format allows to separate the storing of light and heavy data. Different data storage methods are implemented for the latter:
- `AsciiInline`: This format stores the heavy data together with the light data in the xml file. This is only recommended for testing or little data, since its neither fast nor space efficient. It however is the only method that stores everything in one single file
- `XdmfH5Single`: The heavy data is stored in a single hdf5 file. This is the recommended format unless special requirements exist
- `XdmfH5Multiple`: The heavy data is stored in a multiple hdf5 files, one for each time step (and mesh). This creates more files and usually only makes sense when the data is accessed concurrently while its being written.

## Comparison with vtk/vtu

## Issues

- paraview cannot open subset grids

"<https://www.xdmf.org/>"

<https://www.kitware.com/vtk-hdf-reader/>
<https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/>
<https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format>

I am a fan of fail early, so I tried to add validations to the data as much as possible, but reasonble to not affect performance too much.

## Roadmap / planned features

- MPI suport (writing to one file => writing separate independent files can already work if file names passed have ranks)
- SubMesh support
- Reading files. Hopefully even concurrently, perhaps consuming to safe space. This is already kinda planned
- Maybe binary support (could be nice for platforms that dont have hdf installed)

## TODO

- Add docs to at least public API
- More integration tests, should also serve as examples
- Perhaps a real work example comparing to vtk/vtu?
- check h5 file flushing
- test with bigger example
- Mention somewhere in readme that seems that the xdmf is no longer maintained
- Check docs of connectivity
