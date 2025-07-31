# xdmf

This crate does ...
For unstructured meshes (although might also work for others)

xdmf readers: <https://discourse.paraview.org/t/xmdf-reader-names-xdmf2-reader/4756>

## Example

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
