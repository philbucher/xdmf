# Upcoming features:

> **Planning status:** this document is the input wish-list. It has been turned into an ordered set
> of milestones in [`ROADMAP.md`](ROADMAP.md), which also records the eight decisions that were taken
> up front and links the per-feature sub-plans (`01_error_type.md` … `07_mpi.md`).

## Global Context:
- Everything should be tuned towards maximum performance, both reading and writing is in the hot path. 
    - Temporary allocations should be avoided as much as possible, mabye this requires changing the API (e.g. passing pre-allocated vectors so that they can be reused during writing of steps)
    - Compression etc should be tested against real data to find the optimum balance
- A lot of WIP work lives in the "multiple-features" branch.
- Critical features should be checked in the paraview-CI
- Main usages:
    - Arotau: /home/philipp/software/arotau/arotau-core/src/output/xdmf.rs
    - Plain writing of meshes & data from python
- Paraview must be able to read the data! (can be checked locally, or by adding tests in the CI)

## Features for v1.0
- API-improvements: plans/API_IMPROVEMENTS_PLAN.md => this should be checked after the other plans are written down, in case anything needs to be adjusted
- TimeSeriesReader: A draft is in "reader.rs", might need to be adjusted. API should be similar to the reader
    - This obviously requires the implementation of the readers for different formats
- f32 bit floats should be added to `Values`
- Python interface, already exists in the "multiple-features" branch. => this way vibe-coded, needs to be double checked
    - Data must ideally not be copied when going from Rust <=> Python
- Publishing wheels to pypi
- Writing and reading of submeshes => a draft is in the "multiple-features" branch, but this one needs to be done a bit nicer
- Check the entire crate for performance-gains and other optimizations, now is the time! Even if it requires larger refactoring

## Features beyond v1.0
- MPI support! => for now I want to consider this and draft the API, so that we dont miss anything. 
    - First with HDF, as this can already support MPI
    - With Ascii and Binary I would like to write a single file, but that needs MPI-IO, which is currently not supported in rsmpi. I dont think this will impact the API, but I want to mention it for completeness, in v1.1 we will only do MPI with hdf