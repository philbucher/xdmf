# M5 — `TimeSeriesReader`

`README.md`: *"TimeSeriesReader: A draft is in `reader.rs`, might need to be adjusted. API should be
similar to the [writer]. This obviously requires the implementation of the readers for different
formats."*

Decision 2 in `ROADMAP.md`: **guarantee round-tripping of this crate's own output; make a best effort
on a common subset of foreign XDMF2; reject everything else with an explicit `Unsupported` error.**

The sketch that was in `reader.rs` at the repo root is folded into this document — delete that file.

## Order of work: `Format="HDF"` first (2026-08-22)

The milestone is split. **Stage 1 reads the HDF5 storages only** — `Hdf5SingleFile` and
`Hdf5MultipleFiles`, i.e. every `DataItem` with `Format="HDF"` — and is the one to implement first.
Stage 2 adds `Format="XML"` and `Format="Binary"`.

Why that order, beyond the HDF5 storages being the ones that matter in production:

- **Only the HDF5 output is fully reconstructible.** A mesh written with submeshes to an ascii or
  binary storage never writes the mesh's own points: each block carries a copy of the points its
  cells use, so a point no cell uses exists nowhere in the file, and nothing states `num_points`.
  The HDF5 layout writes the mesh's coordinates once and has each block *select* out of them, so
  the global array — every point, in the original order — is right there. See
  "Reading a mesh written with submeshes" below.
- **It is the only path that needs the selection items**, which is the genuinely new reader work
  (`ItemType="HyperSlab"` / `"Coordinates"`). Ascii and binary write blocks as compacted copies, so
  their reader is the simpler, older shape and gains nothing by going first.
- Splitting the format backends is free: the reader dispatches per `DataItem` on `Format`
  (see *Architecture* below), so stage 2 adds two files and touches nothing else. A `Format` with
  no backend yet is `Error::Unsupported`, the same as an unknown one.

Stage 1 still has to parse the *whole* document, including a `data_storage` `Information` naming an
ascii storage — it is the individual heavy-data reads that are unsupported, not the file.

### Stage 1, in order

1. **Deserialize `DataItem`** in every shape it is written in, selections included — risk 1 below,
   and everything else rests on it.
2. **`light_data.rs`**: document → `xdmf_elements` structs, plus the one `Reference="XML"` XPath
   shape (risk 2). At this point `num_points`, `times()` and `submesh_names()` work.
3. **`hdf5_reader.rs`**: `file.h5:group/dataset` → values, honouring `NumberType`/`Precision`, with
   the widening rules from the API notes.
4. **`topology.rs`**: the inverse of `prepare_cells` (risk 3), against the existing
   `prepare_cells_by_celltype` test as a template.
5. **`read_points`/`read_topology` without submeshes**, round-tripping `write_mesh` output — the
   first end-to-end test.
6. **`read_points`/`read_topology` with submeshes**, per the section below: `selection.rs` for the
   two selector shapes, then the scatter through `submesh_cells`.
7. **Per-step data**, reading the global source arrays and ignoring the per-submesh selections.

## The key correction to the draft

The draft has:

```rust
time_series_reader.step(i).read_data(i, Some(&point_data), Some(&cell_data))
```

That cannot work: a time step holds *several named attributes* of different shapes and potentially
different element types (a scalar pressure, a vector velocity, a u64 cell id). They cannot share one
flat output buffer, and the caller cannot size the buffer without first knowing what is in the file.

So the read path is **per attribute, into a caller-provided buffer**, with a metadata query to size
it. That keeps the zero-allocation property the writer has, which matters because `README.md` puts
reading in the hot path too.

## API

**As implemented (2026-08-22), one type rather than the two-phase mirror of the writer originally
sketched below.** The writer's `TimeSeriesWriter` → `write_mesh` → `TimeSeriesDataWriter` split is
forced by a real constraint: writing the mesh is a one-time, irreversible mutation of the file, so
the type system enforcing "mesh written before steps are" earns its keep. Reading has no such
constraint — `TimeSeriesReader::new` already parses the *whole* document, so every read after it
(points, topology, one field of one step) is an independent, repeatable query with nothing to
sequence. Splitting `TimeSeriesReader`/`TimeSeriesDataReader` anyway, purely to match the writer's
shape, would have made reading per-step data require reading the mesh geometry first for no reason
other than imitating a constraint that does not apply — caught in review, not designed in up front.
`read_mesh` is likewise split into `read_points`/`read_topology` rather than filling three buffers
at once: a caller wanting only point positions (or only topology) shouldn't pay for the other, and
the two are independently computable -- points needs only the `Geometry`, while `read_topology`'s
connectivity and cell types are unavoidably produced together by one decode-and-scatter pass and so
stay paired rather than splitting further. Submesh membership (`points_membership`/
`cells_membership`) is computed once in `new`, not by `read_topology`, which is what lets
`read_point_data`/`read_cell_data` work without either mesh method having been called first.

```rust
pub struct TimeSeriesReader { /* parsed light data + base directory + submesh membership */ }

impl TimeSeriesReader {
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self>;

    // Metadata available after `new`, no further heavy data touched
    pub fn num_points(&self) -> usize;
    pub fn num_cells(&self) -> usize;
    pub fn num_steps(&self) -> usize;
    pub fn times(&self) -> &[String];
    pub fn submesh_names(&self) -> &[String];     // empty when the file has no submeshes
    pub fn data_storage(&self) -> Option<DataStorage>;   // informational only, see below

    // Which mesh points/cells a submesh holds, recovered from its own membership (see the
    // reconstruction section below) -- symmetric with what `write_mesh_with_submeshes` takes in.
    pub fn submesh_cells(&self, submesh: usize) -> Result<Vec<usize>>;
    pub fn submesh_points(&self, submesh: usize) -> Result<Vec<usize>>;

    /// Independent of `read_topology`; buffers cleared first, capacity reused.
    pub fn read_points(&self, points: &mut Vec<f64>) -> Result<()>;
    /// Connectivity and cell types are produced together by one decode pass, so they stay paired.
    pub fn read_topology(&self, connectivity: &mut Vec<u64>, cell_types: &mut Vec<CellType>) -> Result<()>;

    /// What is present at a step, so the caller can size buffers and branch on type.
    pub fn point_data_info(&self, step: usize) -> Result<Vec<DataInfo>>;
    pub fn cell_data_info(&self, step: usize) -> Result<Vec<DataInfo>>;

    /// Reads one named attribute of one step into the caller's buffer.
    pub fn read_point_data<T: ValueType>(&self, step: usize, name: &str, into: &mut Vec<T>) -> Result<()>;
    pub fn read_cell_data<T: ValueType>(&self, step: usize, name: &str, into: &mut Vec<T>) -> Result<()>;
}

pub struct DataInfo {
    pub name: String,
    pub attribute: DataAttribute,
    pub number_type: NumberType,
    pub precision: u8,
    pub len: usize,                 // total elements, i.e. num_entities * attribute.size()
}
```

Notes on the shape:

- **`ValueType`.** This is where the sealed `ValueType` trait from the `multiple-features` branch
  earns its place (`03_values_and_f32.md` deliberately did not cherry-pick it). `read_point_data::<f64>`
  into a `Vec<f64>` errors with `NumberTypeMismatch` if the file holds u64, and `DataInfo` is how a
  caller finds out beforehand. Add `impl ValueType for f32` at the same time.
- **Widening is allowed and is not a mismatch.** A `Precision="4"` float dataset read into a
  `Vec<f64>` should succeed (widening f32→f64 loses nothing), as should `u32`-precision integers into
  `Vec<u64>` — the latter is *required*, because that is exactly what the `Binary` backend writes.
  Narrowing (f64 file into `Vec<f32>`) should error rather than silently lose precision; a caller who
  wants that can convert.
- **A convenience whole-step read**, `read_step(step) -> Result<Vec<(String, DataAttribute, Values<'static>)>>`,
  for scripts and for the Python bindings, where the zero-allocation path is less important. Add it
  only once there is a caller (the Python bindings are that caller, in M6) — otherwise it is
  speculative per `CLAUDE.md`.
- **No consuming methods anywhere in this API.** Every method takes `&self`; reading is idempotent
  and repeatable, so nothing needs to be moved or transitioned through a phase.
- **Random access by step index**, not an iterator-only interface — HDF5 and Binary backends can seek
  directly, and the stated use case includes reading one specific step. An `Iterator` adapter over
  steps can be layered on later if wanted; it is not the primitive.

## Reading a mesh written with submeshes (2026-08-22)

A block holds only the points its own cells use, numbered from zero; blocks may overlap; and a mesh
written with `write_mesh_with_submeshes` has no global connectivity of its own. Everything needed to
put the mesh back together is nevertheless in the file. For the HDF5 storages:

| to recover | source |
|---|---|
| the mesh's points, all of them, in the original order | `mesh/points/{0,1,2}` — one array per direction, re-interleave to `[x y z]` triples |
| `num_points` | the `Dimensions` of the source `DataItem` inside any `coords_k_*` item |
| which mesh points block *k* holds | block *k*'s `<Geometry>`: a `HyperSlab` selector's child holds `<start> 1 <len>`; a `Coordinates` selector's child is a `Reference` to the `submesh_points_k` array of indices |
| block *k*'s connectivity, block-local | `connectivity_k` → `mesh/cells/k`, with `TopologyType` / `NodesPerElement` / `NumberOfElements` from its `<Topology>` |
| block *k*'s connectivity, global | `global = points_of_k[local]` — the point list is ascending and a block-local id is a position in it (`submesh_points` sorts, `LocalPoints` numbers by position) |
| which mesh cells block *k* holds | `<Information Name="submesh_cells">`, entry *k*: `<start>:<len>`, or the name of a `submesh_cells_k` `DataItem` |
| block cell *j* ↔ mesh cell | positional: the connectivity is written in the order that list holds the cells, which may deliberately not be ascending |
| cell types | the block's `TopologyType`, or — for `Mixed` — the type code (and, for a poly cell, its point count) prepended per cell inside the connectivity |
| `num_cells` | `1 + max` over all `submesh_cells` entries; sound because `check_all_cells_covered` makes every cell belong to at least one block |

So: allocate `num_points`/`num_cells`, read the global coordinates directly, then walk the blocks
scattering each block's cells to their mesh positions and mapping the block-local point ids back
through the block's point list as they go. Overlapping blocks write the same values twice, which is
why the ids have to be mapped rather than appended.

Three things to be aware of when implementing it:

- **`num_cells` is derived, not stated.** It rests on the writer's coverage invariant, which the
  document does not record. A truncated or hand-edited file therefore reads back as a smaller mesh
  rather than as an error. If that matters, the writer can state it — one
  `<Information Name="mesh" Value="{num_points}:{num_cells}"/>` — and the reader can then *check*
  instead of infer. Decide this when the reader is written; adding it later is a light-data-only
  change, and a reader that treats it as optional stays compatible with files written before it.
- **The block-local ↔ global point convention is a convention.** Nothing in the file says
  "block-local ids are positions in the ascending point list"; it is true by construction. Same for
  the X/Y/Z de-interleaving. Both belong in the reader's own docs.
- **Per-step point data on the HDF5 path is a global array plus per-block selections.** The reader
  should read the source array of the selection — the whole mesh's field, written once — and ignore
  the per-block `HyperSlab`/`Coordinates` items entirely, exactly as it takes the mesh's coordinates
  from the source rather than from the blocks. Cell data is still per block and goes back through
  `submesh_cells`.

For the ascii and binary storages (stage 2) the mesh's coordinates are not written at all: each
block carries a compacted copy of its own points, and the point half of the mapping is the
`<Information Name="submesh_points">` the HDF5 path no longer needs. A point no cell uses is then
absent from the file, and `num_points` has to be derived from the block point lists — so those
storages round-trip the *used* mesh, not necessarily the mesh as passed in. State that limitation in
the docs rather than papering over it. See `09_submesh_references.md` and deviations 7 and 8 in
`04_submeshes.md`.

## Architecture: drive everything off `DataItem`

The important design decision. The reader should **not** branch on `DataStorage`. Every piece of
information needed to locate and decode a heavy-data array is already in the `DataItem` element:
`Format` (XML/HDF/Binary), `Precision`, `NumberType`, `Endian`, `Dimensions`, and the content (inline
text, an `xi:include href`, a relative binary path, or a `file.h5:/group/dataset` reference).

Consequences:

- One `DataReader` trait per *format*, not per `DataStorage` variant — so `Hdf5SingleFile` and
  `Hdf5MultipleFiles` are the same reader (the file path is in the DataItem), and `Ascii` /
  `AsciiInline` differ only in whether the content is inline or an include.
- **Foreign-file support comes almost for free**, because a file written by another tool carries the
  same `DataItem` information. The `Information Name="data_storage"` tag this crate writes becomes
  purely informational — hence `data_storage() -> Option<DataStorage>`.
- A file that mixes formats across DataItems (legal XDMF, and something meshio does) just works.

```
src/reader.rs           TimeSeriesReader, DataInfo, ValueType -- one type, not the two-phase
                         TimeSeriesReader/TimeSeriesDataReader split sketched above, see "API"
                         (`mod.rs` files are a clippy lint in this crate -- see `.clippy.toml`)
src/reader/
    light_data.rs       XML → xdmf_elements structs, reference resolution
    topology.rs         inverse of prepare_cells
    selection.rs        ItemType::HyperSlab / Coordinates -> the source item + the indices, and
                         the general DataItem -> Values dispatcher every other module reads through
                         (as implemented, this subsumed the separate `data_reader.rs`/`DataReader`
                         trait sketched here -- one dispatcher function needed no backend
                         abstraction, since `hdf5_reader::read` is its only leaf so far)
    hdf5_reader.rs      Format::HDF   (cfg(feature = "hdf5"), Unsupported without it) <- stage 1, done
    ascii_reader.rs     Format::XML  (inline text and xi:include)   <- stage 2
    binary_reader.rs    Format::Binary                              <- stage 2
```

## Implementation risks and how to defuse them

### 1. quick-xml + serde deserialization of `DataItem` — test this on day one

`DataItem` uses `#[serde(flatten)]` on a `DataContent` enum whose `Raw` variant is `#[serde(rename =
"$value")]` (`src/xdmf_elements/data_item.rs:34`, `:104`). The structs all derive `Deserialize`
already, but they have only ever been *serialized*. `flatten` combined with `$value` is historically
the fragile corner of quick-xml's serde support.

`DataContent` has since grown a third variant, `Items(Vec<DataItem>)` — the nested children a
`HyperSlab`/`Coordinates` item carries — so the flattened enum now has to choose between `$value`,
`xi:include` and a repeated element. That makes this risk larger than when it was written, and it is
on the critical path for stage 1 rather than a corner.

**First task of the milestone**: deserialize a single hand-written `<DataItem>` of each shape (inline
text, `xi:include` child, reference form, and a `HyperSlab` and a `Coordinates` item with two nested
children each) and confirm it round-trips. If it does not, the fallback is
a hand-rolled `quick_xml::Reader` event loop for `DataItem` only — the rest of the document
(`Xdmf`/`Domain`/`Grid`/`Geometry`/`Topology`/`Attribute`) is plain attributes and children and will
deserialize fine. Budget for the fallback; do not let it surprise the schedule.

### 2. `Reference="XML"` resolution

The writer emits shared mesh DataItems once under `/Xdmf/Domain` and references them from each grid:

```xml
<DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
```

Implement a resolver for exactly this pattern — `/Xdmf/Domain/DataItem[@Name="..."]` — because it is
the only shape this crate produces. Any other XPath expression → `Error::Unsupported`. Do not pull in
an XPath crate for one fixed pattern.

### 3. Inverting `prepare_cells`

`prepare_cells` (`src/time_series_writer.rs:209`) writes `TopologyType::Mixed` connectivity as
interleaved `[cell_type_code, (num_points for poly cells), indices…]`. The reader needs the inverse:
walk the array, map the code back to a `CellType`, skip the extra count for `Vertex`/`Edge`
(`poly_cell_points`), and take `CellType::num_points()` indices.

- The mapping table must be shared with the writer, not duplicated — add a `CellType::from_code(u64)`
  next to the existing discriminants so there is one source of truth.
- **Round-trip test over every `CellType`**, structurally identical to the existing
  `prepare_cells_by_celltype` test (`:587`), which enumerates all 18 types. That test is the template.
- Homogeneous topology types (`Triangle`, `Hexahedron`, …) must also be handled — this crate never
  writes them, but `tests/xdmf_elements.rs` constructs them via the low-level API and foreign files
  use them constantly. Straightforward: no interleaved codes, fixed stride from the topology type.
- **`Polyvertex` is the point-cloud special case.** `write_mesh` with no cells emits a Polyvertex
  topology over `0..num_points` (`:210`). To round-trip, the reader must recognize exactly that
  pattern and return `cell_types = []`, not `num_points` `Vertex` cells. Document the rule; test it.

### 4. Node ordering

`CLAUDE.md`: node ordering follows the VTK convention, not XDMF's historical one, and this is
cross-checked against `vtkio` in `tests/vtk_comparison.rs`. The reader must apply the *same*
convention, so a write→read round-trip is the identity. Foreign files written with XDMF-native
ordering will therefore read back with permuted higher-order nodes — an acknowledged limitation of
"best effort on foreign files", and it should be stated in the reader's docs rather than silently
wrong.

## Explicitly unsupported at 1.0

Each returns `Error::Unsupported { feature, .. }` with a message naming what was found:

- `DataItem` `ItemType` other than `Uniform`, `HyperSlab` and `Coordinates`: `Function`,
  `Collection`, `Tree`. `HyperSlab` and `Coordinates` are **required**, not optional — the crate's
  own HDF5 output uses both, for block geometry and for per-block field data. Only the shapes it
  writes need to be understood: a rank-1 source with a 3-element `<start> <stride> <count>` selector,
  or a rank-1 source with an explicit ascending index list. A rank-2 or strided selector a foreign
  file might carry is `Unsupported`.
- Structured topologies: `2DCoRectMesh`, `3DCoRectMesh`, `2DRectMesh`, `3DRectMesh`, `2DSMesh`, `3DSMesh`.
- `Geometry` types other than `XYZ` and `X_Y_Z`: `XY`, `Origin_DxDyDz`, `Origin_DxDy`. `X_Y_Z` is
  **required**: it is what every block of an HDF5 submesh file uses, since one array per direction is
  what lets X, Y and Z share a single index list (`09_submesh_references.md`).
- `Set` elements, `Tree` grids, multiple `Domain`s.
- `Format="HDF"` when the `hdf5` feature is off — this is `Error::StorageRequiresFeature`, which
  already exists from `01_error_type.md`, and the message must say so clearly since it is a
  compile-time choice manifesting at runtime.
- Big-endian binary data (`Endian="Big"`): actually cheap to support, since `Endian` is right there in
  the DataItem. Do support it — it is a byte-swap in the binary reader and it makes the reader
  meaningfully more robust for files written elsewhere.

## Test matrix

The round-trip suite is the acceptance criterion and the strongest regression net the crate will
have. For each storage mode × each of these cases: write, read, assert equality
(`assert_approx_eq!` for floats per `CLAUDE.md`). **Stage 1 runs the matrix over
`Hdf5SingleFile`/`Hdf5MultipleFiles` only**; stage 2 extends the same table to `Ascii`,
`AsciiInline` and `Binary` rather than writing a second suite, with the one documented difference
that a point no cell uses does not survive those storages.

| case | what it pins |
|------|--------------|
| mesh only | geometry/topology, no temporal collection |
| mesh + point data + cell data, several steps | the common path |
| mixed cell types | `prepare_cells` inversion |
| point cloud (no cells) | the Polyvertex special case |
| every `CellType` | the code→type table |
| f32 attributes (M3) | precision handling |
| u64 attributes | 8-byte integers read back at their declared width (`Binary` refuses them outright, so that row is a rejected write, not a round trip — see `src/paraview.rs`) |
| all `DataAttribute` shapes | including the rank-3 `Matrix` dimension workaround |
| blocks (M4) | block names and membership |
| overlapping, scattered and contiguous submeshes (M4) | the `submesh_cells` mapping in both encodings, the block point lists read out of the `<Geometry>` selectors in both forms, and the local-to-global renumbering |
| a submesh whose cell list is deliberately unordered | that block cell *j* maps positionally, not by sorting |
| a point no cell uses | that the HDF5 round trip is the identity on the points array, unused points included |

Plus a small `tests/reader_fixtures/` directory of hand-written foreign XDMF2 files (e.g. meshio-style
output with a homogeneous topology and a separate HDF5 file) to pin the "common subset" claim, and
one deliberately unsupported file per `Unsupported` category asserting the *specific* error.

## Python

The reader is exposed in M6 alongside the writer. `read_step` returning owned data is the natural
Python shape; reading into a pre-allocated numpy array is the zero-copy shape and is the one worth
having for large data. Detailed in `06_python_bindings.md`.
