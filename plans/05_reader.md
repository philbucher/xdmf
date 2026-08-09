# M5 — `TimeSeriesReader`

`README.md`: *"TimeSeriesReader: A draft is in `reader.rs`, might need to be adjusted. API should be
similar to the [writer]. This obviously requires the implementation of the readers for different
formats."*

Decision 2 in `ROADMAP.md`: **guarantee round-tripping of this crate's own output; make a best effort
on a common subset of foreign XDMF2; reject everything else with an explicit `Unsupported` error.**

The sketch that was in `reader.rs` at the repo root is folded into this document — delete that file.

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

Mirrors the writer's two-phase shape (`TimeSeriesWriter` → `write_mesh` → `TimeSeriesDataWriter`):

```rust
pub struct TimeSeriesReader { /* parsed light data + base directory */ }

impl TimeSeriesReader {
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self>;

    // Metadata available before any heavy data is touched
    pub fn num_points(&self) -> usize;
    pub fn num_cells(&self) -> usize;
    pub fn times(&self) -> &[String];
    pub fn block_names(&self) -> &[String];     // empty when the file has no blocks
    pub fn data_storage(&self) -> Option<DataStorage>;   // informational only, see below

    /// Fills the caller's buffers (cleared first, capacity reused).
    pub fn read_mesh(
        self,
        points: &mut Vec<f64>,
        connectivity: &mut Vec<u64>,
        cell_types: &mut Vec<CellType>,
    ) -> Result<TimeSeriesDataReader>;
}

pub struct TimeSeriesDataReader { /* .. */ }

impl TimeSeriesDataReader {
    pub fn num_steps(&self) -> usize;
    pub fn times(&self) -> &[String];

    /// What is present at a step, so the caller can size buffers and branch on type.
    pub fn point_data_info(&self, step: usize) -> Result<&[DataInfo]>;
    pub fn cell_data_info(&self, step: usize) -> Result<&[DataInfo]>;

    /// Reads one named attribute of one step into the caller's buffer.
    pub fn read_point_data<T: ValueType>(&mut self, step: usize, name: &str, into: &mut Vec<T>) -> Result<()>;
    pub fn read_cell_data<T: ValueType>(&mut self, step: usize, name: &str, into: &mut Vec<T>) -> Result<()>;
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
- **`TimeSeriesReader::read_mesh` consumes `self`**, matching `TimeSeriesWriter::write_mesh`. This is
  what makes the API "similar to the writer" as `README.md` asks.
- **Random access by step index**, not an iterator-only interface — HDF5 and Binary backends can seek
  directly, and the stated use case includes reading one specific step. An `Iterator` adapter over
  steps can be layered on later if wanted; it is not the primitive.

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
src/reader/
    mod.rs              TimeSeriesReader, TimeSeriesDataReader, DataInfo
    light_data.rs       XML → xdmf_elements structs, reference resolution
    topology.rs         inverse of prepare_cells
    data_reader.rs      trait DataReader { fn read(&mut self, item: &DataItem, into: &mut ..) }
    ascii_reader.rs     Format::XML  (inline text and xi:include)
    binary_reader.rs    Format::Binary
    hdf5_reader.rs      Format::HDF   (cfg(feature = "hdf5"))
```

## Implementation risks and how to defuse them

### 1. quick-xml + serde deserialization of `DataItem` — test this on day one

`DataItem` uses `#[serde(flatten)]` on a `DataContent` enum whose `Raw` variant is `#[serde(rename =
"$value")]` (`src/xdmf_elements/data_item.rs:34`, `:104`). The structs all derive `Deserialize`
already, but they have only ever been *serialized*. `flatten` combined with `$value` is historically
the fragile corner of quick-xml's serde support.

**First task of the milestone**: deserialize a single hand-written `<DataItem>` of each shape (inline
text, `xi:include` child, reference form) and confirm it round-trips. If it does not, the fallback is
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

- `DataItem` `ItemType` other than the default `Uniform`: `HyperSlab`, `Function`, `Coordinates`,
  `Collection`, `Tree`. (If `04_submeshes.md` stage B lands, `HyperSlab` moves out of this list.)
- Structured topologies: `2DCoRectMesh`, `3DCoRectMesh`, `2DRectMesh`, `3DRectMesh`, `2DSMesh`, `3DSMesh`.
- `Geometry` types other than `XYZ`: `XY`, `X_Y_Z`, `Origin_DxDyDz`, `Origin_DxDy`.
- `Set` elements, `Tree` grids, multiple `Domain`s.
- `Format="HDF"` when the `hdf5` feature is off — this is `Error::StorageRequiresFeature`, which
  already exists from `01_error_type.md`, and the message must say so clearly since it is a
  compile-time choice manifesting at runtime.
- Big-endian binary data (`Endian="Big"`): actually cheap to support, since `Endian` is right there in
  the DataItem. Do support it — it is a byte-swap in the binary reader and it makes the reader
  meaningfully more robust for files written elsewhere.

## Test matrix

The round-trip suite is the acceptance criterion and the strongest regression net the crate will
have. For each of the five storage modes × each of these cases: write, read, assert equality
(`assert_approx_eq!` for floats per `CLAUDE.md`).

| case | what it pins |
|------|--------------|
| mesh only | geometry/topology, no temporal collection |
| mesh + point data + cell data, several steps | the common path |
| mixed cell types | `prepare_cells` inversion |
| point cloud (no cells) | the Polyvertex special case |
| every `CellType` | the code→type table |
| f32 attributes (M3) | precision handling |
| u64 attributes | the Binary backend's u32 narrowing widening back correctly |
| all `DataAttribute` shapes | including the rank-3 `Matrix` dimension workaround |
| blocks (M4) | block names and membership |

Plus a small `tests/reader_fixtures/` directory of hand-written foreign XDMF2 files (e.g. meshio-style
output with a homogeneous topology and a separate HDF5 file) to pin the "common subset" claim, and
one deliberately unsupported file per `Unsupported` category asserting the *specific* error.

## Python

The reader is exposed in M6 alongside the writer. `read_step` returning owned data is the natural
Python shape; reading into a pre-allocated numpy array is the zero-copy shape and is the one worth
having for large data. Detailed in `06_python_bindings.md`.
