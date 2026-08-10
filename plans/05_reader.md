# M5 — `TimeSeriesReader`

`README.md`: *"TimeSeriesReader: A draft is in `reader.rs`, might need to be adjusted. API should be
similar to the [writer]. This obviously requires the implementation of the readers for different
formats."*

Decision 2 in `ROADMAP.md`: **guarantee round-tripping of this crate's own output; make a best effort
on a common subset of foreign XDMF2; reject everything else with an explicit `Unsupported` error.**

The sketch that was in `reader.rs` at the repo root is folded into this document — delete that file.

## Prerequisites (do these before writing any reader code)

Two of them, both small, both cheaper now than retrofitted:

1. **The `DataItem` deserialization fix** — done. Risk 1 below has landed: `DataItem` now has
   `text`/`include` fields instead of the flattened `DataContent`, and it deserializes.
2. **The `Values::F32` + sealed `ValueType` slice of M3 — done.** `Values` gained an `F32(Cow<'a,
   [f32]>)` variant, all three writer backends handle it, and the sealed `ValueType` trait
   (`f64`/`f32`/`u64`, cherry-picked from the `multiple-features` branch) plus `Values::as_slice`/
   `as_mut_slice` are in. **`ValueType` itself is deliberately not re-exported from `lib.rs` yet** —
   `as_slice`/`as_mut_slice` are inherent methods on `Values`, so `values.as_slice::<f64>()` compiles
   for an external caller without the trait ever being in scope, and nothing today needs to name it
   otherwise. Export it once `read_point_data<T: ValueType>` (below) actually lands and a caller
   needs to write that bound themselves. `dimensions()` was restructured to compute `len` once and
   match on the attribute alone, per `03_values_and_f32.md`. `ValuesMut` (the mutable mirror
   `DataReader::read` will take) is still to do — it belongs to the reader trait itself, not to this
   slice. The rest of M3 (the opt-in f64→f32 downcast on write, ParaView verification) is not needed
   by the reader and stays in its own milestone.

M2 and M4 do *not* block this milestone. Decision 6 keeps the `.xdmf2` a complete, openable file
after every step, so the append+patch-tail rewrite does not change what the reader parses; blocks are
additive and are called out where they matter below.

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
    /// Addressed by index, not by name — see "the borrow problem" below.
    pub fn num_point_data(&self, step: usize) -> Result<usize>;
    pub fn num_cell_data(&self, step: usize) -> Result<usize>;
    pub fn point_data_info(&self, step: usize, index: usize) -> Result<&DataInfo>;
    pub fn cell_data_info(&self, step: usize, index: usize) -> Result<&DataInfo>;

    /// Name → index lookup, for callers that know what they are after.
    pub fn point_data_index(&self, step: usize, name: &str) -> Result<usize>;
    pub fn cell_data_index(&self, step: usize, name: &str) -> Result<usize>;

    /// Reads one attribute of one step into the caller's buffer.
    pub fn read_point_data<T: ValueType>(&mut self, step: usize, index: usize, into: &mut Vec<T>) -> Result<()>;
    pub fn read_cell_data<T: ValueType>(&mut self, step: usize, index: usize, into: &mut Vec<T>) -> Result<()>;
}

pub struct DataInfo {
    pub name: String,
    pub attribute: DataAttribute,
    pub kind: ValueKind,
    pub len: usize,                 // total elements, i.e. num_entities * attribute.size()
}

/// The element type a caller can ask for, mirroring `Values`' variants.
pub enum ValueKind { F32, F64, U64 }
```

Notes on the shape:

- **`ValueType`.** This is where the sealed `ValueType` trait from the `multiple-features` branch
  earns its place (`03_values_and_f32.md` deliberately did not cherry-pick it). `read_point_data::<f64>`
  into a `Vec<f64>` errors if the file holds u64, and `DataInfo` is how a caller finds out
  beforehand. Add `impl ValueType for f32` at the same time.
- **The generic cannot reach the backend trait.** `read_point_data<T: ValueType>` is a generic
  method; `DataReader` is used as `Box<dyn DataReader>`, so a generic method on it is not
  object-safe and will not compile. The generic therefore lives **only** in the public
  `TimeSeriesDataReader` methods, which convert `&mut Vec<T>` into a `ValuesMut<'_>` enum — the
  mutable mirror of `Values`, i.e. `F64(&mut Vec<f64>)` / `F32(&mut Vec<f32>)` / `U64(&mut Vec<u64>)` —
  and that enum is what `DataReader::read` takes. Decide this before writing the trait; it is the
  signature everything else hangs off.
- **The borrow problem, and why reads are addressed by index.** The obvious API,
  `point_data_info(&self, step) -> Result<&[DataInfo]>` plus `read_point_data(&mut self, …, name: &str, …)`,
  does not compile for the caller: iterating the info slice holds a shared borrow of the reader
  across a call that needs `&mut`, and a `name: &str` taken from a `DataInfo` extends that borrow
  further. Addressing both the info query and the read by `usize` ends each borrow before the next
  call, keeps the loop allocation-free, and costs only a `point_data_index(step, name)` lookup for
  callers who think in names.
- **No `data_storage()` accessor.** The plan previously carried a `data_storage() -> Option<DataStorage>`
  as "informational only". Nothing calls it, so it is speculative API per `CLAUDE.md`. The
  `Information Name="data_storage"` tag stays in the file and stays ignored by the reader; add the
  accessor if M6 turns out to want it.
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
  what makes the API "similar to the writer" as `README.md` asks. Note the asymmetry it introduces,
  though: the writer *must* write a mesh, the reader does not have to read one. Re-reading step data
  for a mesh the caller already holds is a plausible hot-path case, and this shape forces decoding
  points and connectivity to get at any data. An `into_data_reader(self)` that skips the mesh is the
  answer if that case shows up — do not add it before it does, but do not design it out either.
- **Random access by step index**, not an iterator-only interface — HDF5 and Binary backends can seek
  directly, and the stated use case includes reading one specific step. An `Iterator` adapter over
  steps can be layered on later if wanted; it is not the primitive.

## Errors: two variants have to be added first

This plan was written assuming `Error::Unsupported` and `Error::StorageRequiresFeature` had landed
with M1. **Neither exists.** The merged enum (`src/error.rs`, PR #20) is `Io`, `Hdf5`,
`InvalidFileName`, `InvalidConfiguration`, `InvalidMesh`, `InvalidTimeStep`, `InvalidData`,
`IntegerTooLargeForBinary`, `Internal`. Every "returns `Error::Unsupported`" in this document is
therefore a *to-do*, not a reference.

What the reader actually needs:

- **`Unsupported { reason: String }`** — new variant. It earns its place by the `CLAUDE.md` rule for
  `IntegerTooLargeForBinary`: it is a failure a caller genuinely reacts to (the file is valid XDMF,
  this crate just will not read it, so fall back to another tool) rather than one they only report.
- **`InvalidFile { path: PathBuf, reason: String }`** — new variant, for a malformed or
  self-inconsistent file: XML that does not parse, a `DataItem` with no `Dimensions`, a
  `Dimensions`/heavy-data length mismatch, an unresolvable `Reference`, a truncated `.bin`.
  `InvalidMesh` and `InvalidData` both mean *the caller passed something bad*, which is the wrong
  story to tell about a file the caller merely opened. Keeping them separate is what makes the
  variant match in a test meaningful.
- **`Format="HDF"` with the `hdf5` feature off** goes to `InvalidConfiguration`, not to a variant of
  its own — that is where `create_writer` already puts the mirror-image case, and the symmetry is
  worth more than the precision. The `reason` must say plainly that this is a compile-time feature
  choice showing up at runtime.

That takes the enum to 11 variants, past the "under 10" guideline in `CLAUDE.md`'s Errors section.
Update that sentence as part of this milestone rather than letting the guideline erode silently —
the grouping principle (by category, `reason: String` over per-failure fields) is the part that
matters and it is unchanged.

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
  purely informational, and the reader ignores it entirely.
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

### 1. `DataItem` does not deserialize — DONE (2026-08-10)

This was risk 1 as a hypothetical, then it was measured and confirmed real, and it is now fixed and
merged: `DataItem` carries `text: Option<String>` / `include: Option<XInclude>` instead of a
flattened `DataContent`, both directions are covered by tests in `data_item.rs`, and the full suite
(`cargo nextest run`, with and without `hdf5`) plus clippy (both feature sets) and doctests are green.
`DataContent` still exists, but only as a `pub(crate)` writer-internal helper (`Raw`/`Include`) with
an `into_parts()` that splits into the two `DataItem` fields at the three call sites in
`time_series_writer.rs`; the writer backends (`ascii_writer.rs`/`binary_writer.rs`/`hdf5_writer.rs`)
are unchanged, since their trait-level return type (`DataContent`) didn't need to move.

What follows is kept as the record of *why*: it was measured and real, before the fix. `DataItem`
used `#[serde(flatten)]` on a `DataContent` enum whose `Raw` variant was
`#[serde(rename = "$value")]`, and that combination was write-only. Every shape failed with the same
error:

```
no variant of enum DataContent found in flattened data
```

— inline text, `xi:include`, `Reference="XML"`, the `file.h5:/path` form, and whole `.xdmf2`
documents produced by all four storage backends.

**The hand-rolled event-loop fallback is not needed.** Replacing `DataContent` with two plain
optional fields on `DataItem` makes both directions work, with byte-identical serialization
(verified by re-serializing each parsed form and comparing to the original string):

```rust
#[serde(rename = "$text", skip_serializing_if = "Option::is_none", default)]
pub text: Option<String>,

#[serde(rename(serialize = "xi:include", deserialize = "include"),
        skip_serializing_if = "Option::is_none", default)]
pub include: Option<XInclude>,
```

The split `rename` is the non-obvious part and the thing to not "clean up" later: quick-xml's
**serializer** needs the literal `xi:include` to emit the namespace prefix, while its
**deserializer** strips the prefix and reports the field as `include`. With a single name, one
direction silently no-ops — the child parses into nothing and no error is raised. Confirmed by
probing with `#[serde(deny_unknown_fields)]`, which reports ``unknown field `include` ``. Multi-line
inline ASCII content survives the round trip as-is (`"\n0 1 2\n3 4 5\n"`), so the ASCII reader must
`split_whitespace()` rather than assume a single line.

What actually landed, slightly leaner than the original sketch: `DataContent` did not disappear —
it stayed as the writer trait's crate-private return type (`Raw`/`Include`, no longer `Serialize`/
`Deserialize`, no longer `pub`), since `ascii_writer.rs`/`binary_writer.rs`/`hdf5_writer.rs` already
had a natural need for a "raw text or file reference" return value and rewriting all three backends'
signatures bought nothing. Only `DataItem`'s own fields changed; a new `DataContent::into_parts()`
converts to `(Option<String>, Option<XInclude>)` at the three call sites in `time_series_writer.rs`
that build a `DataItem` from a writer's return value. Mutually-exclusive-in-practice is not enforced
by the type system any more — a `DataItem` with both `text` and `include` set is constructible. That
is acceptable: the writers never build one, and the reader must treat `include` as taking precedence
when both are present.

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
- `NumberType` other than `Float` and `UInt`. `Int`, `Char` and `UChar` are all legal XDMF and all
  appear in foreign files, and `Values` cannot represent any of them. `Int` in particular must be
  rejected rather than widened into `U64` — the widening is wrong for exactly the values (negative
  ones) that motivate using a signed type in the first place.
- `Format="HDF"` when the `hdf5` feature is off — `Error::InvalidConfiguration`, per the Errors
  section above (there is no `StorageRequiresFeature` variant, and `create_writer` already routes the
  mirror-image case here). The message must say plainly that this is a compile-time feature choice
  manifesting at runtime.
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
