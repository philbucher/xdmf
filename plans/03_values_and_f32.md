# M3 — `f32` support and the `Values` type

> **Status (2026-08-16): Parts 1 and 3 are done**, on `main`'s working tree. `Values::F32` exists
> with all four `From` impls, every backend handles it, `dimensions()` was restructured, and
> `write_mesh` is generic over the sealed `Coordinate` trait. Verified per the checklist below:
> unit tests, per-backend round trips, and the ParaView smoke test extended to two fixtures and run
> locally against 5.13.2 and 6.1.1 across all five storage backends (20 fixture checks, all green).
> **Part 2 (`with_reduced_precision`) and the measurement gate are still open** — no size claim has
> been made in the docs, which is exactly what the gate governs.

`README.md`: *"f32 bit floats should be added to `Values`"*. Decision 4 in `ROADMAP.md`: add the
variant **and** an opt-in f64→f32 downcast on write. Not a full numeric type set.

Part 3 (added 2026-08-16) extends this to mesh coordinates: `write_mesh` accepts f32 or f64 points.
It depends on Part 1 and is independent of Part 2.

## Why the downcast is worth having as well

Adding `Values::F32` only helps callers that already hold `f32` data. Most solvers compute in f64,
and for *visualization* the extra precision is dead weight — 7 significant digits is more than anyone
reads off a pressure field. A one-line opt-in that halves the attribute payload is cheap to offer, and
it is the kind of thing that has to be designed in rather than bolted on (it interacts with the
`DataItem` `Precision` attribute and with every writer backend).

### Two corrections to the motivation, from `SESSION_2026-08-08_cfd_benchmark.md`

**1. This is not about matching pyvista.** That session recorded an earlier round where f32 was
proposed as parity with a pyvista optimization — there is no such optimization. pyvista/VTK uses
`float64` for points and attribute data and `int64` for connectivity throughout this path, and never
`float32`. The motivation for f32 here is xdmf's own output size, and nothing else. Recorded so it is
not re-derived from a false premise.

**2. "Halves the bytes" is a claim about raw size, and raw size is not the metric.** The same session
demonstrated, directly, that a tighter packing can produce a *larger* compressed archive: the same
connectivity values stored as `uint32` (320 MB raw) compressed with zstd-6 to 87.3 MB, while stored as
`int64` (640 MB raw) they compressed to 49.4 MB. The `int64` version is mostly zero padding — indices
for a 10M-node mesh need ~24 of 64 bits — and padding compresses away for free, whereas tightly packed
`uint32` bytes all carry real entropy. Packing tighter concentrates entropy; it does not reduce it.

That result does **not** transfer directly to f64→f32, and the reason matters:

- `int64`→`uint32` discards *low-entropy* bytes (predictable zero padding the compressor was already
  handling for free), so the compressed size got worse.
- f64→f32 discards 29 mantissa bits, which for real solver output are the *high-entropy* bytes —
  roundoff noise that no compressor can do anything with, and which specifically defeats the
  `shuffle` filter the HDF5 backend depends on (shuffle wins by grouping same-significance bytes
  across values into runs; the low mantissa bytes never form runs).

So the expectation is that f32 reduces compressed size roughly proportionally or better — the opposite
sign from the `uint32` case. **But that is a hypothesis, and the whole point of the finding above is
that this class of hypothesis must be measured rather than reasoned about.**

### Measurement gate

Before `with_reduced_precision()` is documented as a size optimization, measure it the way
`SESSION_2026-08-08_cfd_benchmark.md` establishes: **compressed final-archive size**, on realistic
(noisy) field data, not raw bytes and not on the smooth synthetic duct. Concretely, extend the M2
benchmark (`02_performance.md` part A) with an f64-vs-f32 row at 1e5 and 1e7 for `Hdf5SingleFile`
(shuffle+deflate) and `Binary` (raw + external codec), and report raw *and* compressed size for each.

The feature ships either way — halving the payload is worth having for the uncompressed `Binary` and
`Ascii` paths regardless, and the precision reduction is what the user asked for. What the measurement
decides is what the docs are allowed to *claim*, and whether the shuffle/deflate defaults from
`02_performance.md` part E need different values for 4-byte data.

## Part 1 — `Values::F32`

```rust
pub enum Values<'a> {
    F64(Cow<'a, [f64]>),
    F32(Cow<'a, [f32]>),
    U64(Cow<'a, [u64]>),
}
```

Touch points:

| File | Change |
|------|--------|
| `src/values.rs` | variant; `From<Vec<f32>>` / `From<&'a [f32]>`; `precision()` → 4; `number_type()` → `Float`; `len()`; `dimensions()` |
| `src/ascii_writer.rs` | `values_to_string` arm |
| `src/binary_writer.rs` | `write_f32_le`, with the bulk `bytemuck` path from `02_performance.md` part D |
| `src/hdf5_writer.rs` | `write_values`: `group.new_dataset::<f32>()` and the matching `data_set.write(v)` arm |

### Simplify `dimensions()` while you are in there

`Values::dimensions()` (`src/values.rs:63`) currently matches on the attribute *and then* on the
variant, duplicating the same `Dimensions(vec![..])` construction per variant — six arms today, nine
after adding `F32`. Every arm only uses `v.len()`. Restructure to compute `let len = self.len();`
once and match on the attribute alone. Adding the variant is what forces this, and it turns a
combinatorial match into a linear one. The rank-3 shape workaround for `Matrix`-typed attributes
(and the comment explaining the VTK 9.6 / ParaView 6.1 reader behaviour) is preserved verbatim —
it is load-bearing.

### `Precision` in the XML

`DataItem.precision` is already driven by `Values::precision(format)`, so `F32` → `4` flows through
automatically for the light data. **Verify in ParaView** that `NumberType="Float" Precision="4"`
loads correctly in every storage mode — this is the one place the change could silently produce a
file that opens but shows garbage. See "Verification" below.

## Part 2 — opt-in f64 → f32 downcast

### Where the switch lives

Not on `DataStorage` (orthogonal to storage), and not per-attribute at the `write_data` call site
(a caller wants "everything at reduced precision for visualization", not per-field decisions, and
per-call would add a parameter to the hottest function in the API).

A builder method on the writer:

```rust
let writer = TimeSeriesWriter::new(path, storage)?.with_reduced_precision();
```

Additive, so it does not break the constructor for existing callers, and it reads at the call site.
Internally a `bool` (or a small `FloatPrecision { Full, Reduced }` enum if a third mode ever appears
— start with the enum only if it costs nothing).

### It applies to attribute data, not to coordinates

Deliberate: f32 coordinates on a large domain produce visible geometric jitter in ParaView, because
the absolute coordinate magnitude eats the mantissa. Attribute values do not have that problem —
nobody notices 7 significant digits on a pressure field. So `with_reduced_precision()` affects
`write_data` only, and `write_mesh` keeps whatever precision the caller passed.

Document that reasoning on the method itself. Coordinates are covered by Part 3 below instead, and
in a better form than "a second opt-in": the caller passes the type they already hold, so no f64
coordinate is ever silently degraded, and the jitter caveat becomes a doc note rather than a reason
to withhold the feature.

### Implementation: no per-step allocation

The conversion happens centrally in `time_series_writer.rs`, before dispatch to the backend, so all
five backends get it from one implementation. The scratch buffer lives on `TimeSeriesDataWriter` and
is reused across attributes and steps:

```rust
// borrow-checker: cannot hold &mut self.scratch while calling &mut self.writer
let mut scratch: Vec<f32> = std::mem::take(&mut self.f32_scratch);
scratch.clear();
scratch.extend(src.iter().map(|&v| v as f32));
let result = self.writer.write_data(name, center, &Values::from(scratch.as_slice()));
self.f32_scratch = scratch;
result?
```

Same `mem::take` pattern as the other scratch buffers in `02_performance.md` part C — use the shared
helper introduced there rather than a third open-coded copy. Steady state: zero allocations, one
`Vec<f32>` sized to the largest attribute.

`v as f32` is a saturating cast in Rust (out-of-range → ±inf, NaN → NaN), which is the right
behaviour for visualization data — do not add a range check and do not error. Worth one sentence in
the docs so it is not a surprise, and one test asserting an `f64::MAX` input becomes `f32::INFINITY`
rather than something implementation-defined.

### Interaction with `Values::U64`

Unchanged: integers are not downcast by this switch. Note that the `Binary` backend *already*
narrows u64→u32 unconditionally (a ParaView reader bug workaround, `src/binary_writer.rs:145`), which
is a separate mechanism and stays separate.

## Part 3 — f32 or f64 points in `write_mesh`

Requested 2026-08-16. Supersedes the "do not add it now" note in Part 2: coordinates become
caller-typed rather than switch-controlled.

### Why it forces Part 1 first

`DataWriter` is a `dyn` trait, so `DataWriter::write_mesh` cannot be generic — points have to reach
the backends as an enum, and that enum is `Values`. So Part 1 is a hard prerequisite, and Part 1
alone makes `point_data`/`cell_data` accept f32 as a side effect (wanted anyway).

### The payoff: points stop being a special case

Once the backend method takes `&Values`, every backend writes points through the **same helper it
already uses for attribute data** — `values_to_string`/`values_to_writer` in `ascii_writer.rs`,
`values_to_writer` in `binary_writer.rs`, `write_values` in `hdf5_writer.rs`. No new match arms
anywhere, and `hdf5_writer::write_mesh`'s hand-rolled points dataset (`src/hdf5_writer.rs:316-326`,
duplicating shape/shuffle/deflate) collapses into the existing `write_values` call. There is
deliberately no unreachable `Values::U64` arm to write an `Error::Internal` for — see the API shape
below, which makes u64 points unrepresentable.

### Public API — sealed `Coordinate` trait (decided 2026-08-16)

```rust
pub fn write_mesh<C: Coordinate>(
    self,
    points: &[C],
    connectivity: &[u64],
    cell_types: &[CellType],
) -> Result<TimeSeriesDataWriter>
```

Sealed, with impls for `f32` and `f64` only and one private method converting `&[Self]` into a
borrowed `Values`. u64 points are a compile error rather than a runtime `InvalidMesh`, which is
what makes the unreachable-arm question above disappear. Existing `&[f64]` and `&[0.0; N]` call
sites keep compiling untouched — an unconstrained float literal still falls back to `f64` before the
trait bound is checked.

The rejected alternative was `points: impl Into<Values<'_>>`, mirroring `point_data`/`cell_data`
exactly and adding zero public surface, but demoting a type mistake to a runtime error. Not
speculative API per `CLAUDE.md`: this request is its caller. Related but separate — the sealed
`ValueType` trait on `multiple-features` stays deferred to M5; unify the two then if they overlap.

### Touch points

| File | Change |
|------|--------|
| `src/values.rs` | the sealed `Coordinate` trait and its two impls |
| `src/lib.rs` | `DataWriter::write_mesh(&mut self, points: &Values<'_>, cells: &[u64])` |
| `src/{ascii,binary,hdf5}_writer.rs` | delegate points to the existing value helper; drop hdf5's duplicate dataset code |
| `src/time_series_writer.rs` | generic `write_mesh`; coords `DataItem` takes `number_type()`/`precision(format)` from the values instead of the hardcoded `NumberType::Float` / `Some(8)` (`:96-105`); `validate_points_and_cells` takes the point count instead of `&[f64]` (it only reads `len()`/`is_empty()`); the two `DummyWriter` mocks (`:1631`, `:1765`) follow the trait |

### Docs on `write_mesh`

f32 halves the geometry payload; f32 coordinates on a large-magnitude domain produce visible jitter
in ParaView because the absolute coordinate eats the mantissa. Same reasoning as Part 2, stated once
here since this is now where a caller chooses.

## Verification

0. **Spike first, before any of the above.** Hand-write a `.xdmf2` with `NumberType="Float"
   Precision="4"` geometry per storage mode and open it with the local pvpython 5.13 and 6.1. This
   is the one way the feature fails silently — a file that opens and shows garbage geometry. See the
   `paraview-install-locations` note for where those live.
1. **Unit tests** in `src/values.rs` mirroring the existing `vec_f64`/`vec_u64` tests for `F32`:
   precision per format, number type, dimensions for each `DataAttribute`, `len`.
2. **Round-trip through each backend**: write f32 attribute data *and* f32 points, read the file
   back (HDF5 via the `hdf5` crate, binary via raw bytes, ascii via text) and compare with
   `assert_approx_eq!` per the `CLAUDE.md` convention. Assert the coords `DataItem` carries
   `Precision="4"` for f32 points and `"8"` for f64, and that the existing f64 call sites compile
   unchanged.
3. **Downcast tests**: f64 input + `with_reduced_precision()` produces the same bytes as the
   equivalent f32 input without it; `DataItem` `Precision="4"` appears in the XML; the saturating-cast
   edge case.
3b. **The measurement gate above** — an f64-vs-f32 row in the M2 benchmark reporting raw *and*
   compressed size on noisy data, before any size claim goes into the docs.
4. **ParaView smoke test — required.** Per `ROADMAP.md`'s verification rules, this changes what bytes
   land in the file, so extend `examples/paraview_smoke.rs` and `expected.json` rather than only
   adding Rust tests. `paraview_smoke.rs` writes one fixture per invocation today; make it write
   **two** — the existing f64 one plus `fixture_<storage>_f32` with f32 coordinates, an f32 point
   attribute and an f32 cell attribute — turn `expected.json` into a list of fixtures, and have
   `verify_with_pvpython.py` loop over it. Keeping both preserves ParaView coverage of the f64
   default, which a single converted fixture would silently drop. This runs across all 10 existing
   matrix jobs at no extra CI cost and needs no workflow change, which is exactly the situation the
   "extend the fixture, not the matrix" rule is for.

## Noted, not scheduled

- **`U32` / `I64` / `I32` / `U8` variants.** A `U32` variant would let callers avoid the binary
  backend's narrowing check entirely, and `U8` would suit cell-type/flag arrays. Rejected for now
  (decision 4): each new variant multiplies the match arms in every writer *and* every reader
  backend. Revisit if a real caller needs one.
- **`ValueType` sealed trait + `as_slice::<T>()`** (present on the `multiple-features` branch): not
  cherry-picked here. Its caller is the reader — see `05_reader.md`. Adding it now would be
  speculative public API per `CLAUDE.md`.
