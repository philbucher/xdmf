# M3 — `f32` support and the `Values` type

`README.md`: *"f32 bit floats should be added to `Values`"*. Decision 4 in `ROADMAP.md`: add the
variant **and** an opt-in f64→f32 downcast on write. Not a full numeric type set.

> **Part 1 DONE (2026-08-10)** — landed early as an M5 prerequisite (`05_reader.md`'s "Prerequisites"
> section), not in milestone order. `Values::F32`, the writer-backend arms, and the sealed
> `ValueType` trait (cherry-picked from `origin/multiple-features`, contrary to the note below that
> it deliberately wasn't) are all in — see the `2026-08-10` entry in `ROADMAP.md`'s Progress log.
> **Part 2, the opt-in f64→f32 downcast and its measurement gate, is still unstarted**; everything
> below about it is still the plan to execute.

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
`write_data` only, and `write_mesh` keeps f64 coordinates.

Document that reasoning on the method itself. If someone later wants f32 coordinates for a small,
origin-centred domain, that is a *second*, separately named opt-in — do not add it now (no caller).

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

## Verification

1. **Unit tests** in `src/values.rs` mirroring the existing `vec_f64`/`vec_u64` tests for `F32`:
   precision per format, number type, dimensions for each `DataAttribute`, `len`.
2. **Round-trip through each backend**: write f32 attribute data, read the file back (HDF5 via the
   `hdf5` crate, binary via raw bytes, ascii via text) and compare with `assert_approx_eq!` per the
   `CLAUDE.md` convention.
3. **Downcast tests**: f64 input + `with_reduced_precision()` produces the same bytes as the
   equivalent f32 input without it; `DataItem` `Precision="4"` appears in the XML; the saturating-cast
   edge case.
3b. **The measurement gate above** — an f64-vs-f32 row in the M2 benchmark reporting raw *and*
   compressed size on noisy data, before any size claim goes into the docs.
4. **ParaView smoke test — required.** Per `ROADMAP.md`'s verification rules, this changes what bytes
   land in the file, so extend `examples/paraview_smoke.rs` and `expected.json` rather than only
   adding Rust tests: add an f32-valued point attribute and an f32-valued cell attribute to the
   fixture, and assert their values in `verify_with_pvpython.py`. This runs across all 10 existing
   matrix jobs at no extra CI cost, which is exactly the situation the "extend the fixture, not the
   matrix" rule is for.

## Noted, not scheduled

- **`U32` / `I64` / `I32` / `U8` variants.** A `U32` variant would let callers avoid the binary
  backend's narrowing check entirely, and `U8` would suit cell-type/flag arrays. Rejected for now
  (decision 4): each new variant multiplies the match arms in every writer *and* every reader
  backend. Revisit if a real caller needs one.
- **`ValueType` sealed trait + `as_slice::<T>()`** (present on the `multiple-features` branch): the
  reasoning below turned out to not apply. Originally deferred as speculative public API per
  `CLAUDE.md` with "its caller is the reader", but `05_reader.md`'s prerequisites made it a named,
  written-down dependency of M5 before any reader code exists — a real, if not-yet-written, caller —
  so it was cherry-picked in on 2026-08-10 alongside `Values::F32`. See the DONE callout at the top
  of this file.
