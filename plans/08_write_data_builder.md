# M7 — per-attribute `write_data` builder

`README.md`: *"Temporary allocations should be avoided as much as possible, maybe this requires
changing the API (e.g. passing pre-allocated vectors so that they can be reused during writing of
steps)"*.

`02_performance.md` part F declares the API surface of buffer reuse **done** — a caller keeps one
`Vec<f64>` **per field** alive for the whole run and passes `buf.as_slice().into()`. This plan closes
the remaining half: reuse of **one** buffer across the fields of a single step, which the current
call shape makes impossible.

It also removes `Values` from the signature a normal Rust caller has to type, which is the
longer-standing complaint.

> **Status: IMPLEMENTED 2026-08-15**, on `main` rather than on `reader` — deliberately, so the
> shape could be explored before `reader` lands. Deviations from the design below, all decided
> during implementation:
>
> - **The step-finishing method is `write()`, not `commit()` as designed below.** "Commit" implied
>   transactional semantics the crate does not provide: heavy data is already on disk by then, so
>   nothing rolls back — only the light data is deferred. `write` also matches the crate's existing
>   vocabulary (`write_mesh`, `point_data`).
>   The private whole-document serializer `TimeSeriesDataWriter::write()` was renamed to
>   `write_xdmf_file()` to keep the two distinguishable.
> - **`Values` was NOT widened.** Built against `main`'s two-variant `Values` (`F64`/`U64`). The
>   builder is agnostic to the variant count — `impl Into<Values<'a>>` is unchanged either way —
>   so this stays `reader`'s work and is not duplicated here. See "Prerequisites".
> - **`write_data_initialize` is deferred to the first attribute**, rather than running eagerly in
>   `time_step()`. This dissolves the `MultipleFilesHdf5Writer` orphan-file question instead of
>   answering it: a step that writes nothing never creates a per-step file, so there is nothing to
>   clean up. A step that writes at least one attribute and is then abandoned still leaves
>   unreferenced heavy data, which is the pre-existing behaviour for a mid-write failure.
> - `From<&Vec<T>>` and `From<&[T; N]>` impls were appended to `values.rs` (see the gotcha below).
>   Kept deliberately append-only, since `reader` rewrites that file into a macro — reconciling is
>   "add two lines inside the macro".
>
> Verified: `cargo nextest run` (129 tests, and 116 with `--no-default-features`), `--release`,
> `cargo test --doc`, clippy with `-D warnings` on both feature sets, `cargo doc` with
> `-D warnings -D missing_docs`, `cargo +nightly fmt --check`. The `test_write_data_preserve_order`
> golden XML is byte-identical. All five storage backends were regenerated through
> `examples/paraview_smoke.rs` and verified in ParaView 6.1.1 via `verify_with_pvpython.py`.

## Where this sits in the roadmap

Between M5 (reader) and M6 Part 2 (wheels). It is a breaking change to the Rust writer API, so per
the ROADMAP's "M6 last" constraint it must land **before** the wheels go out. M6 Part 1 (the
bindings themselves) has already landed on `reader`, so this milestone includes updating them —
that work is small and is described under "Python bindings" below.

Sequence it **together with `02_performance.md` part B**, not independently. The two changes want
the same shape and doing them separately means building the step-fragment machinery twice; see
"Interaction with part B".

## Prerequisites

None, as built — see the status note. The design below was originally written assuming the `reader`
branch's widened `Values`, but the builder turned out to be independent of it: `impl Into<Values<'a>>`
is the same signature whether `Values` has two variants or six, and the widening touches `values.rs`
plus the backends' match arms, which the builder does not.

- The widening stays `reader`'s work: `F64/F32/U64/U32/I64/I32`, macro-generated, plus the sealed
  `ValueType` trait. Re-doing it on `main` would create a second implementation to reconcile rather
  than avoiding a conflict.
- Note this widening drifted past **ROADMAP decision 4** (*"Add `F32` … not a full numeric type
  set"*). The decision table should be updated to record what was actually built, so it does not
  read as still-open later.

---

## The problem

`write_data` takes both attribute lists up front:

```rust
pub fn write_data<'a>(
    &mut self,
    time: &str,
    point_data: impl IntoIterator<Item = (&'a str, DataAttribute, Values<'a>)>,
    cell_data:  impl IntoIterator<Item = (&'a str, DataAttribute, Values<'a>)>,
) -> Result<()>
```

Three consequences:

1. **Every buffer in a step must be simultaneously live.** A solver with one scratch array cannot
   fill-and-write, fill-and-write; it needs one allocation per field. This is precisely the
   allocation pattern `README.md` asks to remove, and part F cannot fix it without changing the call
   shape.
2. **`Values` is unavoidable in caller code.** Every call site carries `.into()` noise
   (`tests/time_series_writer.rs:60-90` is six consecutive tuples of it).
3. **The list cannot be made generic to fix (2).** A `write_data<T: ValueType>(…, &[T])` would force
   one element type per list, and `examples/paraview_smoke.rs:131-145` already mixes `u64`
   (`region_id`) and `f64` (`stress`) in the same `cell_data`. Splitting into two calls is not
   available either: a repeated time is rejected by `validate_data` (`src/time_series_writer.rs:436`,
   `Error::InvalidTimeStep`). So the tuple-list shape and per-call genericity are mutually exclusive.

The backends are already ready for the fix. Every `DataWriter::write_data` implementation fully
consumes its input before returning — HDF5 creates and writes the dataset (`src/hdf5_writer.rs:121`),
ascii/binary create-write-flush a per-attribute file (`src/ascii_writer.rs:128`), and
`AsciiInlineWriter` formats to a `String` immediately. **Nothing retains a borrow.** The only thing
forcing simultaneous liveness is the public signature.

---

## The design

A step becomes a short-lived builder borrowing the writer:

```rust
impl TimeSeriesDataWriter {
    pub fn time_step(&mut self, time: &str) -> Result<TimeStep<'_>>;
}

impl TimeStep<'_> {
    pub fn point_data<'a>(&mut self, name: &str, attribute: DataAttribute, data: impl Into<Values<'a>>) -> Result<()>;
    pub fn cell_data <'a>(&mut self, name: &str, attribute: DataAttribute, data: impl Into<Values<'a>>) -> Result<()>;
    pub fn write(self) -> Result<()>;
}
```

Usage, with the reuse this exists for:

```rust
let mut buf = vec![0.0; num_points * 3];
let mut step = ts_writer.time_step("0.1")?;

fill_velocity(&mut buf);
step.point_data("velocity", DataAttribute::Vector, &buf[..])?;
fill_displacement(&mut buf);                                   // same allocation
step.point_data("displacement", DataAttribute::Vector, &buf[..])?;
step.cell_data("region_id", DataAttribute::Scalar, &region_ids[..])?;

step.write()?;
```

Each `point_data`/`cell_data` writes its heavy data before returning, so the borrow ends at the
call. `write` is the light-data commit point.

### Why `impl Into<Values<'a>>` and not `<T: ValueType>(data: &[T])`

Both were considered. `impl Into<Values<'a>>` wins because it serves the runtime-typed caller
(the pyo3 bridge, and any Rust caller whose dtype comes from a config file) with the *same* method —
`Values` satisfies its own `Into` via the blanket `impl<T> From<T> for T`, so the bridge passes
`guard.to_values()?` directly and needs no dispatch at all.

**Implementation gotcha, do not skip:** deref coercion does **not** fire through a generic `Into`
bound. `step.point_data("v", Vector, &buf)` with `buf: Vec<f64>` will not compile — the compiler
must choose `T = &Vec<f64>` and look for `From<&Vec<f64>> for Values`, rather than coercing to
`&[f64]`. Two options, pick one and be consistent:

- add `impl<'a> From<&'a Vec<$ty>> for Values<'a>` (and `From<&'a [$ty; N]>`, const-generic) to the
  existing `values!` macro — four lines per arm, and `&buf` then works; **or**
- require `&buf[..]` / `buf.as_slice()` at call sites and document it.

Prefer the first: the whole point of the milestone is removing call-site noise, and trading `.into()`
for `[..]` is not much of a win.

The rejected alternative, `<T: ValueType>(data: &[T])`, has better bare ergonomics (deref coercion
*does* work through a generic slice parameter, so `&buf` is free) but forces every runtime-typed
caller into a six-arm match. That is a real cost paid by the bindings and it buys only the `[..]`.

### `Values` stays `pub`

It is not merely a crate-split artifact. `TimeSeriesWriter` holds `Box<dyn DataWriter>`, and generic
methods are not object-safe, so a type-erased value **must** cross that boundary — `Values` is the
monomorphization sink, and the generic public layer funnels into it. On the `reader` branch it is
additionally the declared input type of `write_mesh`.

So the goal is *not* to hide it. The goal is that callers never have to **write** it, which
`impl Into<Values<'a>>` achieves. Keep it public, documented as the runtime-typed escape hatch, and
keep it out of the README/doctests/examples so it stops being the first thing a new user meets.

---

## Interaction with `02_performance.md` part B

Part B replaces the full-document rewrite with append-a-fragment-and-patch-the-tail, and states that
**"a failed step must leave no trace"** — the fragment is built in memory and written only on
success. That is exactly a step builder:

- `time_step()` validates the time and calls `DataWriter::write_data_initialize`.
- Each `point_data`/`cell_data` writes heavy data and pushes one `attribute::Attribute` onto the
  step's own small `Vec`.
- `write()` serializes that step's `<Grid>` fragment plus the tail into one `Vec<u8>`, issues the
  single `write_all` part B specifies, and calls `write_data_finalize`.

`TimeSeriesDataWriter::attributes` (the unbounded `Vec<(String, Vec<Attribute>)>` retaining every
attribute of every step) disappears, replaced by a per-step buffer that dies at `write`. Part B
wanted that anyway; here it falls out of the type.

"A failed step leaves no trace" also becomes literally true for the light data, because a `TimeStep`
that is never committed never appends anything.

### Naming

Part B introduces `finish(self) -> Result<()>` on **`TimeSeriesDataWriter`** (flushing the appended
XML; `06_python_bindings.md` item 7 maps it to `close()`/`__exit__`). The step-level method must
therefore not also be called `finish`. `write()` is used throughout this document — it is accurate
(it is the point at which the step's fragment is appended) and unambiguous against the writer-level
`finish()`.

---

## Things to get right

- **An abandoned `TimeStep`.** Dropping without `write` must leave the writer usable: the backends
  reject a second `write_data_initialize` (`Error::Internal("writing data was already
  initialized")`, `src/binary_writer.rs`, `src/hdf5_writer.rs:135`), so `Drop` has to call
  `write_data_finalize`. `Drop` cannot report errors — swallow it there, and mark `TimeStep`
  `#[must_use]` so the common mistake is caught at compile time instead. Test that a dropped step is
  followed by a working `time_step()` for the same *and* a different time.
- **`MultipleFilesHdf5Writer` creates a file eagerly** in `write_data_initialize`
  (`src/hdf5_writer.rs:249`), so an abandoned step leaves an empty `data_t_{time}.h5`. Decide
  explicitly: delete it on abandon, or document it as harmless (it is unreferenced by the XML).
  Prefer deleting — an orphan file that ParaView ignores is still a file the user has to explain.
- **Validation moves, and must not be lost.** Current `validate_data`/`collect_data`
  (`src/time_series_writer.rs:436`/`:501`) split three ways:

  | check | new home |
  |---|---|
  | time parses; time not already written | `time_step()`, **before** `write_data_initialize` touches disk |
  | size vs. `num_points`/`num_cells`; name charset; `DataWriter::validate_values` (binary's u64→u32 range) | the individual `point_data`/`cell_data` call |
  | duplicate attribute name | running `HashSet` on the `TimeStep` |
  | "at least one of point_data or cell_data must be provided" | `write()` |

  Every existing error variant and `reason` string keeps its meaning; only the call that raises it
  moves. The `src/error.rs` `mod error_messages` tests are unaffected.
- **`validate_values` up front is no longer required, and that is fine.** `lib.rs:106-112` justifies
  the pre-pass by "would otherwise leave the writer poisoned". What it actually protects is the
  `write_data_initialize`/`write_data_finalize` pairing, which the builder now guarantees
  structurally (via `write` and `Drop`). Update that doc comment rather than leaving it asserting a
  rationale that no longer holds.
- **Error atomicity is unchanged, not weakened.** Today a failure mid-`write_attributes` already
  leaves earlier attributes on disk, because `attributes.push` and `self.write()` both run *after*
  `write_result?`. The XML is the commit point either way.
- **Attribute ordering.** The builder naturally preserves call order and permits interleaving
  point/cell attributes. XDMF is fine with it (each `<Attribute>` carries its own `Center`) and HDF5
  grouping is by `center_to_data_tag` regardless. To keep part B's byte-identical golden-file
  criterion, **migrate the existing tests to call all `point_data` before all `cell_data`**, which
  reproduces today's output exactly. Do not reorder internally to force it.
- **`Cow` in `Values` stops being load-bearing.** With immediate writes the borrowed variant is
  always sufficient. Keep `Cow` anyway — `From<Vec<T>>` is a genuine convenience for a caller
  passing a temporary, and dropping it is a gratuitous extra breaking change in a milestone that
  already has one. Revisit separately if it ever gets in the way.
- **GIL granularity** (bindings): `py.detach` moves from once-per-step to once-per-attribute. Finer
  is better here; the acquire/release cost is nanoseconds against a heavy-data write.

---

## Python bindings

The bindings get **simpler**, not harder. `python/src/writer.rs` currently needs a two-phase dance:

```rust
let point_named  = extract_guards(point_data)?;   // keep every PyReadonlyArrayDyn alive...
let point_values = build_values(&point_named)?;   // ...so Values can borrow from them
```

That split, and the `'g` lifetime on `build_values`, exist *only* because the Rust API needs every
array simultaneously live. With the builder each call is extract-one-guard → `to_values()` → write →
drop guard, so **both helpers and the lifetime disappear**. Zero-copy is untouched: `NumpyArray`
still holds the guard and `to_values()` still borrows the slice.

The Python-facing surface is a **free choice**, because the homogeneity constraint that killed the
tuple-list in Rust does not exist in Python — pyo3 extracts each array dynamically into the
`NumpyArray` enum, so a list of mixed-dtype tuples is fine regardless. Two options:

```python
with writer.time_step("0.1") as step:              # mirrors the Rust builder
    step.point_data("velocity", DataAttribute.Vector, vel)
    step.cell_data("region_id", DataAttribute.Scalar, ids)
```

or keep today's `write_data(time, point_data=[...], cell_data=[...])` as a thin loop over the
builder. **Recommendation: ship both** — the context manager for the reuse case (and it matches the
`__enter__`/`__exit__` idiom already established for `close()`), the list form because it is the
more pythonic default and needs no `write` discipline from users. This is not a dual code path in
the CLAUDE.md sense; the list form is ten lines of `for` loop in the binding layer.

`xdmf.pyi` and `python/tests/test_basic.py` need the corresponding additions.

---

## Migration

Breaking, deliberately, per CLAUDE.md's pre-1.0 rule — no deprecated `write_data` alias.

```rust
// before
writer.write_data("0.1",
    [("velocity", DataAttribute::Vector, vel.as_slice().into())],
    [("region_id", DataAttribute::Scalar, ids.as_slice().into())])?;

// after
let mut step = writer.time_step("0.1")?;
step.point_data("velocity", DataAttribute::Vector, &vel)?;
step.cell_data("region_id", DataAttribute::Scalar, &ids)?;
step.write()?;
```

Call sites to update: `tests/time_series_writer.rs`, `tests/binary_writer.rs`,
`tests/vtk_comparison.rs`, `examples/paraview_smoke.rs`, `README.md`, the `write_data` doctest, and
`python/src/writer.rs`.

**Downstream note (arotau).** `arotau-core/src/output/xdmf.rs` does
`par_iter().map(..).collect::<Vec<f64>>()` per variable per step. `02_performance.md` part F already
recommends persistent per-variable buffers; with this milestone the stronger form becomes available —
**one** buffer filled with `par_iter_mut()` and written between fills. Put both in the migration
notes.

`examples/paraview_smoke.rs:120` also stops needing
`.iter().flatten().copied().collect::<Vec<f64>>()` if the `From<&'a [$ty; N]>` impls land; worth
doing in the same pass since it is the ugliest call site in the crate.

---

## Testing

- Buffer reuse compiles and round-trips: one `Vec`, refilled between two `point_data` calls, both
  attributes correct on read-back (M5's reader makes this a real assertion, not an XML string check).
- Mixed dtypes within one step (`f64` point data + `u32` cell data) — the case that blocks the
  generic tuple-list, so it must be locked down by a test.
- Abandoned step: drop without `write`, then successfully write a later step; and confirm no
  `<Grid>` fragment for the abandoned time appears in the XML.
- Every validation row in the table above keeps its error variant, asserted with `assert_matches!`
  on variant + field per CLAUDE.md.
- `test_write_data_preserve_order`'s golden XML stays byte-identical after migration.
- The ParaView smoke matrix passes unchanged.
- `python/tests/test_basic.py`: context manager, list form, mixed dtypes, abandoned step.

## Acceptance criteria

1. A steady-state write loop with **one** reusable buffer allocates zero per step, for all five
   storage backends — same instrument as `02_performance.md` criterion 3.
2. No `Values` appears in `README.md`, the doctests, or `examples/`.
3. Mixed-dtype attributes within a single time step still work.
4. `test_write_data_preserve_order`'s golden XML is byte-identical.
5. An abandoned `TimeStep` leaves the writer usable and the XML untouched.
6. The pyo3 layer no longer contains `extract_guards`/`build_values`.

## Explicitly out of scope

- **The XML rewrite cost.** That is `02_performance.md` part B / ROADMAP decision 6. This plan is
  shaped to fit it and should land with it, but does not restate or replace it.
- **Widening `Values`.** Done on `reader`; see Prerequisites.
- **Removing `Cow` from `Values`.** Argued above; revisit separately if ever.
- **An XML-rewrite-frequency knob** (rewrite every *k* steps). `write()` would make it expressible
  for the first time, which is exactly why it should be resisted until a measured workload asks —
  part B makes the per-step cost O(1), so the motivation may never arrive.
