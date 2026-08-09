# M2 — Performance

`README.md`: *"Everything should be tuned towards maximum performance, both reading and writing is in
the hot path"*, *"Temporary allocations should be avoided as much as possible"*, *"Check the entire
crate for performance-gains and other optimizations, now is the time! Even if it requires larger
refactoring."*

This milestone is that pass. It is ordered before submeshes (M4) and the reader (M5) because both
build on the machinery changed here.

---

## A. Benchmark harness — prerequisite, do this first

Nothing else in this milestone should be started before there is a baseline. The existing benchmarks
(`python/benchmarks/` on the `multiple-features` branch) need the Python bindings, which are M6.

### What to build

`benches/write_time_series.rs` (criterion), plus a shared mesh generator.

**Mesh generator** — port `python/benchmarks/mesh_gen.py` from the branch to Rust as
`benches/common/mod.rs`: a `nx × ny × nz` hexahedral duct with quad boundary patches
(`inlet`/`outlet`/`sides`), parabolic velocity + linear pressure fields, and the block index lists.
It is already a good synthetic CFD case, the boundary patches exercise mixed cell types, and the
block lists feed M4 directly. Sizes: `1e3` (10×10×10), `1e5` (10×10×1000), `1e7` (100×100×1000).

**Benches:**

| bench | what it isolates |
|-------|------------------|
| `write_mesh` × storage × size | one-shot mesh cost |
| `write_data` × storage × size | steady-state per-step cost — the number that matters |
| `steps_scaling` (N = 10, 100, 1000, 5000, one storage) | **the O(N²) light-data cost**; this should be flat per step after part B and is not today |
| `allocations_per_step` | see below |

Criterion's default 100 samples is unusable at `1e7`. Configure per-bench sample sizes, and keep
`1e7` out of the criterion set entirely: put it in `examples/bench_cfd.rs`, run manually, reported as
a table like `CFD_BENCHMARK_PLAN.md` does.

**Allocation counting.** This is the metric `README.md` actually asks for, and wall-clock alone will
not show it. Install a counting `#[global_allocator]` wrapper in the bench binary:

```rust
struct Counting;                       // wraps System, bumps an AtomicUsize in alloc()
```

Then assert-style report: allocations per `write_data` call, per storage backend. The target after
part C is **a small constant, independent of mesh size and of the number of attributes** — ideally
zero in steady state. A regression here is easy to introduce and invisible otherwise; consider
promoting the strictest case to an actual `#[test]` with an upper bound so CI catches regressions.

**Also record:** bytes written per step, and peak RSS for the `steps_scaling` bench (the current
design retains every step's attributes in memory forever — part B fixes that, and RSS is how you see
it).

**Report compressed size, not just raw.** Per part E's governing principle, any size number the
benchmark emits must include the final-archive size under at least one real codec, because raw size
has already been shown to mislead. Build the codec sweep *into the driver* rather than running it by
hand — `SESSION_2026-08-08_cfd_benchmark.md` flags that the previous sweep was done ad hoc in scratch
files and consequently did not survive.

**The fairness rule, if a cross-tool comparison is run** (it will be, in M6): apply the *same* set of
external codecs to *each tool's real default output*. Not xdmf-raw-plus-a-strong-codec against
pyvista-default-plus-a-weak-one, and not pyvista with its compression stripped out to manufacture a
"raw" baseline — both of those were tried and discarded in the earlier session. And verify what the
other tool actually does before reasoning about why it differs: the assumption that pyvista used f32
somewhere was wrong and sent one round of analysis in the wrong direction.

Add the bench commands to `CLAUDE.md`'s command list.

**Process note for whoever runs this.** Standing instruction from the user, recorded in
`SESSION_2026-08-08_cfd_benchmark.md`: for exploratory and benchmarking work, explain what a command
will do *before* running it — especially long-running ones — so it can be stopped before it starts.
The 1e7 cases here run for minutes at a time; this applies directly.

---

## B. Light-data (XML) writing: O(steps²) → O(steps)

**This is the single largest win in the milestone.** Decision 6 in `ROADMAP.md`: append + patch tail.

### The problem

`TimeSeriesDataWriter::write()` (`src/time_series_writer.rs:343`) runs after *every* `write_data` and
rebuilds the entire document from scratch:

- `self.grid.clone()` once per accumulated time step (`:352`), each clone carrying a full
  `Geometry` + `Topology`;
- `attributes.clone()` per step (`:357`) — every attribute of every step ever written;
- `self.data_items.clone()` (`:383`) — and for `AsciiInline` **these hold the entire mesh as text**,
  so the full point and connectivity arrays are cloned and re-serialized on every single time step;
- serialize the whole thing to a temp file, then rename.

So an N-step run does O(N²) serialization work and writes O(N²) bytes. `self.attributes` also grows
without bound, retaining every attribute of every step for the life of the writer.

Concretely: 10k steps with a ~1 KB grid fragment each is ~50 GB of cumulative rewriting. For
`AsciiInline` on a 1e5 mesh it is far worse, because the ~10 MB of inline mesh text is part of every
one of those rewrites.

### The design

The document has a fixed shape:

```
<Xdmf ...><Domain>
    <DataItem Name="coords" .../>            ← written once, after write_mesh
    <DataItem Name="connectivity" .../>      ← written once
    <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">
        <Grid Name="time_series-t0.0" ...>   ← appended per step
        <Grid Name="time_series-t1.0" ...>   ← appended per step
    </Grid>                                  ┐
</Domain>                                    ├ the tail: fixed, known, short
<Information .../><Information .../></Xdmf>  ┘
```

So:

1. `write_mesh` writes the header and the shared `DataItem`s, records `body_end = stream_position()`,
   writes the tail, flushes. The file is already a valid mesh-only XDMF at this point — which is what
   `write_mesh` guarantees today.
2. Each `write_data` builds **the single step's `<Grid>` fragment plus the tail into one in-memory
   `Vec<u8>`**, seeks to `body_end`, issues **one** `write_all`, advances `body_end` by the fragment
   length, and flushes.

Per step this is O(fragment), and the file on disk is a complete, openable XDMF after every step —
preserving the property that matters in practice: opening a running simulation's output in ParaView.

`self.attributes` and the per-step `Grid` clones disappear entirely. Memory becomes O(1) in the
number of steps instead of O(N).

### Things to get right

- **Torn reads.** Today's temp-file + atomic rename means a concurrent reader never sees a partial
  file; appending gives that up. Mitigation: the fragment and the tail go out in a *single*
  `write_all` of a few KB. That is not a formal atomicity guarantee, but the window shrinks from
  "the entire document" to "one small write", and the failure mode is a manual ParaView reload that
  the user retries. This is a deliberate trade and should be stated in the `TimeSeriesDataWriter`
  docs. If a caller genuinely needs the old guarantee, the escape hatch is that they can copy the
  file; do not add a mode flag for it unless someone asks.
- **The file only ever grows**, because the tail is a constant string and fragments are only ever
  appended before it. No `set_len` truncation is needed. Assert this in a test rather than assuming
  it (a future variable-length tail — e.g. an `Information` element recording the step count — would
  silently break it).
- **A failed step must leave no trace.** The fragment is built in memory and only written on success,
  which strengthens item 1 of `API_IMPROVEMENTS_PLAN.md` (mid-write error poisoning) rather than
  conflicting with it: with streaming, "the call had never happened" becomes literally true for the
  light data.
- **Indentation.** `Xdmf::write_to` uses `quick_xml::Writer::new_with_indent(w, b' ', 4)`. Serializing
  a `Grid` fragment needs it rendered at depth 3. **Spike this before committing to the design** —
  check whether `quick-xml` exposes an initial indent level. If it does not, serialize the fragment to
  a `String` and prefix each line with 12 spaces (cheap, the fragment is small), or hand-roll the
  fragment writing. This is the main implementation risk in part B.
- **Golden-file regression.** `test_write_data_preserve_order`
  (`src/time_series_writer.rs:1187`) already asserts the exact full-document XML for a 4-step run.
  The new writer must produce **byte-identical** output. That test is the acceptance criterion; do
  not relax it to accommodate the implementation.
- **`Drop`.** The open `File`/`BufWriter` must be flushed. `Drop` cannot report errors, so also add
  an explicit `finish(self) -> Result<()>` and document that `Drop` flushes best-effort. (Relevant to
  Python too — see `06_python_bindings.md` on the context-manager protocol.)
- **`write_mesh`-only files.** The "no attributes yet → write the uniform grid directly, not a
  temporal collection" branch (`:369`) has to survive: the header written by `write_mesh` opens the
  collection, so a file with zero steps would be an empty `<Grid GridType="Collection">`. Either
  emit the mesh-only form and rewrite the header once on the first `write_data`, or verify that an
  empty temporal collection opens cleanly in ParaView and simplify to always-collection. Prefer the
  latter if ParaView accepts it; verify in the smoke test.

---

## C. Allocations in the per-step path

Everything below is per `write_data` call, i.e. multiplied by the number of time steps. Part B
removes several of these outright; the rest are individually small but the goal is a *constant*,
ideally zero, steady-state allocation count.

| # | Site | Today | Fix |
|---|------|-------|-----|
| 1 | `collect_data` (`:436`) | a `Vec` + a `HashSet<&str>` per call, ×2 (point + cell) | The attribute count is small (typically < 20). Replace the `HashSet` with an O(n²) scan over the collected slice — no allocation, and faster at these sizes. Keep the `Vec` (needed to validate before writing) but hold it as a reusable buffer on the writer. |
| 2 | `new_attributes` + one `String` per attribute name + a 1-element `Vec<DataItem>` per attribute (`:308`–`:331`) | allocated and then **retained forever** in `self.attributes` | Gone with part B: attributes are serialized into the fragment immediately and dropped. |
| 3 | `self.attributes.push((time.to_string(), ..))` (`:335`) | unbounded growth | Gone with part B. Only `written_times` persists. |
| 4 | `Values::dimensions()` (`src/values.rs:63`) | a `Vec<usize>` inside `Dimensions` per attribute per step | ≤ 3 elements. Low priority; do not add a `smallvec` dependency for it. Revisit only if it shows on the profile. |
| 5 | `hdf5_writer::write_data` (`:108`) | `format!("{}/t_{time}/{}", ..)` per attribute per step, plus `link_exists` + `create_group` + `group()` — three HDF5 round-trips per attribute | Open the time-step group **once** in `write_data_initialize` and cache the `Group` handle; reuse a `String` scratch buffer for the dataset name. |
| 6 | `binary_writer::write_data` (`:104`) | `format!("data_t_{time}_{tag}_{name}.bin")` + a `PathBuf` join per attribute per step | Reusable `String` + `PathBuf` scratch on the writer; truncate and re-push rather than reallocating. |
| 7 | `ascii_writer` | `array_to_string_fmt` builds one giant `String` for the whole array | Real, but ASCII is not the performance-critical backend. Switch `array_to_writer_fmt` to a reusable formatting buffer and leave the inline path alone. Note it, do not gold-plate it. |

**Implementation note for the scratch buffers.** Borrowing a scratch buffer off `self` while also
calling `&mut self.writer` fails the borrow checker. Use `let mut buf = std::mem::take(&mut self.buf);`
… `self.buf = buf;`. This is the standard trick and shows up again in M3 (the f32 downcast buffer) and
M4 (the block gather buffer) — worth putting one commented helper in place rather than three
ad-hoc copies.

---

## D. Bulk binary encoding

`binary_writer::write_f64_le` (`:138`) and `write_u64_as_u32_le` (`:149`) call `write_all` **once per
element**. For a 10M-element case that is tens of millions of calls through `BufWriter`. The
benchmark on the branch measured 2.24 s to write the `1e7` case in `Binary` mode; this is very likely
most of it.

Fix: on little-endian targets an `&[f64]` is already its own little-endian byte representation, so the
whole slice can go out in a single `write_all`:

```rust
#[cfg(target_endian = "little")]
writer.write_all(bytemuck::cast_slice(values))?;
#[cfg(target_endian = "big")]
// existing per-element loop
```

The crate currently contains **zero** `unsafe` (verified), and that is worth keeping — use `bytemuck`
(tiny, no proc macro needed for `cast_slice` on primitives) rather than `slice::from_raw_parts`.
Consider adding `#![forbid(unsafe_code)]` to `lib.rs` at the same time to make the property explicit
and enforced.

`write_u64_as_u32_le` still needs a per-element loop because it narrows and range-checks, but it
should encode into a reusable `Vec<u32>`/byte buffer in chunks (e.g. 8 K elements) and write in bulk.
Note that item 1 of `API_IMPROVEMENTS_PLAN.md` moves the range check to up-front validation, after
which the write loop itself is a pure narrowing and can go through `bytemuck` on a converted buffer.

The same bulk path applies to `f32` once M3 lands.

---

## E. HDF5 tuning

`README.md`: *"Compression etc should be tested against real data to find the optimum balance."*

### The governing principle: measure compressed size, never raw size

`SESSION_2026-08-08_cfd_benchmark.md` established this the hard way and it should constrain every
decision in this part, and in `03_values_and_f32.md`.

The same connectivity values, isolated and compressed with zstd-6:

| representation | raw | zstd-6 |
|---|---:|---:|
| `int64` (VTK/pyvista) | 640 MB | **49.4 MB** |
| `uint32` (xdmf `Binary`) | 320 MB | 87.3 MB |

The *objectively smaller* representation produced the *larger* compressed archive, by 1.77×. `int64`
indices for a 10M-node mesh use ~24 of 64 bits, so five bytes of every index are always zero — free
for the compressor. `uint32` has no such slack, and every byte carries entropy. **Packing tighter
concentrates entropy; it does not reduce it.**

Two consequences that must not be lost:

1. **This explains a result that would otherwise look like a defect.** Against pyvista at 1e7,
   xdmf's `Binary` output plus an external codec loses on final size at deflate, zstd-6, and bzip2,
   and only wins with lzma (22.9 MB vs 52.5 MB) at ~1.7× the time. That is not xdmf writing badly —
   it is xdmf writing *densely*, and it is largely the `uint32` connectivity narrowing doing it.
2. **The narrowing cannot simply be removed.** It exists because ParaView's legacy Xdmf2 reader
   silently misreads 64-bit integers in `Format="Binary"` (`src/binary_writer.rs:145`) — it is a
   correctness workaround, not a size optimization, and it stays. What changes is that nobody should
   "optimize" by packing other arrays tighter without measuring the compressed result, and the docs
   should not claim `Binary` produces smaller archives than VTK. It produces smaller *files*, which
   is a different and less useful property.

This is also the strongest argument for **HDF5 with `shuffle` + `deflate` being the recommended
default storage**, not `Binary`: it compresses the values at write time, on the values themselves,
where the shuffle filter can exploit cross-value byte structure — which is exactly what an external
codec applied to an already-packed byte stream cannot do. The branch measurements back this up
(1e7: 10.9 s total / 8.96 MB for HDF5+shuffle+deflate, versus 23.6 s / 124 MB for `Binary`+zip and
23.7 s / 100 MB for pyvista). Consider making that recommendation explicit in the crate docs as part
of this milestone.

### Existing evidence and what it does not cover

The existing evidence (`CFD_BENCHMARK_PLAN.md` and `python/benchmarks/`, both on
`origin/multiple-features` — see `ROADMAP.md` for the cherry-pick) established:
`shuffle() + deflate()` is the only filter combination that opens in stock ParaView (Blosc is
confirmed broken — do not revisit without new information), and level 3 beats level 6 on write time
for noisy data at a ~1.3 % size cost, which is why `DEFAULT_DEFLATE_LEVEL = 3`.

What that evidence does **not** cover:

1. **Chunk size.** `write_values`/`write_mesh` (`src/hdf5_writer.rs:268`, `:297`) call
   `.shape(n).shuffle().deflate(k).create(..)` and never set a chunk size, so `hdf5-metno` picks one.
   Auto-chunking heuristics are frequently poor for long 1D arrays, and chunk size interacts strongly
   with both compression ratio and write throughput. Sweep it (e.g. targeting 256 KiB / 1 MiB / 4 MiB
   chunks) at 1e5 and 1e7. **This is the highest-expected-value item in part E** — a 2-5× factor from
   chunking alone is common, and it has never been measured here.
2. **Real data.** Every measurement so far used a synthetic duct (parabolic velocity, linear
   pressure) or pseudo-random noise. Real solver output sits between the two. Action: get one real
   arotau result field (or any real CFD/FEM result) and re-run the level and chunk sweeps on it.
   Until that exists, `random_benchmark.py`'s noisy fields are the conservative proxy and should be
   the one the default is tuned against.
3. **f32.** Byte-shuffle groups same-significance bytes across values, so its behaviour changes with
   element width; re-check the level/chunk defaults once M3 lands rather than assuming they carry
   over. `03_values_and_f32.md` has a measurement gate that depends on this and should be run in the
   same sweep.
4. **`szip`** is available in stock ParaView, unlike Blosc. It is worth one measurement, but note the
   licensing restriction on the *encoder* before adopting it — most likely conclusion is "measured,
   not adopted", and that conclusion should be written down so it is not re-litigated.

Re-validate `DEFAULT_DEFLATE_LEVEL` after chunking is tuned; the current value was chosen with
default chunking underneath it, so it is not independent.

---

## F. API-level buffer reuse

`README.md` suggests this *"maybe requires changing the API (e.g. passing pre-allocated vectors so
that they can be reused during writing of steps)"*. The API surface part of this is **done and
merged** (2026-08-09, PR #18, commit `aa3c501` — see `API_IMPROVEMENTS_PLAN.md`); what remains is
documentation:

- `Values<'a>` is `Cow`-backed, so `Values::from(buf.as_slice())` borrows with no copy, and
  `write_data` takes `impl IntoIterator<Item = (&str, DataAttribute, Values)>`. A caller keeps one
  `Vec<f64>` per field alive for the whole run, overwrites it in place each step, and passes
  `buf.as_slice().into()`. **Zero allocations on the caller side.** The README example and the
  `write_data` doctest both demonstrate the pattern now. The owned `From<Vec<f64>>`/`From<Vec<u64>>`
  impls also gained a doc note warning that they move the buffer, so forgetting `.as_slice()` in a
  reuse loop — which fails to compile with `E0382` pointing at "moved ... in previous iteration of
  loop" — is at least explained at the point where a caller would type `.into()`.
- What is still missing: a crate-level doc section and a dedicated `examples/reuse_buffers.rs`
  showing the pattern end-to-end (the README/doctest show it inline, but there's no standalone
  example). Low priority now that the doctest already carries the pattern.
- **Do not** switch `write_data` to take `&Values` (as the `multiple-features` branch did). With
  owned `Values<'a>` the caller's buffer stays a plain `Vec<T>` and the `Values` wrapper is a
  free borrow constructed at the call site. Taking `&Values` would push callers to hold `Values`
  objects and mutate through `as_mut_slice`, which is strictly more machinery for the same result.
  This also means the branch's `ValueType::as_mut_slice` should **not** be cherry-picked here — the
  reader (M5) is what justifies `as_slice`, and `as_mut_slice` may turn out to have no caller at all.

**Downstream note (arotau).** `arotau-core/src/output/xdmf.rs` currently does
`par_iter().map(..).collect::<Vec<f64>>()` per variable per step, i.e. a fresh allocation of the full
field every step, then hands it over owned. Once this milestone lands, the recommended shape is
persistent per-variable buffers filled with `par_iter_mut()`. That is an arotau-side change, but it
is the change that realises most of the benefit, and it should be written into the migration notes.

---

## G. Explicitly out of scope

- **Parallelising attribute writes with rayon.** HDF5 is not thread-safe without the threadsafe build;
  the binary/ascii backends write to separate files and could overlap, but this adds a `rayon`
  dependency and a concurrency model to a crate that does not otherwise have one. Revisit post-1.0,
  and only if the benchmark shows per-attribute I/O (not compression) dominating.
- **Memory-mapped writes.** No evidence they would help here.

---

## Acceptance criteria

1. `steps_scaling` shows **flat per-step cost** from 10 to 5000 steps (today it grows linearly).
2. Peak RSS in `steps_scaling` is flat in the number of steps.
3. Steady-state allocations per `write_data` are a small constant, independent of mesh size and
   attribute count, for all five storage backends — enforced by a test with an upper bound.
4. `1e7` `Binary` write time improves measurably against the 2.24 s branch baseline (part D).
5. `test_write_data_preserve_order`'s golden XML is byte-identical.
6. The ParaView smoke matrix passes unchanged, plus a new assertion that a file is readable
   *mid-run* (after step k of n) — this is the property part B trades atomicity for, so it should be
   tested, not assumed.
