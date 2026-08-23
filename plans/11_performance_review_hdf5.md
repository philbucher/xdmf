# Performance review — Python write → Rust read, HDF5 storage

Static review (2026-08-23, `submeshes` branch). Scope: the path a caller actually takes when
writing from Python and reading back in Rust, both with `DataStorage::Hdf5*`. Every finding was
read off the code rather than profiled; the exception is finding 2, which was measured afterwards
(see "Measured" under it). Findings 1, 3, 4 and 5 have since been implemented — see the status
column and the note under each.

Overlaps with `02_performance.md` are marked — parts B and C5 of that plan are still unimplemented
and are still the two biggest write-side items.

---

## Summary

The **Python → Rust boundary is already clean**: numpy buffers are borrowed, never copied, and the
GIL is released around the write. Nothing in `python/src/` needs work.

The costs are all in the core crate:

| # | Where | Cost | Priority | Status |
|---|-------|------|----------|--------|
| 1 | `read_topology` | ~5× the connectivity's own size in peak RSS, 4 full copies | **high** (read) | **done** (a, b and c) |
| 2 | `write_xdmf_file` per step | O(steps²) serialization + O(steps) retained memory | **high** (write) — `02_performance.md` part B | open, **measured** |
| 3 | `prepare_cells` uniform arm | full copy of the connectivity, avoidable | **high** (write) | **done** |
| 4 | `hdf5_reader::read` | one `H5File::open` per array read | medium (read) | **done** |
| 5 | `TimeSeriesReader::read_data` | one full extra copy of every field read | medium (read) | **done** |
| 6 | `read_points` | drops the caller's buffer instead of reusing it | medium (read) | **done** |
| 7 | `SingleFileHdf5Writer::write_data` | 3 HDF5 round-trips + 2 `String`s per attribute per step | low — `02_performance.md` C5 | open |
| 8 | default chunk shape | 64 MB minimum chunk, one chunk for most arrays | low, but **measure** | open |
| 9 | `Membership::apply` | reads a whole array to keep a slice of it | low today, high if per-submesh reads land | open |

**Findings 1, 3, 4, 5 and 6 were implemented on 2026-08-23.** Verified afterwards: `cargo nextest run`
261/261 in debug and release, `--no-default-features` 218/218, `cargo test --doc` 7/7, `cargo clippy
--all-targets -- -D warnings` clean with and without the `hdf5` feature, `cargo doc` under `-D
warnings -D missing_docs` clean, `cargo +nightly fmt --check` clean, `typos` clean, and the
`python/` crate still checks. What changed is recorded under each finding.

---

## 1. `read_topology` allocates ~5× the connectivity (read path, highest)

`src/reader.rs:360` → `read_topology_plain` → `topology::decode` → `reader.rs:387`.

For a file whose connectivity is `u32`, read into a `Vec<u32>`, with C = the connectivity's byte
size:

1. `hdf5_reader::read_raw::<u32>` → `Vec<u32>` (C)
2. `topology::widen_to_u64` (`topology.rs:178`) → `Vec<u64>` (2C) — unconditional, even when the
   file's type and the requested type are the same
3. `decode_uniform` (`topology.rs:49`) → `connectivity: values.to_vec()` (another 2C) — `values` is
   already an owned `Vec<u64>` that is dropped immediately after, so this copy is pure waste
4. `read_topology`'s conversion loop → the caller's `Vec<u32>` (C)

Peak inside `decode` is C + 2C + 2C = **5C**, and there are four full passes over the data. On a
10M-hex mesh with `u32` connectivity (80M indices, 320 MB) that is ~1.6 GB peak to deliver 320 MB.

Fixes, in increasing order of work:

- **(a) one line, no API change:** make `decode_uniform`/`decode` take `values: Vec<u64>` by value
  and move it into `DecodedTopology.connectivity`. Kills step 3 outright (−2C peak, −1 pass).
  `decode_mixed` genuinely needs a separate output buffer and keeps its `&[u64]`.
- **(b)** in `widen_to_u64`, take `Values` by value and `into_owned()` the `U64` arm, so a `u64`
  file costs nothing there. Today `Values::U64(v) => Ok(v.to_vec())` copies even an owned `Cow`.
- **(c) the real fix:** make the decode generic over the connectivity type the way `read_points` is
  generic over `Coordinate`, so a `u32` file read as `Vec<u32>` never materialises a `u64` array at
  all. The widening exists because `read_topology` re-derives indices against the reassembled mesh
  and range-checks them (`to_connectivity_index`), but that check is a comparison — it does not need
  the values widened into their own array first. Bigger change; do (a) and (b) first and measure
  whether (c) is still worth it.

**Done, all three.** (a) and (b) went in first — `decode` and `widen_to_u64` taking their values by
value — and (c) then removed `widen_to_u64` outright:

- `SealedIndex` gained `indices_from_values`, which takes *any* integer array and checks the values
  (`MAX_INDEX`, and not negative) rather than the element type, and `as_index`. It moves the array
  when the file already holds the requested type, walking it once to catch a negative index rather
  than copying it.
- `topology::decode_in_place` replaced `decode`: the caller's buffer arrives holding the file's array
  and leaves holding the mesh's. A uniform topology moves nothing at all (only `cell_types` is
  produced); `Mixed` is compacted in place with `copy_within` as each cell's type code is dropped,
  which is safe because the write position always trails the read one.
- `read_topology`, `read_topology_plain`, `read_topology_with_submeshes` and `decode_submesh_topology`
  are generic over `I: ConnectivityIndex` throughout, so no `u64` array exists on the path any more.

**Peak for the plain path is now 1C — the caller's buffer, filled by `read_into_raw`** — against the
5C this finding opened with. `re_reading_the_mesh_allocates_nothing_of_its_size` guards it (checked
against a staged copy: it fails with "allocated 1199997 bytes against the 599976 bytes of the
connectivity itself").

The submesh path (`read_topology_with_submeshes`) was worse in a different way: it held **every**
submesh's decoded topology in memory simultaneously plus the global
`connectivity`/`cell_types`/`offsets`/`covered` arrays — the whole mesh twice over plus one `bool`
and one `usize` per cell.

**Also done.** It is now two passes over the submeshes, because the mesh's cell offsets need *every*
cell's type before *any* cell's points can be placed:

1. Fill the caller's `cell_types` (used directly as the global array, so it is no longer replaced
   wholesale at the end). A **uniform** submesh states its one cell type in the light data, so
   `topology::uniform_cell_type` answers it with **no heavy-data read at all**; only `Mixed` is
   decoded here, and therefore decoded twice.
2. Decode each submesh into a scratch pair reused across submeshes, and scatter its connectivity.

So for everything this crate writes, each submesh's array is still read exactly once — in pass 2 —
and peak drops from 2C to 1C plus the largest single submesh. `decode_submesh_topology` also gained
the cell-count check that `cell_of_submesh` (now gone) used to make per cell, which additionally
catches a submesh `Topology` holding *fewer* cells than `submesh_cells` names for it.

What remains proportional to the mesh on this path is inherent to the algorithm: `offsets`
(8 B/cell), `covered` (1 B/cell) and `cell_types` (1 B/cell). Against a `u32` tet connectivity
(16 B/cell) that bookkeeping is not negligible, but it cannot be folded away without losing the
random access into `offsets` that the scatter needs.

`re_reading_a_mesh_with_submeshes_holds_one_submesh_at_a_time` (`tests/reader.rs`) guards both this
and the points change below, under the same counting allocator. Measured on 20 000 hexahedra with a
`u64` connectivity split into 4 submeshes: topology 1 806 KB → 505 KB, points 640 KB → 160 KB. Both
halves were checked against a reintroduction of the old behaviour and fail against it.

## 2. The XDMF file is re-serialized after every time step (write path)

`TimeSeriesDataWriter::write_xdmf_file` (`time_series_writer.rs:1785`) runs from `TimeStep::finish`,
i.e. once per `write_time_step`, and serializes `self.xdmf` — which holds **every step written so
far** — into a temp file, then renames.

- Work: O(steps²) in total. ~1 KB of `<Grid>` per step × 10k steps ≈ 50 GB of cumulative rewriting.
- Memory: `self.xdmf` retains every step's `Grid` and every `Attribute` for the writer's lifetime.
- It also calls `self.writer.flush()` → `H5Fflush` per step.

This is `02_performance.md` part B (append + patch tail) and is still the single largest write-side
win. Nothing has changed since that plan was written except the line numbers. Note that the
submesh document shape (one temporal collection *per submesh*, `wrap_first_step`, `:1635`) means the
fragment is appended in N places rather than one — part B's design needs to account for that, which
it currently does not.

### Measured, 2026-08-23

`Hdf5SingleFile`, release build, throwaway driver (not kept). A 64-point mesh, so the per-step HDF5
cost sits at its floor and the light-data term is what moves:

| steps | attrs | final `.xdmf2` | fragment/step | cumulative XML written | wall | us/step |
|---|---|---|---|---|---|---|
| 100 | 1 | 73 KB | 727 B | 3.7 MB | 55 ms | 551 |
| 500 | 1 | 366 KB | 730 B | 92 MB | 660 ms | 1 317 |
| 1000 | 1 | 732 KB | 731 B | 367 MB | 2.56 s | 2 553 |
| 2000 | 1 | 1.5 MB | 733 B | 1.5 GB | 10.19 s | 5 093 |
| 4000 | 1 | 2.9 MB | 734 B | **5.9 GB** | 39.97 s | 9 987 |
| 4000 | 5 | 6.8 MB | 1 693 B | **13.5 GB** | 86.38 s | 21 589 |

Per-step time doubles when the step count doubles: O(steps) per step, O(steps²) over a run, exactly
as the plan predicted. Fitting `per_step = a + b × steps` gives a constant `a` of ~310 µs for one
attribute (HDF5 dataset + `H5Fflush` + temp-file rename) and a slope `b` of ~4.8 µs per
already-written step; with five attributes, ~419 µs and ~10.6 µs.

**The slope is simply the document being re-serialized at ~150 MB/s** — `b` divided by the fragment
size comes to ~0.0066 µs/byte in both rows. So the useful way to state the cost is: *every step pays
(current `.xdmf2` size) ÷ 150 MB/s, on top of its own work.*

That puts the crossover — where rewriting history costs more than writing the step — at roughly
`steps ≈ a / 5 µs`. A second run, 200 steps of two point fields, puts real numbers on `a`:

| mesh | per-step heavy data | first 10% | last 10% | growth over 200 steps | crossover |
|---|---|---|---|---|---|
| 64 points | ~0 | — | — | — | ~70 steps |
| 10 000 points | 240 KB | 2 425 µs | 3 322 µs | **1.37×** | ~500 steps |
| 200 000 points | 4.8 MB | 41 087 µs | 42 427 µs | **1.03×** | ~8 000 steps |

**So the impact depends entirely on the ratio of steps to mesh size**, because the light-data term is
independent of mesh size while the heavy-data term is not:

- Small mesh, long run (0D/1D models, reduced-order runs, boundary-only meshes, parameter sweeps):
  severe. At 4000 steps on the 64-point mesh ~97% of the runtime is re-serializing history, and the
  last step spends 19 ms of XML on 310 µs of real data.
- Large mesh (the 1e7 CFD case, where one step is ~1 s of compression): the crossover lands around
  10⁵ steps, so part B buys close to nothing there.

Part B is therefore worth doing for the scaling being *correct* rather than for the headline CFD
number — and the unbounded memory growth (`self.xdmf` retains every step's `Grid` and `Attribute`:
6.8 MB of XML at 4000 steps × 5 attributes, held as structs) is arguably the better reason.


Independent of part B: `H5Fflush` per step is worth measuring separately. It is what makes a
half-written run openable in ParaView, so it should stay, but if it dominates for cheap steps a
`flush_every_n` knob is a smaller change than part B.

## 3. `prepare_cells` copies the whole connectivity for a uniform mesh (write path)

`time_series_writer.rs:1468`:

```rust
return Ok((TopologyType::from(*first), connectivity.to_vec()));
```

The uniform arm — the common case, and the one `perf: support uniform topology` (#28) added
specifically to make cheap — copies the caller's entire connectivity array. For a 10M-hex mesh with
`u64` indices that is a 640 MB allocation and memcpy per `write_mesh`, and from Python it is a copy
of a numpy buffer that was deliberately borrowed zero-copy two frames up.

`PreparedMesh.cells` is only ever *read* afterwards (`points_data_item`, `connectivity_data_item`,
`submesh_points`, `extract_connectivity`), so changing it to `Cow<'a, [I]>` — `Borrowed` in the
uniform arm, `Owned` in the `Mixed` and polyvertex arms — removes the copy with no other change to
the call sites. This is the cheapest large win on the write path.

**Done.** `prepare_cells` returns `Cow<'c, [I]>` and `PreparedMesh` holds it; every call site took
the change through deref coercion. `prepare_cells_borrows_a_uniform_connectivity` asserts the
uniform arm hands back the caller's own pointer, and the existing value assertions go through a
`prepare_cells_vec` test helper that owns the result.

While there: `write_mesh` walks the connectivity three times before it reaches HDF5
(`validate_points_and_cells` bounds check, `prepare_cells`, then `paraview::validate`). The third is
free for every type except `u64` (`paraview.rs:57` returns immediately otherwise), so only fix it if
`u64` connectivity turns out to be a case worth caring about.

## 4. Every heavy-data read reopens the HDF5 file (read path)

`reader/hdf5_reader.rs`'s `read_hdf5` calls `H5File::open(file_path)` on **every** `read`. Nothing is cached on
`TimeSeriesReader`.

For `Hdf5SingleFile` — where it is the same file every time — a loop over S steps × F fields costs
S×F opens (superblock + root group parse each). `TimeSeriesReader::new` itself does up to 2×
(number of submeshes) opens before returning, because `submesh_points_membership` and
`parse_submesh_cells` read their index arrays eagerly.

Fix: a small open-file cache on the reader keyed by resolved path (`HashMap<PathBuf, H5File>`, or
just "remember the last file opened", which handles the single-file case and the multi-file case
where a step's fields are all in one file). `hdf5_reader::read` would need the cache passed in —
it is `pub(super)` and has three call sites, so this is contained.

Measure first: an `H5File::open` on a warm page cache is likely tens of microseconds, so this only
matters against small fields or many steps. It is cheap enough to be worth doing anyway.

**Done.** `hdf5_reader::FileCache` holds one slot -- the last file opened -- behind a `Mutex`, so
`TimeSeriesReader` stays `Sync` while every read method keeps taking `&self`. It lives on
`Document`, which is why `read_data_item`/`parse_selector` and the reader's helpers now take a
`&Document` rather than a `&Path` base dir. One slot rather than a map deliberately: a map would
hold one open file descriptor per time step. The trade is that a reader keeps its heavy-data file
open until it is dropped, which `TimeSeriesReader`'s docs now state.

## 5. Every field read is copied twice (read path)

`TimeSeriesReader::read_data` (`reader.rs:461`):

```rust
let converted = T::from_values(values)?;   // moves out of the Cow — free
into.clear();
into.extend(converted);                    // full copy into the caller's buffer, then drops `converted`
```

`from_values` correctly *moves* when the types match, and then the result is copied element-wise
into `into` and thrown away. So a 10M-point `f64` field read costs an 80 MB allocation and an 80 MB
memcpy on top of the HDF5 read.

Two options:

- **cheap:** when `into.is_empty()` (which it is — `read_data` clears it), `*into = converted`. That
  gives up the caller's capacity, which is the point of the `&mut Vec` API, so this is only a
  half-fix.
- **right:** read directly into the caller's buffer. `hdf5::Container::read_into_raw(&mut [T])`
  exists in `hdf5-metno` 0.14 and does exactly this. It needs the element type known before the
  read, which the reader knows (`T: ValueType`) but `hdf5_reader::read` currently discards by
  dispatching on the dataset's dtype into an owned `Values`. A `read_into::<T>(item, base_dir, buf)`
  fast path for the same-type case, falling back to today's `Values` route for widening
  (`f32` → `Vec<f64>` etc.), would make the common read allocation-free after the first call.

**Done, the `read_into_raw` way.** `hdf5_reader::read_into` resizes the caller's `Vec` to the
dataset and lets HDF5 fill it in place whenever the dataset's own dtype is already `T`; a dataset of
another type still goes through `Values`, where `ValueType` decides whether it widens or is a
mismatch. `selection::read_data_item_into` is the recursive wrapper -- a selection cannot be filled
in place, since it has to read its whole source before it knows what to keep, so that shape falls
back to the owned route -- and `TimeSeriesReader::read_data`/`read_submesh_field` fill the caller's
buffer directly. The `H5Type` bound this needs is a `cfg`-gated supertrait of
`reader::sealed::SealedValueType`, which is crate-private, so no public bound changed.

`tests/reader.rs`'s `reading_a_field_repeatedly_allocates_nothing_of_its_size` guards it with a
counting `#[global_allocator]`, asserting in *bytes* rather than allocation count (the path and name
handling makes counts brittle). Checked against the previous behaviour, where it fails with
"allocated 400039 bytes, about the 400000 bytes of the field itself".

`read_points_with_submeshes` built three full `Vec<C>` direction arrays and then interleaved them
into `points` — 2× peak. **Also done:** it now sizes `points` from the first direction and reads one
direction at a time into a reused scratch, scattering each straight into its stride of the output.
Peak 1.33×, and directions 2 and 3 refill the buffer the first one allocated — they also pick up the
`read_exact_into` in-place path, which the old `read_data_item` + convert bypassed.

## 6. `read_points`/`read_topology` drop the caller's buffer (read path)

Both are documented as "cleared first, so its existing capacity is reused". `read_points` clears
(`reader.rs:339`) and then `read_points_plain` does `*points = C::from_values(values)?`
(`reader.rs:702`) — replacing the `Vec` wholesale, freeing the caller's allocation and keeping the
one from the HDF5 read. Same for `*cell_types = decoded.cell_types` (`:722`) and
`*cell_types = global_cell_types` (`:860`).

Behaviourally harmless, but it means the buffer-reuse the API advertises does not happen on the
plain path (it does on the submesh path, which uses `reserve` + push). Either make it true, or drop
the sentence from the docs.

**Done, with finding 1(c).** The obstacle was that `Coordinate` and `ConnectivityIndex` carry their
own conversions with their own wording, so the fallback arm could not just reuse `ValueType`'s. The
fix was to make that arm a parameter: `hdf5_reader::read_exact_into` does the in-place fill and
reports whether the dataset's type matched, and `selection::read_data_item_into` takes a `convert`
closure for everything else — `T::from_values`, `C::coordinates_from_values` or
`I::indices_from_values` at the three call sites. `read_points_plain` and `read_topology_plain` now
fill the caller's buffer, so the sentence in their docs is true.

The three conversions had to be named apart (`SealedValueType::from_values` stayed;
`SealedCoordinate::from_values` and the new `SealedIndex` one became
`coordinates_from_values`/`indices_from_values`) because `SealedValueType` is now a supertrait of
both and `T::from_values` would otherwise be ambiguous.

`read_points_with_submeshes` now fills the caller's buffer too, by sizing it from the first
direction and scattering each direction into its stride (see finding 3).

## 7. Per-attribute HDF5 overhead in the writer (write path)

`SingleFileHdf5Writer::write_data` (`hdf5_writer.rs:148`) does, for **every attribute of every
step**: `time_group_name(time)` (a `format!`), `link_exists`, possibly `create_group`, `group()`,
`index.to_string()`, and `self.h5_file_name.to_string_lossy()` + `full_path` (another `format!`).
That is three HDF5 round-trips and 2–3 `String`s per attribute.

This is `02_performance.md` C5, unchanged: open the step's `Group` once in `write_data_initialize`,
cache the handle, and keep a scratch `String` for the dataset name.

`MultipleFilesHdf5Writer::write_data` (`:337`) has the same shape plus a `data_file.filename()`
call — an HDF5 API round-trip returning a fresh `String` — per attribute, to rebuild a path that is
constant for the whole step. Cache the relative name alongside the handle in
`write_data_initialize`.

Small in absolute terms (microseconds per attribute), but it is the per-step constant the plan's
`allocations_per_step` bench is meant to drive to zero.

## 8. The default chunk shape is 64 MB — measure this

`create_and_write` (`hdf5_writer.rs:448`) sets `.shuffle().deflate(level)` and never sets a chunk
shape, so `hdf5-metno` infers one: `Chunk::MinKB(DEFAULT_CHUNK_SIZE_KB)` where that constant is
`64 * 1024` **KB = 64 MB** (`hdf5-metno-0.14.0/src/hl/dataset.rs:36,465`).

For a 1-D dataset `compute_chunk_shape` then takes the whole extent when it is under ~2× the wanted
size, so in practice:

- arrays under ~8M elements (`f64`) → **one chunk covering the whole dataset**
- larger arrays → ~8M-element chunks

Consequences worth measuring rather than assuming:

- **Write:** the filter pipeline compresses one 64 MB block in a single `deflate` call, with a
  matching transient buffer. That is a memory spike and a serialization point.
- **Read:** the HDF5 chunk cache defaults to 1 MB, so a chunk this size always bypasses it. Harmless
  for today's read-everything pattern; fatal for the partial reads in finding 9.
- **Compression ratio** is essentially flat across chunk sizes in this range, so smaller chunks are
  close to free on size.

Suggested experiment: sweep an explicit `.chunk([n])` at 1, 4, 16 MB against the default on the
1e7 CFD case from `SESSION_2026-08-08_cfd_benchmark.md`, reporting write time, read time, peak RSS
and compressed size. If a smaller chunk is neutral-to-better on all four, set it explicitly — it
also unblocks finding 9. Note the existing constraint from `CLAUDE.md`: ParaView's bundled HDF5
supports only core filters, and chunk shape does not affect that.

## 9. Selections read the whole array (read path, latent)

`Membership::apply` (`reader/selection.rs:53`) evaluates a `HyperSlab`/`Coordinates` selection by
reading the **entire** source array via `read_data_item` and then slicing or gathering out of it in
memory (`slice_owned`/`gather_owned`, `:189`/`:200`).

Today this is mostly off the hot path — `read_point_data`/`read_cell_data` on a mesh with submeshes
deliberately read the *source* of a selection rather than applying it (`read_submesh_field`,
`reader.rs:505`), and `read_points_with_submeshes` does the same. So `apply` fires only for a
foreign document, or a selection this crate did not write.

It becomes the hot path the moment per-submesh reads land (the open `[ ]` item in
`10_submeshes_reader_review.md`): reading one submesh's field would read and decompress the whole
mesh's array, then keep a fraction of it. Reading all K submeshes that way is K× the whole array.

The fix when that happens is `Dataset::read_slice` with an HDF5 hyperslab (contiguous case) or a
point selection (scattered case), pushing the selection down to HDF5 instead of doing it in Rust —
which needs the chunk shape from finding 8 to be smaller than the whole dataset to pay off. Worth
noting now so per-submesh reads are not built on `apply`.

## Not findings

- **`python/src/arrays.rs` / `writer.rs`.** Numpy buffers are borrowed via `PyReadonlyArrayDyn`
  and handed to the core crate as `Values::*(Cow::Borrowed(..))`; non-contiguous arrays are rejected
  rather than silently copied; the write runs under `py.detach`. There is no copy between numpy and
  HDF5 on the attribute path. `extract_cell_types` and `extract_submesh_cells` do copy, but run once
  per mesh and say so.
- **`GatherBuffers`** (`values.rs:147`) is reused across submeshes and across steps, and both the
  mesh-write and per-step gathers borrow rather than allocate. The contiguous case takes
  `Values::slice`, which is a borrow. This is right as it is.
- **`LocalPoints`** (`time_series_writer.rs:1290`) — the lookup-array-over-binary-search decision is
  documented with the measurement behind it, and is only built for non-contiguous submeshes.
- **`hdf5::read_raw`** is a single allocation and a direct read into it; no intermediate `ndarray`.
- **`Analysis`/`GridPath`** resolution is O(depth), not O(submeshes), as the module doc claims.
- **`paraview::validate`** is `Ok(())` immediately for `Format::HDF` on every type except `u64`, so
  it is not a per-step pass over the data for the usual `f64`/`f32` fields.

---

## What is left, in order

1. Finding 8 (chunk sweep) — measurement, no code, and it gates finding 9.
2. Finding 2 (`02_performance.md` part B) — now measured: worth it for small-mesh/long-run cases and
   for the unbounded memory, close to worthless for the 1e7 CFD case. Note that part B was written
   before submeshes and does not account for the one-temporal-collection-per-submesh document shape.
3. Findings 7, 9 — per-step write constants, and pushing selections down to HDF5 before per-submesh
   reads get built on `Membership::apply`.

The findings implemented here were verified by the test suite and by two targeted allocation
tests; none of them has been *benchmarked*. `02_performance.md` part A's harness is still the
prerequisite for numbers — the finding-2 driver above was thrown away rather than kept, and is the
kind of thing part A should own permanently.
