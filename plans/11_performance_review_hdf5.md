# Performance review — Python write → Rust read, HDF5 storage

Static review (2026-08-23, `submeshes` branch). Scope: the path a caller actually takes when
writing from Python and reading back in Rust, both with `DataStorage::Hdf5*`. Nothing was
measured; every claim below is read off the code, and each finding names what to measure.

Overlaps with `02_performance.md` are marked — parts B and C5 of that plan are still unimplemented
and are still the two biggest write-side items.

---

## Summary

The **Python → Rust boundary is already clean**: numpy buffers are borrowed, never copied, and the
GIL is released around the write. Nothing in `python/src/` needs work.

The costs are all in the core crate:

| # | Where | Cost | Priority |
|---|-------|------|----------|
| 1 | `read_topology` | ~5× the connectivity's own size in peak RSS, 4 full copies | **high** (read) |
| 2 | `write_xdmf_file` per step | O(steps²) serialization + O(steps) retained memory | **high** (write) — `02_performance.md` part B |
| 3 | `prepare_cells` uniform arm | full copy of the connectivity, avoidable | **high** (write) |
| 4 | `hdf5_reader::read` | one `H5File::open` per array read | medium (read) |
| 5 | `TimeSeriesReader::read_data` | one full extra copy of every field read | medium (read) |
| 6 | `read_points` | drops the caller's buffer instead of reusing it | medium (read) |
| 7 | `SingleFileHdf5Writer::write_data` | 3 HDF5 round-trips + 2 `String`s per attribute per step | low — `02_performance.md` C5 |
| 8 | default chunk shape | 64 MB minimum chunk, one chunk for most arrays | low, but **measure** |
| 9 | `Membership::apply` | reads a whole array to keep a slice of it | low today, high if per-submesh reads land |

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

The submesh path (`read_topology_with_submeshes`, `reader.rs:783`) is worse in a different way: it
holds **every** submesh's decoded topology in memory simultaneously (`decoded`, `:800`) plus the
global `connectivity`/`cell_types`/`offsets`/`covered` arrays. Peak is roughly the whole mesh twice
over plus one `bool` and one `usize` per cell. Streaming one submesh at a time into the global
arrays would drop that to ~1× — the scatter loop already visits them independently.

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

Same applies to `read_points_with_submeshes` (`reader.rs:730`), which builds three full
`Vec<C>` direction arrays and then interleaves them into `points` — 2× peak. Interleaving needs all
three, so that one is inherent; but `points.reserve` + push is correct there and worth keeping as
the model for the rest.

## 6. `read_points`/`read_topology` drop the caller's buffer (read path)

Both are documented as "cleared first, so its existing capacity is reused". `read_points` clears
(`reader.rs:339`) and then `read_points_plain` does `*points = C::from_values(values)?`
(`reader.rs:702`) — replacing the `Vec` wholesale, freeing the caller's allocation and keeping the
one from the HDF5 read. Same for `*cell_types = decoded.cell_types` (`:722`) and
`*cell_types = global_cell_types` (`:860`).

Behaviourally harmless, but it means the buffer-reuse the API advertises does not happen on the
plain path (it does on the submesh path, which uses `reserve` + push). Either make it true, or drop
the sentence from the docs. Making it true falls out of finding 5's `read_into_raw` fix.

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

## Suggested order

1. Finding 3 (`Cow` in `prepare_cells`) — smallest diff, largest single write-path allocation.
2. Finding 1 (a) and (b) — two small changes, −2C peak on every topology read.
3. Finding 5 + 6 (`read_into_raw` fast path) — makes repeated reads allocation-free and makes the
   documented buffer reuse true.
4. Finding 4 (open-file cache) — contained, helps every read.
5. Finding 8 (chunk sweep) — measurement, no code, but gates finding 9.
6. Finding 2 (`02_performance.md` part B) — largest win, largest change, needs the submesh document
   shape accounted for.
7. Findings 7, 9 — per-step constants and future-proofing.

Each of 1–4 should show up on the `write_mesh` / `read_*` benches and on the allocation counter that
`02_performance.md` part A specifies; that harness is still the prerequisite for confirming any of
this.
