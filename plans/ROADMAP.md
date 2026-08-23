# Roadmap to 1.0

Master plan. `README.md` in this folder is the feature wish-list; this document turns it into an
ordered set of milestones, records the decisions that were made up front, and links the per-feature
sub-plans.

## Decisions taken (2026-08-08)

These were open questions before planning; they are settled and every sub-plan is written assuming
them. Revisit deliberately, not by drift.

| # | Question | Decision |
|---|----------|----------|
| 1 | Error type (`API_IMPROVEMENTS_PLAN.md` item 5) | **Dedicated `Error` enum**, landed *before* the reader so reader errors are typed from day one. |
| 2 | Reader scope at 1.0 | **Round-trip own output + best-effort common subset** of foreign XDMF2. Anything else → explicit `Unsupported` error. |
| 3 | Python wheels and HDF5 | **Static HDF5 built into the wheel**, so `pip install` gives working HDF5 storage with no system library. |
| 4 | `Values` numeric types | **Add `F32`**, plus an opt-in f64→f32 downcast on write. Not a full numeric type set. Unchanged by the benchmark findings, but the *size claim* is now gated on a measurement — see `03_values_and_f32.md`. **Amended 2026-08-16:** `write_mesh` also becomes generic over f32/f64 points via a sealed `Coordinate` trait (Part 3 of that plan), replacing the earlier "coordinates stay f64" position. |
| 5 | Submeshes / blocks | **Overlapping blocks stay allowed**; the per-step copies get optimized away instead of the semantics being restricted. |
| 6 | Light-data (XML) writing | **Append + patch tail.** The `.xdmf2` stays a complete, openable file after every step, but the per-step cost becomes O(1) instead of O(steps). |
| 7 | `multiple-features` branch | **Cherry-pick selectively**, treat the rest as a reference implementation. Main has diverged too far to merge. |
| 8 | MPI (post-1.0) | Design around **one global grid, ranks writing hyperslabs** into shared HDF5 datasets. |

## Progress

- **2026-08-09, PR #18 "Several improvements to the API" (`aa3c501`, merged to `main`).** Landed
  part of M0: the `write_data` rework (`Values<'a>` is now `Cow`-backed, `write_data` takes flat
  `(&str, DataAttribute, Values)` triples instead of `DataMap`) and `API_IMPROVEMENTS_PLAN.md`
  item 3 (flattened `write_mesh`'s cell tuple). Details and links in `API_IMPROVEMENTS_PLAN.md`'s
  top-of-file status note and in `02_performance.md` part F.
- **2026-08-09, PR #19 "small fixes and improvements" (`fa2bbee`, merged to `main`).** The
  remaining `API_IMPROVEMENTS_PLAN.md` items: item 1 (writer poisoning — `write_data_finalize`
  now always runs, and a new `DataWriter::validate_values` lets `BinaryWriter` catch its u64→u32
  range check before `write_data_initialize` rather than mid-write), item 2 (time dedup now keyed
  on `f64::to_bits` instead of the string), and item 4 (`deflate_level` validated in
  `create_writer` before any writer is built). Full details in each item's own "DONE" note in
  `API_IMPROVEMENTS_PLAN.md`. **This closes out M0's items**, except for the O(steps²) light-data
  rewrite (`02_performance.md` part B, decision 6), which is a separate, larger piece of M0/M2 and
  was never part of `API_IMPROVEMENTS_PLAN.md`.

- **2026-08-16, M3 Parts 1 and 3.** `Values::F32` and f32-or-f64 points in `write_mesh` (sealed
  `Coordinate` trait), including the `dimensions()` restructure and the HDF5 backend's points
  dataset now going through `write_values` like any other data. The ParaView smoke fixture writes
  an f32 fixture alongside the f64 one; both verified locally on 5.13.2 and 6.1.1 across all five
  storage backends. Part 2 (`with_reduced_precision`) and its measurement gate remain open — see
  `03_values_and_f32.md`.

- **2026-08-18, M6 Part 1 (writer only), branch `python-interface`.** The Python bindings, rebuilt
  against the current API rather than cherry-picked — both reference implementations
  (`origin/multiple-features`, `origin/reader`) predate the `TimeStep` builder and `paraview.rs`.
  `python/` is now a workspace member: numpy arrays are borrowed with no copy for points,
  connectivity and attribute data, at whichever of the six dtypes they are passed in; `DataWriter`
  gained a `Send + Sync` supertrait so writes release the GIL (measured 2.6× on 4 threads); the data
  writer is a context manager; errors map per `Error` variant, with `IntegerOutOfRange` →
  `OverflowError`; `xdmf.pyi` stubs ship with the wheel and a test fails if the module outgrows them.
  46 pytest tests, plus a CI job (`cargo clippy -p xdmf-python` + `pip install ./python[test]` +
  `pytest`), and the strict clippy list moved to `[workspace.lints]` so it covers the new crate.
  A post-merge review then found nine issues, eight fixed on 2026-08-19 (transposed point arrays
  silently written as a wrong mesh, `int32` cell codes rejected, a rejected `write_mesh` burning the
  writer, unhashable value types, the contiguity message not naming the offending array, and three
  smaller ones); the `deflate_level` range check was written as an `i64` fix and then deliberately
  reverted back to `u8`, so it stays the core's alone rather than duplicating its bound and message.
  One fix is a core-crate bug: `num_entities * DataAttribute::size()` was multiplied out unchecked,
  so in a release build an absurd `Generic`/`Matrix` size could wrap back onto the real array length
  and be written as a corrupt shape, and a zero one divided by zero — both now one `InvalidData`.
  **Open:** the reader bindings (M5 first), the wheels (Part 2), the pyvista re-run (Part 3) — and
  M2/M4/M5 will each move the Rust API this layer wraps. Details in `06_python_bindings.md`'s
  top-of-file status note.

- **2026-08-19, M4 Stage A + optimizations 1 and 2.** `write_mesh_with_submeshes`: one shared
  geometry, one uniform sub-grid per submesh inside a spatial collection, overlapping submeshes
  allowed (decision 5), cell data still passed globally and sliced by the writer. Contiguous cell
  index runs collapse to `{start, len}` so their share of a cell field is a borrow of the caller's
  array; scattered ones gather into one reused buffer per element type. `DataWriter::write_mesh`
  split into `write_points` + `write_connectivity`. Named "submesh", not "block". Verified in
  ParaView 5.13.2 and 6.1.1 across all five storages. **Open:** hyperslab references (optimization
  3), the allocation test (needs M2's harness), and the reader side — the ParaView smoke fixture
  and the Python bindings followed on 2026-08-21, see below. Details in `04_submeshes.md`'s
  top-of-file status note.

- **2026-08-20, rebased onto `perf: support uniform topology` (#28) and `some python improvements`
  (#27).** Main independently landed the same `checked_mul` fix for item 7 above (word-for-word,
  so the rebase merged it without a conflict) and gained a uniform-`TopologyType` fast path in
  `prepare_cells` that skips the per-cell type prefix when every cell shares one `CellType`. That
  fast path changes the connectivity's byte layout, so `cell_offsets` (which slices a submesh's
  share of it) needed a matching uniform-stride branch alongside its existing `Mixed` one, and
  `nodes_per_element` is now computed once per mesh and reused for every submesh grid, since a
  submesh's cells are always a subset of the same uniformly-typed connectivity. Covered by a new
  `write_xdmf_with_submeshes_of_a_uniform_topology` test.

- **2026-08-21, M4 finished off: Python bindings, the smoke fixture, and flat heavy-data naming.**
  `TimeSeriesWriter.write_mesh_with_submeshes` is exposed to Python (cell indices as a numpy
  integer array of any dtype or a sequence of `int`), and `paraview_smoke` gained a multi-block
  fixture — a single-cell, a contiguous, a gathered and an overlapping submesh, plus a `Vector`
  cell field — that CI verifies through `pvpython` against ParaView 5.13.3 and 6.1.1 for every
  storage. Separately, the backends stopped naming heavy data after the caller's field name and
  now number it (deviation 5 in `04_submeshes.md`), which is what allows solver-style names such as
  `Quantity('SOOT DENSITY')` and is checked by a fixture of its own.

- **2026-08-21, the review's code findings.** Two are fixed. Point data is now written once per
  step into a named `Domain`-level `DataItem` that every block references, instead of each block
  carrying a copy — for `DataStorage::AsciiInline`, where the item *is* the data, that turns one
  copy of the field per block per step into one per step. Verified in ParaView 5.13.3: the
  reference resolves inside an `<Attribute>`, but only carries its extent if the referencing
  `DataItem` keeps its `Dimensions`, which the geometry and topology references do not need — the
  reader logs `Dimensions of Attribute not set` and reads nothing otherwise. `prepare_submeshes`
  no longer holds a mesh-sized `Vec<usize>` (8 bytes/cell) to answer its two membership questions:
  two `CellBitSet`s cost a quarter of a byte per cell between them, the per-submesh one being
  cleared by walking that submesh's own indices rather than the mesh.
  `TimeSeriesDataWriter` also now keeps the `Xdmf` document as its state and appends one `<Grid>`
  per completed step, rather than rebuilding the document from stored attributes on every rewrite.
  **That is a simplification, not the speedup it was expected to be:** rewriting the file after
  every step costs O(steps² × submeshes), and measurement puts that cost in serialization, not in
  the cloning it removed — 50 submeshes over 400 steps still takes ~50 s, and the last step alone
  spends 228 ms regenerating a 19 MB document of which ~48 KB is new. That is decision 6's territory
  (`02_performance.md` part B), and the measurement sharpens it: the per-step cost has to drop by
  not re-serializing the history, so each completed step's `<Grid>` wants caching as rendered bytes
  that the rewrite splices in — which needs an escape hatch for pre-rendered markup in
  `xdmf_elements`, since serde cannot emit raw XML. Submeshes multiply the constant by the block
  count, so M4 makes this more pressing than it was. **Open.**

- **2026-08-21, block names made stable across time steps.** ParaView's Xdmf2 reader makes a grid
  name unique across the whole document, so the `Temporal`-of-`Spatial` nesting M4 first used —
  which names every submesh once per step — gave back `quad`, `quad[1]`, `quad[2]`, ... as the
  animation advanced. The block changed identity per step, losing its per-block visibility and
  colouring in the Multi-block Inspector, and the `submesh_fixtures` check failed from the second
  step on for every storage. Inverted to a `Spatial` collection of one `Temporal` collection per
  submesh, so each block name occurs once in the document (deviation 6 in `04_submeshes.md`). The
  full smoke suite now passes on 5.13.3 for all five storages, where the submesh fixture previously
  failed on all five. Guarded in Rust by `write_xdmf_with_submeshes_names_each_block_once`, which
  is cheap enough to run without ParaView.

- **2026-08-21, the reader's `Grids` list is not a submesh selector.** Measured on 5.13.3: the
  Properties panel's `Grids` checkboxes list the file's *uniform* grids, which for a time series is
  one per (submesh, step) — `edge-t0.5`, `edge-t1.5`, ... — so a selection made there holds for one
  step only, and the reader applies it shifted against time (keeping just `edge-t0.5` gives `edge`
  at t=1.5 and everything *but* `edge` at t=0.5 and t=2.5). Not submesh-specific: a plain time
  series lists `time_series-t0.5, ...` the same way. Two layouts that could shorten that list were
  tested and are worse — naming every per-step grid after its submesh gives five uniquified entries
  for nine grids, omitting the names gives `Grid_20, Grid_26, ...` — while block names and cell
  counts stay correct in all three. Nothing to change in the writer: the Multi-block Inspector and
  `Extract Block` select by block name and were measured stable across every step, so `README.md`
  and `examples/submeshes.rs` now point at those and warn off the `Grids` list.

- **2026-08-21, submeshes record which cells they hold.** Points keep their identity across blocks
  (one shared array, indexed globally), cells did not: a block's connectivity is indexed locally,
  blocks may overlap, and a mesh written with submeshes writes no connectivity of its own -- so
  nothing in the file said which cell of the mesh a block cell was, and M5 could only have returned
  the cells permuted against the caller's global indexing. `DataWriter::write_submesh_cells` now
  writes a scattered submesh's global cell indices once, at mesh-write time, and one `<Information
  Name="submesh_cells">` lists per submesh either `<start>:<len>` (contiguous -- no array at all,
  which is what mesh generators produce) or that item's name. Kept a side channel rather than a
  `global_cell_id` cell attribute: an unreferenced `DataItem` plus an `Information` was measured
  ignored by ParaView 5.13.3 (same blocks, same cells, no extra array), so users do not get a field
  they did not ask for next to their own. Disallowing overlap was considered and rejected -- it
  would still need the permutation recorded, at the same cost, and decision 5 keeps overlap.

- **2026-08-21, ParaView 6.1 renames gathered blocks, not selectable ones.** The submesh fixture
  failed on 6.1 for every storage with `expected blocks ['both', ...], got ['Block 0_both', ...]`.
  Measured on 5.13.3 and 6.1.0 side by side: `vtkXdmfReader` emits the same block names in both, and
  so does the block hierarchy ParaView selects by (`/Root/quad`, what the Multi-block Inspector
  lists and `BlockSelectors` takes) — only `servermanager.Fetch`'s client-side gather differs, 6.1
  folding the outer collection's generated name into each leaf's. Nothing in the file or the writer
  is version-dependent, and probing four other nestings (no spatial wrapper, unnamed wrapper, a
  third level, `Temporal`-of-`Spatial`) changed nothing, so there is no layout that avoids it.
  `verify_with_pvpython.py` strips the gather's `Block <n>_` prefix and, so that the stripping
  cannot hide a real rename, now checks the block *paths* exactly, per time step — which is the
  stronger assertion anyway, since those paths are what a user's block selection is made of.

- **2026-08-21, what many small blocks cost in the viewer.** Every block references the whole
  `coords` `DataItem`, so ParaView materializes the full point set and every point field once per
  block: measured `blocks × mesh` exactly, ~1.24 MB per block on a 40,401-point mesh regardless of
  what the block holds — 2.8 MB at one block, 317.8 MB at 256, identical on 5.13.3 and 6.1.0. A
  hand-written compacted variant of the same mesh (each block carrying only the points its cells
  touch) stays at 4.8 MB, at the price of +73% heavy data, +28% light data, 4× the files and 2×
  the load time at 256 blocks, plus a per-block point gather per step and a `submesh_points` side
  channel for the reader. **Implemented the same day** (deviation 8 in `04_submeshes.md`), which
  reproduced the predicted numbers exactly: 4.8 MB at 256 blocks against 317.8 MB, with the writer's
  own output. Point data is cut per submesh from now on, as cell data already was, so the shared
  `Domain`-level point-data item is gone; `submesh_points` joins `submesh_cells` as the side channel
  a reader needs, since a block's points are no longer the mesh's own. Verified against ParaView
  5.13.3, 6.1.0 and 6.1.1 for all five storages, with the smoke fixture now checking each block's
  points, its cells in that block's own numbering, and its share of the point data. The README's
  "hundreds of tiny blocks are worth avoiding" is retired with it.

- **2026-08-22, what a submesh may reference instead of copying — measured, nothing implemented.**
  The gate `04_submeshes.md` left open for optimization 3, answered for the whole question behind it:
  `ItemType="HyperSlab"` and `ItemType="Coordinates"` both work in `Geometry`, `Topology` and
  `Attribute` — but **only for `Format="HDF"`**. `Format="Binary"` honours `HyperSlab` and silently
  ignores `Coordinates`; `Format="XML"` ignores both and reads from the start of the array, which is
  the `paraview.rs` class of defect and rules a selection out for the ascii storages entirely.
  Crucially, a selection does *not* bring back the `submeshes × mesh` memory of the old shared
  geometry: ParaView materializes only the selected values, so a referencing layout holds exactly
  what the compacted one holds. Four layouts measured on the 40,401-point mesh at 16/64/256
  submeshes over 10 steps, on 5.13.2 and 6.1.1 alike. Recommended: keep compacted geometry, write
  each field **once per step** globally and let every submesh select its share — 18% less heavy data
  at 64 contiguous submeshes, 41% at 256, 56% on a scattered split, with the per-step cost no longer
  depending on the submesh count or on overlap, and no per-submesh gather left on the writer's hot
  path. Not recommended: referencing the geometry too (the literal goal — works, smallest on disk,
  but 2.4–4× the light data and up to 5× the read time for a few percent). Full support matrix,
  numbers and the reproducible harness in
  [`09_submesh_references.md`](09_submesh_references.md).

- **2026-08-22, the recommendation implemented: HDF5 submeshes select, they no longer copy.** Each
  field now goes to the heavy data once per step, exactly as the caller passed it, and every
  submesh's `<Attribute>` selects its share -- a `HyperSlab` for a run of entities, a
  `Coordinates` selection through the submesh's index list otherwise. Measured with the writer's
  own output at 64 submeshes over 10 steps: 27.2 MB -> 18.4 MB of heavy data contiguous, 65.1 MB
  -> 24.2 MB strided, for +60% light data and about a third longer to step through the animation;
  same blocks, same values, same viewer memory on ParaView 5.13.2 and 6.1.1. Four things the
  implementation had to settle beyond the measurement: the selection is one-dimensional whatever
  shape the field has (`ParaView` matches its rank against the `HDF5` dataset's, and this crate
  writes every array flat), which also halves the index arrays; a scalar field needs no new array
  at all, since one index per entity is what `submesh_points`/`submesh_cells` already hold for the
  reader; those lists are written signed now that `ParaView` reads them; and a submesh whose cells
  are not ascending still gets a copy, because a `Coordinates` selection comes back in array order
  rather than in the order it named. The ascii and binary storages are untouched -- a selection
  there is silently wrong, and a test guards that none is emitted. Details in
  [`09_submesh_references.md`](09_submesh_references.md)'s "As implemented".

- **2026-08-22, the geometry too: HDF5 submeshes select their points out of the mesh's.** The one
  layout that measurement had recommended against, adopted after all -- not for its few percent of
  disk, but because it retires the side channel: the mesh's coordinates are written once, as one
  array per direction, and a submesh's `<Geometry GeometryType="X_Y_Z">` selects the points it
  holds through `submesh_points`, so the list of mesh points a block holds *is* part of its
  geometry rather than an `<Information>` nothing references -- which that `<Information>` is no
  longer written at all on this path, the geometry having taken its job. Split by direction because that is
  what lets all three selections share one index list; a submesh whose points are a run needs no
  array at all. At 64 submeshes over 10 steps: 18.4 MB -> 17.2 MB of heavy data contiguous, 24.2 MB
  -> 20.2 MB strided (65.1 MB before any of this), for +10% light data and a further ~10% of read
  time. Same blocks, same points materialized, same 4.9 / 10.6 MB in ParaView 5.13.2 and 6.1.1 --
  referencing the mesh's coordinates through a selection does not bring back the memory blow-up
  that referencing them whole caused. The ascii and binary storages keep their compacted copies.
  See [`09_submesh_references.md`](09_submesh_references.md)'s "As implemented, geometry".

- **2026-08-22, M5 Stage 1 landed: `TimeSeriesReader`/`TimeSeriesDataReader` for `Format="HDF"`.**
  `src/reader/{mod.rs -> reader.rs, light_data, hdf5_reader, selection, topology}.rs`, per
  `05_reader.md`. Risk 1 confirmed and fixed on day one: `quick-xml`'s serde cannot deserialize
  `#[serde(flatten)]` combined with a `$value` variant (`"no variant of enum DataContent found in
  flattened data"` on every shape), so `DataItem` keeps its derived `Serialize` but gained a
  hand-written `Deserialize` against a non-flattened intermediate struct -- fixes every shape
  except a nested `xi:include` (irrelevant to Stage 1, `Format="HDF"` never writes one; noted as a
  stage 2 follow-up in `data_item.rs`). `CellType::from_code` added next to the existing
  discriminants as the inverse of `prepare_cells`'s `as u8` cast. `Error` gained three variants
  (`InvalidDocument`, `Unsupported`, `NumberTypeMismatch`) rather than the larger set
  `01_error_type.md` originally sketched, following the grouped-by-category shape the error type
  actually shipped with. Named consistently with the writer throughout ("submesh", not "block",
  the plan's own API sketch used "block" and the first pass here followed it uncritically).
  `TimeSeriesReader::new` eagerly computes `num_points`/`num_cells`/`times`/`submesh_names` (one
  read of a scattered submesh's own `submesh_cells` array in the worst case) so those accessors
  stay infallible, deviating from the plan's fallible-`Result`-free sketch only there. Reading a
  mesh with submeshes follows the plan's reconstruction table exactly: global coordinates read
  directly from `mesh/points/{0,1,2}`, submesh cell/point membership from `submesh_cells` and each
  submesh's `<Geometry>` selector, connectivity scattered through both after decoding each
  submesh's own topology. Per-step field reads prefer a submesh's selection source (the whole
  field, written once) over scattering from every submesh's own share when one exists --
  discovered during testing to be load-bearing, not just an optimization: it is the only path that
  recovers a point no cell uses, which the scatter-only approach silently zeroed. `DataInfo`'s
  `attribute: DataAttribute` collapses `Tensor6`/`Matrix(n, m)`/`Generic` back to `Generic(size)`
  on read, since `AttributeType::Matrix` does not distinguish them in the file; documented as a
  known lossy spot rather than guessed at. `tests/reader.rs` covers the Stage 1 test matrix (mesh
  only, point cloud, all 19 `CellType`s, several steps of point/cell data, f32/u64 attributes,
  every `DataAttribute` shape, contiguous/scattered/overlapping submeshes with an unused point, and
  a submesh with deliberately unordered cells) across both `Hdf5SingleFile`/`Hdf5MultipleFiles`.
  Stage 2 (`Format="XML"`/`"Binary"`) is open.

- **2026-08-22, M5's reader API collapsed to one type.** Review question: does splitting
  `TimeSeriesReader`/`TimeSeriesDataReader` (mirroring `TimeSeriesWriter`/`TimeSeriesDataWriter`)
  actually earn its keep on the read side? No -- the writer's split is forced by writing the mesh
  being a one-time, irreversible file mutation; `TimeSeriesReader::new` already parses the whole
  document, so reading has no such phase to enforce, and the split only made per-step reads require
  reading the mesh geometry first for no real reason. Collapsed into a single `TimeSeriesReader`:
  submesh membership moved from `read_mesh` into `new` (the cell half was already computed there
  for `num_cells`), every method takes `&self`, and `read_mesh` itself split into `read_points`
  (fully independent of everything else) and `read_topology` (connectivity and cell types stay
  paired -- decoding `Mixed` connectivity is one pass that produces both, and a submesh mesh can't
  place connectivity before cell types are scattered first). `submesh_cells`/`submesh_points`
  (recovering which mesh cells/points a submesh holds, symmetric with what
  `write_mesh_with_submeshes` takes in) were added in the same pass -- the first cut of the reader
  had reconstructed the merged mesh but left no way back to the split. `tests/reader.rs` and the
  README's reading example updated to match; `05_reader.md`'s API section keeps the original
  two-phase sketch alongside a note on why it changed.

## Sub-plans

| Plan | Milestone | Covers |
|------|-----------|--------|
| [`API_IMPROVEMENTS_PLAN.md`](API_IMPROVEMENTS_PLAN.md) | M0 | Pre-existing defects and interface cleanups (see its "Adjustments" section) |
| [`01_error_type.md`](01_error_type.md) | M1 | The `Error` enum, migration of 29 message assertions and 49 `IoResult` signatures |
| [`02_performance.md`](02_performance.md) | M2 | Benchmark harness, O(n²) light-data write, allocation elimination, HDF5 tuning |
| [`03_values_and_f32.md`](03_values_and_f32.md) | M3 | `Values::F32`, f32-or-f64 points in `write_mesh`, opt-in f64→f32 downcast, ParaView verification |
| [`04_submeshes.md`](04_submeshes.md) | M4 | `write_mesh_with_blocks`, zero-copy fast paths, block validation |
| [`05_reader.md`](05_reader.md) | M5 | `TimeSeriesReader` / `TimeSeriesDataReader`, per-format `DataReader` backends; split into stage 1 (`Format="HDF"`) and stage 2 (`XML`/`Binary`) |
| [`06_python_bindings.md`](06_python_bindings.md) | M6 | Review of the vibe-coded bindings, GIL release, abi3 + static-HDF5 wheels on PyPI |
| [`07_mpi.md`](07_mpi.md) | post-1.0 | API draft only: collective sizing, global node ids, verification strategy |
| [`08_write_data_builder.md`](08_write_data_builder.md) | M7 (before M6 Part 2) | Per-attribute `TimeStep` builder replacing `write_data`'s tuple lists; intra-step buffer reuse; `Values` out of caller code |
| [`09_submesh_references.md`](09_submesh_references.md) | M4b (after M4, before M5) | Submeshes referencing shared arrays: what ParaView's Xdmf2 reader supports per storage format, four layouts measured; `select` implemented, then its geometry half too |

### Historical records (not plans to execute)

| Document | What it is |
|----------|------------|
| [`SESSION_2026-07-18_python_bindings.md`](SESSION_2026-07-18_python_bindings.md) | How the `multiple-features` Python bindings were built and why; input to `06_python_bindings.md` |
| [`SESSION_2026-08-08_cfd_benchmark.md`](SESSION_2026-08-08_cfd_benchmark.md) | The xdmf-vs-pyvista benchmark, the "final archive" fairness rule, and the entropy finding that now governs `02_performance.md` part E |

## Milestones and why they are in this order

```
M0  cleanup + defects  ──┐
                         ├──▶ M1  error type ──▶ M2  performance ──▶ M3  f32 ──┬─▶ M4  submeshes ──┐
                         │                                                     └─▶ M5  reader ─────┤
                         │                                                                         │
                         └─────────────────────────────────────────────────────────────────────────┴──▶ M6  python + wheels ──▶ 1.0 ──▶ MPI
```

The hard ordering constraints, each of which exists for a reason:

- **M1 before M5.** The reader roughly doubles the number of distinct failure modes. Designing them
  as `io::Error` strings and converting later means designing the reader's error surface twice.
- **M2 before M4.** Blocks multiply the per-step light-data and per-step gather work by the number of
  blocks. Doing them on top of the O(steps²) writer means building the expensive thing on the
  expensive foundation, and it makes the block work's own cost impossible to measure.
- **M3 before M5.** `f32` is a `Values` variant the reader must handle. Adding it after the reader
  means touching every `DataReader` backend twice.
- **M6 last.** The bindings wrap the final public API surface. Every earlier milestone is a breaking
  change to the Rust API, and therefore to the pyo3 layer that wraps it.
- **Benchmark harness inside M2, in Rust.** The existing benchmarks (`python/benchmarks/` on the
  branch) need the Python bindings, which are M6. A Rust-native harness unblocks M2 and is a better
  instrument anyway — it can count allocations, which is the actual goal from `README.md`.

M4 and M5 are independent of each other and can be done in either order or in parallel. The reader
does need to grow block support once M4 exists; that is called out in `05_reader.md`.

- **M4b before M5** (added 2026-08-22, now that M4b has landed). The HDF5 submesh layout is what the
  reader reconstructs a mesh from, and it changed twice during M4b — writing it against the
  pre-M4b layout would have been thrown away. It also decides where M5 starts: `05_reader.md`'s
  stage 1 reads `Format="HDF"` only, because that is the storage whose output round-trips exactly.

## What to take from the `multiple-features` branch

Decision 7: cherry-pick, do not merge. `origin/main...origin/multiple-features` is 31 files, and
`values.rs` / `time_series_writer.rs` have been substantially rewritten on *both* sides since the
merge base.

**Take:**

- `python/benchmarks/{mesh_gen,cfd_benchmark,random_benchmark,bench_common}.py` — the duct-mesh
  generator and the "final archive" benchmark methodology are sound and reusable. The mesh generator
  gets ported to Rust for M2; the Python ones stay for the M6 pyvista comparison.
- `CFD_BENCHMARK_PLAN.md` → move into `plans/` as a findings record. It is the evidence behind the
  current `DEFAULT_DEFLATE_LEVEL = 3` and behind the "don't use Blosc" conclusion, and that evidence
  should not live only on a side branch.

  > **Correction to `SESSION_2026-08-08_cfd_benchmark.md`.** That summary states these files "were
  > not found in this working tree" and concludes that redoing the benchmark means rebuilding from
  > scratch. They are not in the working tree, but they *are* committed on `origin/multiple-features`
  > — verified with `git ls-tree -r --name-only origin/multiple-features`, which lists
  > `CFD_BENCHMARK_PLAN.md` and all four `python/benchmarks/*.py`. Retrieve them with
  > `git show origin/multiple-features:<path>`; do not rewrite them from scratch. This matters
  > because the branch's `CFD_BENCHMARK_PLAN.md` also carries the HDF5-with-compression measurements
  > and the Blosc/ParaView failure analysis, neither of which is in the session summary.
- `PYTHON_BINDINGS_PLAN.md` → move into `plans/` as reference input for `06_python_bindings.md`.
- `python/src/*.rs` and `python/{Cargo,pyproject}.toml` — as a **reference implementation to review
  line-by-line**, not as commits. See `06_python_bindings.md` for the review checklist.
- `examples/submesh_blocks.rs` — as a reference for what the block API should feel like.
- `src/values.rs`'s sealed `ValueType` trait + `as_slice::<T>()` — but **only when M5 needs it**.
  Cherry-picking it now would be speculative public API (`CLAUDE.md`); the reader is the caller that
  justifies it.

**Drop:** everything else under `src/` (superseded by current main plus the `Cow` rework merged in
PR #18), `src/binary_writer.rs` (already on main), `.github/workflows/paraview.yml` (already on main).

**Then delete the branch**, so there is one place to look.

## Repository housekeeping (do this first, it is five minutes)

- ~~`reader.rs` and `xdmf_write_{data,mesh}.xdmf2` are untracked files sitting in the repo root~~ —
  **done.** Both are gone from the working tree as of 2026-08-09.
- ~~`plans/` itself (this file included) is untracked~~ — **done.** Committed in `efc2db1`.

## Release strategy

Every milestone from M1 onward is a breaking change. The crate is `0.1.x` and explicitly not keeping
backward compatibility (`CLAUDE.md`), so:

- Release a `0.2`, `0.3`, … per milestone rather than accumulating one enormous 1.0 diff. Each is
  small enough to review and to bisect against, and arotau can adopt them one at a time.
- Keep a `CHANGELOG.md` from M1 onward, with a "how to migrate" section per breaking release. There
  is one known external consumer (arotau) plus the Python package; both need the migration notes.
- 1.0 is cut when M6 lands and the API has been stable through at least one arotau upgrade cycle.
- `CLAUDE.md` needs updating at M1 (the error-assertion testing convention changes) and at M2 (the
  benchmark commands become part of the standard command list).

## Verification strategy (applies to every milestone)

`README.md` is explicit that ParaView must be able to read the output, and that this can be checked
locally or in CI. The existing `.github/workflows/paraview.yml` matrix is
`{v5.13.3, v6.1.1} × {Ascii, AsciiInline, Hdf5SingleFile, Hdf5MultipleFiles, Binary}` = 10 jobs,
driven by `examples/paraview_smoke.rs` + `tests/paraview_smoke/verify_with_pvpython.py`.

Rules for the milestones below:

1. Anything that changes **what bytes end up in the file** must extend the smoke fixture, not just the
   Rust tests. That is: f32 attributes (M3), blocks (M4), and the restructured light-data output (M2).
2. Prefer extending the *fixture* over adding matrix dimensions — the matrix is already 10 jobs, and
   `expected.json` can carry arbitrarily many assertions per run.
3. Local checks against the real ParaView installs are faster for iterating; see the
   `paraview-install-locations` note for where 5.13/6.1 live and how to drive them with `pvpython`.
4. Round-trip tests (M5) are the strongest regression net the crate can have and should be wired into
   the normal `cargo nextest run`, not just CI.

## Cross-cutting risks

| Risk | Milestone | Mitigation |
|------|-----------|------------|
| `quick-xml` cannot serialize a `Grid` fragment at a chosen indent depth | M2 | Spike this before committing to the design; fallback is manual indentation. See `02_performance.md`. |
| `quick-xml` + serde `#[serde(flatten)]` on `DataContent` does not round-trip on deserialize — now with a third variant, the nested `Items(Vec<DataItem>)` a selection carries | M5 | Test deserialization of one `DataItem` per shape, selections included, on day one; fallback is a hand-rolled event-loop parser for `DataItem` only. |
| Static HDF5 build in manylinux/macOS/Windows wheels | M6 | Spike Linux-only first, before building out the platform matrix. |
| `hdf5::File` may not be `Send`, blocking GIL release | M6 | Check early; if it is not, the Python HDF5 path keeps the GIL and only the other backends release it. |
| ParaView misreading a new XML construct (hyperslab block references) | M4 | Two-stage plan with an explicit go/no-go: ship duplication first, add the hyperslab fast path only after CI proves it. |
