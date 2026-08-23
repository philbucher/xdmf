# M4b — Referencing shared arrays from submeshes

> **Status (2026-08-22): measured, and the recommendation is implemented.** This is the
> "optimization 3" verification gate that `04_submeshes.md` left open, widened to the question
> behind it: can a submesh *reference* points, cells and per-step data that are stored once,
> instead of carrying its own copy? The answer is yes, for HDF5 storage only, and the recommended
> subset is smaller than what the reader turns out to support. See
> [Recommendation](#recommendation) for what was adopted and [As
> implemented](#as-implemented) for what the writer now does.

## The question

`write_mesh_with_submeshes` currently gives every submesh its own copy of everything: its points,
its locally renumbered connectivity, and, per time step, its share of every field. That is what
keeps ParaView's memory flat (deviation 8 in `04_submeshes.md`; the shared-geometry layout it
replaced cost `submeshes × mesh`), but it duplicates every point on a submesh boundary and every
value of every field, on every step.

The wanted layout is the opposite one: points and cells written once, each submesh naming the
indices it holds. It is also what the reader (M5) wants, since those index lists are the mesh's own
identity for the entities a submesh holds -- which is why `submesh_points`/`submesh_cells` already
exist as an `<Information>` side channel.

## Method

Hand-written XDMF2 fixtures, not writer output, so a layout could be measured before implementing
it: `plans/spikes/submesh_references/`. Measured on both local installs, **ParaView 5.13.2 and
6.1.1**, which agreed on every result below; every number quoted is 6.1.1's.

Scale fixture: a 201×201 quad mesh (40,401 points / 40,000 cells) — the same mesh the deviation-8
measurements used — split into `B` submeshes, `T` = 10 steps, carrying a point scalar, a point
vector and a cell scalar. Two splits: **contiguous** (`B` equal cell ranges, what mesh generators
produce) and **scattered** (cell `i` → submesh `i % B`, the worst case). Field values are functions
of the *global* point/cell id, and the checker recomputes the expected value from each block's own
coordinates, per block and per step — so a layout that silently reads the wrong slice fails rather
than merely differing.

## What the reader actually supports

`ItemType="HyperSlab"` (start/stride/count per rank) and `ItemType="Coordinates"` (an explicit index
list) are the two XDMF2 selection constructs. Whether they work **depends on the heavy-data format**,
which is the finding that decides everything else:

| construct | `Format="HDF"` | `Format="Binary"` | `Format="XML"` (ascii, inline or file) |
|---|---|---|---|
| `HyperSlab` in an `Attribute` | ✅ | ✅ | ❌ **silently** reads from the start of the array |
| `Coordinates` in an `Attribute` | ✅ | ❌ **silently** reads from the start | ❌ inline: from the start; file: garbage |
| `HyperSlab` in `Geometry` / `Topology` | ✅ | not measured | — |
| `Coordinates` in `Geometry`, `XYZ` (index *pairs*) | ✅ | — | — |
| `Coordinates` in `Geometry`, `X_Y_Z` (1-D index list per component) | ✅ | — | — |
| `Function="$0 - k"` (offsetting a connectivity slab) | ✅ | — | — |
| `Function="JOIN($0,$1,$2)"` over three selections | ✅ | — | — |
| selector as `Reference="XML"` XPath to a `Domain`-level `DataItem` | ✅ | — | — |
| selector inline as `Format="XML"`, source in HDF5 | ✅ | — | — |
| selector of `Precision="8"` ints, `f32`/int source data | ✅ | — | — |
| `Coordinates` over a rank-1 source for an `XYZ` geometry | ❌ HDF5 "selection + offset not within extent" | — | — |

The ❌ rows are the reason none of this can be a storage-independent feature: the ascii and binary
storages do not *fail* on a selection, they return a different array than the file describes, which
is exactly the class of defect `paraview.rs` exists to refuse. **A selection may only ever be
emitted for `Hdf5SingleFile`/`Hdf5MultipleFiles`** — plus `HyperSlab` for `Binary`, which does work
(verified rank-1 and rank-2).

Two smaller findings from the same sweep:

- The selector array is cheap to place: it can live in the HDF5 file, or inline in the XML, or be
  written once at `Domain` level and referenced by XPath from every grid that uses it.
- A submesh's connectivity still has to be written per submesh. `HyperSlab` does work in `Topology`,
  but the values it selects are the mesh's *global* point ids, and a submesh's geometry is its own
  point set, so they have to be renumbered. `Function="$0 - k"` renumbers a submesh whose points are
  one contiguous run (structured splits only) — noted, not proposed.

## The four layouts, measured

`compact` is today's writer. `shared` is the pre-deviation-8 layout, kept as the memory baseline.
`select` keeps compacted geometry but writes each field **once per step** globally and lets each
submesh reference its share. `shared_sel` is the full ideal: coordinates written once as `X_Y_Z`,
every submesh's geometry *and* data a selection out of the global arrays.

40,401 points / 40,000 cells, T = 10 steps. "data" is ParaView's own
`GetDataInformation().GetMemorySize()`, "RSS" the peak resident set of `pvpython` while stepping
through all ten steps, "time" that traversal's wall clock.

| split | layout | heavy MB | light MB | data MB | peak RSS Δ MB | time s |
|---|---|---|---|---|---|---|
| B=16 contiguous | compact | 19.11 | 0.14 | 4.2 | 23 | 0.28 |
| | select | 18.83 | 0.22 | 4.2 | 24 | 0.29 |
| | shared_sel | 17.78 | 0.34 | 4.2 | 28 | 0.51 |
| | shared | 17.78 | 0.14 | **41.0** | — | — |
| B=64 contiguous | compact | 23.44 | 0.55 | 4.9 | 38 | 0.78 |
| | select | 19.14 | 0.89 | 4.9 | 43 | 0.81 |
| | shared_sel | 17.85 | 1.34 | 4.9 | 55 | 1.63 |
| | shared | 17.85 | 0.56 | **159.3** | 1511 | — |
| B=256 contiguous | compact | 38.13 | 2.23 | 7.2 | 103 | 2.82 |
| | select | 22.58 | 3.68 | 7.2 | 256 | 3.96 |
| | shared_sel | 20.54 | 5.39 | 7.2 | 357 | 13.69 |
| | shared | 18.12 | 2.23 | **632.8** | 1511 | 4.68 |
| B=64 scattered | compact | 60.21 | 0.56 | 10.6 | 51 | 0.79 |
| | select | 26.40 | 0.94 | 10.6 | 324 | 2.87 |
| | shared_sel | 22.54 | 1.36 | 10.6 | 454 | 5.56 |
| | shared | 17.85 | 0.56 | **159.3** | 1511 | — |

Every layout except `shared` returned the right values for every block on every step, on both
ParaView versions. (`shared` is only a memory reference; its per-submesh cell field is the whole
global array, which is what made it cheap on disk and useless in the viewer.)

What the table says:

1. **A selection does not reintroduce the memory explosion.** `select` and `shared_sel` hold exactly
   as much as `compact` — ParaView materializes only the selected values, so the 159 MB / 633 MB
   figures of the shared-geometry layout do not come back. This was the risk that made the whole
   idea look dead, and it is not real.
2. **The saving is in the per-step data, and it is large.** `select` writes each field once per step
   instead of once per submesh per step: −18% heavy data at 64 contiguous submeshes, −41% at 256,
   **−56%** on the scattered split (26.4 MB against 60.2). Per-step heavy data becomes independent
   of the submesh count and of how much the submeshes overlap — which is the property worth having,
   since steps are the axis that grows.
3. **Referencing the geometry too buys little.** `shared_sel` reaches the theoretical minimum on
   disk (it *is* `shared`'s heavy data at ≤64 submeshes) but only 4–13% below `select`, because
   geometry is written once at mesh time while data is written per step. It pays for that with
   2.4–4× the light data, 2–5× the traversal time, and the highest transient RSS.
4. **Selections are not free at read time.** They cost transient memory (256 MB against 103 at 256
   submeshes, 324 against 51 on the scattered split — the reader appears to touch the whole source
   array per block) and time on the scattered split (2.87 s against 0.79 s for ten steps). Nothing
   here is resident, and at ≤64 contiguous submeshes both costs are in the noise.

## Recommendation

**Adopt `select` for the HDF5 storages; leave everything else as it is.**

- **Per-step attribute data is written once, globally** — the caller's array, exactly as handed in.
  Each submesh's `<Attribute>` selects its share: `HyperSlab` when the submesh's indices are a
  contiguous run (no index array at all — the start/stride/count triple goes inline into the XML),
  `Coordinates` otherwise.
- **Geometry stays compacted.** A submesh keeps its own points and its renumbered connectivity, as
  today. It is a one-off cost, it reads back fastest, and deviation 8 already measured it.
- **The point-data selector is `submesh_points`** — the global point ids the file already carries for
  the reader. For rank ≥ 2 point fields either a pairs selector (2 ints per selected component) or
  the measured `JOIN($0,$1,$2)`-of-column-selections form, which reuses the 1-D selector and costs
  light data instead of heavy; pick when implementing, both are verified.
- **`Ascii`/`AsciiInline` keep the current per-submesh copies.** A selection there is silently wrong,
  so the writer must not emit one. `Binary` may use `HyperSlab` for contiguous submeshes and must
  never use `Coordinates`.

Two things fall out of this that are worth as much as the file-size win:

- **The writer gets simpler and faster.** With data written once per step, there is no per-submesh
  gather on the hot path at all — so optimizations 1 and 2 of `04_submeshes.md` (contiguous-run
  collapsing to borrow a subslice, the reused gather buffer) stop being needed on the HDF5 path;
  contiguity is then only a light-data question. What is retained per submesh shrinks to what the
  side channel already needs.
- **The reader gets the fields whole.** A global per-step array plus each submesh's index list is
  precisely M5's input, instead of one permuted copy per submesh to be stitched back together.

The costs to accept, all measured above: +60% light data (the selection markup, per submesh per
step — which makes `02_performance.md` part B's per-step XML cost more pressing, not less), higher
transient RSS in the viewer, and slower stepping when submeshes are scattered.

**Not recommended at the time: `shared_sel`.** It is the layout the goal describes literally, it
works, and it is the smallest on disk — but the gain over `select` is small, and it is the slowest
to read and the heaviest in light data.

*Adopted anyway, 2026-08-22* — see [As implemented, geometry](#as-implemented-geometry-2026-08-22).
The reason the recommendation above missed: it weighed only bytes, and on bytes the geometry is a
rounding error next to the per-step data. What it buys instead is that `submesh_points` stops being
a side channel — the array a submesh's `<Geometry>` selects through *is* the list of mesh points it
holds, so the file carries each submesh's point identity as part of its geometry rather than as an
`<Information>` nothing references. The measured read-time cost was accepted for that.

## Also learned

- **The `suneth_cube` reference file** (`.../research/xdmf_stuff/suneth_cube`, Kratos HDF5 output) is
  the `shared` layout: 53 grids, each referencing the whole `/ModelData/Nodes/Local/Coordinates`
  array, with sub-model-parts expressed as `Polyvertex` node sets rather than cell sets. It is
  therefore the memory behaviour deviation 8 removed, not a way around it. Its one use of
  `ItemType="Coordinates"` (gathering the per-element `Ids` attribute) is the construct measured
  above, and works — but only because the file is HDF5; the same file written with any of the ascii
  storages would show wrong ids without any error.
- **`Function` works in the Xdmf2 reader** (`$0 - k` and `JOIN`), which is more than expected; it is
  what would let a contiguous-point submesh reference the global connectivity instead of carrying a
  renumbered copy. Structured splits only, so not proposed.

## Open, if this is implemented

- Where the choice lives: automatic per storage (HDF5 → selections, ascii/binary → copies), since
  nothing about it is a user decision, and `DataStorage` is already what selects a backend. It does
  mean the heavy-data layout differs per storage, which the fixtures must cover.
- `Hdf5MultipleFiles`: the per-step file holds the global field, the mesh file the selectors; the
  XPath/`Reference` form of the selector is the one that survives that split.
- The ParaView smoke fixture needs a selection case per HDF5 storage, asserting values per block per
  step — a layout that reads from the start of the array instead of the selected slice is exactly
  what a smoke test that only checks shapes would miss.
- Whether `Binary` should get the contiguous-only `HyperSlab` half, or stay on copies for
  simplicity's sake.

## As implemented (2026-08-22)

`TimeStep::write_data_selected` in `src/time_series_writer.rs`. A field written to a storage whose
`DataWriter::supports_selections` is `true` -- the two HDF5 ones -- goes to the heavy data once,
exactly as the caller passed it, and each submesh's `<Attribute>` carries a `DataItem` selecting
its share. Everything else is unchanged: the geometry stays compacted, the grids and their nesting
stay as M4 left them, and `write_data`'s signature and the callers' side of it are untouched.

Five things the implementation had to settle that the measurement had not:

1. **The selection is one-dimensional, whatever shape the field has.** `ParaView` matches the rank
   of a selection against the rank of the `HDF5` dataset, and this crate writes every array as one
   flat run of values -- so a rank-2 selection over a `Vector` field failed with *"Dataset has rank
   2 ... but the array's selection has rank 1"*. The source item inside the selection is therefore
   written flat and the *shape* is carried by the selection item, which is what `ParaView` reads
   the component count from. Verified to come back with the right components and values.
   The alternative -- storing multi-component data as genuinely rank-2 `HDF5` datasets -- would
   have changed the layout of every file the crate writes, `Matrix`-shaped fields included, whose
   declared shape is a rank-3 `ParaView` workaround.
2. **Flat selection positions halve the index arrays** the plan budgeted for: a `(entity,
   component)` pair per component becomes one flat position per value, so a `Vector` field costs 3
   indices per entity rather than 6.
3. **A scalar field needs no new array at all.** One index per entity is exactly what
   `submesh_points`/`submesh_cells` already hold for the reader, so those items are now referenced
   by the selections as well -- the side channel became load-bearing. Wider fields get one array
   per (submesh, centering, component count), written at the step that first carries such a field
   and referenced by every step after; `SELECTIONS`/`DataWriter::write_selection` put it with the
   mesh's arrays, so a discarded step does not take it along.
4. **Those index arrays are written signed now.** They are read by `ParaView` rather than only by a
   reader, and `NumberType="UInt"` is decoded at 32 bits whatever `Precision` says.
5. **A submesh whose cells are not ascending still gets a copy.** `ParaView` hands back the values
   a `Coordinates` selection names in the order the *array* holds them, not in the order they were
   named -- caught by the smoke fixture's `reversed` submesh, which came back with its two cells
   swapped. Such a submesh takes the gather path per field; if *no* submesh can select, the field
   is not written whole at all. `IndexList::is_ascending` is the test, and
   `write_xdmf_with_an_unordered_submesh_writes_its_share_out` guards it.

Measured with the writer's own output, uncompressed, 40,401 points / 40,000 cells, 64 submeshes,
10 steps, a point scalar + a point vector + a cell scalar:

| split | heavy before | heavy after | light before | light after |
|---|---|---|---|---|
| contiguous | 27.20 MB | **18.45 MB** (-32%) | 0.82 MB | 1.34 MB (+62%) |
| strided | 65.08 MB | **24.20 MB** (-63%) | 0.84 MB | 1.35 MB (+60%) |

Verified block by block and step by step against ParaView 5.13.2 and 6.1.1 at that scale (same
blocks, same points materialized, same 4.9 / 10.6 MB in the viewer as before), and the full
`paraview_smoke` suite passes for all five storages on both versions. The read-time cost the
measurement predicted is there: stepping through those ten steps takes 4.1 s against 3.1 s
contiguous, 6.7 s against 3.6 s strided.

Tests: `write_xdmf_with_submeshes_writes_hdf5_data_once_and_selects_it`,
`write_xdmf_with_submeshes_reuses_one_selection_array_per_field_width`,
`write_xdmf_with_an_unordered_submesh_writes_its_share_out`, and
`write_xdmf_with_submeshes_never_selects_from_a_storage_that_misreads_it` -- the last one being the
guard that matters most, since a selection emitted for an ascii storage is silently wrong rather
than an error.

**Still open:** `Binary`'s contiguous-only `HyperSlab` half (it works, it is just not emitted); the
reader (M5), which now finds each field whole and one index list per submesh; and the light-data
growth, which lands on `02_performance.md` part B.

## As implemented, geometry (2026-08-22)

`write_mesh_coordinates`/`selected_coordinates` in `src/time_series_writer.rs`, on top of the data
selections above. For a storage whose `DataWriter::supports_selections` is `true`, the mesh's
coordinates go to the heavy data **once**, as three arrays -- `mesh/points/0`, `/1`, `/2` -- and each
submesh's `<Geometry GeometryType="X_Y_Z">` references three `Domain`-level `DataItem`s that select
the points it holds out of them. Every other storage is unchanged: it still gets a compacted copy
of each submesh's points, written as one interleaved `XYZ` array.

Three things this settled that the `shared_sel` measurement had not:

1. **The coordinates must be split by direction.** An `X_Y_Z` geometry is what lets all three of a
   submesh's selections share *one* index list; selecting out of an interleaved array needs an
   index per coordinate, which would make a scattered submesh's selector three times the size of
   the point list the file already carries. The alternatives measured earlier both fail here:
   `XYZ` with a rank-2 selection needs a rank-2 source, and this crate writes every array flat,
   which is the "`Coordinates` over a rank-1 source for an `XYZ` geometry" row of the support
   matrix.
2. **The whole selection is hoisted, not just its selector.** Each of the three items is named
   (`coords_<submesh>_<x|y|z>`) and sits at `Domain` level, so the grid -- cloned once per time
   step -- repeats three short references rather than three selections. Verified to read
   identically on 5.13.2 and 6.1.1 (`plans/spikes/submesh_references/geometry_selection.py`).
3. **Nothing new is written, and one thing less.** A submesh whose points are one run selects with
   a `HyperSlab` whose start and count are three numbers in the XML; a scattered one references
   `submesh_points_<n>`, the index list the mesh already carried for a reader and already used as
   the selector for that submesh's scalar point fields. Since the `<Geometry>` now states which
   points a submesh holds -- in those same two forms -- the `<Information Name="submesh_points">`
   would be the file saying it twice, and is not written on this path. `submesh_cells` still is:
   cell identity has no other home, the connectivity being a renumbered copy either way. A reader
   therefore takes a submesh's points from its geometry for the HDF5 storages and from the
   `<Information>` for the others; its cells from the `<Information>` always.

Measured with the writer's own output, uncompressed, 40,401 points / 40,000 cells, 64 submeshes,
10 steps, a point scalar + a point vector + a cell scalar -- "before" being `select` as landed
above, "original" the compacted layout that preceded both:

| split | heavy original | heavy select | heavy now | light select | light now |
|---|---|---|---|---|---|
| contiguous | 27.20 MB | 18.45 MB | **17.16 MB** (-7.0%) | 1.34 MB | 1.47 MB (+9.7%) |
| strided | 65.08 MB | 24.20 MB | **20.20 MB** (-16.5%) | 1.35 MB | 1.48 MB (+9.6%) |

Read back block by block and step by step on both ParaView versions: same 64 blocks, same 53,120 /
160,000 points materialized, same 4.9 / 10.6 MB held in the viewer as every layout since deviation
8 -- the geometry selection does not bring the memory blow-up back either. Stepping through the ten
steps costs 4.4 s contiguous and 7.4 s strided on 6.1.1, against 4.1 / 6.7 for `select` and 3.1 /
3.6 for the compacted layout. The full `paraview_smoke` suite passes for all five storages on both
versions, and its submesh fixture covers both selector forms (block 1's points are scattered).

Tests: `write_xdmf_with_a_scattered_submesh_selects_its_points_out_of_the_mesh` pins the whole
document for both forms; `write_xdmf_with_submeshes_of_a_uniform_topology` is its counterpart for a
storage that copies instead; `write_xdmf_with_submeshes_for_every_storage` asserts the item count
per storage.
