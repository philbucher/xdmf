# M4 — Submeshes

> **Status (2026-08-21): Stage A plus optimizations 1 and 2 are done.**
> `TimeSeriesWriter::write_mesh_with_submeshes` is on the `submeshes` branch, with contiguous-run
> collapsing, one reused gather buffer per element type for scattered submeshes, the full
> validation set, 26 Rust tests, `examples/submeshes.rs`, the Python binding
> (`TimeSeriesWriter.write_mesh_with_submeshes`) and a multi-block ParaView smoke fixture that CI
> runs against 5.13.3 and 6.1.1 for every storage. **Optimization 3 landed on 2026-08-22, for the
> HDF5 storages only — see [`09_submesh_references.md`](09_submesh_references.md), which supersedes
> the "Optimization 3" section below: a step's data is written once and each submesh selects its
> share, so optimizations 1 and 2 below now only serve the storages that cannot select. The reader
> side is still open**, as is the allocation test, which needs M2's harness.
>
> Eight deviations from the plan as written below, each deliberate:
>
> 1. **"Block" became "submesh"** throughout the API, errors and heavy-data layout — the plan's own
>    word was overloaded (block-structured grids, HDF5 hyperslab blocks) and collided with the
>    HDF5 backend's real groups. `ParaView`'s Multi-block Inspector is still named in the docs,
>    since that is where users find them.
> 2. **`impl IntoIterator<Item = (N, B)>`** with `N: AsRef<str>, B: AsRef<[usize]>`, rather than
>    `&[(&str, &[usize])]`. A caller can hand over a lazy iterator over their model parts and each
>    index list is consumed and dropped as it is converted; contiguous ones then leave nothing
>    behind at all. Considered and rejected: a closure-based gradual API mirroring `write_time_step`
>    — it cannot save memory here, because the writer has to *retain* every submesh's cell indices
>    for the whole run to slice cell data per step, so a reused caller buffer would just be copied
>    into that retained storage.
> 3. **`Error::InvalidMesh` rather than new variants.** The plan called for one variant per
>    validation rule; `CLAUDE.md` asks for the enum to stay under 10 variants and to group by
>    category. A submesh failure is a mesh-definition failure, and the "cells in no submesh" case
>    names no single submesh, so a `name`-carrying variant would have needed an `Option` anyway.
> 4. **`DataWriter::write_mesh` split into `write_points(Option<usize>, …)` +
>    `write_connectivity(Option<usize>, …)`.** The draft called `write_mesh(points, &[])` and added
>    a second `write_mesh_block` method; one path per array that each backend lays out itself is
>    simpler and removes the empty-slice hack. Both take the submesh position since deviation 8,
>    and every mesh array is now named the same way: `mesh/<array>` or `mesh/<array>/<index>`
>    (HDF5), `<array>.{txt,bin}` or `<array>_<index>.{txt,bin}` (ascii/binary), for `points`,
>    `cells`, `submesh_points` and `submesh_cells` alike.
> 5. **Heavy data is numbered, not named.** Every backend names an array by its position rather
>    than by the caller's field name and `Center` — `data_t_<time>_<index>.txt`,
>    `/data/t_<time>/<index>`, `submesh_<index>` — and the per-center groups
>    (`point_data`/`cell_data`) are gone with it. That is what lets a name be any printable string,
>    which solvers need (`Quantity('SOOT DENSITY')`), and what keeps a submesh name off the
>    filesystem. The cost is paid by every user, not just submesh ones: it is a breaking layout
>    change, and an HDF5 file is no longer self-describing under `h5dump`/`h5py` — the
>    field-to-index mapping is only recoverable from the XDMF file's `<Attribute Name=...>`.
>    **Open:** writing the name back as an HDF5 attribute on each dataset would restore that
>    cheaply.
> 6. **A `Spatial` collection of one `Temporal` collection per submesh**, rather than this plan's
>    `Temporal` collection of one `Spatial` collection per step. The plan's nesting names every
>    submesh once per step, and `ParaView` makes a grid name unique across the whole document, so
>    a block came back as `quad`, `quad[1]`, `quad[2]`, ... and changed identity as the animation
>    ran — losing its visibility and colouring in the Multi-block Inspector, and failing the
>    `submesh_fixtures` check from the second step on. Giving each submesh one grid that carries
>    its name, holding that block's per-step grids, keeps the name stable. Measured on 5.13.3
>    against both nestings and a third (the per-step collections sharing one name, which does not
>    help); guarded by `write_xdmf_with_submeshes_names_each_block_once` as well as by the ParaView
>    job. The per-step grids inside a block are named `<submesh>-t<time>` and carry the `<Time>`.
> 7. **Which cells and which points a submesh holds are recorded for the reader**, as
>    `<Information Name="submesh_cells" Value="0:1 1:2 submesh_cells_2"/>` and the matching
>    `submesh_points`: per submesh in order, either `<start>:<len>` for a contiguous list or the
>    name of an unreferenced `Domain`-level `DataItem` holding its indices
>    (`DataWriter::write_submesh_cells`/`write_submesh_points`, at mesh-write time). Without it the
>    file cannot be read back: a submesh's connectivity is indexed locally, submeshes may overlap,
>    and the mesh's own connectivity is not written at all when submeshes are used, so nothing says
>    which cell of the mesh a block cell was -- a reader could only return the cells permuted
>    against the caller's own global indexing. Disallowing overlap does not avoid this (the
>    permutation still has to be recorded, at the same cost) and would go against decision 5 in
>    `ROADMAP.md`. Measured on 5.13.3: an unreferenced `DataItem` and an `Information` are ignored
>    by `ParaView` -- same blocks, same cells, no extra cell array -- so this stays a side channel
>    for a reader rather than a `global_cell_id` attribute users would see next to their own
>    fields. Contiguous submeshes, the case mesh generators produce, write no array at all.
> 8. **Each submesh carries its own points**, not the mesh's whole point set. The plan (and the
>    branch draft) had every block reference one shared `coords` array, which is optimal on disk
>    but makes `ParaView` build a full copy of the point set and of every point field per block:
>    measured `blocks × mesh`, 2.8 MB at one block and 317.8 MB at 256 on a 40,401-point mesh, the
>    same on 5.13.3 and 6.1.0. A block now holds the points its own cells use, ascending, with its
>    connectivity renumbered against them (`submesh_points`/`renumber_connectivity`), which takes
>    the same mesh to 4.8 MB at 256 blocks. The renumbering looks a point up by subtraction when
>    the submesh's points are one run and through a `LocalPoints` array when they are not; a binary
>    search of the point list, which allocates nothing, was measured 6-28% slower over the whole
>    mesh write on a 4M-point mesh and never faster. Point data is cut per submesh from then on, exactly as
>    cell data already was, so `write_shared_point_data` and the `Domain`-level point-data items
>    are gone with it. The costs, measured on the same mesh at 256 blocks: +73% heavy data (points
>    on a block boundary are written once per block touching them), +28% light data, and one file
>    per block per array for the per-file storages. Nothing measurable at 64 blocks or fewer.


`README.md`: *"Writing and reading of submeshes => a draft is in the 'multiple-features' branch, but
this one needs to be done a bit nicer."*

Decision 5 in `ROADMAP.md`: **overlapping blocks stay allowed**; the per-step copying gets optimized
away rather than the semantics being restricted.

## What the branch draft does, and what "nicer" means

The draft (`origin/multiple-features:src/time_series_writer.rs`, `examples/submesh_blocks.rs`) writes
a spatial `Grid` collection, one uniform sub-grid per block. All blocks share one `Geometry` (the
global point array, referenced) — that part is right and stays. What needs work:

1. **Per-block connectivity is materialized and written separately.** Correct and portable, but for a
   mesh split into many blocks the connectivity is written once in full, plus once more per block.
2. **Cell data is gathered into a fresh `Vec` for every (attribute × block × time step)**
   (`Values::gather`, `values.rs` on the branch). This is the real problem: it is an allocation *and*
   a full copy of every cell field, per block, on every step — squarely in the hot path.
3. **`&[(&str, &BTreeSet<usize>)]`** forces the caller into a specific container.
4. Point data is shared verbatim across blocks — correct, keep it, but see the note on XML size below.

## API

Flattened to match `write_mesh`'s signature, which `API_IMPROVEMENTS_PLAN.md` item 3 already
flattened (landed 2026-08-09, ahead of this milestone):

```rust
pub fn write_mesh_with_blocks(
    self,
    points: &[f64],
    connectivity: &[u64],
    cell_types: &[CellType],
    blocks: &[(&str, &[usize])],
) -> Result<TimeSeriesDataWriter>
```

`&[usize]` rather than `&BTreeSet<usize>`: callers usually already have a `Vec<usize>` per block
(element blocks, material zones, boundary patches) and should not have to build a `BTreeSet` to call
this. Sortedness is not required; duplicates *within* one block are rejected. The given order defines
the block's internal cell order, which only affects that block's own rendering order — global cell
data is still supplied globally and sliced by the writer.

Cell data continues to be passed **globally** to `write_data` (indexed by global cell id); the writer
does the per-block slicing. That is the property that makes blocks cheap to adopt: a caller that adds
blocks does not have to restructure how it produces field data.

### Validation (keep the draft's rules)

- Block names valid (same charset rule as data names) and unique.
- Every block non-empty.
- Indices in range; no duplicates within a block.
- **Every cell belongs to at least one block.** Keep this — without it, cells silently vanish from
  the visualization, which is a much worse failure than an error message. The error should name the
  first few orphaned cells and their count, as the draft's does.
- Overlap *between* blocks is explicitly allowed and tested.

Each of these becomes an `Error` variant per `01_error_type.md`.

## Optimization 1 — contiguous blocks are free

The important observation: in practice, block cell index lists are usually **contiguous ascending
runs**. Element blocks come out of mesh generators grouped; the benchmark's own duct case is
`[hexes…][inlet…][outlet…][sides…]`, i.e. four contiguous ranges. So:

```rust
enum BlockCells {
    Contiguous { start: usize, len: usize },   // detected at write_mesh_with_blocks time
    Scattered(Vec<usize>),
}
```

Detection is a single pass at mesh-write time, done once. For a `Contiguous` block:

- **Cell data needs no gather at all** — the block's values are the subslice
  `&global[start * stride .. (start + len) * stride]`, wrapped as a borrowed `Values`. Zero copy,
  zero allocation, per step.
- **Connectivity needs no duplication** in principle — the block's prepared cells are likewise a
  contiguous slice of the global prepared connectivity. Realizing that in the file requires a
  hyperslab reference, which is optimization 3 below and is gated on ParaView verification.

This is roughly ten lines and it makes the common case optimal. Do it first.

## Optimization 2 — scattered blocks reuse one buffer

For genuinely scattered blocks (which overlapping blocks usually are), the gather is unavoidable, but
the *allocation* is not. One scratch buffer on `TimeSeriesDataWriter`, sized once to
`max_block_cells × max_stride`, reused across every attribute, every block, and every step — the same
`std::mem::take` pattern as `02_performance.md` part C and `03_values_and_f32.md`.

After this, a scattered block costs one memcpy-ish gather per attribute per step and no allocations,
versus the draft's allocation + copy.

## Optimization 3 — hyperslab references, gated on ParaView

Two-stage, with an explicit go/no-go, because this is where ParaView compatibility risk lives.

**Stage A (ship this):** every block writes its own connectivity dataset, exactly as the draft does.
Portable, works with any XDMF2 reader, and combined with optimizations 1 and 2 the *per-step* cost is
already optimal — connectivity is one-shot at mesh-write time, not per step.

**Stage B (only if verified):** for `Contiguous` blocks, emit a `DataItem ItemType="HyperSlab"`
selecting `start / stride 1 / count len` out of the shared `connectivity` `DataItem`, instead of a
separate dataset. Same for cell-data attributes. This removes the duplication entirely for the common
case.

> **Ran 2026-08-22 — see [`09_submesh_references.md`](09_submesh_references.md) for the result.**
> Short version: stage B works, for `Format="HDF"` only. `Format="Binary"` honours `HyperSlab` but
> silently ignores `Coordinates`, and `Format="XML"` ignores both and reads from the start of the
> array. `Coordinates` generalizes it to scattered submeshes after all, and the win is bigger on
> per-step *data* than on connectivity, which is what the recommendation is built around.

Verification gate: write a throwaway fixture using hyperslab DataItems, open it in **both** ParaView
5.13 and 6.1 (locally first, then in the CI matrix), and confirm the geometry and the attribute
values are correct — not merely that the file opens. XDMF2's hyperslab support is exactly the kind of
construct the legacy reader handles inconsistently; the crate already carries two ParaView reader
workarounds (`Format="Binary"` 64-bit integers, and the `Matrix` attribute rank-3 dimension shape),
so assume nothing. If it does not verify cleanly, stage A is a perfectly good final answer — say so
in the plan record and stop.

`ItemType="Coordinates"` (arbitrary index list selection) would generalize this to scattered blocks
too. Same gate, lower prior probability of working; test it in the same spike since the fixture is
already written, but do not plan around it.

## XML size

With B blocks and N steps, point-data attributes appear B×N times in the light data (each block's
grid repeats them). The heavy data is written once and every block references the same `DataContent`
string — that part is already right in the draft and must stay right. The XML growth is linear and
small, and the streaming writer from `02_performance.md` part B makes it O(1) per step to emit. No
action needed, but confirm the `steps_scaling` bench is run *with blocks* too, so this is measured
rather than assumed.

## Attribute naming

The draft's split between the **storage name** (`{data_name}__{block_name}`, unique per dataset) and
the **display name** (`data_name`, what ParaView shows) is correct and must be preserved — ParaView
keys on the `Attribute` `Name`, so all blocks must present the *same* display name for a field to be
selectable as one field across the multiblock dataset. Keep the draft's `build_attribute(storage_name,
display_name)` shape.

With stage B or with contiguous blocks referencing shared datasets, the storage-name suffix becomes
unnecessary for those cases; make sure the naming is derived from how the data is actually stored, not
applied unconditionally.

## Reader

M5 must reconstruct blocks: `TimeSeriesReader::block_names()` and per-block cell index recovery.
Round-trip test — write with blocks, read back, assert identical block names and membership — is the
acceptance criterion, and it is listed in `05_reader.md` too. If M4 lands after M5, this is a
follow-up to M5 rather than a change to it.

## Verification

- Rust tests: contiguous blocks, scattered blocks, overlapping blocks, mixed cell types within a
  block, a block containing every cell, and each validation error (asserted by `Error` variant per
  `01_error_type.md`).
- Allocation test: per-step allocations with blocks are a constant, independent of block count — this
  is the whole point of the milestone and should fail loudly if regressed.
- **ParaView smoke fixture extended with a block-structured case**, asserting via `pvpython` the block
  *names*, per-block cell counts, and that a point field and a cell field carry the right values in
  each block. Per `ROADMAP.md`, extend the fixture rather than the matrix.
- Port `examples/submesh_blocks.rs` from the branch to the final API — it is a good example and the
  Multi-block Inspector instructions in its doc comment are genuinely useful.

## Noted, not scheduled

- **Compacted block geometry — measured, then implemented (2026-08-21).** See deviation 8 above:
  every block now carries only the points its own cells use. What the measurement found, on a
  40,401-point / 40,000-cell quad mesh, two steps, identical on ParaView 5.13.3 and 6.1.0
  (`GetDataInformation().GetMemorySize()`), and what the implementation then reproduced exactly:

  | blocks | points materialized | shared geometry | compacted | heavy data compacted |
  |---|---|---|---|---|
  | 1 | 40,401 / 40,401 | 2.8 MB | 2.8 MB | 2.26 MB |
  | 4 | 161,604 / 41,004 | 6.5 MB | 2.8 MB | 2.28 MB |
  | 16 | 646,416 / 43,424 | 21.3 MB | 2.9 MB | 2.38 MB |
  | 64 | 2,585,664 / 53,120 | 80.6 MB | 3.3 MB | 2.78 MB |
  | 256 | 10,342,656 / 80,896 | 317.8 MB | 4.8 MB | 4.25 MB |

  The shared-geometry column is `blocks × mesh` exactly -- ~1.24 MB per block here whatever the
  block holds, so a block of one cell cost as much as one of ten thousand.

  **Superseded for the HDF5 storages (2026-08-22), without giving that back.** A block may
  reference the mesh's coordinates *and* stay compact, by selecting the points it holds out of them
  rather than taking the array whole: same blocks, same points materialized, same 4.9 MB at 64
  blocks. The heavy data is then written once, and `submesh_points` -- the side channel this entry
  ends by naming -- becomes the selector the geometry reads through, so the global identity is no
  longer something the layout loses and an `<Information>` restores. See
  `plans/09_submesh_references.md`; the ascii and binary storages keep the compacted copy described
  above, since `ParaView` misreads a selection out of those.
- **Node sets / point blocks.** Some callers want named *point* sets, not only cell sets. XDMF's
  `Set` elements are the route; per-block point fields, which the compaction above would now make
  natural to express, are the other. No current caller; out of scope.
