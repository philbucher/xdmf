# M4 — Submeshes (blocks)

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

- **Node sets / point blocks.** Some callers want named node sets, not only cell sets. XDMF has `Set`
  elements for this. No current caller; out of scope.
- **Per-block point data.** Currently every block sees the full point array (correct, since geometry
  is shared). A caller wanting per-block point fields would need per-block geometry, which defeats
  the sharing. Not wanted.
