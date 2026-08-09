# API improvements plan

Follow-up work after the `write_data` interface change. Each item below is independent and can
land on its own; the suggested order is by severity, not by size.

Status of the already-completed change, for context:

- `Values` is now `Values<'a>` backed by `Cow`, so callers can hand the writer a borrowed buffer
  and reuse it across time steps.
- `write_data` takes `impl IntoIterator<Item = (&str, DataAttribute, Values)>` for point and cell
  data instead of `Option<&DataMap>`; `DataMap` is gone. Attribute order now follows the caller's
  order instead of being alphabetized, and a repeated name is a hard error.

**Merged to `main` (2026-08-09, PR #18 "Several improvements to the API", commit `aa3c501`).**
That PR also carried item 3 below (the `write_mesh` flattening) and a doc note on the `Vec` `From`
impls warning that they move the buffer — see `02_performance.md` part F for the buffer-reuse
follow-up that is still open.

**Items 1, 2, and 4 are now done too (2026-08-09, not yet committed — see the note in each
section below).** M0 is complete except for the O(steps²) light-data rewrite, which was never
part of this document — see `02_performance.md` part B.

Items 1 and 2 are defects with confirmed reproductions. Items 3 and 4 are interface cleanups.
Item 5 is a decision to make, not a change to schedule.

---

## 1. A mid-write error permanently poisons the writer

**DONE (2026-08-09, not yet committed).** Both parts landed:
`TimeSeriesDataWriter::write_data` now runs the attribute-writing loop through a private
`write_attributes` helper and always calls `self.writer.write_data_finalize()` after it,
regardless of whether it succeeded — the write error (if any) wins over a finalize error rather
than being masked by it. Separately, a new `DataWriter::validate_values` trait method (default
no-op) lets a backend reject values it cannot represent before `write_data_initialize` ever runs;
`BinaryWriter` overrides it to run the existing u64→u32 range check (factored out as
`checked_u32`) up front. In practice this means the confirmed repro (an out-of-range `u64` in
cell/point data) no longer reaches the mid-write failure path at all — it is now a plain upfront
validation error, same as a size or name mismatch. The open question below (delete partially
written files?) is now moot for the binary-range case specifically, since nothing is written
before the check; it stays relevant in principle for other backends that might fail mid-write in
the future. The rest of this section is kept for the record.

**Severity: high.** This is the item to do first.

### Problem

`TimeSeriesDataWriter::write_data` (`src/time_series_writer.rs:294`) calls
`write_data_initialize`, then writes each attribute, then `write_data_finalize`. If any write
fails in between, it returns early and `write_data_finalize` never runs, so the backing writer
keeps `write_time: Some(..)`. Every later time step then fails inside `write_data_initialize`
with a message unrelated to the real failure.

Confirmed with the `Binary` backend, whose u64→u32 range check (`src/binary_writer.rs:149`) is the
one error that fires *during* the write rather than up front:

```
FIRST  (u64 too large for binary) -> "value 4294967296 does not fit in 32 bits: ..."
SECOND (perfectly valid data)     -> Err("Writing data was already initialized")
FILES ON DISK -> ["cells.bin", "data_t_0_cell_data_r.bin", "points.bin"]
```

The orphaned `data_t_0_cell_data_r.bin` is referenced by no XML, because `self.attributes` is
only appended to after all writes succeed.

### Change

Two parts, both worth doing:

1. **Keep the writer usable after a failed step.** In `write_data`, ensure `write_data_finalize`
   runs even on the error path, so a failed time step leaves the writer in the same state as if
   the call had never happened. The natural shape is to move the body into a private helper and
   have `write_data` finalize before propagating, rather than sprinkling cleanup at each `?`.
   Note `write_data_finalize` is itself fallible: on the error path the original error must win,
   the cleanup error must not mask it.
2. **Move the range check up front.** Add the u64→u32 validation to the binary writer's
   validation step so it fires before any file is opened, joining the existing size and name
   checks. That removes the only current source of mid-write failure that is really caller error.

Files: `src/time_series_writer.rs`, `src/binary_writer.rs`.

### Tests

- Binary backend: bad values at `t=0` must return the range error, and a following valid `t=1`
  must succeed — asserting the *specific* message, not just `is_err()`.
- Assert no orphan `.bin` file is left for the failed step.
- Keep `binary_write_data_rejects_u64_too_large_for_u32` (`tests/binary_writer.rs`) passing; its
  expected message should not change.

### Open question

Whether a partially-written step should also delete the files it already wrote. Leaving them is
harmless (nothing references them) but confusing on disk. Deleting is more work and introduces
its own failure mode. Recommend leaving them for now and revisiting only if it bites.

---

## 2. Time-step dedup is textual while validation is numeric

**DONE (2026-08-09, not yet committed).** `written_times` is now `HashMap<u64, String>`, keyed on
`f64::to_bits` of the parsed time (as this section already recommended), with the value being the
spelling first used. The message is `Time step '{time}' has already been written` when the exact
same spelling repeats, or `Time step '{time}' has already been written (as '{existing}')` when a
different spelling of the same value is rejected — chosen over always appending the `(as ...)`
clause because for the same-spelling case it would just repeat the string back at the caller.
`validate_data` returns the parsed bit pattern on success so `write_data` doesn't need to
re-parse (and `unwrap`) a value already known to be valid. The rest of this section is kept for
the record.

**Severity: medium.** Cheap fix, prevents a malformed output file.

### Problem

`written_times` is a `HashSet<String>` (`src/time_series_writer.rs:243`) and the duplicate check
compares strings (`:410`), but validity is decided by `time.parse::<f64>()` (`:401`). So two
spellings of the same instant both pass:

```rust
w.write_data("0.1",  ...)?;  // ok
w.write_data("0.10", ...)?;  // also ok -- confirmed accepted
```

The result is two grids in the temporal collection at the same time value, which is exactly the
duplicate the check exists to prevent.

### Change

Keep `time: &str` — letting the caller own the formatting is the right call and is documented as
deliberate. Change the dedup to key on the parsed `f64`, not the string. `f64` is not `Hash`/`Eq`,
so either store the bit pattern (`f64::to_bits`, fine here since the value came from a successful
`parse` and NaN is already rejected by it) or keep a sorted `Vec<f64>`. `to_bits` in a `HashSet<u64>`
is the simpler option.

The error message should name the conflicting *value*, and ideally the spelling already used, e.g.
`Time step '0.10' has already been written (as '0.1')`. That requires keeping the original string
alongside, so a `HashMap<u64, String>` keyed on bits is probably the right structure.

Files: `src/time_series_writer.rs`.

### Tests

- `"0.1"` then `"0.10"` must be rejected with the new message.
- `"0.1"` then `"0.2"` must still be accepted.
- Existing `Time step '0.1' has already been written` assertion in `test_validate_data` needs
  updating to whatever the final message is.

---

## 3. Flatten `write_mesh`'s cell tuple

**DONE (2026-08-09, merged in the same PR as the `write_data` change above).**
`write_mesh` now takes `(self, points: &[f64], connectivity: &[u64], cell_types: &[CellType])`;
`validate_points_and_cells` and `prepare_cells` were threaded the same way, and every call site
(README, doctests, examples, tests) was updated. The rest of this section is kept for the record.

**Severity: low, but this is the cheapest readability win.**

### Problem

```rust
pub fn write_mesh(self, points: &[f64], cells: (&[u64], &[CellType])) -> IoResult<TimeSeriesDataWriter>
xdmf_writer.write_mesh(&coords, (&connectivity, &cell_types))
```

The tuple is not a named type, does not travel as a unit, and adds parens at every call site.
Internally it is worse: `cells.0` / `cells.1` appear 10 times across `write_mesh` (`:72`),
`validate_points_and_cells` (`:147`) and `prepare_cells` (`:209`), and which is which has to be
re-derived every time.

### Change

```rust
pub fn write_mesh(
    self,
    points: &[f64],
    connectivity: &[u64],
    cell_types: &[CellType],
) -> IoResult<TimeSeriesDataWriter>
```

Thread the same split through `validate_points_and_cells` and `prepare_cells`, replacing every
`cells.0` / `cells.1` with a named parameter. This also makes `write_mesh` consistent with the
flat triples `write_data` now takes.

Files: `src/time_series_writer.rs`, plus every call site: `tests/time_series_writer.rs`,
`tests/binary_writer.rs`, `tests/vtk_comparison.rs`, `examples/paraview_smoke.rs`, `README.md`,
and the three doctests.

### Tests

Purely mechanical — no behavior changes, so the existing suite plus the golden XML comparisons
are the regression check.

### Related, decide separately

`write_mesh(&points, &[], &[])` for a point cloud stays cryptic even after flattening. A one-line
`write_points(&points)` wrapper would say what it means, and point clouds are already a supported
and tested path (`write_xdmf_only_point_mesh`, `validate_points_and_cells_only_points`), so this
is not speculative API. Counter-argument: it is a second entry point for a one-argument
difference, and documenting the empty-slice convention is enough. Judgment call — flag it, do not
bundle it with the mechanical flattening.

---

## 4. Reject invalid `deflate_level` at construction

**DONE (2026-08-09, not yet committed).** `create_writer` (`src/lib.rs`) now calls
`validate_deflate_level` for both `Hdf5SingleFile` and `Hdf5MultipleFiles` before the
feature-gated branch, so it fires the same way with or without the `hdf5` feature enabled (an
invalid level is now reported before the "requires the hdf5 feature" error would otherwise win).
The rest of this section is kept for the record.

**Severity: low.**

### Problem

`DataStorage::Hdf5SingleFile { deflate_level: Some(42) }` is accepted by `TimeSeriesWriter::new`
and only fails later, inside `write_mesh`, with a raw HDF5 message:

```
deflate_level=42 -> new() OK, write_mesh -> "H5Pset_deflate(): invalid deflate level"
```

Valid range is 0–9. A wrong-by-construction value should not survive past the constructor, and
the caller should get the crate's own error rather than an HDF5 internal one.

### Change

Validate the range where the level is resolved (`create_writer`, `src/lib.rs:118`), returning the
usual `InvalidInput` error with a message naming the valid range. Applies to both
`Hdf5SingleFile` and `Hdf5MultipleFiles`.

Files: `src/lib.rs` (or `src/hdf5_writer.rs` constructors, whichever keeps the `hdf5`-feature
cfg cleaner — note the check must not break the `--no-default-features` build, where these
variants already error out for a different reason).

### Tests

- `Some(10)` and `Some(255)` rejected with the crate's message.
- `Some(0)`, `Some(9)`, and `None` all accepted.
- Must be gated on the `hdf5` feature, or written so it still makes sense without it.

### Noted, not scheduled

`DataStorage: FromStr` cannot express `deflate_level` at all — every string form parses to
`None`. Harmless asymmetry, only worth fixing if the string form ever needs to round-trip.

---

## 5. Decision needed: the error type

**DECIDED (2026-08-08): yes, do it — scheduled as M1, see [`01_error_type.md`](01_error_type.md).**
The deciding argument was the one below plus a new one: the reader (M5) roughly doubles the number of
distinct failure modes, and shipping it on `io::Error` means designing its error surface twice. The
original write-up is kept below for the reasoning.

Every error in the crate is a `std::io::Error`, almost always with `ErrorKind::InvalidInput` and a
formatted string. A caller cannot distinguish "your array is the wrong size" from "duplicate data
name" from "the disk is full" without substring-matching the message — which is exactly what the
test suite does, per the pattern documented in `CLAUDE.md`.

A dedicated error enum would fix this, with `io::Error` becoming one variant.

Arguments for waiting: this is the largest churn of anything in this document (every
error-assertion test changes), and for a writer library where most errors are caller mistakes
caught up front, `io::Error` is defensible. Arguments for doing it: the crate is pre-1.0 and
explicitly willing to take breaking changes, and this gets much more expensive to change later.

The point of listing it here is to make it a deliberate yes/no rather than something the crate
drifts into 1.0 with.

---

## Housekeeping

Running `cargo test --doc` writes `xdmf_write_data.xdmf2` and `xdmf_write_mesh.xdmf2` into the
repo root, because the doctests pass bare relative file names to `TimeSeriesWriter::new`. They
are untracked and not ignored, so they show up in `git status` after any doctest run. Either point
the doctests at a temp directory or add the pattern to `.gitignore`.

---

## Suggested sequencing

1. Item 1 (writer poisoning) and item 2 (time dedup) — both are defects, both are small, both are
   confined to `time_series_writer.rs` plus one writer.
2. Item 3 (flatten `write_mesh`) — mechanical, touches many files, best done in isolation so the
   diff stays reviewable. **Done, see above** — landed ahead of items 1/2/4 with no dependency
   violation; nothing in this document makes 1/2/4 depend on 3 or vice versa.
3. Item 4 (deflate range) — trivial, can ride along with anything.
4. Item 5 — a conversation, not a task.

Items 1–4 together are one round of breaking changes. Since the crate is pre-1.0 and not keeping
backward compatibility, landing them close together keeps the number of releases that break
callers to one. In practice this document's items landed as two separate changes: item 3 with the
`write_data` rework on 2026-08-09 (PR #18, merged), and items 1, 2, and 4 together, also on
2026-08-09 but as a second, not-yet-committed change — so M0 ends up as two breaking releases
rather than the one this paragraph originally aimed for.

---

# Adjustments after the rest of the roadmap was written (2026-08-08)

This document was written before the other plans. Re-read against them
([`ROADMAP.md`](ROADMAP.md) and its sub-plans), here is what changes. **Items 1–4 all still stand
and are still the right first thing to do** — they become milestone M0, ahead of everything else.

### Item 1 (mid-write error poisons the writer) — reinforced, not changed

Do it as written. Note that `02_performance.md` part B restructures `write_data` so each step's XML
fragment is built in memory and only written on success. That makes "a failed step leaves the writer
as if the call had never happened" literally true for the light data, so the two changes reinforce
each other. The open question at the end of item 1 (should a partially-written step delete its heavy
data files?) is unaffected — the recommendation to leave them stands.

Moving the binary range check up front (part 2 of item 1) has a second payoff: `02_performance.md`
part D wants to replace the per-element write loop with a bulk `bytemuck` encode, which is only
possible once the loop no longer has to range-check as it goes.

### Item 2 (time dedup is textual) — unchanged, absorbed into M1

The `HashMap<u64, String>` keyed on `f64::to_bits` is still the right structure. The error message
this item proposes (`Time step '0.10' has already been written (as '0.1')`) becomes
`Error::DuplicateTime { time, existing }` in `01_error_type.md`. Land item 2 first with the message
as written; M1 converts it to a variant along with the other 28 assertions.

### Item 3 (flatten `write_mesh`'s cell tuple) — extend to blocks

Unchanged, but `04_submeshes.md` adds `write_mesh_with_blocks` with the *same* flat parameter list.
Ship both in the same breaking release if the ordering allows, so callers do one migration rather
than two. If M4 lands later, keep the flat shape for consistency.

The "related, decide separately" question — whether to add `write_points(&points)` for point clouds —
should be decided at the same time as the block API, since that is when the family of mesh-writing
entry points is being designed as a set. It is also relevant to the reader:
`05_reader.md` has to special-case the Polyvertex-over-all-points pattern to round-trip a point
cloud, and a named `write_points` makes that convention discoverable rather than folkloric.
Recommendation is now mildly **in favour** of adding it, on those grounds.

### Item 4 (reject invalid `deflate_level`) — unchanged, plus one note

Still trivial, still do it. `01_error_type.md` gives it `Error::DeflateLevelOutOfRange { level }`.
One addition: `07_mpi.md` will introduce a second writer-construction path (`new_parallel`), so put
the check where the level is *resolved* (`create_writer`, `src/lib.rs:118`), not in the constructor,
so a future second entry point cannot bypass it.

The "noted, not scheduled" observation that `DataStorage: FromStr` cannot express `deflate_level`
stays unscheduled. Note that `05_reader.md` makes `data_storage()` purely informational on the reader
side (the reader is driven by `DataItem` contents, not by the stored `DataStorage` tag), which
removes the only plausible reason the string form would ever need to round-trip.

### Item 5 (the error type) — decided, scheduled

See the update in the section itself. M1, before the reader.

### Housekeeping — expanded

Still valid. Add to it:

- `reader.rs` in the repo root is the reader sketch; its content is now in `05_reader.md`. Delete it.
- This file has moved into `plans/`; commit the move.
- The `.gitignore` fix for `xdmf_write_{data,mesh}.xdmf2` is the smaller half of the fix — pointing
  the doctests at a temp directory is better, because the current doctests also model bad practice
  for anyone copying them, and `02_performance.md` part F wants the `write_data` doctest rewritten to
  demonstrate buffer reuse anyway.

### One thing this document did not cover, now covered elsewhere

The largest API-adjacent performance problem in the crate is not in items 1–5: `write_data`
re-serializes the entire XDMF document on every call, cloning every prior step's grid and (for
`AsciiInline`) the entire inline mesh text with it. That is O(steps²) work and unbounded memory
growth. It is `02_performance.md` part B.
