# Review findings: `submeshes` branch (submeshes + reader)

Reviewed `origin/main...submeshes` (~9k lines) on 2026-08-23, high-level: interfaces, usage,
correctness. Verified locally: `cargo nextest run` 247/247 pass, `cargo doc` clean,
`cargo +nightly fmt --check` clean, `typos` clean, `cargo clippy -- -D warnings` **fails**.

Each bullet is independent -- delete the ones you don't want.

**Status (implemented):** every `[x]` item below is done. Verified afterwards:
`cargo nextest run` 259/259 (debug, release, and `--no-default-features` 217/217),
`cargo test --doc` 7/7, `cargo clippy --all-targets -- -D warnings` clean with and without the
`hdf5` feature, `cargo doc` clean, `cargo +nightly fmt --check` clean, `typos` clean. The two
`[ ]` items (per-submesh reads, a `SelectionWriter` sub-trait) were left alone, and the
`Format Some(XML)` message was reverted afterwards on request -- judged over the top for what it
buys, so that message still reads as it did.

## Blocking (CI fails today)

- [x] `src/reader.rs:1093` `parse_data_storage` is dead code -> `-D warnings` error, both with and
      without the `hdf5` feature. Either wire it up (see the "reject unreadable files early" item,
      which is what it was clearly written for) or delete it.
- [x] `src/reader.rs:1117` `parse_deflate_level` is dead code, same error, same call.
- [x] `src/reader/light_data.rs:73` and `:76`: `pub(super)` fields on `Analysis` trip
      `clippy::field_scoped_visibility_modifiers` (warn-level in `[lints.clippy]`, error under CI's
      `-D warnings`). Accessor methods, or make the whole struct's fields private and expose what
      the callers need.

## Interfaces

- [x] **Reject unreadable files in `TimeSeriesReader::new`.** Opening an `AsciiInline` file
      succeeds today; `num_points()`, `num_cells()` and `cell_data_info()` all answer, and only
      `read_points` fails. The document already records its `DataStorage` in an `Information`
      element, which is what `parse_data_storage` exists to read -- use it in `new()` and fail with
      "written with `AsciiInline`, only the HDF5 storages can be read".
- [x] **Fix the `Format Some(XML)` message** (`src/reader/hdf5_reader.rs`): a `Debug`-formatted
      `Option` in user-facing text. Print the format name, or "no Format attribute".
- [x] **Make `read_points`/`read_topology` generic over `ValueType`**, as `read_point_data` already
      is. Today they are hardwired to `Vec<f64>`/`Vec<u64>`, so a caller who wrote `f32` coordinates
      and `u32` connectivity -- the case the writer works hardest to preserve exactly -- cannot read
      them back at that width and pays 2x memory. Alternative if this is deliberate: say why in the
      doc comment.
- [ ] **Per-submesh reads.** The reader only reassembles the whole mesh; reading one block means
      reading everything and slicing it with `submesh_cells`/`submesh_points`. Fine as a first cut
      -- but say so in the module doc, since submeshes are the branch's headline feature.
- [x] **Python parity:** `write_mesh_with_submeshes` is bound, `TimeSeriesReader` is not bound at
      all. If that is intentional, note it in `python/README.md`/`xdmf.pyi`.
- [ ] **`DataWriter` is now a partial trait**: six methods plus `supports_selections()`, of which
      `write_point_component` and `write_selection` default to `Err(Error::Internal)`. Correctness
      rests on callers checking `supports_selections()` first, not on the type. `pub(crate)`, so
      low urgency -- but a `SelectionWriter` sub-trait would make the contract checkable if a third
      selection-capable backend ever lands.
- [x] **Large contiguous submeshes allocate on the caller's side.** `B: AsRef<[usize]>` forces a
      caller splitting a 100M-cell mesh into element blocks to materialise the full index lists
      (~800 MB) purely for `collapse_indices` to fold each back into two numbers. Accepting
      something range-shaped would remove that; `IndexList::Contiguous` already models it.

## Correctness / robustness

- [x] **The reader re-parses the document on every call.** `Analysis::build` runs at
      `src/reader.rs:295` (`read_points`), `:316` (`read_topology`) and `:493` (`step_grid`) -- and
      `step_grid` is called once *per submesh* inside `read_submesh_field`, so one field read costs
      O(submeshes x total_grids) = O(S^2 x T) grid collection plus a `Vec<Vec<&Grid>>` allocation
      each time. The module doc promises the opposite ("parses the whole document up front ... every
      read call after it is a plain query"). Fix: store the analysis as *indices* in `new()` --
      storing `&Grid` is what forces the rebuild.
- [x] **`decode_mixed` skips a poly-cell's point count without checking it**
      (`src/reader/topology.rs:73`). Safe for this crate's own files (the count is always the type's
      fixed one), but a foreign `Mixed` file with a real multi-node Polyvertex desynchronises the
      stream and surfaces as a confusing "unknown cell type code". `cell_type_of` already rejects
      the equivalent case for uniform topologies -- one comparison makes the two consistent.
- [x] **Implicit name coupling between geometry and index lists.** `selected_coordinates` emits a
      `Reference` to `submesh_points_{k}` before `finish_mesh` writes that `DataItem`; the link is
      by string name only, and `write_submesh_index_lists` keeps writing those arrays for the
      selecting storages solely because of it. Works, and round-trip tests cover it -- but nothing
      in the types stops a future edit there from producing dangling references. At minimum, a
      comment at both ends naming the other.

## Docs / conventions

- [x] **`CLAUDE.md` was not updated** for this branch: no mention of `src/reader/`, of the reshaped
      `DataWriter` trait, or of the three new error variants.
- [x] **`CLAUDE.md`'s "under 10 variants" claim about `Error` is now false** (12 variants). Either
      update the sentence or fold `InvalidDocument`/`Unsupported`/`NumberTypeMismatch` -- though all
      three look justified as written, so updating the doc is probably the right call.
- [x] **The README "Reading" example is a plain ```rs block, not a doctest**, so nothing checks it
      compiles -- unlike the writer examples above it.
- [x] **No test for reading an `Ascii`/`Binary` file.** `tests/reader.rs` covers both HDF5 storages
      well, but the first wall most users hit -- pointing the reader at a non-HDF file -- has no
      coverage. Pairs with the "reject in `new()`" item above.

## Explicitly *not* findings (checked, fine as designed)

- Selection arrays survive a discarded step: single-file HDF5 unlinks only the time group,
  multi-file writes selections into `mesh.h5` and removes only the step's own file. Both
  deliberate, both commented.
- `write_data_per_submesh`/`write_data_selected` collect attributes and append only after every
  submesh succeeded, so a mid-field failure cannot leave blocks disagreeing about which fields
  they carry.
- `paraview::validate` is still called from the `TimeSeriesWriter`/`TimeStep` layer only; the
  submesh index arrays deliberately bypass it (written signed, read back at the declared width),
  and that is stated at the call site.
- The "every cell must belong to at least one submesh" rule forces catch-all blocks (the example
  admits `front_row` exists only to pick up cell 1), but silently dropping cells from the
  visualization is the worse failure. Documented in both the Rust doc and the README.
