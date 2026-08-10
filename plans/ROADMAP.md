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
| 4 | `Values` numeric types | **Add `F32`**, plus an opt-in f64→f32 downcast on write. Not a full numeric type set. Unchanged by the benchmark findings, but the *size claim* is now gated on a measurement — see `03_values_and_f32.md`. |
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
- **2026-08-10, not yet merged (local, uncommitted).** Started M2 part A (benchmark harness):
  `benches/common/mesh.rs` (Rust port of the CFD duct mesh generator), `benches/write_time_series.rs`
  (criterion: `write_mesh`/`write_data`/`steps_scaling`/`allocations_per_step` at 1e3/1e5),
  `benches/common/counting_alloc.rs` (allocation-counting global allocator), and
  `examples/bench_cfd.rs` (manual 1e7 driver). First allocation-count data point: `Ascii`/`AsciiInline`
  do ~21k allocations per `write_data` call (1e3 mesh) vs. ~300 for `Binary`/HDF5 — a baseline for
  part C. Not done yet: the compressed-size codec sweep and RSS tracking part A also asks for (see
  the code comments in `write_time_series.rs`/`bench_cfd.rs` for what's deferred and why). Parts
  B–F of M2 are still unstarted.

## Sub-plans

| Plan | Milestone | Covers |
|------|-----------|--------|
| [`API_IMPROVEMENTS_PLAN.md`](API_IMPROVEMENTS_PLAN.md) | M0 | Pre-existing defects and interface cleanups (see its "Adjustments" section) |
| [`01_error_type.md`](01_error_type.md) | M1 | The `Error` enum, migration of 29 message assertions and 49 `IoResult` signatures |
| [`02_performance.md`](02_performance.md) | M2 | Benchmark harness, O(n²) light-data write, allocation elimination, HDF5 tuning |
| [`03_values_and_f32.md`](03_values_and_f32.md) | M3 | `Values::F32`, opt-in f64→f32 downcast, ParaView verification |
| [`04_submeshes.md`](04_submeshes.md) | M4 | `write_mesh_with_blocks`, zero-copy fast paths, block validation |
| [`05_reader.md`](05_reader.md) | M5 | `TimeSeriesReader` / `TimeSeriesDataReader`, per-format `DataReader` backends |
| [`06_python_bindings.md`](06_python_bindings.md) | M6 | Review of the vibe-coded bindings, GIL release, abi3 + static-HDF5 wheels on PyPI |
| [`07_mpi.md`](07_mpi.md) | post-1.0 | API draft only: collective sizing, global node ids, verification strategy |

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
- **No `CHANGELOG.md`** (decided 2026-08-10, user preference) — migration notes for arotau and the
  Python package, if needed, travel via PR descriptions instead.
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
| `quick-xml` + serde `#[serde(flatten)]` on `DataContent` does not round-trip on deserialize | M5 | Test deserialization of one `DataItem` on day one; fallback is a hand-rolled event-loop parser for `DataItem` only. |
| Static HDF5 build in manylinux/macOS/Windows wheels | M6 | Spike Linux-only first, before building out the platform matrix. |
| `hdf5::File` may not be `Send`, blocking GIL release | M6 | Check early; if it is not, the Python HDF5 path keeps the GIL and only the other backends release it. |
| ParaView misreading a new XML construct (hyperslab block references) | M4 | Two-stage plan with an explicit go/no-go: ship duplication first, add the hyperslab fast path only after CI proves it. |
