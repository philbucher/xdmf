# Session summary — Python bindings via pyo3/maturin (2026-07-18)

> **Status: historical record.** This documents a past session's implementation, now committed on
> `origin/multiple-features` (not `main`). `ROADMAP.md` Decision 7 treats that branch as a
> **reference implementation to review, not to merge** — see `06_python_bindings.md` for the
> up-to-date, critical review and the plan for re-landing this properly. This file is the "what
> happened and why" record that plan draws on; it is not itself a plan to execute.

## Where the work actually lives

- Branch: `origin/multiple-features`, commit `219635d` ("python bindings"), on top of
  `25c7be0` ("binary writer and block support"). Also present in `48db9d2` ("more features").
- **Not on `main`** and not merged — `main` has since diverged (the `Values` `Cow` rework was
  redone independently on `main`, per `ROADMAP.md`'s "Drop" list).
- Files: `PYTHON_BINDINGS_PLAN.md`, `python/{Cargo,pyproject}.toml`, `python/src/{lib,arrays,enums,
  error,writer}.rs` (~470 lines), `python/tests/test_basic.py` (8 tests, all passing at the time).

## Request and constraints

Build a Python interface for the `xdmf` crate via pyo3/maturin, to be benchmarked against
`pyvista`'s VTK writing. Hard requirement: values handed in from Python as numpy arrays must not be
copied where avoidable — this matters most for per-timestep attribute data (mesh geometry is
written once; attributes are written every step, so that's the hot path for the benchmark).

Decisions taken with the user before implementing:

- Bindings in a **separate workspace crate** (`python/`), not a feature flag on the core crate —
  keeps `pyo3`/`numpy` off the crates.io `xdmf` dependency graph.
- Core crate's `Values` type **refactored to borrow** (`Cow`-backed, lifetime-parameterized), so
  attribute data can cross from Python with zero copies.
- **HDF5 excluded from this first cut** — avoids the CMake/static-linking build complexity for the
  wheel; only `Ascii`, `AsciiInline`, `Binary` exposed to Python.

## What was built

**Core crate (`src/values.rs` and call sites):** `Values` became `Values<'a>`, backed by
`Cow<'a, [f64]>` / `Cow<'a, [u64]>`. `From<Vec<T>>` (→ `Cow::Owned`) kept for existing Rust callers;
`From<&'a [T]>` (→ `Cow::Borrowed`) added for the zero-copy Python path. Propagated `Values<'_>`
through `DataWriter::write_data`, `ascii_writer.rs`, `binary_writer.rs`, `hdf5_writer.rs`, and
`time_series_writer.rs`. Verified clean (`build`/`test`/`clippy -D warnings`) under both `hdf5` and
`--no-default-features`, 130 tests + 4 doctests passing.

**`python/` crate:** `xdmf-python` package, `crate-type = ["cdylib"]`, `[lib] name = "xdmf"`,
`xdmf = { path = "..", default-features = false }` (confirmed this excludes `hdf5-metno`/CMake from
the wheel build). pyo3 0.29.0 + numpy 0.29.0. Maturin backend, tested locally via a venv.

- `error.rs` — `IoError` → `PyErr` (`InvalidInput` → `ValueError`, else → `OSError`/`IOError`).
- `enums.rs` — `DataStorage`, `CellType` (discriminants mirroring `xdmf::CellType`), `DataAttribute`
  (wrapper struct with `SCALAR`/`VECTOR`/`TENSOR`/`TENSOR6` class attrs, `.matrix()`/`.generic()`).
- `arrays.rs` — the zero-copy numpy layer. `FloatArray` requires contiguous `float64`. `UintArray`
  accepts `uint64` (borrowed as-is) or `int64` (borrowed, after an O(n) sign check + bit-reinterpret
  to `u64` via `unsafe { slice::from_raw_parts }` — numpy's default int dtype is signed, so requiring
  `uint64` would force a copy on the common path). Non-contiguous arrays raise `ValueError` rather
  than silently copying.
- `writer.rs` — `PyTimeSeriesWriter` / `PyTimeSeriesDataWriter`, both `#[pyclass(unsendable)]`
  (see deviation below). `write_mesh` consumes `Option<xdmf::TimeSeriesWriter>` via `.take()`,
  matching the Rust API's consuming style; a second call raises `RuntimeError`.

**Verification:** pytest suite (8/8 passing: both storage modes, `int64` connectivity, negative
`int64` rejection, dtype rejection, non-contiguous rejection, double `write_mesh`, blocks); headless
ParaView load via `pvpython` confirmed correct point/cell counts and correct, distinct per-timestep
attribute values. The pyvista comparison itself was **not run** (pyvista wasn't installed).

## Deviation from the plan: GIL is not released during writes

The plan called for `py.allow_threads(...)` (pyo3 0.29 renamed this `py.detach(...)`) around the
actual write calls. This didn't compile: `TimeSeriesWriter`/`TimeSeriesDataWriter` hold a
`Box<dyn DataWriter>`, and `dyn DataWriter` has no `Send` bound, which `detach`'s `Ungil` bound
requires for anything crossing it. The fix (`Send` as a supertrait on `DataWriter`) is safe for
`AsciiWriter`/`AsciiInlineWriter`/`BinaryWriter` but needed checking against the `hdf5` feature
build (`hdf5::File`'s `Send`-ness was unverified), so it wasn't made in passing — writes run under
the GIL. `06_python_bindings.md` item 4 picks this back up.

## Process note (why this file exists as a separate record)

Mid-session the user had to correct the plan-document location twice — a plan written only to the
Claude-internal plan-mode file (`~/.claude/plans/...`) is not "on disk" from their perspective;
"write to disk" means the project's own working directory. `PYTHON_BINDINGS_PLAN.md` at the repo
root (now committed on `multiple-features`) was the fix. Worth remembering for any future planning
docs in this repo: they belong under version control in the project tree, not only in Claude's own
state.

## Everything else the review in `06_python_bindings.md` already covers

Rather than duplicate it: unsafe removal (`bytemuck` instead of raw pointer casts), error-message
precision, `cell_types`/blocks as numpy arrays instead of `Vec`, a context-manager `close()`, error
mapping onto the `Error` enum from `01_error_type.md`, `.pyi` stubs, version-skew checking, and the
abi3 + static-HDF5 wheel-building work — see that file for the actionable version of all of this.
