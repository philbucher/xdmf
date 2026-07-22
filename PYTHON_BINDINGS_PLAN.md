# Python bindings for `xdmf` via pyo3/maturin

**Status: implemented and verified (uncommitted).** This doc now reflects what was actually
built, not just the original plan — see the "Deviations from the original plan" section for the
one place implementation diverged.

## Context

The `xdmf` crate currently only has a Rust API. The user wants a Python interface,
built with pyo3/maturin (a pattern they've used on other projects), to be benchmarked
against `pyvista`'s VTK writing for performance. The explicit requirement is that
values handed in from Python (as numpy arrays) should not be copied where avoidable —
this matters most for per-timestep attribute data, since mesh geometry/connectivity is
written once but attribute data is written repeatedly (once per timestep), so it's the
hot path for the benchmark.

Decisions confirmed with the user before implementation:
- Bindings live in a **separate workspace crate** (`python/`), not a feature flag on
  the core crate — keeps `xdmf` on crates.io free of pyo3/numpy deps.
- The core crate's `Values` type was **refactored to borrow** (`Cow`-backed,
  lifetime-parameterized) so attribute data can be passed from Python with zero copies.
- **HDF5 storage was out of scope for the first cut** (avoids the CMake/static-linking
  build complexity discussed earlier) — only `Ascii`, `AsciiInline`, `Binary` were
  initially exposed to Python. **Update (2026-07-18): wired up.** The concern turned out to
  only bite the `static` (build-from-source) linking mode; this machine already has
  `libhdf5-dev` installed system-wide, so `hdf5-metno`'s default (dynamic, `pkg-config`
  based) linking needs no CMake at all. `python/Cargo.toml` now depends on
  `xdmf` with `features = ["hdf5"]` (still `default-features = false`, so a future
  default-feature addition on the core crate doesn't get silently pulled into the wheel),
  and `PyDataStorage` (`python/src/enums.rs`) gained `Hdf5SingleFile`/`Hdf5MultipleFiles`
  variants. No other code changes were needed — `TimeSeriesWriter`/`TimeSeriesDataWriter`
  are already generic over `DataWriter`. Covered by two new tests in
  `python/tests/test_basic.py` (`h5py` via `pytest.importorskip`, so it's not a hard test
  dependency). Distributing a portable wheel (one that doesn't require the end user to have
  system HDF5 installed) would still need the `static` feature + CMake — not done, since
  this was about enabling local/dev use, not packaging.

## Part 1 — Core crate: make `Values` borrow-capable — DONE

`src/values.rs`: `Values` is now `Values<'a>`, backed by `Cow<'a, [f64]>` / `Cow<'a, [u64]>`:

```rust
pub enum Values<'a> {
    F64(Cow<'a, [f64]>),
    U64(Cow<'a, [u64]>),
}
```

- `From<Vec<f64>>` / `From<Vec<u64>>` (→ `Cow::Owned`) kept for existing Rust callers.
- `From<&'a [f64]>` / `From<&'a [u64]>` (→ `Cow::Borrowed`) added — used by the Python
  binding to wrap a numpy buffer with no copy.
- `ValueType::as_slice`/`as_mut_slice` needed an explicit named lifetime on the *outer*
  reference (`fn as_slice<'v>(values: &'v Values<'_>) -> Option<&'v [Self]>`) — plain
  elision doesn't work here because `&Values<'_>` has two lifetime positions (the `&`
  and `Values`'s own param), so the compiler can't auto-pick which one the return
  borrows from. `as_mut_slice` uses `v.to_mut()` (clone-on-write for a borrowed `Values`).
- `gather` wraps its result in `Cow::Owned(..)`; `write_data`/`collect_data` in
  `time_series_writer.rs` needed a second lifetime param with an explicit `'v: 'a` bound
  (a `&'a Values<'v>` requires the data to outlive the reference to it).
- Propagated `Values<'_>` through `DataWriter::write_data`, `ascii_writer.rs`,
  `binary_writer.rs`, `hdf5_writer.rs`, and `time_series_writer.rs`'s helpers.
  `let x: Values = ...` local bindings (tests, doctests) needed **no** changes — lifetime
  elision in `let` type annotations resolves via inference, unlike item signatures.
- One non-obvious fixup: `hdf5_writer.rs`'s `data_set.write(v)` where `v` is now
  `&Cow<[T]>` doesn't compile via deref coercion (works for concrete `&[T]` params, not
  generic trait-bounded ones) — needed explicit `&v[..]`.

**Verified**: `cargo build`/`test`/`clippy -- -D warnings` clean under both default
(`hdf5` on) and `--no-default-features`. 130 tests + 4 doctests passing.

## Part 2 — Workspace + crate scaffold — DONE

- Root `Cargo.toml` gained `[workspace] members = ["python"]`.
- `python/Cargo.toml`: package `xdmf-python`, `publish = false`, `[lib] name = "xdmf"`
  (so the compiled artifact importable name is `xdmf`, matching the `pyproject.toml`
  project name), `crate-type = ["cdylib"]`. Deps resolved via `cargo add`:
  **pyo3 0.29.0** (`extension-module` feature) and **numpy 0.29.0** (versions track each
  other). `xdmf = { path = "..", default-features = false }` — confirmed this actually
  excludes `hdf5-metno`/CMake from the wheel build (`cargo build -p xdmf-python` alone,
  which is what `maturin` invokes, does not compile `hdf5-metno`; `cargo build
  --workspace` *does* pull it in, because feature-unification unions the root package's
  own default-on `hdf5` feature across all primary targets in that one invocation — a
  harmless `--workspace`-only quirk, not a wheel-build problem).
- `python/pyproject.toml`: maturin backend, project name `xdmf`, dep `numpy>=1.21`,
  pure-extension layout (no `.pyi` stubs).
- Tested locally via a venv at `python/.venv` (gitignored, along with
  `.pytest_cache`/`__pycache__`/`*.egg-info`) with `maturin`, `numpy`, `pytest` installed.

## Part 3 — Bindings (`python/src/`) — DONE

Files: `error.rs` (IoError → PyErr), `enums.rs` (`DataStorage`, `CellType`,
`DataAttribute`), `arrays.rs` (`FloatArray`, `UintArray`, `ValueGuard` — the numpy
conversion layer), `writer.rs` (`PyTimeSeriesWriter`, `PyTimeSeriesDataWriter`), `lib.rs`
(`#[pymodule] fn xdmf`).

- `DataStorage`/`CellType`: fieldless `#[pyclass(eq, eq_int, from_py_object)]` enums.
  `CellType` discriminants mirror `xdmf::CellType` exactly. (`from_py_object` needed
  explicitly — pyo3 0.29 deprecated the implicit auto-derived `FromPyObject` for
  `Clone` pyclasses in favor of opt-in.)
- `DataAttribute`: wrapper struct (`Matrix`/`Generic` carry data, so not a plain enum) —
  `SCALAR`/`VECTOR`/`TENSOR`/`TENSOR6` class attributes, `.matrix(rows, cols)` /
  `.generic(size)` static methods.
- `PyTimeSeriesWriter` wraps `Option<xdmf::TimeSeriesWriter>`, `.take()`s itself on
  `write_mesh`/`write_mesh_with_blocks` (Rust's consuming API); a second call raises
  `RuntimeError`. Both pyclasses are `#[pyclass(unsendable)]` (see deviation below).
- Zero-copy numpy path (`arrays.rs`): `FloatArray` requires contiguous `float64`.
  `UintArray` accepts `uint64` (borrowed as-is) or `int64` (borrowed, after an O(n) sign
  check + bit-reinterpret to `u64` — `numpy`'s default int dtype is signed `int64`, so
  requiring `uint64` would force a copy on the common path). Non-contiguous arrays raise
  `ValueError` rather than being silently copied. `ValueGuard` unifies both for
  `write_data`'s attribute arrays (tries `FloatArray` first, then `UintArray`).
- `cell_types` takes `Vec<PyCellType>` (actual `xdmf.CellType` enum members) — **not**
  bare ints as originally sketched; dropped for scope, see follow-ups.
- `IoError` → `PyErr`: `ErrorKind::InvalidInput` → `PyValueError`, else `PyIOError`.

## Part 4 — Verification — DONE

1. `cargo build -p xdmf-python` / `cargo clippy -p xdmf-python --all-targets` — clean,
   zero warnings.
2. `maturin develop --release` inside `python/.venv` — builds and installs.
3. `python/tests/test_basic.py` (pytest, 8 tests, all passing): mesh + timestep data via
   numpy arrays (Binary and Ascii storage), `int64` connectivity accepted, negative
   `int64` rejected, wrong dtype (`float32`) rejected, non-contiguous array rejected,
   double `write_mesh` rejected, `write_mesh_with_blocks`.
4. Re-verified against the real ParaView install
   (`/home/philipp/software/ParaView-5.13.2-MPI-Linux-Python3.10-x86_64/bin/pvpython`):
   wrote a hexahedron mesh + 3 timesteps of point/cell data via the Python API, loaded it
   headlessly — correct point/cell counts, correct and distinct `height`/`cell_id`
   values per timestep.
5. Optional pyvista smoke comparison — **not run** (pyvista not installed; left for the
   user, this was explicitly optional).

## Deviations from the original plan

- **GIL is not released during writes.** The plan called for `py.allow_threads(...)`
  (now named `py.detach(...)` in pyo3 0.29) around the actual `write_mesh`/`write_data`
  call. This doesn't compile: `xdmf::TimeSeriesWriter`/`TimeSeriesDataWriter` contain a
  `Box<dyn DataWriter>`, and `dyn DataWriter` (no `+ Send` bound) isn't `Send`, which
  `detach`'s `Ungil` bound requires for anything crossing it. Fixing this means adding
  `Send` as a supertrait on the crate-private `DataWriter` trait in `src/lib.rs` — true
  for all four current writer impls (`AsciiWriter`, `AsciiInlineWriter`, `BinaryWriter`
  are trivially `Send`; the two HDF5 writers need checking), but it's a change to a
  trait also used by the `hdf5` feature build, so it wasn't made in passing. Writes
  currently run under the GIL — correct, just not maximally concurrent with other
  Python threads.

## Follow-ups (not done, either dropped for scope or genuinely out of scope)

- Release the GIL during writes (needs the `DataWriter: Send` change above).
- `cell_types` currently only accepts `xdmf.CellType` enum members, not a numpy
  `uint8` array of codes or bare ints — fine for typical meshes (this array is rarely
  the dominant cost) but worth adding if profiling says otherwise.
- Portable wheel distribution of HDF5 storage (needs the `static` feature + CMake so end
  users don't need system HDF5 installed) — local/dev use is already wired up, see above.
- `.pyi` type stubs / a mixed pure-Python wrapper package.
- 2D (`N, 3`) convenience array shapes for points (flat 1D only, matches Rust layout).
- Run the pyvista comparison benchmark.
