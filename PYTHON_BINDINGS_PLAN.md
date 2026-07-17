# Python bindings for `xdmf` via pyo3/maturin

## Context

The `xdmf` crate currently only has a Rust API. The user wants a Python interface,
built with pyo3/maturin (a pattern they've used on other projects), to be benchmarked
against `pyvista`'s VTK writing for performance. The explicit requirement is that
values handed in from Python (as numpy arrays) should not be copied where avoidable —
this matters most for per-timestep attribute data, since mesh geometry/connectivity is
written once but attribute data is written repeatedly (once per timestep), so it's the
hot path for the benchmark.

Decisions already confirmed with the user:
- Bindings live in a **separate workspace crate** (`python/`), not a feature flag on
  the core crate — keeps `xdmf` on crates.io free of pyo3/numpy deps.
- The core crate's `Values` type will be **refactored to borrow** (`Cow`-backed,
  lifetime-parameterized) so attribute data can be passed from Python with zero copies.
- **HDF5 storage is out of scope for this first cut** (avoids the CMake/static-linking
  build complexity discussed earlier) — only `Ascii`, `AsciiInline`, `Binary` are
  exposed to Python for now.

## Part 1 — Core crate: make `Values` borrow-capable

`src/values.rs`: change `Values` to `Values<'a>`, backed by `Cow<'a, [f64]>` /
`Cow<'a, [u64]>` instead of owned `Vec`s:

```rust
pub enum Values<'a> {
    F64(Cow<'a, [f64]>),
    U64(Cow<'a, [u64]>),
}
```

- Keep `From<Vec<f64>>` / `From<Vec<u64>>` (→ `Cow::Owned`) so existing Rust callers
  are source-compatible modulo the lifetime annotation.
- Add `From<&'a [f64]>` / `From<&'a [u64]>` (→ `Cow::Borrowed`) — this is what the
  Python binding will use to wrap a numpy buffer with no copy.
- `ValueType::as_slice`/`as_mut_slice`, `dimensions`, `number_type`, `len`, `precision`,
  `gather` bodies are essentially unchanged (`Cow` derefs to `[T]`); `gather` keeps
  building a new owned `Vec` internally (`Cow::Owned`), which is valid for any `'a`.
  Note in `as_mut_slice`'s doc comment that in-place mutation on a *borrowed* `Values`
  will clone-on-write (only owned `Values` get the "no reallocation" guarantee).
- Propagate `Values<'_>` through: `DataWriter::write_data`, `ascii_writer.rs`,
  `binary_writer.rs`, `hdf5_writer.rs` (`write_data` signatures), and
  `time_series_writer.rs` (`collect_data`, `check_data_size`, `validate_data_name`,
  `build_attribute`, `TimeSeriesDataWriter::write_data`, `validate_data`) — mechanical
  `&Values` → `&Values<'_>` / `&'a Values` → `&'a Values<'v>` fixups.
- Fix doctests in `time_series_writer.rs`/`lib.rs`, `README.md`'s example, and
  `tests/*.rs` the same way.
- Verify with `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings`
  (both default and `--no-default-features`) before moving on — this refactor must land
  clean on its own before the Python crate is added.

## Part 2 — Workspace + crate scaffold

- Add `[workspace]` / `members = ["python"]` to the root `Cargo.toml` (the root package
  stays the workspace root package too — this doesn't affect `cargo publish` for it).
- `python/Cargo.toml`: package `xdmf-python`, `publish = false`,
  `[lib] crate-type = ["cdylib"]`, deps: `xdmf = { path = ".." }`, `pyo3` (latest 0.2x,
  `extension-module` feature), `numpy` (matching pyo3 version).
- `python/pyproject.toml`: `build-backend = "maturin"`, project name `xdmf`,
  `dependencies = ["numpy>=1.21"]`. Pure-extension layout (module *is* the compiled
  `.so`) — no `.pyi` stub package for this first cut (fast-follow).

## Part 3 — Bindings (`python/src/`)

Expose, mirroring the Rust API shape:
- `DataStorage` (Ascii / AsciiInline / Binary only).
- `CellType` — mirrors `xdmf::CellType` (already `#[repr(u8)]` with explicit VTK
  discriminants in `src/xdmf_elements.rs`), exposed as a Python int-enum.
- `DataAttribute` — Scalar/Vector/Tensor/Tensor6 as constants, `Matrix(rows, cols)` and
  `Generic(size)` as static-method constructors (they carry data, so a plain enum
  doesn't fit).
- `TimeSeriesWriter` — wraps `Option<xdmf::TimeSeriesWriter>`. `write_mesh` /
  `write_mesh_with_blocks` do `self.inner.take()` (Rust's `write_mesh` consumes
  `self`), returning a `TimeSeriesDataWriter`; calling either twice raises a clear
  Python exception instead of a Rust move error.
- `TimeSeriesDataWriter` — wraps `xdmf::TimeSeriesDataWriter` directly (`write_data`
  takes `&mut self` in Rust, no consumption, so this is a plain mutable pyclass).

**Numpy → Rust, zero-copy path:**
- `points` and attribute float data: require a contiguous 1D `float64` numpy array,
  use `PyReadonlyArray1<f64>::as_slice()` to get `&[f64]` directly — matches the
  existing flat-layout Rust API (`&coords`, `Values::F64(Cow::Borrowed(..))`).
- `connectivity` / uint attribute data: accept **either** `uint64` (direct
  zero-copy) or `int64` (also zero-copy, via an O(n) sign-check pass — same cost as
  the pass we'd need anyway to write the bytes — then bit-reinterpret as `u64`, since
  indices/counts are never negative). This matters because numpy's default integer
  dtype is signed `int64` on Linux/Mac, so requiring `uint64` would force users into an
  `.astype()` copy on the most common path. Negative values produce a clear
  `ValueError`, not silent wraparound.
- Arrays that aren't C-contiguous are **rejected with a clear error** (e.g. "call
  `np.ascontiguousarray()`"), not silently copied — silent copying would be a perf trap
  given the whole point is predictable zero-copy behavior.
- `cell_types`: accept a plain Python sequence of `CellType`/`int` (small array
  relative to the rest of the mesh data in typical cases; a per-element validating
  conversion to `Vec<CellType>` is fine here and is not the hot path).
- Release the GIL (`py.allow_threads`) around the actual `write_mesh`/`write_data`
  call, after slice extraction — writing (disk I/O, and for `Binary`, the u32-narrowing
  pass) shouldn't block other Python threads. Slices borrowed via `PyReadonlyArray`
  remain valid without the GIL (pyo3-numpy's borrow tracking is independent of GIL
  state) — verify this holds under `maturin develop` with a quick concurrent-write
  smoke test before relying on it.
- `IoError` → `PyErr`: map `ErrorKind::InvalidInput` to `PyValueError`, everything else
  to `PyIOError`, via a small `From`/helper conversion used at every `?` boundary.

## Part 4 — Verification

1. `cargo build --workspace` / `cargo clippy --workspace --all-targets -- -D warnings`
   / `cargo test --workspace`.
2. `maturin develop --release` inside `python/` to build+install into the active venv.
3. `python/tests/test_basic.py` (pytest): write a small mesh + a few timesteps of
   point/cell data using numpy arrays (mirroring the existing Rust doctest example),
   assert on the produced `.xdmf2`/data files; cover the error paths (wrong dtype,
   non-contiguous array, mismatched sizes, negative int64 index).
4. Re-verify the produced file actually loads in ParaView via `pvpython`
   (`/home/philipp/software/ParaView-5.13.2-MPI-Linux-Python3.10-x86_64/bin/pvpython`),
   the same headless approach used earlier to catch the `Format::Binary` reader bug —
   worth reusing here since it's the ground truth for "does this actually work."
5. Optional smoke comparison: a short script writing a sizeable mesh via
   `xdmf.TimeSeriesWriter` vs. `pyvista`, just to sanity-check the zero-copy path is
   actually faster before the user runs their own benchmark.

## Out of scope (this pass)

- HDF5 storage from Python (needs the static-linking work discussed earlier).
- `.pyi` type stubs / a mixed pure-Python wrapper package.
- 2D (`N, 3`) convenience array shapes for points (flat 1D only, matches Rust layout).
