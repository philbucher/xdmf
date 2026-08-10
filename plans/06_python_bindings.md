# M6 — Python bindings and PyPI wheels

`README.md`: *"Python interface, already exists in the 'multiple-features' branch. => this was
vibe-coded, needs to be double checked. Data must ideally not be copied when going from Rust <=>
Python"*, and *"Publishing wheels to pypi"*.

Decision 3 in `ROADMAP.md`: **static HDF5 built into the wheel**, so `pip install xdmf` gives working
HDF5 storage with no system library. Decision 7: the branch's `python/` is a **reference
implementation to review**, not commits to merge.

This is last because every earlier milestone breaks the Rust API this layer wraps.

## Part 1 — Review and re-land the bindings

The branch has `python/src/{lib,arrays,enums,error,writer}.rs` (~470 lines) plus `Cargo.toml`,
`pyproject.toml`, and `tests/test_basic.py` (8 tests). The architecture is sound and should be kept:

- Separate workspace crate (`python/`), `publish = false`, `crate-type = ["cdylib"]`, `[lib] name = "xdmf"`.
  Keeps `pyo3`/`numpy` off the crates.io `xdmf` dependency graph. Correct.
- `xdmf = { path = "..", default-features = false, features = ["hdf5"] }` — explicit rather than
  relying on the core crate's default features. Correct, and the comment explaining why should survive
  the re-land.
- Zero-copy numpy borrowing via `PyReadonlyArray1` + `Values::from(&[T])`. This is the requirement
  from `README.md` and it is met.

### Review findings to fix on the way in

These come from reading the branch code; each is a concrete change, not a general "review it".

1. **`unsafe` in `arrays.rs`.** `UintArray::as_u64_slice` reinterprets `&[i64]` as `&[u64]` via
   `slice::from_raw_parts` after a sign check. The reasoning is correct, but the core crate has zero
   `unsafe` and `02_performance.md` proposes `#![forbid(unsafe_code)]` there. Use
   `bytemuck::cast_slice::<i64, u64>` instead — both are `Pod`, it is the same zero-cost
   reinterpretation, and the sign check stays. Removes the only `unsafe` in the project.
2. **`FloatArray::extract` swallows the real error.** Any extraction failure maps to
   `"expected a 1D numpy array with dtype float64"`, so a 2-D `float64` array reports a dtype problem.
   Report shape and dtype separately.
3. **`ValueGuard::extract` misreports `float32`.** It tries `FloatArray` (f64) then `UintArray`, so a
   `float32` array falls through and is rejected with *"expected uint64 or int64"*. `03_values_and_f32.md`
   fixes the substance (float32 becomes natively supported via `Values::F32`); the error path still
   needs to name the actual dtype it got.
4. **The GIL is held during writes.** The branch documents this as a deviation: `py.detach()` requires
   `Ungil`, hence `Send`, and `Box<dyn DataWriter>` has no `Send` bound. Fix it: add `Send` as a
   supertrait on the crate-private `DataWriter` trait. `AsciiWriter`/`AsciiInlineWriter`/`BinaryWriter`
   are trivially `Send`. **Verify `hdf5::File`** — if `hdf5-metno`'s handle is not `Send`, the HDF5
   backends cannot cross `detach` and the options are (a) keep the GIL only for HDF5, or (b) drop the
   idea. Check this early; a 10-second write that blocks every other Python thread is a real defect in
   a library whose whole selling point is large-data throughput.
   Dropping `#[pyclass(unsendable)]` follows from the same change.
5. **`cell_types` takes `Vec<PyCellType>`.** Fine for typical meshes but it copies and it is
   per-element. Accept a numpy `uint8`/`int64` array of raw codes as an alternative input, using the
   `CellType::from_code` mapping that `05_reader.md` adds anyway.
6. **Blocks take `Vec<usize>`** — same issue; accept numpy index arrays, which is how a caller
   producing blocks in numpy will actually have them.
7. **No `close()` / context manager.** The HDF5 backends hold their files open until the object is
   dropped, and `CFD_BENCHMARK_PLAN.md` already records tripping over this (`del data_writer, writer`
   before reading file sizes). Add `__enter__`/`__exit__` and an explicit `close()`, mapping to the
   `finish()` method that `02_performance.md` part B introduces on the Rust side. This is the single
   most user-visible ergonomic fix in the list.
8. **Error mapping needs rewriting** for the `Error` enum from `01_error_type.md`: validation variants
   → `ValueError`, `Error::Io` → `OSError`, `Error::Unsupported` → `NotImplementedError`. Per-variant,
   not per-`ErrorKind`. Note the variant names here predate the merged enum — check `src/error.rs`
   rather than this list (`StorageRequiresFeature` never landed; that case is `InvalidConfiguration`,
   and `Unsupported` arrives with M5, see `05_reader.md`).
9. **No `.pyi` stubs.** Add them — an extension module with no stubs is invisible to every editor and
   type checker. `pyo3-stub-gen` or hand-written; hand-written is fine at this API size.
10. **Version skew.** `python/pyproject.toml` and `python/Cargo.toml` carry their own `version`,
    independent of the crate. Derive or check it in CI so a wheel cannot claim a version the crate
    does not have.
11. **pyo3/numpy 0.29** on the branch — re-check for the current release at implementation time; pyo3
    moves fast and the `allow_threads` → `detach` rename is exactly the kind of churn to expect.

### New surface to add

- **Reader bindings** (M5): `TimeSeriesReader`, `read_mesh` into numpy arrays, `read_step`, and a
  read-into-preallocated-array path for the large cases.
- **f32** (M3): `Values::F32` means `float32` numpy arrays become a first-class zero-copy input, and
  `with_reduced_precision()` should be exposed as a constructor keyword.
- **2-D array shapes.** Points as `(N, 3)` and vector fields as `(N, 3)` are the natural numpy layouts;
  the current bindings accept flat 1-D only. Accept both, since a C-contiguous `(N, 3)` array has
  exactly the flat memory layout the Rust API wants — so it stays zero-copy, it is purely a shape
  check. This removes a `reshape(-1)` from every realistic call site.

### Tests

Keep and extend `python/tests/test_basic.py` (currently 8 tests, covering both storage modes, `int64`
connectivity, negative-`int64` rejection, dtype rejection, non-contiguous rejection, double
`write_mesh`, and blocks). Add: context manager, reader round-trip, f32, `(N, 3)` shapes, every error
mapping, and a `h5py`-based content check (already done via `pytest.importorskip`, which is the right
pattern).

## Part 2 — Wheels on PyPI

### Build configuration

- **abi3.** Build with `pyo3/abi3-py39` so one wheel per platform covers every Python ≥ 3.9 instead of
  one per (platform × Python version). This cuts the matrix by ~5× and is the difference between a
  CI job that is maintainable and one that is not. The bindings use nothing that abi3 forbids.
- **Platforms:** `manylinux_2_28` x86_64 + aarch64, macOS x86_64 + arm64, Windows x86_64. Plus an
  sdist.
- **Tooling:** `maturin-action` in `.github/workflows/wheels.yml`, triggered on tags and manually.
- **Publishing:** PyPI trusted publishing (OIDC), not a long-lived API token.

### The HDF5 problem — spike before building the matrix

This is the risk in the milestone. `hdf5-metno`'s `static` feature builds HDF5 from source with CMake.
Unknowns to settle in a **Linux-only spike, before any of the platform matrix is written**:

1. Does the `static` build work inside the manylinux image, and is `cmake` available or pip-installable
   there?
2. Does the static build vendor zlib, or does it need a static zlib alongside? Deflate is not optional
   here — it is the entire reason HDF5 storage wins (see `CFD_BENCHMARK_PLAN.md`).
3. What does it do to build time? A from-source HDF5 per platform per CI run is minutes, not seconds;
   caching strategy matters.
4. Wheel size — a statically linked HDF5 is not small. Check it is acceptable.
5. Does the resulting `.h5` output still open in stock ParaView? It should (the file format is the
   file format), but this is cheap to confirm and expensive to discover late.

Only after Linux is green, extend to macOS (both arches) and Windows — Windows CMake/HDF5 is
historically the awkward one, so budget for it separately.

**Fallback if static HDF5 proves impractical on some platform:** ship the non-HDF5 wheel there
(`Ascii`/`AsciiInline`/`Binary` still work, and `is_hdf5_enabled()` already exists to report it), and
document the gap. Do not block the whole release on the worst platform.

### Package name

Check `xdmf` is available on PyPI before anything else — the branch's `pyproject.toml` assumes it. If
taken, decide the alternative (`xdmf-rs` / `pyxdmf`) early, because it propagates into the module name,
the docs, and every example.

## Part 3 — Re-run the pyvista benchmark

Two prior records: `CFD_BENCHMARK_PLAN.md` on `origin/multiple-features` (the HDF5-with-compression
measurements) and `SESSION_2026-08-08_cfd_benchmark.md` (the external-codec sweep against pyvista, and
the entropy finding). Read both before re-running; they measure different things and only together do
they explain the results.

Re-run once M2 (performance) and M3 (f32) have landed:

- The `1e7` numbers should improve from the bulk binary encoding (`02_performance.md` part D) and from
  HDF5 chunk tuning (part E).
- **Do not expect f32 to be a straightforward win on the archive metric, and do not report it as one
  before measuring.** `03_values_and_f32.md` carries the measurement gate and the reasoning: halving
  the raw bytes is not the same as halving the compressed archive, and the same session demonstrated a
  case where a 2× smaller raw representation compressed 1.77× *worse*. The expectation for f64→f32 is
  the opposite sign (it discards high-entropy mantissa noise rather than low-entropy padding) — but
  that is a hypothesis to test, not a result to assume.
- **Set expectations for the headline comparison honestly.** At 1e7, xdmf's `Binary` output plus an
  external codec currently loses to pyvista on final size at deflate, zstd-6 and bzip2, and only wins
  with lzma. The answer to that is HDF5 with `shuffle`+`deflate`, which compresses the values at write
  time — it was the first xdmf mode to beat pyvista on time *and* size simultaneously at 10M elements.
  Make sure the re-run includes the HDF5 modes; the session-summary table only covers `Binary`.
- The pyvista comparison was never run in the wheel-installed configuration; do that here, since it is
  the configuration users will have.

Keep the fairness methodology exactly as documented — the same external codec set applied to each
tool's *real default* output, on both sides. Two variants were tried and discarded before that rule
stuck, and it is the reason the numbers are trustworthy. Likewise, verify the other tool's actual
behaviour before explaining a difference: pyvista uses `float64`/`int64` throughout this path and never
`float32`, and assuming otherwise already misdirected one round of analysis.
