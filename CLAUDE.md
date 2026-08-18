# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust crate implementing the [XDMF](https://www.xdmf.org) file format for writing meshes with time-series data, readable by ParaView/VisIt. XDMF splits storage into light data (XML metadata describing the mesh/data layout) and heavy data (the actual numeric arrays), which can be stored in several formats. The main advantage over VTK-based formats is that the mesh can be written once and referenced by many time steps, instead of being duplicated per step.

## Commands

Tests use `cargo-nextest`, not `cargo test` (nextest doesn't run doctests, so those are run separately).

```bash
# lint (CI runs this with -D warnings; run both with/without the hdf5 feature)
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings

# tests
cargo nextest run
cargo nextest run --no-default-features        # hdf5 feature is default; verify the crate still works without it
cargo nextest run --release

# a single test
cargo nextest run test_name
cargo nextest run -E 'test(module::test_name)'

# doctests (not covered by nextest)
cargo test --doc

# formatting - requires nightly (uses unstable rustfmt options, see .rustfmt.toml)
cargo +nightly fmt --all
cargo +nightly fmt --all --check

# docs (CI treats missing docs and warnings as errors)
RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc --no-deps --document-private-items
```

CI (`.github/workflows/rust.yml`) runs this same matrix across Linux/macOS/Windows, each with and without the `hdf5` feature, in both debug and release. It also checks formatting on nightly and runs `typos` for spell-checking — run `typos` locally before pushing if you've added prose/comments.

`.clippy.toml` allows `unwrap`/`expect`/`panic` in test code, but the crate-level lint list in `Cargo.toml` (`[lints.clippy]`) is strict elsewhere (e.g. `unwrap_used`, `expect_used`, `panic` are warnings promoted to errors by CI's `-D warnings`) — don't introduce those in library code.

## Architecture

### Two-layer model: XDMF elements vs. writers

- **`src/xdmf_elements/`** (`Xdmf`, `Domain`, `Grid`, `Geometry`, `Topology`, `Attribute`, `DataItem`, ...) — typed, serde-serializable structs mirroring the XDMF XML schema itself. These are format-agnostic; they just describe *what* the light-data XML looks like (dimensions, cell types, attribute centers, etc.) and are independent of how the heavy data is stored.
- **`src/*_writer.rs`** (`ascii_writer.rs`, `binary_writer.rs`, `hdf5_writer.rs`) — implementations of the `DataWriter` trait (defined in `src/lib.rs`), each responsible for persisting the *heavy* numeric data (points, connectivity, per-timestep attributes) in one storage format and returning a `DataContent` that the light-data XML can reference. `hdf5_writer.rs` is gated behind the `hdf5` feature; the crate must still compile and work (falling back to Ascii/AsciiInline/Binary) with that feature disabled — this is why CI runs everything twice.
- **`src/paraview.rs`** — every limit the crate places on integer *values*, and nothing else. None of them come from XDMF: each is a defect of ParaView's legacy Xdmf2 reader (`UInt` decoded at 32 bits whatever `Precision` says, in every format; the ascii storages read through a `double`; `Format="Binary"` misreading 64-bit integers) that would otherwise silently show a number the file does not contain. `paraview::validate(data, format)` is the single entry point and dispatches on `Format`, which maps exactly onto the three restriction sets. **The `*_writer.rs` backends validate nothing** — `TimeSeriesWriter`/`TimeStep` call `paraview::validate` before handing data over, so a future path that doesn't care about ParaView (writing data only to read it back, restart files) is one call away. Don't add a check to a backend, to `DataWriter`, or to `validate_points_and_cells`; it belongs here, where it can be skipped with the rest. It only ever *rejects*, never rewrites: `Format::Binary` refuses `i64`/`u64` outright (measured — ParaView reads them at the wrong stride, so narrowing them would put a different type in the file than the caller passed), rather than narrowing as earlier versions did. Note the asymmetry that measurement established: 8-byte integers read back correctly from the ascii and HDF5 storages, so there is nothing to reject or narrow there — only `Binary` is affected.
- **`DataStorage`** (`src/lib.rs`) is the public enum selecting which writer backs a given output: `Ascii`, `AsciiInline`, `Hdf5SingleFile`, `Hdf5MultipleFiles`, `Binary`. `create_writer()` maps a `DataStorage` to a boxed `DataWriter`.

### Public entry point

`TimeSeriesWriter` / `TimeSeriesDataWriter` (`src/time_series_writer.rs`) is the recommended high-level API and the one most usage should go through: `TimeSeriesWriter::new(...)` → `write_mesh(...)` (writes points/cells once, returns a `TimeSeriesDataWriter`) → repeated `write_data(...)` calls per time step. It composes the XDMF element structs with a `DataWriter` under the hood. Lower-level usage (constructing `Xdmf`/`Domain`/`Grid` elements directly) is possible but only recommended for special cases — see `tests/xdmf_elements.rs` for that style.

### Values / DataAttribute

`Values` (`src/values.rs`) is a thin enum wrapper (`F64`/`F32`/`I64`/`I32`/`U64`/`U32`) giving a uniform interface over numeric data regardless of type, including format-dependent precision (e.g. binary format uses 32-bit ints due to a ParaView reader bug — see `tests/binary_writer.rs`). The variants and their four `From` impls each (owned `Vec`, `&[T]`, `&Vec<T>`, `&[T; N]`) are generated by the `define_values!` macro from a one-line type list; the methods on `Values` are deliberately *not* generated, so their exhaustive matches — and those in the `*_writer.rs` backends — make the compiler point at every decision a new element type needs. `Values::precision()` is simply each type's own width — nothing anywhere narrows, widens or casts the caller's data, so what lands in the file is the type that was passed in, and a storage that cannot read a type back says so instead (see `src/paraview.rs`). `Coordinate` (`f32`/`f64`) and `ConnectivityIndex` (`u32`/`u64`/`i32`/`i64`) are the two public sealed traits selecting the element type of a mesh's points and connectivity; both are backed by `values::sealed` traits that convert a slice into `Values`, so points and connectivity flow through the same per-type validation and per-backend writing as attribute data (`DataWriter::write_mesh` takes two `&Values`, not a `&[u64]`). The connectivity type is therefore what caps the mesh size: `SealedIndex::MAX_INDEX` is what the type itself can index (checked against the point count in `validate_points_and_cells`, before the connectivity is assembled, so an unindexable mesh is rejected without first allocating it), while the lower cap ParaView puts on `u64` reaches the connectivity as a restriction on its *values*, through `paraview::validate` like any other integer data — deliberately not duplicated as a point-count check, which could not be skipped with the rest. `DataAttribute` (`src/lib.rs`) describes the tensor shape of a data field (`Scalar`, `Vector`, `Tensor`, `Tensor6`, `Matrix(n, m)`, `Generic(size)`) and how it maps to XDMF's `AttributeType`.

### Errors

`Error` / `Result<T>` (`src/error.rs`) is the crate's error type: a flat `thiserror` enum, deliberately kept small (under 10 variants) by grouping failures by *category* rather than giving every distinct failure reason its own variant. `Error::Io`/`Error::Hdf5` (the latter `cfg`-gated on the `hdf5` feature) wrap the underlying `std::io::Error`/`hdf5::Error` with an operation description and (for `Io`) a path — attached via the `error::io_ctx(operation, path)` helper at every fallible filesystem call, never via a bare `?`. `Error::InvalidFileName`/`InvalidConfiguration`/`InvalidMesh`/`InvalidTimeStep`/`InvalidData` each cover several related validation failures (e.g. `InvalidMesh` covers an empty/malformed points array, an out-of-bounds connectivity index, a connectivity/cell-type size mismatch, and re-writing a mesh) via a `reason: String` built with `format!` at the call site, rather than per-failure struct fields — callers match on the variant to react to the category; the exact wording is covered by `src/error.rs`'s own `mod error_messages`, not part of the API contract. `Error::IntegerOutOfRange { value, reason }` stays its own variant since it's the one failure a caller might actually want to catch and react to (e.g. fall back to a different `DataStorage`); it pairs the offending value — widened to `i128` so `u64` and `i64` inputs both print as written — with a `reason` saying which of the three limits was hit, since they differ in whether another storage would help: `Binary` narrows 64-bit integers to 32 bits, the ascii storages cap `i64` at ±2^53 (`ascii_writer.rs`'s `MAX_EXACT_ASCII_INT`, since ParaView parses their integers through a `double`), and the `u64` cap applies to every backend. The rule behind all three is the same: refuse anything ParaView would read back as a different number, rather than write it. `Error::Internal(&'static str)` covers state-machine invariants inside the `*_writer.rs` backends (e.g. `write_data_finalize` called before `write_data_initialize`) that are not reachable through the public `TimeSeriesWriter` API — its `&'static str` payload can still be pattern-matched exactly in tests since string literals are valid patterns. There is deliberately no blanket `From<std::io::Error> for Error`, since that would lose the operation/path context; there is a `From<Error> for std::io::Error` for callers that plumb `io::Error` throughout their own codebase.

### Node/cell ordering

Node ordering follows the [VTK convention](https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf), not XDMF's own historical ordering — this is tested against `vtkio` output in `tests/vtk_comparison.rs`.

## Testing conventions

- Unit tests live inline in `src/*.rs` (`#[cfg(test)] mod tests`); integration/behavioral tests live in `tests/*.rs`.
- `tests/xdmf_elements.rs` shows the low-level element-construction API; `tests/time_series_writer.rs` shows the recommended high-level API — prefer extending the latter style when adding coverage for new writer/storage behavior.
- `tests/vtk_comparison.rs` cross-checks xdmf output against `vtkio`-written VTK files (fixtures under `tests/xdmf_vtk_comparison/`) for correctness and storage-size/write-time comparison.
- `tests/paraview_smoke/` holds smoke-test fixtures for validating output actually opens correctly in ParaView (relevant when changing HDF5 filter pipelines, e.g. compression/shuffle settings — ParaView's bundled HDF5 doesn't support every filter, notably no dynamically-loaded plugins like Blosc/zfp, only core ones like deflate/shuffle/szip).
- `mpi_safe_create_dir_all` (`src/lib.rs`) exists specifically for correctness under concurrent directory creation on slow/clustered filesystems (see its test using 100 threads) — don't simplify it back to a plain `create_dir_all`.

## Code style

- Doc comments (`///`/`//!`): one short sentence per item is the norm (see `xdmf_elements/attribute.rs`, `lib.rs`). State a non-obvious rationale once, on the single most relevant item, rather than repeating it on every related item. Match the density already present in the file being edited.
- Ascii float output goes through `FormatNumber` (`ascii_writer.rs`), which uses `{:e}` — Rust's shortest exponential form, i.e. the fewest digits that parse back to the same value. Don't swap it for a fixed precision: too few digits silently corrupts the low mantissa bits (8 significant digits loses ~1 in 100 `f32`s), and enough digits for the worst case (17 for `f64`) costs ~24% file size on every value. `ascii_writer.rs`'s `float_round_trip` test guards this by parsing formatted output back over random bit patterns — a sweep by repeated multiplication does *not* catch it.
- Floating-point comparisons in tests: use `float_cmp::assert_approx_eq!` (e.g. `assert_approx_eq!(&[f64], &expected, &actual)`), not manual epsilon checks — see `hdf5_writer.rs` tests. `float_cmp` is also a `#[warn]` clippy lint at the crate level, so raw `==` on floats outside tests will fail CI.
- A test that asserts an operation fails must check *which* error occurred, not just `is_err()`/`unwrap_err()` with nothing else. Errors are the typed `xdmf::Error` (`src/error.rs`), which is not `PartialEq` (it wraps `std::io::Error`/`hdf5::Error`), so assert with `std::assert_matches!` on the **variant and its fields** where the variant has discriminating fields, e.g. `std::assert_matches!(writer.write_data("0.0", data, []).unwrap_err(), Error::InvalidTimeStep { time, .. } if time == "0.1")` — see `time_series_writer.rs`. Several variants (`InvalidMesh`/`InvalidConfiguration`/`InvalidData`/...) carry only a `reason: String` instead of per-failure fields, so their call-site tests match the variant and assert on `reason` (`if reason.contains("...")` or an exact `==` when the whole message is known) rather than on structured fields — this is not the same as the message-text `Display` tests in `src/error.rs`'s `mod error_messages`, which exist purely to guard wording per message family (not one per call site) and are the only place a full `.to_string()` comparison belongs.
- Don't put `use` imports inside function bodies — hoist them to the enclosing module/`mod tests` scope, per `.rustfmt.toml`'s `imports_granularity`/`group_imports` settings and existing practice throughout `src/`.
- Don't add speculative public API (methods, variants, constructors) that nothing currently calls, even if it seems generally useful. Add it when a real caller (production code or a test that needs it for another purpose) needs it. If a method exists only to be exercised by its own dedicated test, that's a sign it's speculative.
- This crate is pre-1.0 (`0.1.x`) and still settling its API (e.g. `DataStorage`'s HDF5 variants gained a `deflate_level` field after release) — prefer a breaking change that simplifies over a backward-compat shim; don't add deprecated aliases or dual code paths to preserve old signatures.
- **Do not silence lints to make code compile.** No `#[allow(...)]`/`#![allow(...)]` (clippy, rustc, or rustdoc) and don't relax lint levels in `Cargo.toml`. This is doubly enforced here: `allow_attributes`/`allow_attributes_without_reason` are themselves warn-level clippy lints in `[lints.clippy]`. If a lint fires, fix the underlying issue; if it feels genuinely wrong, ask before suppressing it.
