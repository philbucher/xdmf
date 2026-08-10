# M1 — Dedicated error type

Resolves item 5 of `API_IMPROVEMENTS_PLAN.md`: **yes, do it**, and do it before the reader.

## Why now

Every fallible operation in the crate returns `std::io::Error`, almost always
`ErrorKind::InvalidInput` with a formatted string. A caller cannot tell "your array is the wrong
size" from "duplicate data name" from "the disk is full" without substring-matching the message —
which is exactly what the test suite does today.

The reader (M5) adds a whole second family of failures (parse errors, missing heavy-data files,
unsupported XDMF constructs, type mismatches). Those are failures a caller genuinely wants to branch
on: "unsupported feature, fall back to another loader" is a reasonable recovery, "the file does not
exist" is not the same thing, and neither is "the connectivity dimension in the XML disagrees with
the HDF5 dataset". Shipping the reader on `io::Error` means designing that surface twice.

Current scale of the migration: **29** `unwrap_err().to_string()` assertions and **49** `IoResult<..>`
signatures. All of it mechanical.

## Design

### Shape

A flat, `#[non_exhaustive]` enum. Flat rather than nested-by-category because the crate has a
bounded, enumerable set of failure modes and two levels of matching buys nothing. `#[non_exhaustive]`
so M5 can add reader variants without another breaking release.

```rust
pub type Result<T> = std::result::Result<T, Error>;

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // --- I/O -------------------------------------------------------------
    #[error("{operation} failed for {path}: {source}")]
    Io { operation: &'static str, path: PathBuf, #[source] source: std::io::Error },

    // --- writer construction ---------------------------------------------
    #[error("File name '{name}' cannot contain the following characters: {INVALID_FILE_NAME_CHARS:?}")]
    InvalidFileNameChars { name: String },
    #[error("File name must not be empty")]
    EmptyFileName,
    #[error("File name must be valid UTF-8")]
    NonUtf8FileName,
    #[error("deflate level {level} is out of range, must be between 0 and 9")]
    DeflateLevelOutOfRange { level: u8 },
    #[error("Using {storage:?} DataStorage requires the '{feature}' feature")]
    StorageRequiresFeature { storage: DataStorage, feature: &'static str },

    // --- mesh -------------------------------------------------------------
    #[error("At least one point is required")]
    NoPoints,
    #[error("Points must have 3 dimensions, but {len} is not a multiple of 3")]
    PointsNotThreeDimensional { len: usize },
    #[error("Connectivity index {index} is out of bounds, the mesh only has {num_points} points")]
    ConnectivityIndexOutOfBounds { index: u64, num_points: usize },
    #[error("Size of connectivity ({actual}) does not match the number expected from the cell types ({expected})")]
    ConnectivitySizeMismatch { actual: usize, expected: usize },
    #[error("Mesh was already written")]
    MeshAlreadyWritten,

    // --- time steps -------------------------------------------------------
    #[error("Time must be a valid float, and not '{time}'")]
    InvalidTime { time: String },
    #[error("Time step '{time}' has already been written (as '{existing}')")]
    DuplicateTime { time: String, existing: String },
    #[error("At least one of point_data or cell_data must be provided")]
    NoData,

    // --- attributes -------------------------------------------------------
    #[error("Size of {center}-data '{name}' must be {expected}, but is {actual}")]
    DataSizeMismatch { center: DataCenter, name: String, expected: usize, actual: usize },
    #[error("Data name '{name}' of {center}-data is not valid, must be non-empty and contain only alphanumeric characters, underscores or dashes")]
    InvalidDataName { center: DataCenter, name: String },
    #[error("Name '{name}' of {center}-data is used more than once")]
    DuplicateDataName { center: DataCenter, name: String },

    // --- storage-specific -------------------------------------------------
    #[error("value {value} does not fit in 32 bits: uncompressed Binary output only supports integer data up to u32 (ParaView's legacy Xdmf2 reader misreads 64-bit integers)")]
    IntegerTooLargeForBinary { value: u64 },

    // --- hdf5 -------------------------------------------------------------
    #[cfg(feature = "hdf5")]
    #[error("HDF5 error while {operation}: {source}")]
    Hdf5 { operation: String, #[source] source: hdf5::Error },
}
```

M5 appends the reader variants (`Xml`, `MissingElement`, `MissingAttribute`, `Unsupported`,
`HeavyDataNotFound`, `HeavyDataSizeMismatch`, `NumberTypeMismatch`) — listed in `05_reader.md`, not
here, so this milestone stays reviewable.

### Supporting decisions

**`DataCenter`.** The messages today use a `label: &str` that is either `"point"` or `"cell"`. As an
error field that should be a type, not a string. `xdmf_elements::attribute::Center` already exists
but spells it `Node`/`Cell` (XDMF's vocabulary), while every user-facing message says "point". Add a
small public `enum DataCenter { Point, Cell }` with a `Display` impl, and convert to
`attribute::Center` internally. It is one extra public type, but it appears in three error variants
and in the reader's `DataInfo`, so it is not speculative.

**`thiserror`.** Use it (version 2). ~20 variants means ~80 lines of hand-written `Display`/`Error`
boilerplate that has to stay in sync with the variants; `thiserror` keeps the message next to the
variant, which is where it belongs. It adds no runtime cost and no *new* transitive build cost —
`serde_derive` and `quick-xml`'s serialize feature already pull `syn`/`quote`/`proc-macro2` into the
graph. If a zero-proc-macro dependency graph is a goal in itself, write the impls by hand; nothing
else in the design changes.

**`From<Error> for std::io::Error`.** Provide it. arotau's `.map_err(ArotauError::output)` takes
anything `Display`, so it does not strictly need this, but the conversion is polite for any consumer
already plumbing `io::Error` and it costs five lines. Map validation-ish variants to
`ErrorKind::InvalidInput` and `Error::Io { source, .. }` back to its own kind.

**`From<std::io::Error>` — do *not* provide a blanket impl.** A bare `?` on a filesystem call would
then produce an `Error::Io` with no path and no operation, which is exactly the context that makes
these errors useful. Force call sites through a helper:

```rust
fn io_ctx(operation: &'static str, path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_
```

so the call reads `File::create(&p).map_err(io_ctx("creating data file", &p))?`. This is the single
biggest quality win in the milestone: today a failed write in the HDF5 multi-file backend surfaces as
a bare "No such file or directory" with no indication of which of the hundreds of files it was.

**HDF5 errors.** Currently laundered through `IoError::other(..)`, losing the type. Keep the real
`hdf5::Error` as `#[source]` and attach an operation string. Note the variant is `cfg`-gated, so the
`--no-default-features` build must still compile — that is what `StorageRequiresFeature` is for.

**`mpi_safe_create_dir_all`** is public and returns `IoResult<()>`. Convert it to `Result<()>` for
consistency; it already formats a `Failed to create directory {path:?}` message that becomes
`Error::Io { operation: "creating directory", path, source }`.

## Test convention change

`CLAUDE.md` currently mandates:

> Since errors here are `std::io::Error` (no `PartialEq`), the established pattern is
> `assert_eq!(res.unwrap_err().to_string(), "expected message")`

That stops being the right advice. New convention, to be written into `CLAUDE.md` as part of this
milestone:

- Assert on the **variant and its fields**:
  ```rust
  assert!(matches!(
      writer.write_data("0.0", data, []).unwrap_err(),
      Error::DataSizeMismatch { expected: 10, actual: 9, .. }
  ));
  ```
- Keep a **small number of `Display` tests** — one per message family, not one per call site — so
  message quality stays guarded without re-coupling every test to exact wording. Put them together in
  one `mod error_messages` in `src/error.rs`, where a reviewer can read all the user-facing strings
  at once.

The 29 existing assertions convert to `matches!` mechanically; roughly a dozen of them are the *only*
assertion in their test, and those are the ones that actually get better, since they currently pass
for the wrong reason whenever a message is reworded.

## Work breakdown

1. `src/error.rs`: the enum, `Result` alias, `DataCenter`, `io_ctx` helper, `From<Error> for io::Error`,
   the `mod error_messages` tests. Re-export `Error` and `Result` from `lib.rs`.
2. Convert `src/lib.rs`, `src/time_series_writer.rs`, `src/values.rs` — 49 signatures, mostly
   `IoResult<T>` → `Result<T>`.
3. Convert the three writers. This is where `io_ctx` gets applied to every `File::create` /
   `write_all` / `flush`, and where the HDF5 `map_err(IoError::other)` calls become
   `Error::Hdf5 { operation, source }`.
4. Convert the tests (`src/**/tests`, `tests/*.rs`) to `matches!`.
5. `cargo clippy --all-targets` and `--no-default-features` both clean; `cargo test --doc` (doctests
   using `.expect(..)` are unaffected).
6. Update `CLAUDE.md`'s "Testing conventions" section.

**Decided (2026-08-10): no `CHANGELOG.md`.** Dropped from the work breakdown at the user's
request; see the note in `ROADMAP.md`'s release strategy section.

## Open questions

- **Should `Error` be `PartialEq`?** It cannot be, because `io::Error` and `hdf5::Error` are not. This
  is why the test convention is `matches!` rather than `assert_eq!`. Deriving `PartialEq` on a
  boxed-source-free subset was considered and rejected as not worth splitting the enum for.
- **`Error` size.** With `PathBuf` and `String` fields the enum will be ~50-60 bytes, so
  `Result<T, Error>` gets fat, and it is returned from `write_data` which is in the hot path. If the
  benchmark from M2 shows this on the profile, box the payload (`Error(Box<ErrorKind>)`) — a standard
  fix, but do not do it preemptively. Worth an explicit measurement in M2 since M1 lands first.
