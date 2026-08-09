# Changelog

All notable changes to this crate are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **Breaking:** replaced `std::io::Result<T>` / `std::io::Error` across the public API with a
  dedicated `xdmf::Error` enum and `xdmf::Result<T>` alias (`src/error.rs`). Every fallible
  function in the crate (`TimeSeriesWriter`/`TimeSeriesDataWriter` methods,
  `mpi_safe_create_dir_all`) now returns `xdmf::Result<T>`. `Error` is a flat, `thiserror`-based
  enum grouped by failure category (`Io`, `Hdf5` (`hdf5` feature only), `InvalidFileName`,
  `InvalidConfiguration`, `InvalidMesh`, `InvalidTimeStep`, `InvalidData`,
  `IntegerTooLargeForBinary`, `Internal`), so callers can match on the category instead of
  substring-matching an error message. Most variants carry a `reason: String` describing the
  specific failure in prose rather than their own dedicated variant/fields.

  **Migration for consumers matching on old messages:** replace
  `err.to_string() == "..."` / substring checks with `matches!(err, xdmf::Error::SomeVariant { .. })`.
  **Migration for consumers plumbing `std::io::Error` through their own error type:** either
  switch to `xdmf::Error` directly, or keep using `io::Error` via the provided
  `From<xdmf::Error> for std::io::Error` (note this is one-directional — there is deliberately no
  `From<std::io::Error> for xdmf::Error`, since that would lose the operation/path context every
  `Error::Io` now carries).
