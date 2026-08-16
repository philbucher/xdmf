//! Typed error type for this crate.

use std::path::{Path, PathBuf};

/// Result alias defaulting to this crate's [`Error`] type.
///
/// The error type is a parameter so that APIs taking a caller-supplied closure can name the
/// caller's own error type; everything else uses the default.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The error type for all fallible operations in this crate.
///
/// Most variants carry a `reason` describing the specific failure in prose rather than as their
/// own variant/fields, so callers wanting to react to a *category* of failure (a bad mesh, a bad
/// time step, ...) can match on the variant, while the exact wording is covered by this crate's
/// own message-family tests rather than being part of the API contract.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A filesystem operation failed.
    #[error("{operation} failed for {path}: {source}", path = path.display())]
    Io {
        /// Short description of what was being attempted, e.g. "creating data file".
        operation: &'static str,
        /// The path the operation was performed on.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// An HDF5 library call failed.
    #[cfg(feature = "hdf5")]
    #[error("HDF5 error while {operation}: {source}")]
    Hdf5 {
        /// Short description of what was being attempted, e.g. "creating mesh group".
        operation: &'static str,
        /// The underlying HDF5 error.
        #[source]
        source: hdf5::Error,
    },
    /// An output path is missing a final component, or that component is not usable as a file
    /// name. Parent directories are not validated, they may legitimately contain characters that
    /// are rejected in the file name itself.
    #[error("invalid file name '{path}': {reason}", path = path.display())]
    InvalidFileName {
        /// The offending path, as passed by the caller.
        path: PathBuf,
        /// What is wrong with it.
        reason: String,
    },
    /// A writer construction option (e.g. `deflate_level`, `DataStorage`) is invalid or requires
    /// a Cargo feature that is not enabled.
    #[error("invalid configuration: {reason}")]
    InvalidConfiguration {
        /// What is wrong with the configuration.
        reason: String,
    },
    /// The points/connectivity/cell types given to `write_mesh` are inconsistent, or a mesh was
    /// already written.
    #[error("invalid mesh: {reason}")]
    InvalidMesh {
        /// What is wrong with the mesh.
        reason: String,
    },
    /// A time step string is not a valid finite float, was already written, or its step was left
    /// without any data.
    #[error("invalid time step '{time}': {reason}")]
    InvalidTimeStep {
        /// The offending time step string.
        time: String,
        /// What is wrong with it.
        reason: String,
    },
    /// A data field written into a time step is invalid (wrong size, bad name, or duplicate name).
    #[error("invalid data: {reason}")]
    InvalidData {
        /// What is wrong with the data.
        reason: String,
    },
    /// An integer value is outside the range that can be written and read back correctly.
    ///
    /// The `reason` says which limit was hit, and with it whether another [`DataStorage`] would
    /// accept the value: the `Binary` backend's 32-bit narrowing is storage-specific, while the
    /// cap on `u64` data applies to every backend.
    ///
    /// [`DataStorage`]: crate::DataStorage
    #[error("integer value {value} is out of range: {reason}")]
    IntegerOutOfRange {
        /// The out-of-range value, widened to `i128` so that both `u64` and `i64` inputs are
        /// reported as written.
        value: i128,
        /// Which limit was hit, and whether a different `DataStorage` would avoid it.
        reason: String,
    },
    /// An internal invariant was violated. Not reachable through the public API; guards against
    /// a future regression in the state-machine pairing between a backend's
    /// `write_data_initialize`/`write_data_finalize` calls, or in this crate's own path handling.
    #[error("internal invariant violated: {0}")]
    Internal(&'static str),
}

/// Attach filesystem-operation context to a [`std::io::Error`], for use with `map_err`.
///
/// A bare `?` on a filesystem call loses which path and which operation failed; every fallible
/// filesystem call in this crate should be routed through this helper instead.
pub(crate) fn io_ctx<'a>(
    operation: &'static str,
    path: &'a Path,
) -> impl FnOnce(std::io::Error) -> Error + 'a {
    move |source| Error::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

/// Converts to a `std::io::Error` for consumers that plumb `io::Error` throughout their own
/// codebase. `Error::Io`'s original [`std::io::ErrorKind`] is preserved; every other variant
/// (a validation failure, not a filesystem failure) becomes [`std::io::ErrorKind::InvalidInput`].
///
/// The [`Error`] is kept as the payload rather than flattened into a string, so the original
/// cause (and with it e.g. `raw_os_error`) stays reachable via [`std::io::Error::get_ref`].
impl From<Error> for std::io::Error {
    fn from(err: Error) -> Self {
        let kind = match &err {
            Error::Io { source, .. } => source.kind(),
            _ => std::io::ErrorKind::InvalidInput,
        };
        Self::new(kind, err)
    }
}

#[cfg(test)]
mod error_messages {
    use super::*;

    #[test]
    fn io() {
        let err = Error::Io {
            operation: "creating data file",
            path: PathBuf::from("/tmp/out/data.txt"),
            source: std::io::Error::other("No such file or directory"),
        };
        assert_eq!(
            err.to_string(),
            "creating data file failed for /tmp/out/data.txt: No such file or directory"
        );
    }

    #[test]
    fn invalid_file_name() {
        assert_eq!(
            Error::InvalidFileName {
                path: PathBuf::from("a:b"),
                reason: "file name component must not contain any of the following characters"
                    .to_string(),
            }
            .to_string(),
            "invalid file name 'a:b': file name component must not contain any of the following \
             characters"
        );
    }

    #[test]
    fn invalid_configuration() {
        assert_eq!(
            Error::InvalidConfiguration {
                reason: "deflate level 10 is out of range, must be between 0 and 9".to_string(),
            }
            .to_string(),
            "invalid configuration: deflate level 10 is out of range, must be between 0 and 9"
        );
        assert_eq!(
            Error::InvalidConfiguration {
                reason: "using Hdf5SingleFile { deflate_level: None } DataStorage requires the 'hdf5' feature".to_string(),
            }
            .to_string(),
            "invalid configuration: using Hdf5SingleFile { deflate_level: None } DataStorage requires the 'hdf5' feature"
        );
    }

    #[test]
    fn invalid_mesh() {
        assert_eq!(
            Error::InvalidMesh {
                reason: "at least one point is required".to_string(),
            }
            .to_string(),
            "invalid mesh: at least one point is required"
        );
    }

    #[test]
    fn invalid_time_step() {
        assert_eq!(
            Error::InvalidTimeStep {
                time: "not_a_float".to_string(),
                reason: "must be a valid float".to_string(),
            }
            .to_string(),
            "invalid time step 'not_a_float': must be a valid float"
        );
        assert_eq!(
            Error::InvalidTimeStep {
                time: "0.10".to_string(),
                reason: "already written (as '0.1')".to_string(),
            }
            .to_string(),
            "invalid time step '0.10': already written (as '0.1')"
        );
    }

    #[test]
    fn invalid_data() {
        assert_eq!(
            Error::InvalidData {
                reason: "size of point_data 'temperature' must be 10, but is 9".to_string(),
            }
            .to_string(),
            "invalid data: size of point_data 'temperature' must be 10, but is 9"
        );
    }

    #[test]
    fn integer_out_of_range() {
        assert_eq!(
            Error::IntegerOutOfRange {
                value: -2_147_483_649,
                reason: "Binary storage narrows 64-bit integers to 32 bits".to_string(),
            }
            .to_string(),
            "integer value -2147483649 is out of range: Binary storage narrows 64-bit integers \
             to 32 bits"
        );
        assert_eq!(
            Error::IntegerOutOfRange {
                value: 4_294_967_296,
                reason: "u64 data must fit in 32 bits".to_string(),
            }
            .to_string(),
            "integer value 4294967296 is out of range: u64 data must fit in 32 bits"
        );
    }

    #[test]
    fn internal() {
        assert_eq!(
            Error::Internal("writing data was not initialized").to_string(),
            "internal invariant violated: writing data was not initialized"
        );
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn hdf5() {
        let err = Error::Hdf5 {
            operation: "creating group",
            source: hdf5::Error::from("boom".to_string()),
        };
        assert_eq!(err.to_string(), "HDF5 error while creating group: boom");
    }

    #[test]
    fn from_error_for_io_error_preserves_io_kind() {
        let err = Error::Io {
            operation: "creating data file",
            path: PathBuf::from("/tmp/out/data.txt"),
            source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        };
        let io_err: std::io::Error = err.into();
        assert_eq!(io_err.kind(), std::io::ErrorKind::PermissionDenied);

        // the original error is kept as the payload, not flattened into a string
        std::assert_matches!(
            io_err.get_ref().and_then(|e| e.downcast_ref::<Error>()),
            Some(Error::Io { operation, .. }) if *operation == "creating data file"
        );
    }

    #[test]
    fn from_error_for_io_error_defaults_to_invalid_input() {
        let io_err: std::io::Error = Error::InvalidMesh {
            reason: "at least one point is required".to_string(),
        }
        .into();
        assert_eq!(io_err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
