//! Typed error type for this crate.

use std::path::{Path, PathBuf};

use crate::{DataStorage, xdmf_elements::attribute};

/// Result alias using this crate's [`Error`] type.
pub type Result<T> = std::result::Result<T, Error>;

/// Characters not allowed in the final path component of an XDMF file name.
pub(crate) const INVALID_FILE_NAME_CHARS: [char; 8] = ['?', '\0', ':', '*', '"', '<', '>', '|'];

/// The error type for all fallible operations in this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // --- I/O -----------------------------------------------------------
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

    // --- writer construction --------------------------------------------
    /// A file name contains a character that is not allowed.
    #[error(
        "File name '{name}' cannot contain the following characters: {INVALID_FILE_NAME_CHARS:?}"
    )]
    InvalidFileNameChars {
        /// The offending file name.
        name: String,
    },
    /// A file name was empty.
    #[error("File name must not be empty")]
    EmptyFileName,
    /// A file name was not valid UTF-8.
    #[error("File name must be valid UTF-8")]
    NonUtf8FileName,
    /// A `deflate_level` was outside the 0-9 range accepted by zlib.
    #[error("deflate level {level} is out of range, must be between 0 and 9")]
    DeflateLevelOutOfRange {
        /// The rejected level.
        level: u8,
    },
    /// The chosen [`DataStorage`] requires a Cargo feature that is not enabled.
    #[error("Using {storage:?} DataStorage requires the '{feature}' feature")]
    StorageRequiresFeature {
        /// The storage variant that was requested.
        storage: DataStorage,
        /// The feature that must be enabled to use it.
        feature: &'static str,
    },

    // --- mesh -------------------------------------------------------------
    /// No points were given when writing a mesh.
    #[error("At least one point is required")]
    NoPoints,
    /// The points array length was not a multiple of 3.
    #[error("Points must have 3 dimensions, but {len} is not a multiple of 3")]
    PointsNotThreeDimensional {
        /// The actual length of the points array.
        len: usize,
    },
    /// A connectivity index referenced a point that does not exist.
    #[error("Connectivity index {index} is out of bounds, the mesh only has {num_points} points")]
    ConnectivityIndexOutOfBounds {
        /// The offending index.
        index: u64,
        /// The number of points in the mesh.
        num_points: usize,
    },
    /// The connectivity array length did not match what the cell types require.
    #[error(
        "Size of connectivity ({actual}) does not match the number expected from the cell types ({expected})"
    )]
    ConnectivitySizeMismatch {
        /// The actual length of the connectivity array.
        actual: usize,
        /// The length expected from the given cell types.
        expected: usize,
    },
    /// `write_mesh` was called more than once on the same writer.
    #[error("Mesh was already written")]
    MeshAlreadyWritten,
    /// Internal invariant: a backend's `write_data`/`write_data_finalize` was called before its
    /// `write_data_initialize`. Not reachable through the public `TimeSeriesDataWriter` API,
    /// which always pairs the two; guards against a future regression in that pairing.
    #[error("Writing data was not initialized")]
    DataWriteNotInitialized,
    /// Internal invariant: a backend's `write_data_initialize` was called again before the
    /// previous call was finalized. See [`Error::DataWriteNotInitialized`].
    #[error("Writing data was already initialized")]
    DataWriteAlreadyInitialized,
    /// Internal invariant: an HDF5 file path constructed by this crate had no resolvable parent
    /// directory and file name.
    #[error("Could not get parent and file name")]
    Hdf5PathResolution,
    /// Internal invariant: a backend derived its data-file name from an output path with no
    /// final path component. Not reachable through the public `TimeSeriesWriter` API, which
    /// already rejects such paths via [`Error::NonUtf8FileName`]/[`Error::EmptyFileName`] before
    /// any backend is constructed.
    #[error("Input file name must have a valid file name")]
    MissingFileNameComponent,

    // --- time steps ---------------------------------------------------------
    /// A time step string could not be parsed as a float.
    #[error("Time must be a valid float, and not '{time}'")]
    InvalidTime {
        /// The unparsable string.
        time: String,
    },
    /// The same time step (by parsed value) was written more than once.
    #[error("Time step '{time}' has already been written (as '{existing}')")]
    DuplicateTime {
        /// The spelling passed this time.
        time: String,
        /// The spelling first used for this time value.
        existing: String,
    },
    /// `write_data` was called with neither point data nor cell data.
    #[error("At least one of point_data or cell_data must be provided")]
    NoData,

    // --- attributes -----------------------------------------------------------
    /// A data field's length did not match what the mesh and `DataAttribute` require.
    #[error("Size of {center}-data '{name}' must be {expected}, but is {actual}")]
    DataSizeMismatch {
        /// Whether the field is point- or cell-data.
        center: DataCenter,
        /// The field's name.
        name: String,
        /// The length required by the mesh size and `DataAttribute`.
        expected: usize,
        /// The actual length given.
        actual: usize,
    },
    /// A data field's name contains characters that are not allowed.
    #[error(
        "Data name '{name}' of {center}-data is not valid, must be non-empty and contain only alphanumeric characters, underscores or dashes"
    )]
    InvalidDataName {
        /// Whether the field is point- or cell-data.
        center: DataCenter,
        /// The offending name.
        name: String,
    },
    /// The same data field name was used more than once in a single `write_data` call.
    #[error("Name '{name}' of {center}-data is used more than once")]
    DuplicateDataName {
        /// Whether the field is point- or cell-data.
        center: DataCenter,
        /// The duplicated name.
        name: String,
    },

    // --- storage-specific -------------------------------------------------
    /// A value does not fit the numeric range the `Binary` backend can represent.
    #[error(
        "value {value} does not fit in 32 bits: uncompressed Binary output only supports integer data up to u32 (ParaView's legacy Xdmf2 reader misreads 64-bit integers)"
    )]
    IntegerTooLargeForBinary {
        /// The out-of-range value.
        value: u64,
    },

    // --- hdf5 ---------------------------------------------------------------
    /// An HDF5 library call failed.
    #[cfg(feature = "hdf5")]
    #[error("HDF5 error while {operation}: {source}")]
    Hdf5 {
        /// Short description of what was being attempted.
        operation: String,
        /// The underlying HDF5 error.
        #[source]
        source: hdf5::Error,
    },
}

/// Whether a data field is associated with mesh points or mesh cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataCenter {
    /// Data associated with mesh points (nodes).
    Point,
    /// Data associated with mesh cells.
    Cell,
}

impl std::fmt::Display for DataCenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Point => write!(f, "point"),
            Self::Cell => write!(f, "cell"),
        }
    }
}

impl From<DataCenter> for attribute::Center {
    fn from(center: DataCenter) -> Self {
        match center {
            DataCenter::Point => Self::Node,
            DataCenter::Cell => Self::Cell,
        }
    }
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
impl From<Error> for std::io::Error {
    fn from(err: Error) -> Self {
        let kind = match &err {
            Error::Io { source, .. } => source.kind(),
            _ => std::io::ErrorKind::InvalidInput,
        };
        Self::new(kind, err.to_string())
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
    fn writer_construction() {
        assert_eq!(
            Error::InvalidFileNameChars {
                name: "a:b".to_string()
            }
            .to_string(),
            "File name 'a:b' cannot contain the following characters: ['?', '\\0', ':', '*', '\"', '<', '>', '|']"
        );
        assert_eq!(
            Error::EmptyFileName.to_string(),
            "File name must not be empty"
        );
        assert_eq!(
            Error::DeflateLevelOutOfRange { level: 10 }.to_string(),
            "deflate level 10 is out of range, must be between 0 and 9"
        );
        assert_eq!(
            Error::StorageRequiresFeature {
                storage: DataStorage::Hdf5SingleFile {
                    deflate_level: None
                },
                feature: "hdf5",
            }
            .to_string(),
            "Using Hdf5SingleFile { deflate_level: None } DataStorage requires the 'hdf5' feature"
        );
    }

    #[test]
    fn mesh() {
        assert_eq!(
            Error::NoPoints.to_string(),
            "At least one point is required"
        );
        assert_eq!(
            Error::PointsNotThreeDimensional { len: 22 }.to_string(),
            "Points must have 3 dimensions, but 22 is not a multiple of 3"
        );
        assert_eq!(
            Error::ConnectivityIndexOutOfBounds {
                index: 70,
                num_points: 11
            }
            .to_string(),
            "Connectivity index 70 is out of bounds, the mesh only has 11 points"
        );
        assert_eq!(
            Error::ConnectivitySizeMismatch {
                actual: 8,
                expected: 10
            }
            .to_string(),
            "Size of connectivity (8) does not match the number expected from the cell types (10)"
        );
        assert_eq!(
            Error::MeshAlreadyWritten.to_string(),
            "Mesh was already written"
        );
        assert_eq!(
            Error::DataWriteNotInitialized.to_string(),
            "Writing data was not initialized"
        );
        assert_eq!(
            Error::DataWriteAlreadyInitialized.to_string(),
            "Writing data was already initialized"
        );
        assert_eq!(
            Error::Hdf5PathResolution.to_string(),
            "Could not get parent and file name"
        );
        assert_eq!(
            Error::MissingFileNameComponent.to_string(),
            "Input file name must have a valid file name"
        );
    }

    #[test]
    fn time_steps() {
        assert_eq!(
            Error::InvalidTime {
                time: "not_a_float".to_string()
            }
            .to_string(),
            "Time must be a valid float, and not 'not_a_float'"
        );
        assert_eq!(
            Error::DuplicateTime {
                time: "0.10".to_string(),
                existing: "0.1".to_string()
            }
            .to_string(),
            "Time step '0.10' has already been written (as '0.1')"
        );
        assert_eq!(
            Error::NoData.to_string(),
            "At least one of point_data or cell_data must be provided"
        );
    }

    #[test]
    fn attributes() {
        assert_eq!(
            Error::DataSizeMismatch {
                center: DataCenter::Point,
                name: "temperature".to_string(),
                expected: 10,
                actual: 9,
            }
            .to_string(),
            "Size of point-data 'temperature' must be 10, but is 9"
        );
        assert_eq!(
            Error::InvalidDataName {
                center: DataCenter::Cell,
                name: "bad name".to_string(),
            }
            .to_string(),
            "Data name 'bad name' of cell-data is not valid, must be non-empty and contain only alphanumeric characters, underscores or dashes"
        );
        assert_eq!(
            Error::DuplicateDataName {
                center: DataCenter::Point,
                name: "duplicate".to_string(),
            }
            .to_string(),
            "Name 'duplicate' of point-data is used more than once"
        );
    }

    #[test]
    fn integer_too_large_for_binary() {
        assert_eq!(
            Error::IntegerTooLargeForBinary {
                value: 4_294_967_296
            }
            .to_string(),
            "value 4294967296 does not fit in 32 bits: uncompressed Binary output only supports \
             integer data up to u32 (ParaView's legacy Xdmf2 reader misreads 64-bit integers)"
        );
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn hdf5() {
        let err = Error::Hdf5 {
            operation: "creating group".to_string(),
            source: hdf5::Error::from("boom".to_string()),
        };
        assert_eq!(err.to_string(), "HDF5 error while creating group: boom");
    }

    #[test]
    fn data_center_display() {
        assert_eq!(DataCenter::Point.to_string(), "point");
        assert_eq!(DataCenter::Cell.to_string(), "cell");
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
    }

    #[test]
    fn from_error_for_io_error_defaults_to_invalid_input() {
        let io_err: std::io::Error = Error::NoPoints.into();
        assert_eq!(io_err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
