//! Conversion from the core crate's [`xdmf::Error`] into a Python exception, matched per variant.

use pyo3::{
    PyErr,
    exceptions::{
        PyNotImplementedError, PyOSError, PyOverflowError, PyRuntimeError, PyTypeError,
        PyValueError,
    },
};
use xdmf::{Error, ErrorKind};

/// Maps every [`xdmf::ErrorKind`] to the Python exception a caller would expect to catch:
/// validation failures the caller can fix, including an unreadable document, to `ValueError`; an
/// integer no storage format can carry back to `OverflowError`, kept apart from `ValueError` since
/// a caller may react by falling back to another `DataStorage`; an unsupported reader construct to
/// `NotImplementedError`; a read into a type that would lose precision to `TypeError`; an internal
/// invariant to `RuntimeError`; and filesystem/HDF5 failures to `OSError`.
pub(crate) fn to_py_err(error: Error) -> PyErr {
    match error.kind() {
        ErrorKind::InvalidFileName
        | ErrorKind::InvalidConfiguration
        | ErrorKind::InvalidMesh
        | ErrorKind::InvalidTimeStep
        | ErrorKind::InvalidData
        | ErrorKind::InvalidDocument => PyValueError::new_err(error.to_string()),
        ErrorKind::IntegerOutOfRange => PyOverflowError::new_err(error.to_string()),
        ErrorKind::Unsupported => PyNotImplementedError::new_err(error.to_string()),
        ErrorKind::NumberTypeMismatch => PyTypeError::new_err(error.to_string()),
        ErrorKind::Internal => PyRuntimeError::new_err(error.to_string()),
        ErrorKind::Io | ErrorKind::Hdf5 => PyOSError::new_err(error.to_string()),
    }
}
