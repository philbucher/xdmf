//! Conversion from the core crate's [`xdmf::Error`] into a Python exception, matched per variant
//! rather than by a shared `io::Error` `ErrorKind`

use pyo3::{
    PyErr,
    exceptions::{PyIOError, PyNotImplementedError, PyOSError, PyRuntimeError, PyValueError},
};
use xdmf::Error;

/// Maps every [`xdmf::Error`] variant to the Python exception a caller would expect to catch:
/// validation failures the caller can fix (`ValueError`), a feature this crate does not implement
/// (`NotImplementedError`), an internal invariant (`RuntimeError`, never reachable through this
/// binding's own API surface, but mapped rather than left to panic), and actual filesystem/HDF5
/// I/O failures (`OSError`).
pub(crate) fn to_py_err(err: Error) -> PyErr {
    match err {
        Error::InvalidFileName { .. }
        | Error::InvalidConfiguration { .. }
        | Error::InvalidMesh { .. }
        | Error::InvalidTimeStep { .. }
        | Error::InvalidData { .. }
        | Error::IntegerTooLargeForBinary { .. }
        | Error::InvalidFile { .. } => PyValueError::new_err(err.to_string()),
        Error::Unsupported { .. } => PyNotImplementedError::new_err(err.to_string()),
        Error::Internal(_) => PyRuntimeError::new_err(err.to_string()),
        Error::Io { .. } => PyOSError::new_err(err.to_string()),
        Error::Hdf5 { .. } => PyIOError::new_err(err.to_string()),
    }
}
