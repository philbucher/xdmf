//! Conversion from the core crate's [`xdmf::Error`] into a Python exception, matched per variant.

use pyo3::{
    PyErr,
    exceptions::{PyOSError, PyOverflowError, PyRuntimeError, PyValueError},
};
use xdmf::Error;

/// Maps every [`xdmf::Error`] variant to the Python exception a caller would expect to catch:
/// validation failures the caller can fix (`ValueError`), an integer no storage format can carry
/// back (`OverflowError` -- deliberately not a `ValueError`, since this is the one failure worth
/// reacting to specifically, e.g. by falling back to another `DataStorage`), an internal invariant
/// (`RuntimeError`, not reachable through this binding's API surface but mapped rather than left to
/// panic), and filesystem/HDF5 failures (`OSError`, which is what Python's `IOError` names too).
pub(crate) fn to_py_err(error: Error) -> PyErr {
    match error {
        Error::InvalidFileName { .. }
        | Error::InvalidConfiguration { .. }
        | Error::InvalidMesh { .. }
        | Error::InvalidTimeStep { .. }
        | Error::InvalidData { .. } => PyValueError::new_err(error.to_string()),
        Error::IntegerOutOfRange { .. } => PyOverflowError::new_err(error.to_string()),
        Error::Internal(_) => PyRuntimeError::new_err(error.to_string()),
        Error::Io { .. } | Error::Hdf5 { .. } => PyOSError::new_err(error.to_string()),
    }
}
