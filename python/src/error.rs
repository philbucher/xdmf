//! Conversion from the core crate's [`xdmf::Error`] into a Python exception, matched per variant.

use pyo3::{
    PyErr,
    exceptions::{
        PyNotImplementedError, PyOSError, PyOverflowError, PyRuntimeError, PyTypeError,
        PyValueError,
    },
};
use xdmf::Error;

/// Maps every [`xdmf::Error`] variant to the Python exception a caller would expect to catch:
/// validation failures the caller can fix (`ValueError`, which a document that does not describe a
/// readable mesh is one of), an integer no storage format can carry back (`OverflowError` --
/// deliberately not a `ValueError`, since this is the one failure worth reacting to specifically,
/// e.g. by falling back to another `DataStorage`), a construct this crate's reader does not
/// support (`NotImplementedError`), a read call asking for a type the file cannot be read as
/// without losing precision (`TypeError`), an internal invariant (`RuntimeError`, not reachable
/// through this binding's API surface but mapped rather than left to panic), and filesystem/HDF5
/// failures (`OSError`, which is what Python's `IOError` names too).
pub(crate) fn to_py_err(error: Error) -> PyErr {
    match error {
        Error::InvalidFileName { .. }
        | Error::InvalidConfiguration { .. }
        | Error::InvalidMesh { .. }
        | Error::InvalidTimeStep { .. }
        | Error::InvalidData { .. }
        | Error::InvalidDocument { .. } => PyValueError::new_err(error.to_string()),
        Error::IntegerOutOfRange { .. } => PyOverflowError::new_err(error.to_string()),
        Error::Unsupported { .. } => PyNotImplementedError::new_err(error.to_string()),
        Error::NumberTypeMismatch { .. } => PyTypeError::new_err(error.to_string()),
        Error::Internal(_) => PyRuntimeError::new_err(error.to_string()),
        Error::Io { .. } => PyOSError::new_err(error.to_string()),
        #[cfg(feature = "hdf5")]
        Error::Hdf5 { .. } => PyOSError::new_err(error.to_string()),
    }
}
