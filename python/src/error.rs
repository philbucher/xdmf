//! Conversion from the core crate's `std::io::Error` (used for all fallible operations) into a
//! Python exception.

use std::io::{Error as IoError, ErrorKind};

use pyo3::{
    PyErr,
    exceptions::{PyIOError, PyValueError},
};

/// `InvalidInput` (used throughout `xdmf` for validation failures, e.g. mismatched sizes,
/// invalid names, out-of-bounds indices) maps to `ValueError`; anything else (actual filesystem
/// I/O failures) maps to `OSError`/`IOError`.
pub(crate) fn to_py_err(err: IoError) -> PyErr {
    match err.kind() {
        ErrorKind::InvalidInput => PyValueError::new_err(err.to_string()),
        _ => PyIOError::new_err(err.to_string()),
    }
}
