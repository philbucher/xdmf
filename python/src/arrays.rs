//! Zero-copy conversion from numpy arrays to the slices `xdmf`'s writer API expects.
//!
//! Points and float attribute data must be a contiguous 1D `float64` array (matches the core
//! crate's flat-layout API exactly). Connectivity and uint attribute data accept either `uint64`
//! (borrowed as-is) or `int64` (also borrowed, after an O(n) sign check and a bit-reinterpret to
//! `u64` -- numpy's default integer dtype is signed `int64`, so requiring `uint64` would force a
//! copy on the most common path). Arrays that aren't C-contiguous are rejected with a clear
//! error rather than silently copied.

use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};

const NOT_CONTIGUOUS: &str =
    "array must be C-contiguous; call `numpy.ascontiguousarray()` on it first";

/// A borrowed 1D `float64` numpy array, checked for contiguity.
pub(crate) struct FloatArray<'py>(PyReadonlyArray1<'py, f64>);

impl<'py> FloatArray<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let arr: PyReadonlyArray1<'py, f64> = obj.extract().map_err(|_| {
            PyValueError::new_err("expected a 1D numpy array with dtype float64")
        })?;
        Ok(Self(arr))
    }

    pub(crate) fn as_slice(&self) -> PyResult<&[f64]> {
        self.0.as_slice().map_err(|_| PyValueError::new_err(NOT_CONTIGUOUS))
    }
}

/// A borrowed 1D integer numpy array (either `uint64` or `int64`), checked for contiguity and,
/// for `int64`, for non-negativity.
pub(crate) enum UintArray<'py> {
    U64(PyReadonlyArray1<'py, u64>),
    I64(PyReadonlyArray1<'py, i64>),
}

impl<'py> UintArray<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'py, u64>>() {
            return Ok(Self::U64(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'py, i64>>() {
            return Ok(Self::I64(arr));
        }
        Err(PyValueError::new_err(
            "expected a 1D numpy array with dtype uint64 or int64",
        ))
    }

    /// Borrows the array as `&[u64]` with no copy. `int64` data is bit-reinterpreted after
    /// verifying every element is non-negative (indices/counts are never negative).
    pub(crate) fn as_u64_slice(&self) -> PyResult<&[u64]> {
        match self {
            Self::U64(arr) => arr.as_slice().map_err(|_| PyValueError::new_err(NOT_CONTIGUOUS)),
            Self::I64(arr) => {
                let signed = arr.as_slice().map_err(|_| PyValueError::new_err(NOT_CONTIGUOUS))?;
                if let Some(&neg) = signed.iter().find(|&&v| v < 0) {
                    return Err(PyValueError::new_err(format!(
                        "value {neg} is negative, but indices/counts must be non-negative"
                    )));
                }
                // SAFETY: `i64` and `u64` have identical size and alignment, and every element
                // was just checked to be non-negative, so reinterpreting the bits as `u64`
                // preserves the numeric value exactly.
                Ok(unsafe {
                    std::slice::from_raw_parts(signed.as_ptr().cast::<u64>(), signed.len())
                })
            }
        }
    }
}

/// Either kind of array `xdmf::Values` can hold, kept alive for the duration of a `write_data`
/// call so the `xdmf::Values` borrowing from it stays valid.
pub(crate) enum ValueGuard<'py> {
    Float(FloatArray<'py>),
    Uint(UintArray<'py>),
}

impl<'py> ValueGuard<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = FloatArray::extract(obj) {
            return Ok(Self::Float(arr));
        }
        UintArray::extract(obj).map(Self::Uint)
    }

    pub(crate) fn to_values(&self) -> PyResult<xdmf::Values<'_>> {
        match self {
            Self::Float(arr) => Ok(xdmf::Values::from(arr.as_slice()?)),
            Self::Uint(arr) => Ok(xdmf::Values::from(arr.as_u64_slice()?)),
        }
    }
}
