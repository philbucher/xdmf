//! Zero-copy conversion from numpy arrays to the `xdmf::Values` this crate's API expects.
//!
//! `NumpyArray` is the single borrowed-array type behind every numeric parameter this crate
//! accepts: `extract` tries every dtype numpy commonly hands us, and `to_values` maps
//! each one straight onto the matching `xdmf::Values` variant without copy. Arrays that aren't
//! C-contiguous are rejected with a clear error rather than silently copied.

use numpy::{Element, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

const NOT_CONTIGUOUS: &str =
    "array must be C-contiguous; call `numpy.ascontiguousarray()` on it first";

/// Describes what `obj` actually is, for error messages that name the real problem (shape vs.
/// dtype vs. "not an array at all") instead of a single generic rejection message.
fn describe(obj: &Bound<'_, PyAny>) -> String {
    match obj.getattr("dtype") {
        Ok(dtype) => format!("a numpy array with dtype {dtype}"),
        Err(_) => obj
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "an unknown type".to_string()),
    }
}

fn contiguous_slice<'a, 'py, T: Element>(arr: &'a PyReadonlyArrayDyn<'py, T>) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(NOT_CONTIGUOUS))
}

/// A borrowed numpy array of one of the six numeric dtypes this crate understands (any
/// dimensionality), kept alive for the duration of a call so the `xdmf::Values` borrowing from it
/// stays valid.
pub(crate) enum NumpyArray<'py> {
    F64(PyReadonlyArrayDyn<'py, f64>),
    F32(PyReadonlyArrayDyn<'py, f32>),
    U64(PyReadonlyArrayDyn<'py, u64>),
    U32(PyReadonlyArrayDyn<'py, u32>),
    I64(PyReadonlyArrayDyn<'py, i64>),
    I32(PyReadonlyArrayDyn<'py, i32>),
}

impl<'py> NumpyArray<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, f64>>() {
            return Ok(Self::F64(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, f32>>() {
            return Ok(Self::F32(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, u64>>() {
            return Ok(Self::U64(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, u32>>() {
            return Ok(Self::U32(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, i64>>() {
            return Ok(Self::I64(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, i32>>() {
            return Ok(Self::I32(arr));
        }
        Err(PyValueError::new_err(format!(
            "expected a numpy array with dtype float64, float32, uint64, uint32, int64, or \
             int32, got {}",
            describe(obj)
        )))
    }

    /// Converts to `xdmf::Values`, for points, connectivity, and attribute data alike -- every
    /// variant maps directly onto the matching `Values` variant with no copy.
    pub(crate) fn to_values(&self) -> PyResult<xdmf::Values<'_>> {
        match self {
            Self::F64(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::F32(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::U64(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::U32(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::I64(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::I32(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
        }
    }
}
