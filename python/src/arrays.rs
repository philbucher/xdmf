//! Zero-copy conversion from numpy arrays to the slices/`Values` `xdmf`'s API expects.
//!
//! Points and attribute float data accept `float64` or `float32` (mirroring `xdmf::Values`'
//! `F64`/`F32` variants); connectivity and uint attribute data accept `uint64` (borrowed as-is) or
//! `int64` (also borrowed, after an O(n) sign check and a `bytemuck` bit-reinterpret to `u64` --
//! numpy's default integer dtype is signed `int64`, so requiring `uint64` would force a copy on
//! the most common path). Any dimensionality is accepted as long as the array is C-contiguous: a
//! contiguous `(N, 3)` array has exactly the flat memory layout the core crate's API wants, so a
//! caller can hand in `(N, 3)` points/vectors without a `reshape(-1)`, and it stays zero-copy.
//! Arrays that aren't C-contiguous are rejected with a clear error rather than silently copied.

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

/// A borrowed `float64` numpy array (any dimensionality), used for points -- the core crate's
/// `write_mesh` always takes `&[f64]`, so unlike attribute data (see `ValueGuard`), `float32`
/// points are not accepted.
pub(crate) struct PointsArray<'py>(PyReadonlyArrayDyn<'py, f64>);

impl<'py> PointsArray<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let arr: PyReadonlyArrayDyn<'py, f64> = obj.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "points must be a numpy array with dtype float64, got {}",
                describe(obj)
            ))
        })?;
        Ok(Self(arr))
    }

    pub(crate) fn as_slice(&self) -> PyResult<&[f64]> {
        contiguous_slice(&self.0)
    }
}

/// A borrowed integer numpy array (`uint64` or `int64`, any dimensionality), used for
/// connectivity: checked for contiguity and, for `int64`, for non-negativity.
pub(crate) enum UintArray<'py> {
    U64(PyReadonlyArrayDyn<'py, u64>),
    I64(PyReadonlyArrayDyn<'py, i64>),
}

impl<'py> UintArray<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, u64>>() {
            return Ok(Self::U64(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, i64>>() {
            return Ok(Self::I64(arr));
        }
        Err(PyValueError::new_err(format!(
            "connectivity must be a numpy array with dtype uint64 or int64, got {}",
            describe(obj)
        )))
    }

    /// Borrows the array as `&[u64]` with no copy. `int64` data is bit-reinterpreted (via
    /// `bytemuck`, no `unsafe`) after verifying every element is non-negative (indices/counts are
    /// never negative).
    pub(crate) fn as_u64_slice(&self) -> PyResult<&[u64]> {
        match self {
            Self::U64(arr) => contiguous_slice(arr),
            Self::I64(arr) => {
                let signed = contiguous_slice(arr)?;
                if let Some(&neg) = signed.iter().find(|&&v| v < 0) {
                    return Err(PyValueError::new_err(format!(
                        "value {neg} is negative, but indices/counts must be non-negative"
                    )));
                }
                Ok(bytemuck::cast_slice(signed))
            }
        }
    }
}

/// Either kind of array `xdmf::Values` can hold, kept alive for the duration of a `write_data`
/// call so the `xdmf::Values` borrowing from it stays valid.
pub(crate) enum ValueGuard<'py> {
    F64(PyReadonlyArrayDyn<'py, f64>),
    F32(PyReadonlyArrayDyn<'py, f32>),
    Uint(UintArray<'py>),
}

impl<'py> ValueGuard<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, f64>>() {
            return Ok(Self::F64(arr));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, f32>>() {
            return Ok(Self::F32(arr));
        }
        if let Ok(arr) = UintArray::extract(obj) {
            return Ok(Self::Uint(arr));
        }
        Err(PyValueError::new_err(format!(
            "expected a numpy array with dtype float64, float32, uint64, or int64, got {}",
            describe(obj)
        )))
    }

    pub(crate) fn to_values(&self) -> PyResult<xdmf::Values<'_>> {
        match self {
            Self::F64(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::F32(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::Uint(arr) => Ok(xdmf::Values::from(arr.as_u64_slice()?)),
        }
    }
}
