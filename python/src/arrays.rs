//! Zero-copy conversion from numpy arrays to the slices/`Values` `xdmf`'s API expects.
//!
//! `NumpyArray` is the single borrowed-array type behind every numeric parameter this crate
//! accepts (points, connectivity, attribute data): `extract` always tries all four dtypes numpy
//! commonly hands us (`float64`, `float32`, `uint64`, `int64` -- the latter, since numpy's default
//! integer dtype is signed, is borrowed as-is and bit-reinterpreted to `u64` via `bytemuck` after
//! an O(n) sign check, so the common `int64` index case still avoids a copy). Which dtypes a given
//! call site actually accepts differs -- points are never integer, connectivity is never float --
//! so that restriction is applied where the array is consumed (`to_values`/`as_u64_slice`) rather
//! than by varying what `extract` accepts per call site: points delegate their integer rejection
//! to the core crate (`Error::InvalidMesh`), so the rule lives in one place instead of being
//! duplicated at the binding boundary. Any dimensionality is accepted as long as the array is
//! C-contiguous: a contiguous `(N, 3)` array has exactly the flat memory layout the core crate's
//! API wants, so a caller can hand in `(N, 3)` points/vectors without a `reshape(-1)`, and it stays
//! zero-copy. Arrays that aren't C-contiguous are rejected with a clear error rather than silently
//! copied.

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

/// A borrowed numpy array of one of the four numeric dtypes this crate understands (any
/// dimensionality), kept alive for the duration of a call so any `xdmf::Values`/`&[u64]` borrowing
/// from it stays valid. Used for points, connectivity, and attribute data alike; see the module
/// docs for how each restricts the dtypes it actually accepts.
pub(crate) enum NumpyArray<'py> {
    F64(PyReadonlyArrayDyn<'py, f64>),
    F32(PyReadonlyArrayDyn<'py, f32>),
    U64(PyReadonlyArrayDyn<'py, u64>),
    I64(PyReadonlyArrayDyn<'py, i64>),
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
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, i64>>() {
            return Ok(Self::I64(arr));
        }
        Err(PyValueError::new_err(format!(
            "expected a numpy array with dtype float64, float32, uint64, or int64, got {}",
            describe(obj)
        )))
    }

    fn dtype_name(&self) -> &'static str {
        match self {
            Self::F64(_) => "float64",
            Self::F32(_) => "float32",
            Self::U64(_) => "uint64",
            Self::I64(_) => "int64",
        }
    }

    /// Converts to `xdmf::Values`, for points and attribute data. Every variant maps to a
    /// `Values` variant (`int64` via the same sign-checked reinterpret as `as_u64_slice`).
    pub(crate) fn to_values(&self) -> PyResult<xdmf::Values<'_>> {
        match self {
            Self::F64(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::F32(arr) => Ok(xdmf::Values::from(contiguous_slice(arr)?)),
            Self::U64(_) | Self::I64(_) => Ok(xdmf::Values::from(self.as_u64_slice()?)),
        }
    }

    /// Borrows the array as `&[u64]` with no copy, for connectivity. `int64` is bit-reinterpreted
    /// (via `bytemuck`, no `unsafe`) after verifying every element is non-negative; `float64`/
    /// `float32` are rejected (connectivity is indices, never coordinates).
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
            Self::F64(_) | Self::F32(_) => Err(PyValueError::new_err(format!(
                "connectivity must be a numpy array with dtype uint64 or int64, got a numpy \
                 array with dtype {}",
                self.dtype_name()
            ))),
        }
    }
}
