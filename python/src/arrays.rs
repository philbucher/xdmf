//! Zero-copy conversion from numpy arrays into what the core crate's API expects.
//!
//! Every numeric parameter of this crate arrives as one of these three borrowed-array types:
//! [`PointArray`] and [`IndexArray`] name exactly the dtypes `xdmf::Coordinate` and
//! `xdmf::ConnectivityIndex` accept, so a dtype the mesh cannot hold is rejected here, by name,
//! instead of further down; [`ValueArray`] covers all six element types attribute data can have
//! and maps each straight onto the matching `xdmf::Values` variant.
//!
//! Any dimensionality is accepted -- a C-contiguous `(N, 3)` array has exactly the flat memory
//! layout the Rust API wants, so the natural numpy layout for points and vector fields needs no
//! `reshape(-1)`. Arrays that are not C-contiguous are rejected rather than silently copied.

use numpy::{Element, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

const NOT_CONTIGUOUS: &str =
    "array must be C-contiguous; call `numpy.ascontiguousarray()` on it first";

/// Describes what `obj` actually is, so a rejection names the real problem ("dtype int16", "a
/// list") instead of only restating what was expected.
fn describe(obj: &Bound<'_, PyAny>) -> String {
    match obj.getattr("dtype") {
        Ok(dtype) => format!("a numpy array with dtype {dtype}"),
        Err(_) => obj.get_type().name().map_or_else(
            |_no_name| "an unknown type".to_string(),
            |name| name.to_string(),
        ),
    }
}

/// Borrows the array's buffer, rejecting a strided view instead of copying it into one.
pub(crate) fn contiguous_slice<'a, T: Element>(
    array: &'a PyReadonlyArrayDyn<'_, T>,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_not_contiguous| PyValueError::new_err(NOT_CONTIGUOUS))
}

// Declares one borrowed-array enum per group of dtypes the API accepts, off a single variant list
// each, so the accepted dtypes and the message naming them cannot drift apart.
macro_rules! numpy_arrays {
    ($(
        $(#[$doc:meta])*
        $name:ident($dtypes:literal) { $($variant:ident($ty:ty)),+ $(,)? }
    )+) => {
        $(
            $(#[$doc])*
            ///
            /// Holds the numpy array itself, keeping it alive (and pinned against reallocation)
            /// for as long as anything borrows its buffer.
            pub(crate) enum $name<'py> {
                $($variant(PyReadonlyArrayDyn<'py, $ty>),)+
            }

            impl<'py> $name<'py> {
                /// Borrows `obj`, or reports what it is and what was expected instead. `role`
                /// names what the array was passed as (e.g. `"points"`), since the same dtype can
                /// be valid in one position and not in another.
                pub(crate) fn extract(obj: &Bound<'py, PyAny>, role: &str) -> PyResult<Self> {
                    $(
                        if let Ok(array) = obj.extract::<PyReadonlyArrayDyn<'py, $ty>>() {
                            return Ok(Self::$variant(array));
                        }
                    )+
                    Err(PyValueError::new_err(format!(
                        "expected {role} as a numpy array with dtype {}, got {}",
                        $dtypes,
                        describe(obj),
                    )))
                }
            }
        )+
    };
}

numpy_arrays! {
    /// Point coordinates: the dtypes `xdmf::Coordinate` is implemented for.
    PointArray("float64 or float32") {
        F64(f64),
        F32(f32),
    }

    /// Connectivity indices: the dtypes `xdmf::ConnectivityIndex` is implemented for.
    IndexArray("uint64, uint32, int64, or int32") {
        U64(u64),
        U32(u32),
        I64(i64),
        I32(i32),
    }

    /// Attribute data: every element type `xdmf::Values` can hold.
    ValueArray("float64, float32, uint64, uint32, int64, or int32") {
        F64(f64),
        F32(f32),
        U64(u64),
        U32(u32),
        I64(i64),
        I32(i32),
    }
}

impl ValueArray<'_> {
    /// Borrows the buffer as `xdmf::Values` -- each dtype maps onto the matching variant, so the
    /// data lands in the file as the type it was passed in, without a copy or a cast.
    pub(crate) fn to_values(&self) -> PyResult<xdmf::Values<'_>> {
        match self {
            Self::F64(array) => Ok(xdmf::Values::from(contiguous_slice(array)?)),
            Self::F32(array) => Ok(xdmf::Values::from(contiguous_slice(array)?)),
            Self::U64(array) => Ok(xdmf::Values::from(contiguous_slice(array)?)),
            Self::U32(array) => Ok(xdmf::Values::from(contiguous_slice(array)?)),
            Self::I64(array) => Ok(xdmf::Values::from(contiguous_slice(array)?)),
            Self::I32(array) => Ok(xdmf::Values::from(contiguous_slice(array)?)),
        }
    }
}
