//! Zero-copy conversion from numpy arrays into what the core crate's API expects.
//!
//! Every numeric parameter arrives as one of three borrowed-array types. [`PointArray`] and
//! [`IndexArray`] name exactly the dtypes `xdmf::Coordinate` and `xdmf::ConnectivityIndex` accept,
//! so a dtype the mesh cannot hold is rejected here by name; [`ValueArray`] covers all six element
//! types attribute data can have and maps each onto the matching `xdmf::Values` variant.
//!
//! Shape is otherwise free: a C-contiguous `(N, 3)` array has the flat memory layout the Rust API
//! wants, so the natural numpy layout for points and vector fields needs no `reshape(-1)`. The
//! exception is [`PointArray::validate_shape`]: points are always 3 components, so a trailing
//! dimension that is not 3 is rejected rather than read as interleaved xyz. An array that is not
//! C-contiguous is rejected rather than silently copied.

use numpy::{Element, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

const NOT_CONTIGUOUS: &str =
    "array must be C-contiguous; call `numpy.ascontiguousarray()` on it first";

/// Describes what `obj` actually is, so a rejection names the real problem ("dtype int16", "a
/// list") instead of only restating what was expected.
///
/// The array case is told apart by the type rather than by having a `dtype`, since a numpy
/// *scalar* has one too.
fn describe(obj: &Bound<'_, PyAny>) -> String {
    if obj.cast::<PyUntypedArray>().is_ok()
        && let Ok(dtype) = obj.getattr("dtype")
    {
        return format!("a numpy array with dtype {dtype}");
    }

    // qualified, so a numpy scalar reads "numpy.float64" rather than a bare "float64" that could
    // be mistaken for the dtype the sentence just asked for (builtins keep their short name)
    obj.get_type().fully_qualified_name().map_or_else(
        |_no_name| "an unknown type".to_string(),
        |name| name.to_string(),
    )
}

/// Borrows the array's buffer, rejecting a strided view instead of copying it into one.
pub(crate) fn contiguous_slice<'a, T: Element>(
    array: &'a PyReadonlyArrayDyn<'_, T>,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_not_contiguous| PyValueError::new_err(NOT_CONTIGUOUS))
}

/// Rejects a point array whose trailing dimension is not 3.
///
/// Shape is otherwise ignored, since a C-contiguous `(N, 3)` array is the same memory as the flat
/// one. The transposed `(3, N)` layout is C-contiguous too, so without this check it would be read
/// as interleaved xyz.
fn validate_point_shape(shape: &[usize]) -> PyResult<()> {
    match shape {
        // a flat array is the layout the Rust API takes, so its length is a count, not a
        // component width -- only a multi-dimensional array names components in its last axis
        [] | [_] | [.., 3] => Ok(()),
        [.., other] => Err(PyValueError::new_err(format!(
            "expected points as a flat array of x/y/z coordinates or one shaped (..., 3), got \
             shape {shape:?} whose last dimension is {other}; if this is a (3, N) array of \
             separate x/y/z rows, transpose it with `numpy.ascontiguousarray(points.T)`"
        ))),
    }
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

impl PointArray<'_> {
    /// Checks the shape, which only points constrain (they are always 3 components).
    pub(crate) fn validate_shape(&self) -> PyResult<()> {
        match self {
            Self::F64(array) => validate_point_shape(array.shape()),
            Self::F32(array) => validate_point_shape(array.shape()),
        }
    }
}

impl ValueArray<'_> {
    /// Borrows the buffer as `xdmf::Values`. Each dtype maps onto the matching variant, so the
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
