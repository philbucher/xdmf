//! This module contains the wrapper type for using a common interface for different data types.

use std::borrow::Cow;

use crate::{
    DataAttribute,
    xdmf_elements::{
        data_item::{Format, NumberType},
        dimensions::Dimensions,
    },
};

/// Wrapper around different types of data, used to provide a unified interface.
///
/// Backed by [`Cow`] rather than an owned `Vec`, so a caller that already holds the data in a
/// slice can wrap it without copying and hand the same buffer to every time step.
#[derive(Debug)]
pub enum Values<'a> {
    /// f64 values
    F64(Cow<'a, [f64]>),
    /// f32 values
    F32(Cow<'a, [f32]>),
    /// u64 values
    U64(Cow<'a, [u64]>),
}

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for u64 {}
}

/// Marker for the element types a [`Values`] can hold. Sealed (cannot be implemented outside
/// this crate), so adding a new supported type only requires a new `impl ValueType for ...`
/// here, without growing [`Values`]'s public accessor surface.
///
/// Not re-exported from `lib.rs`: callers use `values.as_slice::<f64>()` etc. with a concrete
/// type and never need to name this trait, since `as_slice`/`as_mut_slice` are inherent methods
/// on `Values`, not trait methods dispatched through it. Export it only once something (M5's
/// `read_point_data<T: ValueType>` is the known future case) needs a caller to write `T:
/// ValueType` themselves.
pub trait ValueType: private::Sealed + Sized {
    #[doc(hidden)]
    fn as_slice<'v>(values: &'v Values<'_>) -> Option<&'v [Self]>;
    #[doc(hidden)]
    fn as_mut_slice<'v>(values: &'v mut Values<'_>) -> Option<&'v mut [Self]>;
}

impl ValueType for f64 {
    fn as_slice<'v>(values: &'v Values<'_>) -> Option<&'v [Self]> {
        match values {
            Values::F64(v) => Some(v),
            Values::F32(_) | Values::U64(_) => None,
        }
    }

    fn as_mut_slice<'v>(values: &'v mut Values<'_>) -> Option<&'v mut [Self]> {
        match values {
            Values::F64(v) => Some(v.to_mut()),
            Values::F32(_) | Values::U64(_) => None,
        }
    }
}

impl ValueType for f32 {
    fn as_slice<'v>(values: &'v Values<'_>) -> Option<&'v [Self]> {
        match values {
            Values::F32(v) => Some(v),
            Values::F64(_) | Values::U64(_) => None,
        }
    }

    fn as_mut_slice<'v>(values: &'v mut Values<'_>) -> Option<&'v mut [Self]> {
        match values {
            Values::F32(v) => Some(v.to_mut()),
            Values::F64(_) | Values::U64(_) => None,
        }
    }
}

impl ValueType for u64 {
    fn as_slice<'v>(values: &'v Values<'_>) -> Option<&'v [Self]> {
        match values {
            Values::U64(v) => Some(v),
            Values::F64(_) | Values::F32(_) => None,
        }
    }

    fn as_mut_slice<'v>(values: &'v mut Values<'_>) -> Option<&'v mut [Self]> {
        match values {
            Values::U64(v) => Some(v.to_mut()),
            Values::F64(_) | Values::F32(_) => None,
        }
    }
}

/// Moves `vec` into the value. If the same buffer is reused across multiple `write_data` calls,
/// borrow it instead (`buf.as_slice().into()`)
impl From<Vec<f64>> for Values<'_> {
    fn from(vec: Vec<f64>) -> Self {
        Self::F64(Cow::Owned(vec))
    }
}

/// Moves `vec` into the value; see the `f64` impl above for the buffer-reuse caveat.
impl From<Vec<f32>> for Values<'_> {
    fn from(vec: Vec<f32>) -> Self {
        Self::F32(Cow::Owned(vec))
    }
}

/// Moves `vec` into the value; see the `f64` impl above for the buffer-reuse caveat.
impl From<Vec<u64>> for Values<'_> {
    fn from(vec: Vec<u64>) -> Self {
        Self::U64(Cow::Owned(vec))
    }
}

impl<'a> From<&'a [f64]> for Values<'a> {
    fn from(slice: &'a [f64]) -> Self {
        Self::F64(Cow::Borrowed(slice))
    }
}

impl<'a> From<&'a [f32]> for Values<'a> {
    fn from(slice: &'a [f32]) -> Self {
        Self::F32(Cow::Borrowed(slice))
    }
}

impl<'a> From<&'a [u64]> for Values<'a> {
    fn from(slice: &'a [u64]) -> Self {
        Self::U64(Cow::Borrowed(slice))
    }
}

impl Values<'_> {
    pub(crate) fn precision(&self, format: Format) -> u8 {
        match self {
            Self::F64(_) => 8,
            Self::F32(_) => 4,
            Self::U64(_) => format.uint_precision(),
        }
    }

    pub(crate) fn number_type(&self) -> NumberType {
        match self {
            Self::F64(_) | Self::F32(_) => NumberType::Float,
            Self::U64(_) => NumberType::UInt,
        }
    }

    pub(crate) fn dimensions(&self, attribute: DataAttribute) -> Dimensions {
        let len = self.len();
        match attribute {
            DataAttribute::Scalar => Dimensions(vec![len]),
            // written as a rank-3 shape ("<count> <size> 1") rather than "<count> <size>": VTK's
            // XDMF2 reader (vtkXdmfHeavyData, since https://github.com/Kitware/VTK/commit/7199be5854,
            // shipped in VTK 9.6 / ParaView 6.1) computes an AttributeType="Matrix" attribute's
            // component count as the product of its *last two* Dimensions entries, so a 2D
            // "<count> <size>" shape gets misread as one giant tuple. Appending a trailing 1 keeps
            // that product equal to `size` while `count` is used for the tuple count.
            DataAttribute::Tensor6 | DataAttribute::Matrix(_, _) | DataAttribute::Generic(_) => {
                Dimensions(vec![len / attribute.size(), attribute.size(), 1])
            }
            _ => Dimensions(vec![len / attribute.size(), attribute.size()]),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::U64(v) => v.len(),
        }
    }

    /// Returns the underlying data as a `T` slice, or `None` if this `Values` holds a different
    /// type. Useful for reading a `Values` without needing to match on its variant.
    pub fn as_slice<T: ValueType>(&self) -> Option<&[T]> {
        T::as_slice(self)
    }

    /// Returns the underlying data as a mutable `T` slice, or `None` if this `Values` holds a
    /// different type. Useful for overwriting a `Values` in place (e.g. across time steps)
    /// without reallocating it, and without needing to match on its variant.
    pub fn as_mut_slice<T: ValueType>(&mut self) -> Option<&mut [T]> {
        T::as_mut_slice(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_f64() {
        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];

        let values = vec_f64.into();
        std::assert_matches!(values, Values::F64(_));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(Format::XML), 8);
        assert_eq!(values.precision(Format::Binary), 8);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Vector),
            Dimensions(vec![2, 3])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Tensor6),
            Dimensions(vec![1, 6, 1])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Matrix(3, 2)),
            Dimensions(vec![1, 6, 1])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_f32() {
        let vec_f32 = vec![1_f32, 2., 3., 4., 5., 6.];

        let values = vec_f32.into();
        std::assert_matches!(values, Values::F32(_));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(Format::XML), 4);
        assert_eq!(values.precision(Format::Binary), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Vector),
            Dimensions(vec![2, 3])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_u64() {
        let vec_u64 = vec![1_u64, 2, 3, 4, 5, 6];
        let values = vec_u64.into();
        std::assert_matches!(values, Values::U64(_));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.precision(Format::XML), 8);
        assert_eq!(values.precision(Format::HDF), 8);
        assert_eq!(values.precision(Format::Binary), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn borrowed_slices() {
        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];
        let values = Values::from(vec_f64.as_slice());
        std::assert_matches!(values, Values::F64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(
            values.dimensions(DataAttribute::Vector),
            Dimensions(vec![2, 3])
        );
        assert_eq!(values.len(), 6);

        let vec_f32 = vec![1_f32, 2., 3.];
        let values = Values::from(vec_f32.as_slice());
        std::assert_matches!(values, Values::F32(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.len(), 3);

        let vec_u64 = vec![1_u64, 2, 3];
        let values = Values::from(vec_u64.as_slice());
        std::assert_matches!(values, Values::U64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn as_slice_and_as_mut_slice() {
        let mut f64_values: Values = vec![1.0, 2.0].into();
        assert_eq!(f64_values.as_slice::<f64>(), Some([1.0, 2.0].as_slice()));
        assert_eq!(f64_values.as_slice::<f32>(), None);
        assert_eq!(f64_values.as_slice::<u64>(), None);

        f64_values.as_mut_slice::<f64>().expect("holds f64 data")[0] = 5.0;
        assert_eq!(f64_values.as_slice::<f64>(), Some([5.0, 2.0].as_slice()));
        assert_eq!(f64_values.as_mut_slice::<f32>(), None);
        assert_eq!(f64_values.as_mut_slice::<u64>(), None);

        let mut f32_values: Values = vec![1.0_f32, 2.0].into();
        assert_eq!(f32_values.as_slice::<f32>(), Some([1.0, 2.0].as_slice()));
        assert_eq!(f32_values.as_slice::<f64>(), None);

        f32_values.as_mut_slice::<f32>().expect("holds f32 data")[0] = 5.0;
        assert_eq!(f32_values.as_slice::<f32>(), Some([5.0, 2.0].as_slice()));
        assert_eq!(f32_values.as_mut_slice::<u64>(), None);

        let mut u64_values: Values = vec![1_u64, 2].into();
        assert_eq!(u64_values.as_slice::<u64>(), Some([1, 2].as_slice()));
        assert_eq!(u64_values.as_slice::<f64>(), None);

        u64_values.as_mut_slice::<u64>().expect("holds u64 data")[0] = 5;
        assert_eq!(u64_values.as_slice::<u64>(), Some([5, 2].as_slice()));
        assert_eq!(u64_values.as_mut_slice::<f64>(), None);
        assert_eq!(u64_values.as_mut_slice::<f32>(), None);
    }
}
