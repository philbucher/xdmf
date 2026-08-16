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

// The `&Vec<T>` and `&[T; N]` impls below are not redundant with the `&[T]` ones above: the
// `impl Into<Values<'_>>` arguments of `TimeStep::point_data`/`cell_data` are resolved by trait
// matching, which does not deref-coerce, so passing a `&vec` or a `&[1.0, 2.0]` needs its own
// impl.

impl<'a> From<&'a Vec<f64>> for Values<'a> {
    fn from(vec: &'a Vec<f64>) -> Self {
        Self::F64(Cow::Borrowed(vec))
    }
}

impl<'a> From<&'a Vec<f32>> for Values<'a> {
    fn from(vec: &'a Vec<f32>) -> Self {
        Self::F32(Cow::Borrowed(vec))
    }
}

impl<'a> From<&'a Vec<u64>> for Values<'a> {
    fn from(vec: &'a Vec<u64>) -> Self {
        Self::U64(Cow::Borrowed(vec))
    }
}

impl<'a, const N: usize> From<&'a [f64; N]> for Values<'a> {
    fn from(array: &'a [f64; N]) -> Self {
        Self::F64(Cow::Borrowed(array))
    }
}

impl<'a, const N: usize> From<&'a [f32; N]> for Values<'a> {
    fn from(array: &'a [f32; N]) -> Self {
        Self::F32(Cow::Borrowed(array))
    }
}

impl<'a, const N: usize> From<&'a [u64; N]> for Values<'a> {
    fn from(array: &'a [u64; N]) -> Self {
        Self::U64(Cow::Borrowed(array))
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

    // Only the number of values matters here, never their type, so the length is taken once and
    // the match is on the attribute alone -- matching on both would be one arm per (attribute,
    // variant) pair for the same `Dimensions`.
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
}

// Sealed so that `Coordinate` names exactly the two types the XDMF geometry can hold, and stays
// closed to outside impls. The conversion lives here rather than on `Coordinate` itself, so it is
// callable inside the crate without becoming public API.
pub(crate) mod sealed {
    use std::borrow::Cow;

    use super::Values;

    /// Conversion backing [`Coordinate`](super::Coordinate), not nameable outside the crate
    pub trait Sealed: Sized {
        /// Borrow a slice of coordinates as [`Values`]
        fn as_values(points: &[Self]) -> Values<'_>;
    }

    impl Sealed for f64 {
        fn as_values(points: &[Self]) -> Values<'_> {
            Values::F64(Cow::Borrowed(points))
        }
    }

    impl Sealed for f32 {
        fn as_values(points: &[Self]) -> Values<'_> {
            Values::F32(Cow::Borrowed(points))
        }
    }
}

/// A type usable as a point coordinate: `f32` or `f64`
pub trait Coordinate: sealed::Sealed {}

impl Coordinate for f64 {}

impl Coordinate for f32 {}

#[cfg(test)]
mod tests {
    use super::{sealed::Sealed, *};

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
        let vec_f32 = vec![1., 2., 3., 4., 5., 6.0_f32];

        let values = vec_f32.into();
        std::assert_matches!(values, Values::F32(_));

        // same NumberType as f64, only the precision distinguishes the two in the light data
        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(Format::XML), 4);
        assert_eq!(values.precision(Format::HDF), 4);
        assert_eq!(values.precision(Format::Binary), 4);
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

        let vec_f32 = vec![1., 2., 3., 4., 5., 6.0_f32];
        let values = Values::from(vec_f32.as_slice());
        std::assert_matches!(values, Values::F32(Cow::Borrowed(_)));

        assert_eq!(values.precision(Format::Binary), 4);
        assert_eq!(values.len(), 6);

        let vec_u64 = vec![1_u64, 2, 3];
        let values = Values::from(vec_u64.as_slice());
        std::assert_matches!(values, Values::U64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn coordinates_borrow_as_values() {
        let points_f64 = [0.0_f64, 1.0, 2.0];
        std::assert_matches!(f64::as_values(&points_f64), Values::F64(Cow::Borrowed(_)));

        let points_f32 = [0.0_f32, 1.0, 2.0];
        std::assert_matches!(f32::as_values(&points_f32), Values::F32(Cow::Borrowed(_)));
    }
}
