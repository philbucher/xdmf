//! This module contains the wrapper type for using a common interface for different data types.

use std::borrow::Cow;

use crate::{
    DataAttribute,
    xdmf_elements::{
        data_item::{Format, NumberType},
        dimensions::Dimensions,
    },
};

macro_rules! define_values {
    ($($variant:ident($ty:ty)),+ $(,)?) => {
        /// Wrapper around different types of data, used to provide a unified interface.
        ///
        /// Backed by [`Cow`] rather than an owned `Vec`, so a caller that already holds the data
        /// in a slice can wrap it without copying and hand the same buffer to every time step.
        #[derive(Debug)]
        pub enum Values<'a> {
            $(
                #[doc = concat!("`", stringify!($ty), "` values")]
                $variant(Cow<'a, [$ty]>),
            )+
        }

        $(
            /// Moves `vec` into the value. If the same buffer is reused across multiple
            /// `write_data` calls, borrow it instead (`buf.as_slice().into()`)
            impl From<Vec<$ty>> for Values<'_> {
                fn from(vec: Vec<$ty>) -> Self {
                    Self::$variant(Cow::Owned(vec))
                }
            }

            impl<'a> From<&'a [$ty]> for Values<'a> {
                fn from(slice: &'a [$ty]) -> Self {
                    Self::$variant(Cow::Borrowed(slice))
                }
            }

            // The `&Vec<T>` and `&[T; N]` impls are not redundant with the `&[T]` one: the
            // `impl Into<Values<'_>>` arguments of `TimeStep::point_data`/`cell_data` are resolved
            // by trait matching, which does not deref-coerce, so passing a `&vec` or a
            // `&[1.0, 2.0]` needs its own impl.
            impl<'a> From<&'a Vec<$ty>> for Values<'a> {
                fn from(vec: &'a Vec<$ty>) -> Self {
                    Self::$variant(Cow::Borrowed(vec))
                }
            }

            impl<'a, const N: usize> From<&'a [$ty; N]> for Values<'a> {
                fn from(array: &'a [$ty; N]) -> Self {
                    Self::$variant(Cow::Borrowed(array))
                }
            }
        )+
    };
}

define_values!(F64(f64), F32(f32), I64(i64), I32(i32), U64(u64), U32(u32));

impl Values<'_> {
    pub(crate) fn precision(&self, format: Format) -> u8 {
        match self {
            Self::F64(_) => 8,
            Self::F32(_) | Self::I32(_) | Self::U32(_) => 4,
            Self::I64(_) | Self::U64(_) => format.int_precision(),
        }
    }

    pub(crate) fn number_type(&self) -> NumberType {
        match self {
            Self::F64(_) | Self::F32(_) => NumberType::Float,
            Self::I64(_) | Self::I32(_) => NumberType::Int,
            Self::U64(_) | Self::U32(_) => NumberType::UInt,
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
            Self::I64(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::U32(v) => v.len(),
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
        // narrowed to 32 bits for binary, see `Format::int_precision`
        assert_eq!(values.precision(Format::Binary), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_u32() {
        let vec_u32 = vec![1_u32, 2, 3, 4, 5, 6];
        let values = vec_u32.into();
        std::assert_matches!(values, Values::U32(_));

        // same NumberType as u64, and 32 bits wide in every format
        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.precision(Format::XML), 4);
        assert_eq!(values.precision(Format::HDF), 4);
        assert_eq!(values.precision(Format::Binary), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_i64() {
        let vec_i64 = vec![-1_i64, 2, 3, 4, 5, 6];
        let values = vec_i64.into();
        std::assert_matches!(values, Values::I64(_));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.precision(Format::XML), 8);
        assert_eq!(values.precision(Format::HDF), 8);
        // narrowed to 32 bits for binary, see `Format::int_precision`
        assert_eq!(values.precision(Format::Binary), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Vector),
            Dimensions(vec![2, 3])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_i32() {
        let vec_i32 = vec![-1_i32, 2, 3, 4, 5, 6];
        let values = vec_i32.into();
        std::assert_matches!(values, Values::I32(_));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.precision(Format::XML), 4);
        assert_eq!(values.precision(Format::HDF), 4);
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

        let vec_u32 = vec![1_u32, 2, 3];
        let values = Values::from(vec_u32.as_slice());
        std::assert_matches!(values, Values::U32(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.len(), 3);

        let vec_i64 = vec![1_i64, 2, 3];
        let values = Values::from(vec_i64.as_slice());
        std::assert_matches!(values, Values::I64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.len(), 3);

        let vec_i32 = vec![1_i32, 2, 3];
        let values = Values::from(vec_i32.as_slice());
        std::assert_matches!(values, Values::I32(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.len(), 3);
    }

    // the `&Vec<T>`/`&[T; N]` impls are generated for every element type, so a compile-time check
    // that they resolve is the point here
    #[test]
    fn borrowed_vecs_and_arrays() {
        let vec_i32 = vec![1_i32, 2, 3];
        std::assert_matches!(Values::from(&vec_i32), Values::I32(Cow::Borrowed(_)));
        std::assert_matches!(Values::from(&[1_u32, 2]), Values::U32(Cow::Borrowed(_)));
        std::assert_matches!(Values::from(&[1_i64, 2]), Values::I64(Cow::Borrowed(_)));
    }

    #[test]
    fn coordinates_borrow_as_values() {
        let points_f64 = [0.0_f64, 1.0, 2.0];
        std::assert_matches!(f64::as_values(&points_f64), Values::F64(Cow::Borrowed(_)));

        let points_f32 = [0.0_f32, 1.0, 2.0];
        std::assert_matches!(f32::as_values(&points_f32), Values::F32(Cow::Borrowed(_)));
    }
}
