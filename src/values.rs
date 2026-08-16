//! This module contains the wrapper type for using a common interface for different data types.

use std::borrow::Cow;

use crate::{
    DataAttribute,
    xdmf_elements::{data_item::NumberType, dimensions::Dimensions},
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
    /// Width in bytes each element type is written at, which is simply its own width.
    ///
    /// No storage writes anything narrower: a backend that cannot carry a type says so through
    /// [`crate::paraview`] instead of quietly storing fewer bytes than the caller handed over.
    pub(crate) fn precision(&self) -> u8 {
        match self {
            Self::F64(_) | Self::I64(_) | Self::U64(_) => 8,
            Self::F32(_) | Self::I32(_) | Self::U32(_) => 4,
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

// Sealed so that `Coordinate` and `ConnectivityIndex` name exactly the types the XDMF geometry and
// topology can hold, and stay closed to outside impls. The conversions live here rather than on the
// public traits, so they are callable inside the crate without becoming public API.
pub(crate) mod sealed {
    use std::borrow::Cow;

    use super::Values;

    /// Conversion backing [`Coordinate`](super::Coordinate), not nameable outside the crate
    pub trait SealedCoordinate: Sized {
        /// Borrow a slice of coordinates as [`Values`]
        fn as_values(points: &[Self]) -> Values<'_>;
    }

    impl SealedCoordinate for f64 {
        fn as_values(points: &[Self]) -> Values<'_> {
            Values::F64(Cow::Borrowed(points))
        }
    }

    impl SealedCoordinate for f32 {
        fn as_values(points: &[Self]) -> Values<'_> {
            Values::F32(Cow::Borrowed(points))
        }
    }

    /// Conversion backing [`ConnectivityIndex`](super::ConnectivityIndex), not nameable outside
    /// the crate
    pub trait SealedIndex: Copy + Sized {
        /// The largest index this type can hold.
        ///
        /// Deliberately the type's own limit and nothing else: the lower cap `ParaView` puts on
        /// `UInt` connectivity is a restriction on the *values*, so it is enforced once, with
        /// every other one, by [`crate::paraview`] -- not duplicated here, where it could not be
        /// skipped along with the rest.
        const MAX_INDEX: i128;

        /// Borrow a slice of indices as [`Values`]
        fn as_values(cells: &[Self]) -> Values<'_>;

        /// A cell type code or poly-cell point count, small enough for every index type
        fn from_u8(value: u8) -> Self;

        /// The index as this type, `None` when the type cannot hold it
        fn from_index(index: usize) -> Option<Self>;

        /// Widened for bounds checking, so signed and unsigned indices compare the same way
        fn as_i128(self) -> i128;
    }

    macro_rules! impl_sealed_index {
        ($($variant:ident($ty:ty, $max_index:expr)),+ $(,)?) => {
            $(
                impl SealedIndex for $ty {
                    const MAX_INDEX: i128 = $max_index;

                    fn as_values(cells: &[Self]) -> Values<'_> {
                        Values::$variant(Cow::Borrowed(cells))
                    }

                    fn from_u8(value: u8) -> Self {
                        Self::from(value)
                    }

                    fn from_index(index: usize) -> Option<Self> {
                        Self::try_from(index).ok()
                    }

                    fn as_i128(self) -> i128 {
                        i128::from(self)
                    }
                }
            )+
        };
    }

    impl_sealed_index!(
        U32(u32, u32::MAX as i128),
        U64(u64, u64::MAX as i128),
        I32(i32, i32::MAX as i128),
        I64(i64, i64::MAX as i128),
    );
}

/// A type usable as a point coordinate: `f32` or `f64`
pub trait Coordinate: sealed::SealedCoordinate {}

impl Coordinate for f64 {}

impl Coordinate for f32 {}

/// A type usable as a connectivity index: `u32`, `u64`, `i32` or `i64`
///
/// The connectivity is written as the type it is passed in, so this choice is what sets the
/// largest mesh that can be written -- see
/// [`TimeSeriesWriter::write_mesh`](crate::TimeSeriesWriter::write_mesh).
pub trait ConnectivityIndex: sealed::SealedIndex {}

impl ConnectivityIndex for u32 {}

impl ConnectivityIndex for u64 {}

impl ConnectivityIndex for i32 {}

impl ConnectivityIndex for i64 {}

#[cfg(test)]
mod tests {
    use super::{
        sealed::{SealedCoordinate, SealedIndex},
        *,
    };

    #[test]
    fn vec_f64() {
        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];

        let values = vec_f64.into();
        std::assert_matches!(values, Values::F64(_));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(), 8);
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
        assert_eq!(values.precision(), 4);
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
        assert_eq!(values.precision(), 8);
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

        // same NumberType as u64, but half the width
        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.precision(), 4);
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
        assert_eq!(values.precision(), 8);
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
        assert_eq!(values.precision(), 4);
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

        assert_eq!(values.precision(), 4);
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
        std::assert_matches!(
            SealedCoordinate::as_values(&points_f64),
            Values::F64(Cow::Borrowed(_))
        );

        let points_f32 = [0.0_f32, 1.0, 2.0];
        std::assert_matches!(
            SealedCoordinate::as_values(&points_f32),
            Values::F32(Cow::Borrowed(_))
        );
    }

    #[test]
    fn connectivity_indices_borrow_as_values() {
        std::assert_matches!(
            SealedIndex::as_values(&[0_u32, 1]),
            Values::U32(Cow::Borrowed(_))
        );
        std::assert_matches!(
            SealedIndex::as_values(&[0_u64, 1]),
            Values::U64(Cow::Borrowed(_))
        );
        std::assert_matches!(
            SealedIndex::as_values(&[0_i32, 1]),
            Values::I32(Cow::Borrowed(_))
        );
        std::assert_matches!(
            SealedIndex::as_values(&[0_i64, 1]),
            Values::I64(Cow::Borrowed(_))
        );
    }

    #[test]
    fn connectivity_index_limits() {
        // each type's own limit -- the lower one ParaView puts on `u64` is checked on the values,
        // by `validate_paraview_uint_range`, so that it can be skipped with the rest
        assert_eq!(u32::MAX_INDEX, i128::from(u32::MAX));
        assert_eq!(u64::MAX_INDEX, i128::from(u64::MAX));
        assert_eq!(i32::MAX_INDEX, i128::from(i32::MAX));
        assert_eq!(i64::MAX_INDEX, i128::from(i64::MAX));

        assert_eq!(u32::from_index(7), Some(7_u32));
        assert_eq!(i32::from_index(usize::MAX), None);
        assert_eq!(i32::from_u8(9), 9_i32);
        assert_eq!((-1_i64).as_i128(), -1_i128);
    }
}
