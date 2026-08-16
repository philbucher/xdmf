//! This module contains the wrapper type for using a common interface for different data types.

use std::borrow::Cow;

use crate::{
    DataAttribute, Error, Result,
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

/// Width in bytes that `NumberType="UInt"` data is written at, in *every* format.
///
/// `ParaView`'s Xdmf2 reader builds a 32-bit array for `UInt` whatever `Precision` the light data
/// declares, so a `u64` above `u32::MAX` is read back truncated (ascii formats) or clamped to
/// `u32::MAX` (HDF5) -- silently, without any reader error. Rather than write 8 bytes the reader
/// then ignores, `u64` data is capped at `u32::MAX` and stored narrowed: the upper 4 bytes could
/// only ever be zeros. Unlike the `Binary` backend's narrowing of `i64`, this is a property of the
/// reader, so switching `DataStorage` does not lift it -- `i64` is decoded at the full 64 bits and
/// is the way to store integers beyond 32 bits.
pub(crate) const UINT_PRECISION: u8 = 4;

const UINT_RANGE_REASON: &str = "u64 data must fit in 32 bits, since ParaView decodes UInt data \
                                 as 32-bit whatever precision is declared; no DataStorage avoids \
                                 this, use i64 for integers beyond 32 bits";

/// Narrow a `u64` to the 32 bits [`UINT_PRECISION`] writes, rejecting anything that does not fit.
pub(crate) fn checked_uint(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_err| Error::IntegerOutOfRange {
        value: i128::from(value),
        reason: UINT_RANGE_REASON.to_string(),
    })
}

impl Values<'_> {
    pub(crate) fn precision(&self, format: Format) -> u8 {
        match self {
            Self::F64(_) => 8,
            // `u64` sits with the 4-byte types rather than with `i64`: it is written narrowed to
            // 32 bits in every format (see `UINT_PRECISION`), so its values and its declared
            // precision agree everywhere.
            Self::F32(_) | Self::I32(_) | Self::U32(_) | Self::U64(_) => UINT_PRECISION,
            Self::I64(_) => format.int_precision(),
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

    // Checked up front, before any backend touches disk, so a value that could not be read back
    // is reported as a caller error rather than written and silently misread.
    pub(crate) fn validate_uint_range(&self) -> Result<()> {
        let Self::U64(values) = self else {
            return Ok(());
        };

        for &value in values.iter() {
            checked_uint(value)?;
        }
        Ok(())
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
        /// The largest index that survives being read back, which is not always what the type
        /// itself can hold -- see the `u64` impl.
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
        // not `u64::MAX`: `UInt` connectivity is decoded at 32 bits whatever precision the light
        // data declares, so a wider index is read back as a different point (see `UINT_PRECISION`)
        U64(u64, u32::MAX as i128),
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
        // narrowed to 32 bits in *every* format, unlike i64 -- see `UINT_PRECISION`
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
        // the unsigned types share a limit: `u64` is capped by the reader, not by the type
        assert_eq!(u32::MAX_INDEX, i128::from(u32::MAX));
        assert_eq!(u64::MAX_INDEX, i128::from(u32::MAX));
        assert_eq!(i32::MAX_INDEX, i128::from(i32::MAX));
        assert_eq!(i64::MAX_INDEX, i128::from(i64::MAX));

        assert_eq!(u32::from_index(7), Some(7_u32));
        assert_eq!(i32::from_index(usize::MAX), None);
        assert_eq!(i32::from_u8(9), 9_i32);
        assert_eq!((-1_i64).as_i128(), -1_i128);
    }
}
