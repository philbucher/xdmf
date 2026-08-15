//! This module contains the wrapper type for using a common interface for different data types.

use std::borrow::Cow;

use crate::{
    DataAttribute,
    xdmf_elements::{
        data_item::{Format, NumberType},
        dimensions::Dimensions,
    },
};

// Generates `Values`, its mutable mirror `ValuesMut`, and their shared per-variant boilerplate
// (`number_type`, `len`, and the `From<Vec<T>>`/`From<&[T]>` impls). `precision` is deliberately
// NOT generated here.
macro_rules! values {
    ($($variant:ident($ty:ty, $doc:literal) => $number_type:expr;)+) => {
        /// Wrapper around different types of data, used to provide a unified interface.
        ///
        /// Backed by [`Cow`] rather than an owned `Vec`, so a caller that already holds the data
        /// in a slice can wrap it without copying and hand the same buffer to every time step.
        #[derive(Debug)]
        pub enum Values<'a> {
            $(
                #[doc = $doc]
                $variant(Cow<'a, [$ty]>),
            )+
        }

        impl Values<'_> {
            pub(crate) fn number_type(&self) -> NumberType {
                match self {
                    $(Self::$variant(_) => $number_type,)+
                }
            }

            pub(crate) fn len(&self) -> usize {
                match self {
                    $(Self::$variant(v) => v.len(),)+
                }
            }
        }

        $(
            /// Moves `vec` into the value. If the same buffer is reused across multiple
            /// `write_data` calls, borrow it instead (`buf.as_slice().into()`).
            impl From<Vec<$ty>> for Values<'_> {
                fn from(vec: Vec<$ty>) -> Self {
                    Self::$variant(Cow::Owned(vec))
                }
            }

            /// Borrows `slice` into the value, for zero-copy reuse of a caller-owned buffer.
            impl<'a> From<&'a [$ty]> for Values<'a> {
                fn from(slice: &'a [$ty]) -> Self {
                    Self::$variant(Cow::Borrowed(slice))
                }
            }
        )+

        /// The mutable mirror of [`Values`], used at the reader's format-backend boundary to
        /// write parsed heavy data directly into a caller-owned buffer.
        pub enum ValuesMut<'a> {
            $(
                #[doc = $doc]
                $variant(&'a mut Vec<$ty>),
            )+
        }
    };
}

values! {
    F64(f64, "f64 values") => NumberType::Float;
    F32(f32, "f32 values") => NumberType::Float;
    U64(u64, "u64 values") => NumberType::UInt;
    U32(u32, "u32 values") => NumberType::UInt;
    I64(i64, "i64 values") => NumberType::Int;
    I32(i32, "i32 values") => NumberType::Int;
}

mod private {
    pub trait Sealed {}
}

/// Marker for the element types a [`Values`]/[`ValuesMut`] can hold. Sealed (cannot be
/// implemented outside this crate), so adding a new supported type only requires a new
/// `impl_value_type!` invocation below, without growing [`Values`]'s public accessor surface.
pub trait ValueType: private::Sealed + Sized {
    #[doc(hidden)]
    fn as_slice<'v>(values: &'v Values<'_>) -> Option<&'v [Self]>;
    #[doc(hidden)]
    fn as_mut_slice<'v>(values: &'v mut Values<'_>) -> Option<&'v mut [Self]>;
    #[doc(hidden)]
    fn as_values_mut(vec: &mut Vec<Self>) -> ValuesMut<'_>;
}

// The `_` arms below intentionally aren't spelled out per other variant (unlike some other matches
// in this crate, e.g. the reader's `Values::U32|I64|I32` catch-alls): "does this `Values` hold a
// `$ty`" is correct for `_ => None` no matter how many variants `Values` grows to, so there is no
// exhaustiveness safety net worth keeping here.
macro_rules! impl_value_type {
    ($ty:ty, $variant:ident) => {
        impl private::Sealed for $ty {}

        impl ValueType for $ty {
            fn as_slice<'v>(values: &'v Values<'_>) -> Option<&'v [Self]> {
                match values {
                    Values::$variant(v) => Some(v),
                    _ => None,
                }
            }

            fn as_mut_slice<'v>(values: &'v mut Values<'_>) -> Option<&'v mut [Self]> {
                match values {
                    Values::$variant(v) => Some(v.to_mut()),
                    _ => None,
                }
            }

            fn as_values_mut(vec: &mut Vec<Self>) -> ValuesMut<'_> {
                ValuesMut::$variant(vec)
            }
        }
    };
}

impl_value_type!(f64, F64);
impl_value_type!(f32, F32);
impl_value_type!(u64, U64);
impl_value_type!(u32, U32);
impl_value_type!(i64, I64);
impl_value_type!(i32, I32);

impl Values<'_> {
    pub(crate) fn precision(&self, format: Format) -> u8 {
        match self {
            Self::F64(_) => 8,
            Self::F32(_) => 4,
            Self::U64(_) => format.int64_precision(),
            Self::U32(_) => 4,
            Self::I64(_) => format.int64_precision(),
            Self::I32(_) => 4,
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

    /// Returns the underlying data as a `T` slice, or `None` if this `Values` holds a different type.
    pub fn as_slice<T: ValueType>(&self) -> Option<&[T]> {
        T::as_slice(self)
    }

    /// Returns the underlying data as a mutable `T` slice, or `None` if this `Values` holds a different type.
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
    fn vec_u32() {
        let vec_u32 = vec![1_u32, 2, 3, 4, 5, 6];
        let values = vec_u32.into();
        std::assert_matches!(values, Values::U32(_));

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
        assert_eq!(values.precision(Format::Binary), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
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

        let vec_u32 = vec![1_u32, 2, 3];
        let values = Values::from(vec_u32.as_slice());
        std::assert_matches!(values, Values::U32(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.len(), 3);

        let vec_i64 = vec![-1_i64, 2, 3];
        let values = Values::from(vec_i64.as_slice());
        std::assert_matches!(values, Values::I64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.len(), 3);

        let vec_i32 = vec![-1_i32, 2, 3];
        let values = Values::from(vec_i32.as_slice());
        std::assert_matches!(values, Values::I32(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Int);
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

        let mut u32_values: Values = vec![1_u32, 2].into();
        assert_eq!(u32_values.as_slice::<u32>(), Some([1, 2].as_slice()));
        assert_eq!(u32_values.as_slice::<u64>(), None);

        u32_values.as_mut_slice::<u32>().expect("holds u32 data")[0] = 5;
        assert_eq!(u32_values.as_slice::<u32>(), Some([5, 2].as_slice()));
        assert_eq!(u32_values.as_mut_slice::<i32>(), None);

        let mut i64_values: Values = vec![-1_i64, 2].into();
        assert_eq!(i64_values.as_slice::<i64>(), Some([-1, 2].as_slice()));
        assert_eq!(i64_values.as_slice::<u64>(), None);

        i64_values.as_mut_slice::<i64>().expect("holds i64 data")[0] = 5;
        assert_eq!(i64_values.as_slice::<i64>(), Some([5, 2].as_slice()));
        assert_eq!(i64_values.as_mut_slice::<i32>(), None);

        let mut i32_values: Values = vec![-1_i32, 2].into();
        assert_eq!(i32_values.as_slice::<i32>(), Some([-1, 2].as_slice()));
        assert_eq!(i32_values.as_slice::<i64>(), None);

        i32_values.as_mut_slice::<i32>().expect("holds i32 data")[0] = 5;
        assert_eq!(i32_values.as_slice::<i32>(), Some([5, 2].as_slice()));
        assert_eq!(i32_values.as_mut_slice::<u32>(), None);
    }
}
