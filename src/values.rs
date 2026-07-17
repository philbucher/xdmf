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
/// Backed by [`Cow`] rather than an owned `Vec`, so callers that already have the data in a
/// borrowed slice (e.g. a numpy buffer borrowed from Python) can wrap it via `From<&[f64]>`/
/// `From<&[u64]>` without copying. `From<Vec<f64>>`/`From<Vec<u64>>` are still available for
/// owned data and work for any `'a` (owned data has no borrow to satisfy).
pub enum Values<'a> {
    /// vector of f64 values
    F64(Cow<'a, [f64]>),
    /// vector of u64 values
    U64(Cow<'a, [u64]>),
}

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for u64 {}
}

/// Marker for the element types a [`Values`] can hold. Sealed (cannot be implemented outside
/// this crate), so adding a new supported type only requires a new `impl ValueType for ...`
/// here, without growing [`Values`]'s public accessor surface.
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
            Values::U64(_) => None,
        }
    }

    fn as_mut_slice<'v>(values: &'v mut Values<'_>) -> Option<&'v mut [Self]> {
        match values {
            Values::F64(v) => Some(v.to_mut()),
            Values::U64(_) => None,
        }
    }
}

impl ValueType for u64 {
    fn as_slice<'v>(values: &'v Values<'_>) -> Option<&'v [Self]> {
        match values {
            Values::F64(_) => None,
            Values::U64(v) => Some(v),
        }
    }

    fn as_mut_slice<'v>(values: &'v mut Values<'_>) -> Option<&'v mut [Self]> {
        match values {
            Values::F64(_) => None,
            Values::U64(v) => Some(v.to_mut()),
        }
    }
}

impl<'a> From<Vec<f64>> for Values<'a> {
    fn from(vec: Vec<f64>) -> Self {
        Self::F64(Cow::Owned(vec))
    }
}

impl<'a> From<Vec<u64>> for Values<'a> {
    fn from(vec: Vec<u64>) -> Self {
        Self::U64(Cow::Owned(vec))
    }
}

impl<'a> From<&'a [f64]> for Values<'a> {
    fn from(slice: &'a [f64]) -> Self {
        Self::F64(Cow::Borrowed(slice))
    }
}

impl<'a> From<&'a [u64]> for Values<'a> {
    fn from(slice: &'a [u64]) -> Self {
        Self::U64(Cow::Borrowed(slice))
    }
}

impl<'a> Values<'a> {
    pub(crate) fn precision(&self, format: Format) -> u8 {
        match self {
            Self::F64(_) => 8,
            Self::U64(_) => format.uint_precision(),
        }
    }

    pub(crate) fn number_type(&self) -> NumberType {
        match self {
            Self::F64(_) => NumberType::Float,
            Self::U64(_) => NumberType::UInt,
        }
    }

    pub(crate) fn dimensions(&self, attribute: DataAttribute) -> Dimensions {
        match attribute {
            DataAttribute::Scalar => match self {
                Self::F64(v) => Dimensions(vec![v.len()]),
                Self::U64(v) => Dimensions(vec![v.len()]),
            },
            _ => match self {
                Self::F64(v) => Dimensions(vec![v.len() / attribute.size(), attribute.size()]),
                Self::U64(v) => Dimensions(vec![v.len() / attribute.size(), attribute.size()]),
            },
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
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

    /// Gather the entries at `indices` (each `stride` elements wide) into a new `Values`,
    /// preserving variant and order. Used to slice per-entity (e.g. per-cell) data down to
    /// the subset belonging to a single block, without needing any XDMF-level windowing.
    pub(crate) fn gather<'idx>(
        &self,
        stride: usize,
        indices: impl IntoIterator<Item = &'idx usize>,
    ) -> Self {
        match self {
            Self::F64(v) => Self::F64(Cow::Owned(gather_strided(v, stride, indices))),
            Self::U64(v) => Self::U64(Cow::Owned(gather_strided(v, stride, indices))),
        }
    }
}

fn gather_strided<'a, T: Copy>(
    values: &[T],
    stride: usize,
    indices: impl IntoIterator<Item = &'a usize>,
) -> Vec<T> {
    let mut out = Vec::new();
    for &idx in indices {
        out.extend_from_slice(&values[idx * stride..(idx + 1) * stride]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_f64() {
        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];

        let values = vec_f64.into();
        matches!(values, Values::F64(_));

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
            Dimensions(vec![1, 6])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Matrix(3, 2)),
            Dimensions(vec![1, 6])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_u64() {
        let vec_u64 = vec![1_u64, 2, 3, 4, 5, 6];
        let values = vec_u64.into();
        matches!(values, Values::U64(_));

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
    fn gather_f64_scalar() {
        let values: Values = vec![10.0, 11.0, 12.0, 13.0].into();
        let gathered = values.gather(1, &[2, 0]);
        match gathered {
            Values::F64(v) => assert_eq!(v.into_owned(), vec![12.0, 10.0]),
            Values::U64(_) => panic!("expected F64"),
        }
    }

    #[test]
    fn gather_f64_vector() {
        // 3 cells, 2 components each: cell0=[0,1], cell1=[2,3], cell2=[4,5]
        let values: Values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into();
        let gathered = values.gather(2, &[2, 2, 0]);
        match gathered {
            Values::F64(v) => assert_eq!(v.into_owned(), vec![4.0, 5.0, 4.0, 5.0, 0.0, 1.0]),
            Values::U64(_) => panic!("expected F64"),
        }
    }

    #[test]
    fn gather_u64() {
        let values: Values = vec![1_u64, 2, 3, 4].into();
        let gathered = values.gather(1, &[3, 1]);
        match gathered {
            Values::U64(v) => assert_eq!(v.into_owned(), vec![4, 2]),
            Values::F64(_) => panic!("expected U64"),
        }
    }

    #[test]
    fn gather_empty_indices() {
        let values: Values = vec![1.0, 2.0].into();
        let gathered = values.gather(1, &[]);
        assert_eq!(gathered.len(), 0);
    }

    #[test]
    fn as_slice_and_as_mut_slice() {
        let mut f64_values: Values = vec![1.0, 2.0].into();
        assert_eq!(f64_values.as_slice::<f64>(), Some([1.0, 2.0].as_slice()));
        assert_eq!(f64_values.as_slice::<u64>(), None);

        f64_values.as_mut_slice::<f64>().expect("holds f64 data")[0] = 5.0;
        assert_eq!(f64_values.as_slice::<f64>(), Some([5.0, 2.0].as_slice()));
        assert_eq!(f64_values.as_mut_slice::<u64>(), None);

        let mut u64_values: Values = vec![1_u64, 2].into();
        assert_eq!(u64_values.as_slice::<u64>(), Some([1, 2].as_slice()));
        assert_eq!(u64_values.as_slice::<f64>(), None);

        u64_values.as_mut_slice::<u64>().expect("holds u64 data")[0] = 5;
        assert_eq!(u64_values.as_slice::<u64>(), Some([5, 2].as_slice()));
        assert_eq!(u64_values.as_mut_slice::<f64>(), None);
    }
}
