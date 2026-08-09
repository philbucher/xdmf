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

impl<'a> From<&'a [u64]> for Values<'a> {
    fn from(slice: &'a [u64]) -> Self {
        Self::U64(Cow::Borrowed(slice))
    }
}

impl Values<'_> {
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
            // written as a rank-3 shape ("<count> <size> 1") rather than "<count> <size>": VTK's
            // XDMF2 reader (vtkXdmfHeavyData, since https://github.com/Kitware/VTK/commit/7199be5854,
            // shipped in VTK 9.6 / ParaView 6.1) computes an AttributeType="Matrix" attribute's
            // component count as the product of its *last two* Dimensions entries, so a 2D
            // "<count> <size>" shape gets misread as one giant tuple. Appending a trailing 1 keeps
            // that product equal to `size` while `count` is used for the tuple count.
            DataAttribute::Tensor6 | DataAttribute::Matrix(_, _) | DataAttribute::Generic(_) => {
                match self {
                    Self::F64(v) => {
                        Dimensions(vec![v.len() / attribute.size(), attribute.size(), 1])
                    }
                    Self::U64(v) => {
                        Dimensions(vec![v.len() / attribute.size(), attribute.size(), 1])
                    }
                }
            }
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

        let vec_u64 = vec![1_u64, 2, 3];
        let values = Values::from(vec_u64.as_slice());
        std::assert_matches!(values, Values::U64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.len(), 3);
    }
}
