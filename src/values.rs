//! This module contains the wrapper type for using a common interface for different data types.

use crate::{
    DataAttribute,
    xdmf_elements::{data_item::NumberType, dimensions::Dimensions},
};

/// Wrapper around different types of data, used to provide a unified interface.
pub enum Values {
    /// vector of f64 values
    F64(Vec<f64>),
    /// vector of u64 values
    U64(Vec<u64>),
}

impl From<Vec<f64>> for Values {
    fn from(vec: Vec<f64>) -> Self {
        Self::F64(vec)
    }
}

impl From<Vec<u64>> for Values {
    fn from(vec: Vec<u64>) -> Self {
        Self::U64(vec)
    }
}

impl Values {
    pub(crate) fn precision(&self) -> u8 {
        match self {
            Self::F64(_) => 8,
            Self::U64(_) => 8,
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
        assert_eq!(values.precision(), 8);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(values.len(), 6);
    }
}
