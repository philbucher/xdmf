use xdmf_elements::data_item::NumberType;
use xdmf_elements::dimensions::Dimensions;

pub enum Values {
    F64(Vec<f64>),
    U64(Vec<u64>),
}

impl From<Vec<f64>> for Values {
    fn from(vec: Vec<f64>) -> Self {
        Values::F64(vec)
    }
}

impl From<Vec<u64>> for Values {
    fn from(vec: Vec<u64>) -> Self {
        Values::U64(vec)
    }
}

impl Values {
    pub(crate) fn precision(&self) -> u8 {
        match self {
            Values::F64(_) => 8,
            Values::U64(_) => 8,
        }
    }

    pub(crate) fn number_type(&self) -> NumberType {
        match self {
            Values::F64(_) => NumberType::Float,
            Values::U64(_) => NumberType::UInt,
        }
    }

    pub(crate) fn dimensions(&self) -> Dimensions {
        match self {
            Values::F64(v) => Dimensions(vec![v.len()]),
            Values::U64(v) => Dimensions(vec![v.len()]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_f64() {
        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];

        let values = vec_f64.into();
        matches!(values, Values::F64(_));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(), 8);
        assert_eq!(values.dimensions(), Dimensions(vec![6]));
    }

    #[test]
    fn test_vec_u64() {
        let vec_u64 = vec![1u64, 2, 3, 4, 5, 6];
        let values = vec_u64.into();
        matches!(values, Values::U64(_));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.precision(), 8);
        assert_eq!(values.dimensions(), Dimensions(vec![6]));
    }
}
