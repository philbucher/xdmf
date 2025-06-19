use ndarray::{ArrayView1, ArrayView2, ArrayViewD};

use xdmf_elements::data_item::NumberType;
use xdmf_elements::dimensions::Dimensions;

pub enum Values<'a> {
    // f64 variants
    View1Df64(ArrayView1<'a, f64>),
    View2Df64(ArrayView2<'a, f64>),
    ViewDynf64(ArrayViewD<'a, f64>),
}

impl<'a> From<ArrayView1<'a, f64>> for Values<'a> {
    fn from(view: ArrayView1<'a, f64>) -> Self {
        Values::View1Df64(view)
    }
}

impl<'a> From<ArrayView2<'a, f64>> for Values<'a> {
    fn from(view: ArrayView2<'a, f64>) -> Self {
        Values::View2Df64(view)
    }
}

impl<'a> From<ArrayViewD<'a, f64>> for Values<'a> {
    fn from(view: ArrayViewD<'a, f64>) -> Self {
        Values::ViewDynf64(view)
    }
}

impl Values<'_> {
    pub(crate) fn precision(&self) -> u8 {
        match self {
            Values::View1Df64(_) => 8,
            Values::View2Df64(_) => 8,
            Values::ViewDynf64(_) => 8,
        }
    }

    pub(crate) fn number_type(&self) -> NumberType {
        match self {
            Values::View1Df64(_) => NumberType::Float,
            Values::View2Df64(_) => NumberType::Float,
            Values::ViewDynf64(_) => NumberType::Float,
        }
    }

    pub(crate) fn dimensions(&self) -> Dimensions {
        match self {
            Values::View1Df64(d) => Dimensions(vec![d.len()]),
            Values::View2Df64(d) => Dimensions(d.shape().to_vec()),
            Values::ViewDynf64(d) => Dimensions(d.shape().to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_1d_f64() {
        let data_f64 = vec![1., 2., 3., 4., 5., 6.];
        let view_f64 = ArrayView1::from(&data_f64);
        assert_eq!(view_f64.shape(), &[6]);
        assert_eq!(view_f64.len(), data_f64.len());

        let values: Values = view_f64.into();
        matches!(values, Values::View1Df64(_));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(), 8);
        assert_eq!(values.dimensions(), Dimensions(vec![6]));
    }

    #[test]
    fn test_view_2d_f64() {
        let data_f64 = vec![1., 2., 3., 4., 5., 6.];
        let view_f64 = ArrayView2::from_shape((2, 3), &data_f64).unwrap();
        assert_eq!(view_f64.shape(), &[2, 3]);
        assert_eq!(view_f64.len(), data_f64.len());

        let values: Values = view_f64.into();
        matches!(values, Values::View2Df64(_));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(), 8);
        assert_eq!(values.dimensions(), Dimensions(vec![2, 3]));
    }

    #[test]
    fn test_view_dyn_f64() {
        let data_f64 = vec![1., 2., 3., 4., 5., 6.];
        let view_f64 = ArrayViewD::from_shape(vec![3, 2], &data_f64).unwrap();
        assert_eq!(view_f64.shape(), &[3, 2]);
        assert_eq!(view_f64.len(), data_f64.len());

        let values: Values = view_f64.into();
        matches!(values, Values::View2Df64(_));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(), 8);
        assert_eq!(values.dimensions(), Dimensions(vec![3, 2]));
    }
}
