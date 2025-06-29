use std::io::Result as IoResult;

use ndarray::{ArrayView, ArrayView1, ArrayView2, Dimension};

use crate::DataWriter;
use crate::values::Values;

use xdmf_elements::data_item::Format;

pub(crate) struct XmlWriter {}

impl XmlWriter {
    pub fn new() -> Self {
        XmlWriter {}
    }

    fn values_to_string(&self, data: &Values) -> String {
        match data {
            Values::View1Df64(view) => array_to_string_fmt(view),
            Values::View2Df64(view) => array_to_string_fmt(view),
            Values::ViewDynf64(view) => array_to_string_fmt(view),
        }
    }
}

impl DataWriter for XmlWriter {
    fn format(&self) -> Format {
        Format::XML
    }

    fn write_mesh(
        &mut self,
        points: &ArrayView2<f64>,
        cells: &ArrayView1<usize>,
    ) -> IoResult<(String, String)> {
        Ok((array_to_string_fmt(points), array_to_string_fmt(cells)))
    }

    fn write_data(&mut self, _time: &str, data: &Values) -> IoResult<String> {
        Ok(self.values_to_string(data))
    }
}

pub trait FormatNumber {
    fn format_number(&self) -> String;
}

macro_rules! impl_format_number {
    ($t:ty, $format:expr) => {
        impl FormatNumber for $t {
            fn format_number(&self) -> String {
                format!($format, self)
            }
        }
    };
}

// Implement FormatNumber for various types
// taken from meshio
impl_format_number!(f32, "{:.7e}");
impl_format_number!(f64, "{:.16e}");
impl_format_number!(i8, "{}");
impl_format_number!(i16, "{}");
impl_format_number!(i32, "{}");
impl_format_number!(i64, "{}");
impl_format_number!(isize, "{}");
impl_format_number!(u8, "{}");
impl_format_number!(u16, "{}");
impl_format_number!(u32, "{}");
impl_format_number!(u64, "{}");
impl_format_number!(usize, "{}");

/// Generic formatter for ndarray arrays of either f64 or i32
pub fn array_to_string_fmt<T, D>(array: &ArrayView<T, D>) -> String
where
    T: FormatNumber,
    D: Dimension,
{
    array
        .iter()
        .map(|elem| elem.format_number())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::ArrayViewD;

    #[test]
    fn test_format() {
        assert_eq!(XmlWriter::new().format(), Format::XML);
    }

    #[test]
    fn test_write_mesh() {
        let mut writer = XmlWriter::new();
        let points = &ArrayView2::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap();
        let cells = ArrayView1::from(&[0, 1, 2, 0, 2, 3]);

        let result = writer.write_mesh(points, &cells).unwrap();
        assert_eq!(
            result,
            (
                "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0".to_string(),
                "0 1 2 0 2 3".to_string()
            )
        );
    }

    #[test]
    fn test_write_data_view_1d_f64() {
        let mut writer = XmlWriter::new();
        let raw_data = vec![1.0, 2.0, 3.0];
        let data = Values::View1Df64(ArrayView1::from(&raw_data));

        let result = writer.write_data("0.0", &data).unwrap();
        assert_eq!(
            result,
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0"
        );
    }

    #[test]
    fn test_write_data_view_2d_f64() {
        let mut writer = XmlWriter::new();
        let raw_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data: Values<'_> =
            Values::View2Df64(ArrayView2::from_shape((2, 3), &raw_data).unwrap());

        let result = writer.write_data("0.0", &data).unwrap();
        assert_eq!(
            result,
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0"
        );
    }

    #[test]
    fn test_write_data_view_dyn_f64() {
        let mut writer = XmlWriter::new();
        let raw_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data: Values<'_> =
            Values::ViewDynf64(ArrayViewD::from_shape(ndarray::IxDyn(&[3, 2]), &raw_data).unwrap());

        let result = writer.write_data("0.0", &data).unwrap();
        assert_eq!(
            result,
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0"
        );
    }
}
