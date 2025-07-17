use std::io::Result as IoResult;

use crate::{
    DataWriter,
    values::Values,
    xdmf_elements::{attribute, data_item::Format},
};

pub(crate) struct XmlWriter {}

impl XmlWriter {
    pub fn new() -> Self {
        Self {}
    }

    fn values_to_string(&self, data: &Values) -> String {
        match data {
            Values::F64(v) => array_to_string_fmt(v),
            Values::U64(v) => array_to_string_fmt(v),
        }
    }
}

impl DataWriter for XmlWriter {
    fn format(&self) -> Format {
        Format::XML
    }

    fn write_mesh(&mut self, points: &[f64], cells: &[u64]) -> IoResult<(String, String)> {
        Ok((array_to_string_fmt(points), array_to_string_fmt(cells)))
    }

    #[cfg(feature = "unstable-submesh-api")]
    fn write_submesh(
        &mut self,
        _name: &str,

        point_indices: &[u64],
        cell_indices: &[u64],
    ) -> IoResult<(String, String)> {
        Ok((
            array_to_string_fmt(point_indices),
            array_to_string_fmt(cell_indices),
        ))
    }

    fn write_data(
        &mut self,
        _name: &str,
        _center: attribute::Center,
        data: &Values,
    ) -> IoResult<String> {
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
pub fn array_to_string_fmt<T>(vec: &[T]) -> String
where
    T: FormatNumber,
{
    vec.iter()
        .map(|elem| elem.format_number())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format() {
        assert_eq!(XmlWriter::new().format(), Format::XML);
    }

    #[test]
    fn write_mesh() {
        let mut writer = XmlWriter::new();
        let points = vec![1., 2., 3., 4., 5., 6.];
        let cells = vec![0_u64, 1, 2, 0, 2, 3];

        let result = writer.write_mesh(&points, &cells).unwrap();
        pretty_assertions::assert_eq!(
            result,
            (
                "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0".to_string(),
                "0 1 2 0 2 3".to_string()
            )
        );
    }

    #[test]
    fn write_data_vec_f64() {
        let mut writer = XmlWriter::new();
        let raw_data = vec![1.0, 2.0, 3.0];
        let data = raw_data.into();

        let result = writer
            .write_data("dummy", attribute::Center::Node, &data)
            .unwrap();
        pretty_assertions::assert_eq!(
            result,
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0"
        );
    }
}
