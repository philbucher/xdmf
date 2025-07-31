use std::{
    io::Result as IoResult,
    path::{Path, PathBuf},
};

use crate::{
    DataStorage, DataWriter,
    values::Values,
    xdmf_elements::{attribute, data_item::Format},
};

pub(crate) struct AsciiInlineWriter {}

impl AsciiInlineWriter {
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

impl DataWriter for AsciiInlineWriter {
    fn format(&self) -> Format {
        Format::XML
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::AsciiInline
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

/// This writer uses the XML format, but instead of writing the data directly into the xdmf file,
/// it writes it to a separate file and includes it in the xdmf file using an `xi:include` tag.
pub(crate) struct AsciiDataWriter {
    txt_files_dir: PathBuf,
    write_time: Option<String>,
}

impl AsciiDataWriter {
    pub fn new(base_file_name: impl AsRef<Path>) -> IoResult<Self> {
        let txt_files_dir = base_file_name.as_ref().to_path_buf().with_extension("txt");

        crate::mpi_safe_create_dir_all(&txt_files_dir)?;

        Ok(Self {
            txt_files_dir,
            write_time: None,
        })
    }

    fn values_to_string(&self, data: &Values) -> String {
        match data {
            Values::F64(v) => array_to_string_fmt(v),
            Values::U64(v) => array_to_string_fmt(v),
        }
    }
}

impl DataWriter for AsciiDataWriter {
    fn format(&self) -> Format {
        Format::XML
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::Ascii
    }

    fn write_mesh(&mut self, points: &[f64], cells: &[u64]) -> IoResult<(String, String)> {
        // create files for points and cells
        std::fs::write("coords.txt", array_to_writer_fmt(points))?;
        std::fs::write("cells.txt", array_to_writer_fmt(cells))?;
        Ok((xinclude_data("coords.txt"), xinclude_data("cells.txt")))
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
        name: &str,
        center: attribute::Center,
        data: &Values,
    ) -> IoResult<String> {
        let time = self
            .write_time
            .as_ref()
            .ok_or_else(|| std::io::Error::other("Writing data was not initialized"))?;

        let file_name = self.txt_files_dir.join(format!(
            "data_t_{}_{}_{name}.txt",
            time,
            attribute::center_to_data_tag(center)
        ));

        // TODO write data to file

        Ok(xinclude_data(file_name.to_string_lossy().as_ref()))
    }

    fn write_data_initialize(&mut self, time: &str) -> IoResult<()> {
        if self.write_time.is_some() {
            return Err(std::io::Error::other(
                "Writing data was already initialized",
            ));
        }

        self.write_time = Some(time.to_string());
        Ok(())
    }

    fn write_data_finalize(&mut self) -> IoResult<()> {
        if self.write_time.is_none() {
            return Err(std::io::Error::other("Writing data was not initialized"));
        }

        self.write_time = None;
        Ok(())
    }
}

fn xinclude_data(name: &str) -> String {
    format!("<xi:include href=\"{name}\" parse=\"text\"/>")
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

/// Generic formatter for arrays of either f64 or i32
pub fn array_to_string_fmt<T>(vec: &[T]) -> String
where
    T: FormatNumber,
{
    vec.iter()
        .map(|elem| elem.format_number())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Generic formatter for arrays of either f64 or i32
pub fn array_to_writer_fmt<T>(vec: &[T]) -> String
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
        assert_eq!(AsciiInlineWriter::new().format(), Format::XML);
    }

    #[test]
    fn write_mesh() {
        let mut writer = AsciiInlineWriter::new();
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
        let mut writer = AsciiInlineWriter::new();
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
