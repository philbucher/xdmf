use std::{
    fs::File,
    io::{BufWriter, Result as IoResult, Write},
    path::{Path, PathBuf},
};

use crate::{
    DataStorage, DataWriter,
    values::Values,
    xdmf_elements::{
        attribute,
        data_item::{DataContent, Format, XInclude},
    },
};

pub(crate) struct AsciiInlineWriter {}

impl AsciiInlineWriter {
    pub fn new() -> Self {
        Self {}
    }
}

impl DataWriter for AsciiInlineWriter {
    fn format(&self) -> Format {
        Format::XML
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::AsciiInline
    }

    fn write_mesh(
        &mut self,
        points: &[f64],
        cells: &[u64],
    ) -> IoResult<(DataContent, DataContent)> {
        Ok((
            array_to_string_fmt(points).into(),
            array_to_string_fmt(cells).into(),
        ))
    }

    #[cfg(feature = "unstable-submesh-api")]
    fn write_submesh(
        &mut self,
        _name: &str,
        point_indices: &[u64],
        cell_indices: &[u64],
    ) -> IoResult<(String, String)> {
        unimplemented!()
    }

    fn write_data(
        &mut self,
        _name: &str,
        _center: attribute::Center,
        data: &Values,
    ) -> IoResult<DataContent> {
        Ok(values_to_string(data).into())
    }
}

/// This writer uses the XML format, but instead of writing the data directly into the xdmf file,
/// it writes it to a separate file and includes it in the xdmf file using an `xi:include` tag.
pub(crate) struct AsciiWriter {
    txt_files_dir: PathBuf,
    file_name: PathBuf,
    write_time: Option<String>,
}

impl AsciiWriter {
    pub fn new(base_file_name: impl AsRef<Path>) -> IoResult<Self> {
        let txt_files_dir = base_file_name.as_ref().to_path_buf().with_extension("txt");

        let raw_file_name = txt_files_dir.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Base file name must have a valid file name",
            )
        })?;

        crate::mpi_safe_create_dir_all(&txt_files_dir)?;

        Ok(Self {
            file_name: PathBuf::from(raw_file_name),
            txt_files_dir,
            write_time: None,
        })
    }
}

impl DataWriter for AsciiWriter {
    fn format(&self) -> Format {
        Format::XML
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::Ascii
    }

    fn write_mesh(
        &mut self,
        points: &[f64],
        cells: &[u64],
    ) -> IoResult<(DataContent, DataContent)> {
        // create files for points and cells
        let points_file_name = "points.txt";
        let cells_file_name = "cells.txt";

        let mut file_points =
            BufWriter::new(File::create(self.txt_files_dir.join(points_file_name))?);
        let mut file_cells =
            BufWriter::new(File::create(self.txt_files_dir.join(cells_file_name))?);

        array_to_writer_fmt(points, &mut file_points)?;
        array_to_writer_fmt(cells, &mut file_cells)?;

        // explicitly flush the buffers to ensure all data is written and errors are caught
        file_points.flush()?;
        file_cells.flush()?;

        Ok((
            XInclude::new(
                self.file_name.join(points_file_name).to_string_lossy(),
                true,
            )
            .into(),
            XInclude::new(self.file_name.join(cells_file_name).to_string_lossy(), true).into(),
        ))
    }

    #[cfg(feature = "unstable-submesh-api")]
    fn write_submesh(
        &mut self,
        _name: &str,
        point_indices: &[u64],
        cell_indices: &[u64],
    ) -> IoResult<(String, String)> {
        unimplemented!()
    }

    fn write_data(
        &mut self,
        name: &str,
        center: attribute::Center,
        data: &Values,
    ) -> IoResult<DataContent> {
        let time = self
            .write_time
            .as_ref()
            .ok_or_else(|| std::io::Error::other("Writing data was not initialized"))?;

        let data_file_name = format!(
            "data_t_{time}_{}_{name}.txt",
            attribute::center_to_data_tag(center)
        );

        let mut data_file = BufWriter::new(File::create(self.txt_files_dir.join(&data_file_name))?);

        values_to_writer(data, &mut data_file)?;

        // explicitly flush the buffers to ensure all data is written and errors are caught
        data_file.flush()?;

        Ok(XInclude::new(self.file_name.join(data_file_name).to_string_lossy(), true).into())
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

/// Generic formatter for arrays of scalar numeric types
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
pub fn array_to_writer_fmt<T, W>(vec: &[T], writer: &mut W) -> IoResult<()>
where
    T: FormatNumber,
    W: Write,
{
    let mut iter = vec.iter().peekable();

    while let Some(elem) = iter.next() {
        write!(writer, "{}", elem.format_number())?;
        if iter.peek().is_some() {
            write!(writer, " ")?;
        }
    }

    // final newline
    writeln!(writer)
}

fn values_to_string(data: &Values) -> String {
    match data {
        Values::F64(v) => array_to_string_fmt(v),
        Values::U64(v) => array_to_string_fmt(v),
    }
}

fn values_to_writer(data: &Values, writer: &mut impl Write) -> IoResult<()> {
    match data {
        Values::F64(v) => array_to_writer_fmt(v, writer),
        Values::U64(v) => array_to_writer_fmt(v, writer),
    }
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
                "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0".into(),
                "0 1 2 0 2 3".into()
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
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0".into()
        );
    }
}
