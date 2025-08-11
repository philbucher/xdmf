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
    folder_name: PathBuf,
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
            folder_name: raw_file_name.into(),
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
                self.folder_name.join(points_file_name).to_string_lossy(),
                true,
            )
            .into(),
            XInclude::new(
                self.folder_name.join(cells_file_name).to_string_lossy(),
                true,
            )
            .into(),
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

        Ok(XInclude::new(
            self.folder_name.join(data_file_name).to_string_lossy(),
            true,
        )
        .into())
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
    use crate::xdmf_elements::data_item::XInclude;

    #[test]
    fn format_number_all_types() {
        // floating point numbers
        let num: f32 = 3.141_590_4;
        assert_eq!(num.format_number(), "3.1415904e0");
        let num: f64 = 1.234_567_89;
        assert_eq!(num.format_number(), "1.2345678899999999e0");

        // signed integer types
        let num: i8 = -5;
        assert_eq!(num.format_number(), "-5");
        let num: i16 = -32768;
        assert_eq!(num.format_number(), "-32768");
        let num: i32 = 42;
        assert_eq!(num.format_number(), "42");
        let num: i64 = -1_234_567_890_123_456_789;
        assert_eq!(num.format_number(), "-1234567890123456789");
        let num: isize = -987_654_321;
        assert_eq!(num.format_number(), "-987654321");

        // unsigned integer types
        let num: u8 = 255;
        assert_eq!(num.format_number(), "255");
        let num: u16 = 65535;
        assert_eq!(num.format_number(), "65535");
        let num: u32 = 4_294_967_295;
        assert_eq!(num.format_number(), "4294967295");
        let num: u64 = 1000;
        assert_eq!(num.format_number(), "1000");
        let num: usize = 123_456_789;
        assert_eq!(num.format_number(), "123456789");
    }

    #[test]
    fn array_to_string_fmt_multiple_types() {
        let vec_f64 = vec![1.0, 2.0, 3.0];
        let result_f64 = array_to_string_fmt(&vec_f64);
        assert_eq!(
            result_f64,
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0"
        );

        let vec_u64 = vec![1_u64, 2, 3];
        let result_u64 = array_to_string_fmt(&vec_u64);
        assert_eq!(result_u64, "1 2 3");
    }

    #[test]
    fn array_to_writer_fmt_multiple_types() {
        let vec_f64 = vec![1.0, 2.0, 3.0];
        let mut buffer = Vec::new();
        array_to_writer_fmt(&vec_f64, &mut buffer).unwrap();
        assert_eq!(
            String::from_utf8(buffer).unwrap(),
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0\n"
        );

        let vec_u64 = vec![1_u64, 2, 3];
        let mut buffer = Vec::new();
        array_to_writer_fmt(&vec_u64, &mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), "1 2 3\n");
    }

    #[test]
    fn values_to_string_multiple_types() {
        let data_f64 = Values::F64(vec![1.0, 2.0, 3.0]);
        let result_f64 = values_to_string(&data_f64);
        assert_eq!(
            result_f64,
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0"
        );

        let data_u64 = Values::U64(vec![1_u64, 2, 3]);
        let result_u64 = values_to_string(&data_u64);
        assert_eq!(result_u64, "1 2 3");
    }

    #[test]
    fn values_to_writer_multiple_types() {
        let data_f64 = Values::F64(vec![1.0, 2.0, 3.0]);
        let mut buffer = Vec::new();
        values_to_writer(&data_f64, &mut buffer).unwrap();
        assert_eq!(
            String::from_utf8(buffer).unwrap(),
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0\n"
        );

        let data_u64 = Values::U64(vec![1_u64, 2, 3]);
        let mut buffer = Vec::new();
        values_to_writer(&data_u64, &mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), "1 2 3\n");
    }

    #[test]
    fn ascii_inline_writer_write_mesh() {
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
    fn ascii_inline_writer_write_data_vec_f64() {
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

    #[test]
    fn ascii_writer_write_data_init_fin() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = AsciiWriter::new(file_name).unwrap();

        assert!(writer.write_time.is_none());

        let res_fin = writer.write_data_finalize();
        assert_eq!(
            res_fin.unwrap_err().to_string(),
            "Writing data was not initialized"
        );

        let res_write = writer.write_data(
            "test_data",
            attribute::Center::Node,
            &Values::F64(vec![1.0, 2.0]),
        );
        assert_eq!(
            res_write.unwrap_err().to_string(),
            "Writing data was not initialized"
        );

        writer.write_data_initialize("120.05").unwrap();
        assert_eq!(writer.write_time.clone().unwrap(), "120.05");

        let res_init = writer.write_data_initialize("0.0");
        assert_eq!(
            res_init.unwrap_err().to_string(),
            "Writing data was already initialized"
        );

        writer.write_data_finalize().unwrap();
        assert!(writer.write_time.is_none());
    }

    #[test]
    fn ascii_writer_new() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let writer = AsciiWriter::new(&file_name).unwrap();
        let exp_dir_name = file_name.with_extension("txt");
        assert_eq!(writer.txt_files_dir, exp_dir_name);
        assert!(writer.txt_files_dir.exists());
        assert!(writer.txt_files_dir.is_dir());
        assert!(writer.write_time.is_none());
    }

    #[test]
    fn ascii_writer_write_mesh() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = AsciiWriter::new(file_name).unwrap();
        let points_file = writer.txt_files_dir.join("points.txt");
        let cells_file = writer.txt_files_dir.join("cells.txt");
        assert!(!points_file.exists());
        assert!(!cells_file.exists());

        let points = vec![0.0, 1.0, 2.0];
        let cells = vec![0, 1, 2];
        let (points_path, cells_path) = writer.write_mesh(&points, &cells).unwrap();
        assert!(points_file.exists());
        assert!(cells_file.exists());

        assert_eq!(
            points_path,
            XInclude::new("test.txt/points.txt", true).into()
        );
        assert_eq!(cells_path, XInclude::new("test.txt/cells.txt", true).into());

        // read back the data to verify
        let points_data = std::fs::read_to_string(&points_file).unwrap();
        let cells_data = std::fs::read_to_string(&cells_file).unwrap();

        assert_eq!(
            points_data,
            "0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0\n"
        );
        assert_eq!(cells_data, "0 1 2\n");
    }

    #[test]
    fn ascii_writer_write_data() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = AsciiWriter::new(file_name).unwrap();
        let write_time = "12.258";
        let point_data_name = "dummy_point_data";
        let cell_data_name = "some_cell_data";
        let data_file_points = writer.txt_files_dir.join(format!(
            "data_t_{write_time}_point_data_{point_data_name}.txt"
        ));
        let data_file_cells = writer.txt_files_dir.join(format!(
            "data_t_{write_time}_cell_data_{cell_data_name}.txt"
        ));
        assert!(!data_file_points.exists());
        assert!(!data_file_cells.exists());

        writer.write_data_initialize(write_time).unwrap();
        assert!(!data_file_points.exists());
        assert!(!data_file_cells.exists());

        // write points data
        let data_points = vec![0.0, 1.0, 2.0];
        let data_path_points = writer
            .write_data(
                point_data_name,
                attribute::Center::Node,
                &Values::F64(data_points),
            )
            .unwrap();

        assert!(data_file_points.exists());
        assert!(!data_file_cells.exists());

        // write cell data
        let data_cells = vec![-9.0, 1.0, 2.0, 55.87];
        let data_path_cells = writer
            .write_data(
                "some_cell_data",
                attribute::Center::Cell,
                &Values::F64(data_cells),
            )
            .unwrap();
        assert!(data_file_points.exists());
        assert!(data_file_cells.exists());

        writer.write_data_finalize().unwrap();

        assert_eq!(
            data_path_points,
            XInclude::new(
                "test.txt/data_t_12.258_point_data_dummy_point_data.txt",
                true
            )
            .into()
        );
        assert_eq!(
            data_path_cells,
            XInclude::new("test.txt/data_t_12.258_cell_data_some_cell_data.txt", true).into()
        );

        // read back the data to verify
        let points_data = std::fs::read_to_string(&data_file_points).unwrap();
        let cells_data = std::fs::read_to_string(&data_file_cells).unwrap();

        assert_eq!(
            points_data,
            "0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0\n"
        );
        assert_eq!(
            cells_data,
            "-9.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0 5.5869999999999997e1\n"
        );
    }
}
