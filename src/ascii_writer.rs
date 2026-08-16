//! Implementations of writers for ASCII data storage (inline and in separate files).

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::{
    DataStorage, DataWriter, Error, Result,
    error::io_ctx,
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
        points: &Values<'_>,
        cells: &[u64],
    ) -> Result<(DataContent, DataContent)> {
        Ok((
            values_to_string(points).into(),
            array_to_string_fmt(cells).into(),
        ))
    }

    fn write_data(
        &mut self,
        _name: &str,
        _center: attribute::Center,
        data: &Values<'_>,
    ) -> Result<DataContent> {
        Ok(values_to_string(data).into())
    }
}

/// This writer uses the XML format, but instead of writing the data directly into the xdmf file,
/// it writes it to a separate file and includes it in the xdmf file using an `xi:include` tag.
pub(crate) struct AsciiWriter {
    txt_files_dir: PathBuf,
    folder_name: PathBuf,
    write_time: Option<String>,
    // Files written for the time step currently in progress, so an abandoned step can remove
    // them again instead of leaving them behind unreferenced.
    step_files: Vec<PathBuf>,
}

impl AsciiWriter {
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self> {
        let txt_files_dir = file_name.as_ref().to_path_buf().with_extension("txt");

        let folder_name = txt_files_dir
            .file_name()
            .ok_or(Error::Internal("output path has no file name component"))?;

        crate::mpi_safe_create_dir_all(&txt_files_dir)?;

        Ok(Self {
            folder_name: folder_name.into(),
            txt_files_dir,
            write_time: None,
            step_files: Vec::new(),
        })
    }

    // Built with an explicit `/` rather than `PathBuf::join`/`to_string_lossy`, so the path
    // embedded in the XDMF file is valid on every OS regardless of which OS wrote it (e.g. no
    // backslashes from a Windows `PathBuf` ending up in a file read back on Linux).
    fn relative_path(&self, file_name: &str) -> String {
        format!("{}/{file_name}", self.folder_name.to_string_lossy())
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
        points: &Values<'_>,
        cells: &[u64],
    ) -> Result<(DataContent, DataContent)> {
        // create files for points and cells
        let points_file_name = "points.txt";
        let cells_file_name = "cells.txt";
        let points_path = self.txt_files_dir.join(points_file_name);
        let cells_path = self.txt_files_dir.join(cells_file_name);

        let mut file_points = BufWriter::new(
            File::create(&points_path).map_err(io_ctx("creating points file", &points_path))?,
        );
        let mut file_cells = BufWriter::new(
            File::create(&cells_path).map_err(io_ctx("creating cells file", &cells_path))?,
        );

        values_to_writer(points, &mut file_points)
            .map_err(io_ctx("writing points data", &points_path))?;
        array_to_writer_fmt(cells, &mut file_cells)
            .map_err(io_ctx("writing cells data", &cells_path))?;

        // explicitly flush the buffers to ensure all data is written and errors are caught
        file_points
            .flush()
            .map_err(io_ctx("flushing points file", &points_path))?;
        file_cells
            .flush()
            .map_err(io_ctx("flushing cells file", &cells_path))?;

        Ok((
            XInclude::new(self.relative_path(points_file_name), true).into(),
            XInclude::new(self.relative_path(cells_file_name), true).into(),
        ))
    }

    fn write_data(
        &mut self,
        name: &str,
        center: attribute::Center,
        data: &Values<'_>,
    ) -> Result<DataContent> {
        let time = self
            .write_time
            .as_ref()
            .ok_or(Error::Internal("writing data was not initialized"))?;

        let data_file_name = format!(
            "data_t_{time}_{}_{name}.txt",
            attribute::center_to_data_tag(center)
        );
        let data_path = self.txt_files_dir.join(&data_file_name);

        let data_file =
            File::create(&data_path).map_err(io_ctx("creating data file", &data_path))?;

        // Recorded once the file exists
        self.step_files.push(data_path.clone());

        let mut data_file = BufWriter::new(data_file);

        values_to_writer(data, &mut data_file).map_err(io_ctx("writing data", &data_path))?;

        // explicitly flush the buffers to ensure all data is written and errors are caught
        data_file
            .flush()
            .map_err(io_ctx("flushing data file", &data_path))?;

        Ok(XInclude::new(self.relative_path(&data_file_name), true).into())
    }

    fn write_data_initialize(&mut self, time: &str) -> Result<()> {
        if self.write_time.is_some() {
            return Err(Error::Internal("writing data was already initialized"));
        }

        self.write_time = Some(time.to_string());
        self.step_files.clear();
        Ok(())
    }

    fn write_data_finalize(&mut self) -> Result<()> {
        if self.write_time.is_none() {
            return Err(Error::Internal("writing data was not initialized"));
        }

        self.write_time = None;
        self.step_files.clear();
        Ok(())
    }

    fn write_data_discard(&mut self) -> Result<()> {
        if self.write_time.is_none() {
            return Err(Error::Internal("writing data was not initialized"));
        }

        // the step is over either way, so the writer is reset before the removals are reported
        self.write_time = None;

        crate::remove_step_files(&mut self.step_files)
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
pub fn array_to_writer_fmt<T, W>(vec: &[T], writer: &mut W) -> std::io::Result<()>
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

fn values_to_string(data: &Values<'_>) -> String {
    match data {
        Values::F64(v) => array_to_string_fmt(v),
        Values::F32(v) => array_to_string_fmt(v),
        Values::I64(v) => array_to_string_fmt(v),
        Values::I32(v) => array_to_string_fmt(v),
        Values::U64(v) => array_to_string_fmt(v),
        Values::U32(v) => array_to_string_fmt(v),
    }
}

fn values_to_writer(data: &Values<'_>, writer: &mut impl Write) -> std::io::Result<()> {
    match data {
        Values::F64(v) => array_to_writer_fmt(v, writer),
        Values::F32(v) => array_to_writer_fmt(v, writer),
        Values::I64(v) => array_to_writer_fmt(v, writer),
        Values::I32(v) => array_to_writer_fmt(v, writer),
        Values::U64(v) => array_to_writer_fmt(v, writer),
        Values::U32(v) => array_to_writer_fmt(v, writer),
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
        let data_f64: Values = vec![1.0, 2.0, 3.0].into();
        let result_f64 = values_to_string(&data_f64);
        assert_eq!(
            result_f64,
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0"
        );

        let data_f32: Values = vec![1.0_f32, 2.0, 3.0].into();
        let result_f32 = values_to_string(&data_f32);
        assert_eq!(result_f32, "1.0000000e0 2.0000000e0 3.0000000e0");

        let data_u64: Values = vec![1_u64, 2, 3].into();
        let result_u64 = values_to_string(&data_u64);
        assert_eq!(result_u64, "1 2 3");
    }

    #[test]
    fn values_to_writer_multiple_types() {
        let data_f64: Values = vec![1.0, 2.0, 3.0].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_f64, &mut buffer).unwrap();
        assert_eq!(
            String::from_utf8(buffer).unwrap(),
            "1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0\n"
        );

        let data_f32: Values = vec![1.0_f32, 2.0, 3.0].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_f32, &mut buffer).unwrap();
        assert_eq!(
            String::from_utf8(buffer).unwrap(),
            "1.0000000e0 2.0000000e0 3.0000000e0\n"
        );

        let data_u64: Values = vec![1_u64, 2, 3].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_u64, &mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), "1 2 3\n");
    }

    #[test]
    fn ascii_inline_writer_write_mesh() {
        let mut writer = AsciiInlineWriter::new();
        let points = vec![1., 2., 3., 4., 5., 6.];
        let cells = vec![0_u64, 1, 2, 0, 2, 3];

        let result = writer
            .write_mesh(&points.as_slice().into(), &cells)
            .unwrap();
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
    fn ascii_writer_write_data_discard() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = AsciiWriter::new(&file_name).unwrap();

        std::assert_matches!(
            writer.write_data_discard().unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        let txt_dir = file_name.with_extension("txt");
        let node_file = txt_dir.join("data_t_0.5_point_data_discarded.txt");
        let cell_file = txt_dir.join("data_t_0.5_cell_data_discarded.txt");

        writer.write_data_initialize("0.5").unwrap();
        writer
            .write_data(
                "discarded",
                attribute::Center::Node,
                &Values::F64(vec![1.0, 2.0].into()),
            )
            .unwrap();
        writer
            .write_data(
                "discarded",
                attribute::Center::Cell,
                &Values::F64(vec![3.0].into()),
            )
            .unwrap();
        assert!(node_file.exists());
        assert!(cell_file.exists());

        writer.write_data_discard().unwrap();

        // every file written for the step is removed, not just the last one
        assert!(!node_file.exists());
        assert!(!cell_file.exists());
        assert!(writer.write_time.is_none());
        assert!(writer.step_files.is_empty());

        // the time can be written again afterwards, and finalizing keeps what it wrote
        writer.write_data_initialize("0.5").unwrap();
        writer
            .write_data(
                "kept",
                attribute::Center::Node,
                &Values::F64(vec![4.0].into()),
            )
            .unwrap();
        writer.write_data_finalize().unwrap();

        assert!(txt_dir.join("data_t_0.5_point_data_kept.txt").exists());
        assert!(!node_file.exists());
    }

    #[test]
    fn ascii_writer_write_data_discard_removes_every_file_despite_a_failure() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = AsciiWriter::new(&file_name).unwrap();

        let txt_dir = file_name.with_extension("txt");
        let first_file = txt_dir.join("data_t_0.5_point_data_first.txt");
        let second_file = txt_dir.join("data_t_0.5_point_data_second.txt");

        writer.write_data_initialize("0.5").unwrap();
        for name in ["first", "second"] {
            writer
                .write_data(
                    name,
                    attribute::Center::Node,
                    &Values::F64(vec![1.0].into()),
                )
                .unwrap();
        }

        // removing the first file fails (it is already gone), which must not stop the second one
        // from being removed
        std::fs::remove_file(&first_file).unwrap();

        std::assert_matches!(
            writer.write_data_discard().unwrap_err(),
            Error::Io { operation, path, .. }
                if operation == "removing discarded data file" && path == first_file
        );

        assert!(!second_file.exists());
        assert!(writer.write_time.is_none());
        assert!(writer.step_files.is_empty());
    }

    #[test]
    fn ascii_writer_write_data_discard_after_a_failed_create() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = AsciiWriter::new(&file_name).unwrap();

        // a directory in the place of the data file makes `File::create` fail
        let txt_dir = file_name.with_extension("txt");
        std::fs::create_dir(txt_dir.join("data_t_0.5_point_data_boom.txt")).unwrap();

        writer.write_data_initialize("0.5").unwrap();
        std::assert_matches!(
            writer
                .write_data(
                    "boom",
                    attribute::Center::Node,
                    &Values::F64(vec![1.0].into()),
                )
                .unwrap_err(),
            Error::Io { operation, .. } if operation == "creating data file"
        );

        // no file was created, so the failed attribute recorded nothing and the discard has
        // nothing to remove -- recording the path any earlier would fail here instead
        writer.write_data_discard().unwrap();
    }

    #[test]
    fn ascii_writer_write_data_init_fin() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = AsciiWriter::new(file_name).unwrap();

        assert!(writer.write_time.is_none());

        let res_fin = writer.write_data_finalize();
        std::assert_matches!(
            res_fin.unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        let res_write = writer.write_data(
            "test_data",
            attribute::Center::Node,
            &Values::F64(vec![1.0, 2.0].into()),
        );
        std::assert_matches!(
            res_write.unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        writer.write_data_initialize("120.05").unwrap();
        assert_eq!(writer.write_time.clone().unwrap(), "120.05");

        let res_init = writer.write_data_initialize("0.0");
        std::assert_matches!(
            res_init.unwrap_err(),
            Error::Internal("writing data was already initialized")
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
        assert_eq!(writer.folder_name, PathBuf::from("test.txt"));
    }

    #[test]
    fn relative_path_uses_forward_slash() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let writer = AsciiWriter::new(file_name).unwrap();
        assert_eq!(writer.relative_path("data.txt"), "test.txt/data.txt");
        assert!(!writer.relative_path("data.txt").contains('\\'));
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
        let (points_path, cells_path) = writer
            .write_mesh(&points.as_slice().into(), &cells)
            .unwrap();
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
    fn ascii_writer_write_mesh_f32_points() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = AsciiWriter::new(file_name).unwrap();
        let points_file = writer.txt_files_dir.join("points.txt");

        let points = vec![0.0_f32, 1.0, 2.5];
        let cells = vec![0_u64, 1, 2];
        writer
            .write_mesh(&points.as_slice().into(), &cells)
            .unwrap();

        // f32 coordinates are written with f32's digit count, not f64's
        assert_eq!(
            std::fs::read_to_string(&points_file).unwrap(),
            "0.0000000e0 1.0000000e0 2.5000000e0\n"
        );
    }

    #[test]
    fn ascii_writer_write_data_f32() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = AsciiWriter::new(file_name).unwrap();

        writer.write_data_initialize("0.1").unwrap();
        let raw_data = vec![1.0_f32, 2.0, 3.0];
        let result = writer
            .write_data("temperature", attribute::Center::Node, &raw_data.into())
            .unwrap();

        assert_eq!(
            result,
            XInclude::new("test.txt/data_t_0.1_point_data_temperature.txt", true).into()
        );
        assert_eq!(
            std::fs::read_to_string(
                writer
                    .txt_files_dir
                    .join("data_t_0.1_point_data_temperature.txt")
            )
            .unwrap(),
            "1.0000000e0 2.0000000e0 3.0000000e0\n"
        );
    }

    #[test]
    fn ascii_inline_writer_write_mesh_f32_points() {
        let mut writer = AsciiInlineWriter::new();
        let points = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cells = vec![0_u64, 1, 2, 0, 2, 3];

        let result = writer
            .write_mesh(&points.as_slice().into(), &cells)
            .unwrap();
        pretty_assertions::assert_eq!(
            result,
            (
                "1.0000000e0 2.0000000e0 3.0000000e0 4.0000000e0 5.0000000e0 6.0000000e0".into(),
                "0 1 2 0 2 3".into()
            )
        );
    }

    #[test]
    fn ascii_inline_writer_write_data_vec_f32() {
        let mut writer = AsciiInlineWriter::new();
        let raw_data = vec![1.0_f32, 2.0, 3.0];
        let data = raw_data.into();

        let result = writer
            .write_data("dummy", attribute::Center::Node, &data)
            .unwrap();
        pretty_assertions::assert_eq!(result, "1.0000000e0 2.0000000e0 3.0000000e0".into());
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
                &Values::F64(data_points.as_slice().into()),
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
                &Values::F64(data_cells.as_slice().into()),
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
