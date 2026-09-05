//! Writers for ASCII data storage, inline and in separate files.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::{
    CELLS, DataStorage, DataWriter, Error, POINTS, Result, SUBMESH_CELLS, SUBMESH_POINTS,
    error::io_ctx,
    mesh_file_name,
    values::Values,
    xdmf_elements::data_item::{DataContent, Format, XInclude},
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

    // Which submesh this is plays no part here, for the points as for every other mesh array:
    // inline data is identified by where it sits in the XML, not by a name or a number of its own.
    fn write_points(
        &mut self,
        _submesh: Option<usize>,
        points: &Values<'_>,
    ) -> Result<DataContent> {
        Ok(values_to_string(points).into())
    }

    fn write_connectivity(
        &mut self,
        _submesh: Option<usize>,
        cells: &Values<'_>,
    ) -> Result<DataContent> {
        Ok(values_to_string(cells).into())
    }

    fn write_submesh_cells(&mut self, _submesh: usize, cells: &Values<'_>) -> Result<DataContent> {
        Ok(values_to_string(cells).into())
    }

    fn write_submesh_points(
        &mut self,
        _submesh: usize,
        points: &Values<'_>,
    ) -> Result<DataContent> {
        Ok(values_to_string(points).into())
    }

    fn write_data(&mut self, _index: usize, data: &Values<'_>) -> Result<DataContent> {
        Ok(values_to_string(data).into())
    }
}

/// Writes ASCII data to a separate file per array, referenced from the xdmf file with an
/// `xi:include` tag rather than inlined.
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

    // Shared by the points and every connectivity array, which differ only in the file they go to.
    fn write_mesh_file(&self, file_name: &str, values: &Values<'_>) -> Result<DataContent> {
        let path = self.txt_files_dir.join(file_name);

        let mut file =
            BufWriter::new(File::create(&path).map_err(io_ctx("creating mesh file", &path))?);

        values_to_writer(values, &mut file).map_err(io_ctx("writing mesh data", &path))?;

        // explicitly flush the buffer to ensure all data is written and errors are caught
        file.flush().map_err(io_ctx("flushing mesh file", &path))?;

        Ok(XInclude::new(self.relative_path(file_name), true).into())
    }
}

impl DataWriter for AsciiWriter {
    fn format(&self) -> Format {
        Format::XML
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::Ascii
    }

    fn write_points(&mut self, submesh: Option<usize>, points: &Values<'_>) -> Result<DataContent> {
        self.write_mesh_file(&mesh_file_name(POINTS, submesh, "txt"), points)
    }

    fn write_connectivity(
        &mut self,
        submesh: Option<usize>,
        cells: &Values<'_>,
    ) -> Result<DataContent> {
        self.write_mesh_file(&mesh_file_name(CELLS, submesh, "txt"), cells)
    }

    fn write_submesh_cells(&mut self, submesh: usize, cells: &Values<'_>) -> Result<DataContent> {
        self.write_mesh_file(&mesh_file_name(SUBMESH_CELLS, Some(submesh), "txt"), cells)
    }

    fn write_submesh_points(&mut self, submesh: usize, points: &Values<'_>) -> Result<DataContent> {
        self.write_mesh_file(
            &mesh_file_name(SUBMESH_POINTS, Some(submesh), "txt"),
            points,
        )
    }

    fn write_data(&mut self, index: usize, data: &Values<'_>) -> Result<DataContent> {
        let time = self
            .write_time
            .as_ref()
            .ok_or(Error::Internal("writing data was not initialized"))?;

        let data_file_name = format!("data_t_{time}_{index}.txt");
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

// `{:e}` rather than a fixed digit count: Rust's exponential formatting without a precision emits
// the *shortest* digit string that parses back to the same value, which is both exact and short.
// A fixed count has to be picked for the worst case -- 9 significant digits for f32, 17 for f64
// (`FLT_DECIMAL_DIG`/`DBL_DECIMAL_DIG`) -- and then pays it on every value, spelling 10.5 as
// "1.05000000e1" and 1.23456789 as "1.2345678899999999e0". Too few digits is worse than verbose
// though: 8 for f32 flips the last mantissa bit on roughly one value in a hundred, which is what
// `float_round_trip` guards, since the shortest-representation behaviour is what this relies on.
impl_format_number!(f32, "{:e}");
impl_format_number!(f64, "{:e}");
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

/// Formatter for arrays of scalar numeric types.
pub fn array_to_string_fmt<T>(vec: &[T]) -> String
where
    T: FormatNumber,
{
    vec.iter()
        .map(|elem| elem.format_number())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The same, written straight to `writer` instead of collected into a `String`.
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
        // floating point numbers: only as many digits as it takes to read the value back, so one
        // that is round in decimal stays short instead of carrying trailing zeros or roundoff
        let num: f32 = 3.141_590_4;
        assert_eq!(num.format_number(), "3.1415904e0");
        let num: f64 = 1.234_567_89;
        assert_eq!(num.format_number(), "1.23456789e0");
        let num: f64 = 10.5;
        assert_eq!(num.format_number(), "1.05e1");
        let num: f32 = 0.0;
        assert_eq!(num.format_number(), "0e0");

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

    // Deterministic bit patterns rather than a `rand` dependency, so a failure is reproducible.
    // Drawing the *bits* and reinterpreting is what makes this a real test: sweeping values by
    // arithmetic (`value *= 1.1`) walks a thin, well-behaved path through the space and lets a
    // digit count that is one too low pass, while random mantissas hit the awkward cases at the
    // rate they actually occur -- about one f32 in a hundred.
    fn pseudo_random_bits(seed: u64) -> impl Iterator<Item = u64> {
        let mut state = seed;
        std::iter::repeat_with(move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        })
    }

    // The point of the digit counts above is that a reader gets the value that was written back,
    // bit for bit. Asserting the formatted text alone would not catch a count that is one too low:
    // the output still looks like a plausible float, it is just no longer the same one.
    #[test]
    fn float_round_trip() {
        let mut checked_f32 = 0;
        for bits in pseudo_random_bits(1).take(200_000) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the low 32 bits are the sample; this is a bit pattern, not a number"
            )]
            let value = f32::from_bits(bits as u32);
            if !value.is_finite() {
                continue;
            }
            let parsed: f32 = value.format_number().parse().unwrap();
            assert_eq!(
                parsed.to_bits(),
                value.to_bits(),
                "f32 {value:e} did not survive '{}'",
                value.format_number()
            );
            checked_f32 += 1;
        }

        let mut checked_f64 = 0;
        for bits in pseudo_random_bits(2).take(200_000) {
            let value = f64::from_bits(bits);
            if !value.is_finite() {
                continue;
            }
            let parsed: f64 = value.format_number().parse().unwrap();
            assert_eq!(
                parsed.to_bits(),
                value.to_bits(),
                "f64 {value:e} did not survive '{}'",
                value.format_number()
            );
            checked_f64 += 1;
        }

        // most bit patterns are finite, so a filter that started rejecting everything would show up
        assert!(checked_f32 > 100_000 && checked_f64 > 100_000);

        // and the extremes of each type, where the exponent is widest
        for value in [
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            0.0,
            -0.0,
        ] {
            let parsed: f32 = value.format_number().parse().unwrap();
            assert_eq!(parsed.to_bits(), value.to_bits());
        }
        for value in [
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            0.0,
            -0.0,
        ] {
            let parsed: f64 = value.format_number().parse().unwrap();
            assert_eq!(parsed.to_bits(), value.to_bits());
        }
    }

    #[test]
    fn array_to_string_fmt_multiple_types() {
        let vec_f64 = vec![1.0, 2.0, 3.0];
        let result_f64 = array_to_string_fmt(&vec_f64);
        assert_eq!(result_f64, "1e0 2e0 3e0");

        let vec_u64 = vec![1_u64, 2, 3];
        let result_u64 = array_to_string_fmt(&vec_u64);
        assert_eq!(result_u64, "1 2 3");
    }

    #[test]
    fn array_to_writer_fmt_multiple_types() {
        let vec_f64 = vec![1.0, 2.0, 3.0];
        let mut buffer = Vec::new();
        array_to_writer_fmt(&vec_f64, &mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), "1e0 2e0 3e0\n");

        let vec_u64 = vec![1_u64, 2, 3];
        let mut buffer = Vec::new();
        array_to_writer_fmt(&vec_u64, &mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), "1 2 3\n");
    }

    #[test]
    fn values_to_string_multiple_types() {
        let data_f64: Values = vec![1.0, 2.0, 3.0].into();
        let result_f64 = values_to_string(&data_f64);
        assert_eq!(result_f64, "1e0 2e0 3e0");

        let data_f32: Values = vec![1.0_f32, 2.0, 3.0].into();
        let result_f32 = values_to_string(&data_f32);
        assert_eq!(result_f32, "1e0 2e0 3e0");

        let data_u64: Values = vec![1_u64, 2, 3].into();
        let result_u64 = values_to_string(&data_u64);
        assert_eq!(result_u64, "1 2 3");
    }

    #[test]
    fn values_to_writer_multiple_types() {
        let data_f64: Values = vec![1.0, 2.0, 3.0].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_f64, &mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), "1e0 2e0 3e0\n");

        let data_f32: Values = vec![1.0_f32, 2.0, 3.0].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_f32, &mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), "1e0 2e0 3e0\n");

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

        let points_data = writer
            .write_points(None, &points.as_slice().into())
            .unwrap();
        let cells_data = writer
            .write_connectivity(None, &cells.as_slice().into())
            .unwrap();
        pretty_assertions::assert_eq!(points_data, "1e0 2e0 3e0 4e0 5e0 6e0".into());
        pretty_assertions::assert_eq!(cells_data, "0 1 2 0 2 3".into());
    }

    #[test]
    fn ascii_inline_writer_write_data_vec_f64() {
        let mut writer = AsciiInlineWriter::new();
        let raw_data = vec![1.0, 2.0, 3.0];
        let data = raw_data.into();

        let result = writer.write_data(0, &data).unwrap();
        pretty_assertions::assert_eq!(result, "1e0 2e0 3e0".into());
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
        let first_file = txt_dir.join("data_t_0.5_0.txt");
        let second_file = txt_dir.join("data_t_0.5_1.txt");

        writer.write_data_initialize("0.5").unwrap();
        writer
            .write_data(0, &Values::F64(vec![1.0, 2.0].into()))
            .unwrap();
        writer
            .write_data(1, &Values::F64(vec![3.0].into()))
            .unwrap();
        assert!(first_file.exists());
        assert!(second_file.exists());

        writer.write_data_discard().unwrap();

        // every file written for the step is removed, not just the last one
        assert!(!first_file.exists());
        assert!(!second_file.exists());
        assert!(writer.write_time.is_none());
        assert!(writer.step_files.is_empty());

        // the time can be written again afterwards, and finalizing keeps what it wrote. A
        // different array number, so the file it keeps is distinguishable from the discarded one
        writer.write_data_initialize("0.5").unwrap();
        writer
            .write_data(2, &Values::F64(vec![4.0].into()))
            .unwrap();
        writer.write_data_finalize().unwrap();

        assert!(txt_dir.join("data_t_0.5_2.txt").exists());
        assert!(!first_file.exists());
    }

    #[test]
    fn ascii_writer_write_data_discard_removes_every_file_despite_a_failure() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = AsciiWriter::new(&file_name).unwrap();

        let txt_dir = file_name.with_extension("txt");
        let first_file = txt_dir.join("data_t_0.5_0.txt");
        let second_file = txt_dir.join("data_t_0.5_1.txt");

        writer.write_data_initialize("0.5").unwrap();
        for index in [0, 1] {
            writer
                .write_data(index, &Values::F64(vec![1.0].into()))
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
        std::fs::create_dir(txt_dir.join("data_t_0.5_0.txt")).unwrap();

        writer.write_data_initialize("0.5").unwrap();
        std::assert_matches!(
            writer
                .write_data(0, &Values::F64(vec![1.0].into()))
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

        let res_write = writer.write_data(0, &Values::F64(vec![1.0, 2.0].into()));
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
        let cells = vec![0_u64, 1, 2];
        let points_path = writer
            .write_points(None, &points.as_slice().into())
            .unwrap();
        let cells_path = writer
            .write_connectivity(None, &cells.as_slice().into())
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

        assert_eq!(points_data, "0e0 1e0 2e0\n");
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
            .write_points(None, &points.as_slice().into())
            .unwrap();
        writer
            .write_connectivity(None, &cells.as_slice().into())
            .unwrap();

        // f32 coordinates are written with f32's digit count, not f64's
        assert_eq!(
            std::fs::read_to_string(&points_file).unwrap(),
            "0e0 1e0 2.5e0\n"
        );
    }

    #[test]
    fn ascii_writer_write_data_f32() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = AsciiWriter::new(file_name).unwrap();

        writer.write_data_initialize("0.1").unwrap();
        let raw_data = vec![1.0_f32, 2.0, 3.0];
        let result = writer.write_data(0, &raw_data.into()).unwrap();

        assert_eq!(
            result,
            XInclude::new("test.txt/data_t_0.1_0.txt", true).into()
        );
        assert_eq!(
            std::fs::read_to_string(writer.txt_files_dir.join("data_t_0.1_0.txt")).unwrap(),
            "1e0 2e0 3e0\n"
        );
    }

    #[test]
    fn ascii_inline_writer_write_mesh_f32_points() {
        let mut writer = AsciiInlineWriter::new();
        let points = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cells = vec![0_u64, 1, 2, 0, 2, 3];

        let points_data = writer
            .write_points(None, &points.as_slice().into())
            .unwrap();
        let cells_data = writer
            .write_connectivity(None, &cells.as_slice().into())
            .unwrap();
        pretty_assertions::assert_eq!(points_data, "1e0 2e0 3e0 4e0 5e0 6e0".into());
        pretty_assertions::assert_eq!(cells_data, "0 1 2 0 2 3".into());
    }

    #[test]
    fn ascii_inline_writer_write_data_vec_f32() {
        let mut writer = AsciiInlineWriter::new();
        let raw_data = vec![1.0_f32, 2.0, 3.0];
        let data = raw_data.into();

        let result = writer.write_data(0, &data).unwrap();
        pretty_assertions::assert_eq!(result, "1e0 2e0 3e0".into());
    }

    #[test]
    fn ascii_writer_write_data() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = AsciiWriter::new(file_name).unwrap();
        let write_time = "12.258";
        let points_index = 0;
        let cells_index = 1;
        let data_file_points = writer
            .txt_files_dir
            .join(format!("data_t_{write_time}_{points_index}.txt"));
        let data_file_cells = writer
            .txt_files_dir
            .join(format!("data_t_{write_time}_{cells_index}.txt"));
        assert!(!data_file_points.exists());
        assert!(!data_file_cells.exists());

        writer.write_data_initialize(write_time).unwrap();
        assert!(!data_file_points.exists());
        assert!(!data_file_cells.exists());

        // write points data
        let data_points = vec![0.0, 1.0, 2.0];
        let data_path_points = writer
            .write_data(points_index, &Values::F64(data_points.as_slice().into()))
            .unwrap();

        assert!(data_file_points.exists());
        assert!(!data_file_cells.exists());

        // write cell data
        let data_cells = vec![-9.0, 1.0, 2.0, 55.87];
        let data_path_cells = writer
            .write_data(cells_index, &Values::F64(data_cells.as_slice().into()))
            .unwrap();
        assert!(data_file_points.exists());
        assert!(data_file_cells.exists());

        writer.write_data_finalize().unwrap();

        assert_eq!(
            data_path_points,
            XInclude::new("test.txt/data_t_12.258_0.txt", true).into()
        );
        assert_eq!(
            data_path_cells,
            XInclude::new("test.txt/data_t_12.258_1.txt", true).into()
        );

        // read back the data to verify
        let points_data = std::fs::read_to_string(&data_file_points).unwrap();
        let cells_data = std::fs::read_to_string(&data_file_cells).unwrap();

        assert_eq!(points_data, "0e0 1e0 2e0\n");
        assert_eq!(cells_data, "-9e0 1e0 2e0 5.587e1\n");
    }
}
