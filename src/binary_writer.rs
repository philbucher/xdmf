//! Implementation of a writer for uncompressed raw binary data storage in separate files.

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
        data_item::{DataContent, Format},
    },
};

/// Writes uncompressed, little-endian raw binary data to separate files, referenced from the
/// XDMF file by a plain relative path.
pub(crate) struct BinaryWriter {
    bin_files_dir: PathBuf,
    folder_name: PathBuf,
    write_time: Option<String>,
}

impl BinaryWriter {
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self> {
        let bin_files_dir = file_name.as_ref().to_path_buf().with_extension("bin");

        let folder_name = bin_files_dir
            .file_name()
            .ok_or(Error::Internal("output path has no file name component"))?;

        crate::mpi_safe_create_dir_all(&bin_files_dir)?;

        Ok(Self {
            folder_name: folder_name.into(),
            bin_files_dir,
            write_time: None,
        })
    }

    // Built with an explicit `/` rather than `PathBuf::join`/`to_string_lossy`, so the path
    // embedded in the XDMF file is valid on every OS regardless of which OS wrote it (e.g. no
    // backslashes from a Windows `PathBuf` ending up in a file read back on Linux).
    fn relative_path(&self, file_name: &str) -> String {
        format!("{}/{file_name}", self.folder_name.to_string_lossy())
    }
}

impl DataWriter for BinaryWriter {
    fn format(&self) -> Format {
        Format::Binary
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::Binary
    }

    fn write_mesh(&mut self, points: &[f64], cells: &[u64]) -> Result<(DataContent, DataContent)> {
        let points_file_name = "points.bin";
        let cells_file_name = "cells.bin";
        let points_path = self.bin_files_dir.join(points_file_name);
        let cells_path = self.bin_files_dir.join(cells_file_name);

        let mut file_points = BufWriter::new(
            File::create(&points_path).map_err(io_ctx("creating points file", &points_path))?,
        );
        let mut file_cells = BufWriter::new(
            File::create(&cells_path).map_err(io_ctx("creating cells file", &cells_path))?,
        );

        write_f64_le(points, &mut file_points, &points_path)?;
        write_u64_as_u32_le(cells, &mut file_cells, &cells_path)?;

        // explicitly flush the buffers to ensure all data is written and errors are caught
        file_points
            .flush()
            .map_err(io_ctx("flushing points file", &points_path))?;
        file_cells
            .flush()
            .map_err(io_ctx("flushing cells file", &cells_path))?;

        Ok((
            self.relative_path(points_file_name).into(),
            self.relative_path(cells_file_name).into(),
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
            "data_t_{time}_{}_{name}.bin",
            attribute::center_to_data_tag(center)
        );
        let data_path = self.bin_files_dir.join(&data_file_name);

        let mut data_file = BufWriter::new(
            File::create(&data_path).map_err(io_ctx("creating data file", &data_path))?,
        );

        values_to_writer(data, &mut data_file, &data_path)?;

        // explicitly flush the buffers to ensure all data is written and errors are caught
        data_file
            .flush()
            .map_err(io_ctx("flushing data file", &data_path))?;

        Ok(self.relative_path(&data_file_name).into())
    }

    fn write_data_initialize(&mut self, time: &str) -> Result<()> {
        if self.write_time.is_some() {
            return Err(Error::Internal("writing data was already initialized"));
        }

        self.write_time = Some(time.to_string());
        Ok(())
    }

    fn write_data_finalize(&mut self) -> Result<()> {
        if self.write_time.is_none() {
            return Err(Error::Internal("writing data was not initialized"));
        }

        self.write_time = None;
        Ok(())
    }

    fn validate_values(&self, data: &Values<'_>) -> Result<()> {
        if let Values::U64(v) = data {
            for &value in v.iter() {
                checked_u32(value)?;
            }
        }
        Ok(())
    }
}

fn write_f64_le(vec: &[f64], writer: &mut impl Write, path: &Path) -> Result<()> {
    for &v in vec {
        writer
            .write_all(&v.to_le_bytes())
            .map_err(io_ctx("writing binary data", path))?;
    }
    Ok(())
}

// Integers are narrowed to 4 bytes (matching `Format::uint_precision()` in the `DataItem`) because
// Paraview's legacy Xdmf reader silently misreads 64-bit ones, see `Error::IntegerTooLargeForBinary`.
// Values that don't fit are rejected rather than silently truncated.
fn checked_u32(v: u64) -> Result<u32> {
    u32::try_from(v).map_err(|_err| Error::IntegerTooLargeForBinary { value: v })
}

fn write_u64_as_u32_le(vec: &[u64], writer: &mut impl Write, path: &Path) -> Result<()> {
    for &v in vec {
        let v32 = checked_u32(v)?;
        writer
            .write_all(&v32.to_le_bytes())
            .map_err(io_ctx("writing binary data", path))?;
    }
    Ok(())
}

fn values_to_writer(data: &Values<'_>, writer: &mut impl Write, path: &Path) -> Result<()> {
    match data {
        Values::F64(v) => write_f64_le(v, writer, path),
        Values::U64(v) => write_u64_as_u32_le(v, writer, path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_f64_le_multiple_values() {
        let vec_f64 = vec![1.0_f64, -2.5];
        let mut buffer = Vec::new();
        write_f64_le(&vec_f64, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0_f64.to_le_bytes());
        expected.extend_from_slice(&(-2.5_f64).to_le_bytes());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn write_u64_as_u32_le_multiple_values() {
        let vec_u64 = vec![1_u64, 2];
        let mut buffer = Vec::new();
        write_u64_as_u32_le(&vec_u64, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1_u32.to_le_bytes());
        expected.extend_from_slice(&2_u32.to_le_bytes());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn write_u64_as_u32_le_rejects_values_too_large_for_u32() {
        let vec_u64 = vec![1_u64, u64::from(u32::MAX) + 1];
        let mut buffer = Vec::new();
        let res = write_u64_as_u32_le(&vec_u64, &mut buffer, Path::new("test.bin"));
        std::assert_matches!(
            res.unwrap_err(),
            Error::IntegerTooLargeForBinary {
                value: 4_294_967_296
            }
        );
    }

    #[test]
    fn values_to_writer_multiple_types() {
        let data_f64: Values = vec![1.0, 2.0].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_f64, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0_f64.to_le_bytes());
        expected.extend_from_slice(&2.0_f64.to_le_bytes());
        assert_eq!(buffer, expected);

        let data_u64: Values = vec![1_u64, 2].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_u64, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1_u32.to_le_bytes());
        expected.extend_from_slice(&2_u32.to_le_bytes());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn binary_writer_new() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let writer = BinaryWriter::new(&file_name).unwrap();
        let exp_dir_name = file_name.with_extension("bin");
        assert_eq!(writer.bin_files_dir, exp_dir_name);
        assert!(writer.bin_files_dir.exists());
        assert!(writer.bin_files_dir.is_dir());
        assert!(writer.write_time.is_none());
        assert_eq!(writer.folder_name, PathBuf::from("test.bin"));
    }

    #[test]
    fn binary_writer_write_mesh() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = BinaryWriter::new(file_name).unwrap();
        let points_file = writer.bin_files_dir.join("points.bin");
        let cells_file = writer.bin_files_dir.join("cells.bin");
        assert!(!points_file.exists());
        assert!(!cells_file.exists());

        let points = vec![0.0, 1.0, 2.0];
        let cells = vec![0_u64, 1, 2];
        let (points_content, cells_content) = writer.write_mesh(&points, &cells).unwrap();
        assert!(points_file.exists());
        assert!(cells_file.exists());

        assert_eq!(points_content, "test.bin/points.bin".into());
        assert_eq!(cells_content, "test.bin/cells.bin".into());

        // read back the raw bytes to verify
        let points_bytes = std::fs::read(&points_file).unwrap();
        let cells_bytes = std::fs::read(&cells_file).unwrap();

        let mut expected_points = Vec::new();
        for p in &points {
            expected_points.extend_from_slice(&p.to_le_bytes());
        }
        let mut expected_cells = Vec::new();
        for &c in &cells {
            expected_cells.extend_from_slice(&(c as u32).to_le_bytes());
        }

        assert_eq!(points_bytes, expected_points);
        assert_eq!(cells_bytes, expected_cells);
    }

    #[test]
    fn binary_writer_write_data() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = BinaryWriter::new(file_name).unwrap();

        writer.write_data_initialize("1.5").unwrap();

        let data: Values = vec![1.0, -2.0, 3.5].into();
        let content = writer
            .write_data("temperature", attribute::Center::Node, &data)
            .unwrap();

        assert_eq!(
            content,
            "test.bin/data_t_1.5_point_data_temperature.bin".into()
        );

        let bytes = std::fs::read(
            writer
                .bin_files_dir
                .join("data_t_1.5_point_data_temperature.bin"),
        )
        .unwrap();
        let mut expected = Vec::new();
        for v in [1.0_f64, -2.0, 3.5] {
            expected.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(bytes, expected);

        writer.write_data_finalize().unwrap();
    }

    #[test]
    fn binary_writer_write_data_init_fin() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = BinaryWriter::new(file_name).unwrap();

        assert!(writer.write_time.is_none());

        let res_fin = writer.write_data_finalize();
        std::assert_matches!(
            res_fin.unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        let res_write =
            writer.write_data("test_data", attribute::Center::Node, &vec![1.0, 2.0].into());
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
    fn relative_path_uses_forward_slash() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let writer = BinaryWriter::new(file_name).unwrap();
        assert_eq!(writer.relative_path("data.bin"), "test.bin/data.bin");
        assert!(!writer.relative_path("data.bin").contains('\\'));
    }

    #[test]
    fn format_and_data_storage() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let writer = BinaryWriter::new(file_name).unwrap();
        assert_eq!(writer.format(), Format::Binary);
        assert_eq!(writer.data_storage(), DataStorage::Binary);
    }
}
