//! Implementation of a writer for uncompressed raw binary data storage in separate files.

use std::{
    fs::File,
    io::{
        BufWriter, Error as IoError,
        ErrorKind::{InvalidFilename, InvalidInput},
        Result as IoResult, Write,
    },
    path::{Path, PathBuf},
};

use crate::{
    DataStorage, DataWriter,
    values::Values,
    xdmf_elements::{
        attribute,
        data_item::{DataContent, Format},
    },
};

/// Writes uncompressed, little-endian raw binary data to separate files, referenced from the
/// XDMF file by a plain relative path. A `Binary` `DataItem`'s content *is* that path (unlike
/// `XML`, where the content is the data itself), so there is no inline variant analogous to
/// `AsciiInlineWriter`.
pub(crate) struct BinaryWriter {
    bin_files_dir: PathBuf,
    folder_name: PathBuf,
    write_time: Option<String>,
}

impl BinaryWriter {
    pub fn new(file_name: impl AsRef<Path>) -> IoResult<Self> {
        let bin_files_dir = file_name.as_ref().to_path_buf().with_extension("bin");

        let folder_name = bin_files_dir.file_name().ok_or_else(|| {
            IoError::new(
                InvalidFilename,
                "Input file name must have a valid file name",
            )
        })?;

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

    fn write_mesh(
        &mut self,
        points: &[f64],
        cells: &[u64],
    ) -> IoResult<(DataContent, DataContent)> {
        let points_file_name = "points.bin";
        let cells_file_name = "cells.bin";

        let mut file_points =
            BufWriter::new(File::create(self.bin_files_dir.join(points_file_name))?);
        let mut file_cells =
            BufWriter::new(File::create(self.bin_files_dir.join(cells_file_name))?);

        write_f64_le(points, &mut file_points)?;
        write_u64_as_u32_le(cells, &mut file_cells)?;

        // explicitly flush the buffers to ensure all data is written and errors are caught
        file_points.flush()?;
        file_cells.flush()?;

        Ok((
            self.relative_path(points_file_name).into(),
            self.relative_path(cells_file_name).into(),
        ))
    }

    fn write_mesh_block(&mut self, name: &str, cells: &[u64]) -> IoResult<DataContent> {
        let block_file_name = format!("block_{name}_cells.bin");

        let mut file_block =
            BufWriter::new(File::create(self.bin_files_dir.join(&block_file_name))?);

        write_u64_as_u32_le(cells, &mut file_block)?;
        file_block.flush()?;

        Ok(self.relative_path(&block_file_name).into())
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
            .ok_or_else(|| IoError::other("Writing data was not initialized"))?;

        let data_file_name = format!(
            "data_t_{time}_{}_{name}.bin",
            attribute::center_to_data_tag(center)
        );

        let mut data_file =
            BufWriter::new(File::create(self.bin_files_dir.join(&data_file_name))?);

        values_to_writer(data, &mut data_file)?;

        // explicitly flush the buffers to ensure all data is written and errors are caught
        data_file.flush()?;

        Ok(self.relative_path(&data_file_name).into())
    }

    fn write_data_initialize(&mut self, time: &str) -> IoResult<()> {
        if self.write_time.is_some() {
            return Err(IoError::other("Writing data was already initialized"));
        }

        self.write_time = Some(time.to_string());
        Ok(())
    }

    fn write_data_finalize(&mut self) -> IoResult<()> {
        if self.write_time.is_none() {
            return Err(IoError::other("Writing data was not initialized"));
        }

        self.write_time = None;
        Ok(())
    }
}

fn write_f64_le(vec: &[f64], writer: &mut impl Write) -> IoResult<()> {
    for &v in vec {
        writer.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

// Paraview's legacy Xdmf2 reader (the one `.xdmf2` files are pinned to, see `README.md`)
// silently misreads 64-bit integers in `Format="Binary"` `DataItem`s: connectivity comes back
// empty and attribute data comes back with corrupted values. Narrowing to 4 bytes (and matching
// `Format::uint_precision()` in the `DataItem`) is what actually loads correctly in Paraview.
// Values that don't fit in 32 bits are rejected rather than silently truncated.
fn write_u64_as_u32_le(vec: &[u64], writer: &mut impl Write) -> IoResult<()> {
    for &v in vec {
        let v = u32::try_from(v).map_err(|err| {
            IoError::new(
                InvalidInput,
                format!(
                    "value {v} does not fit in 32 bits: uncompressed Binary output only \
                     supports integer data up to u32 (Paraview's legacy Xdmf2 reader misreads \
                     64-bit integers): {err}"
                ),
            )
        })?;
        writer.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn values_to_writer(data: &Values, writer: &mut impl Write) -> IoResult<()> {
    match data {
        Values::F64(v) => write_f64_le(v, writer),
        Values::U64(v) => write_u64_as_u32_le(v, writer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_f64_le_multiple_values() {
        let vec_f64 = vec![1.0_f64, -2.5];
        let mut buffer = Vec::new();
        write_f64_le(&vec_f64, &mut buffer).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0_f64.to_le_bytes());
        expected.extend_from_slice(&(-2.5_f64).to_le_bytes());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn write_u64_as_u32_le_multiple_values() {
        let vec_u64 = vec![1_u64, 2];
        let mut buffer = Vec::new();
        write_u64_as_u32_le(&vec_u64, &mut buffer).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1_u32.to_le_bytes());
        expected.extend_from_slice(&2_u32.to_le_bytes());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn write_u64_as_u32_le_rejects_values_too_large_for_u32() {
        let vec_u64 = vec![1_u64, u64::from(u32::MAX) + 1];
        let mut buffer = Vec::new();
        let res = write_u64_as_u32_le(&vec_u64, &mut buffer);
        assert_eq!(
            res.unwrap_err().to_string(),
            "value 4294967296 does not fit in 32 bits: uncompressed Binary output only \
             supports integer data up to u32 (Paraview's legacy Xdmf2 reader misreads \
             64-bit integers): out of range integral type conversion attempted"
        );
    }

    #[test]
    fn values_to_writer_multiple_types() {
        let data_f64 = Values::F64(vec![1.0, 2.0]);
        let mut buffer = Vec::new();
        values_to_writer(&data_f64, &mut buffer).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0_f64.to_le_bytes());
        expected.extend_from_slice(&2.0_f64.to_le_bytes());
        assert_eq!(buffer, expected);

        let data_u64 = Values::U64(vec![1_u64, 2]);
        let mut buffer = Vec::new();
        values_to_writer(&data_u64, &mut buffer).unwrap();
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
    fn binary_writer_write_mesh_block() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = BinaryWriter::new(file_name).unwrap();

        let cells = vec![3_u64, 4, 5];
        let content = writer.write_mesh_block("my_block", &cells).unwrap();

        assert_eq!(content, "test.bin/block_my_block_cells.bin".into());

        let bytes = std::fs::read(writer.bin_files_dir.join("block_my_block_cells.bin")).unwrap();
        let mut expected = Vec::new();
        for &c in &cells {
            expected.extend_from_slice(&(c as u32).to_le_bytes());
        }
        assert_eq!(bytes, expected);
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
