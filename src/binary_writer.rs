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
    xdmf_elements::data_item::{DataContent, Format},
};

/// Writes uncompressed, little-endian raw binary data to separate files, referenced from the
/// XDMF file by a plain relative path.
pub(crate) struct BinaryWriter {
    bin_files_dir: PathBuf,
    folder_name: PathBuf,
    write_time: Option<String>,
    // Files written for the time step currently in progress, so an abandoned step can remove
    // them again instead of leaving them behind unreferenced.
    step_files: Vec<PathBuf>,
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

impl DataWriter for BinaryWriter {
    fn format(&self) -> Format {
        Format::Binary
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::Binary
    }

    fn write_mesh(
        &mut self,
        points: &Values<'_>,
        cells: &Values<'_>,
    ) -> Result<(DataContent, DataContent)> {
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

        values_to_writer(points, &mut file_points, &points_path)?;
        values_to_writer(cells, &mut file_cells, &cells_path)?;

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

    fn write_data(&mut self, index: usize, data: &Values<'_>) -> Result<DataContent> {
        let time = self
            .write_time
            .as_ref()
            .ok_or(Error::Internal("writing data was not initialized"))?;

        let data_file_name = format!("data_t_{time}_{index}.bin");
        let data_path = self.bin_files_dir.join(&data_file_name);

        let data_file =
            File::create(&data_path).map_err(io_ctx("creating data file", &data_path))?;

        // Recorded once the file exists
        self.step_files.push(data_path.clone());

        let mut data_file = BufWriter::new(data_file);

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

// `to_le_bytes` is an inherent method on each numeric type, not a trait method, so writing all
// widths through one generic function takes this small trait
trait LeBytes {
    type Bytes: AsRef<[u8]>;

    fn le_bytes(&self) -> Self::Bytes;
}

macro_rules! impl_le_bytes {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl LeBytes for $ty {
                type Bytes = [u8; size_of::<$ty>()];

                fn le_bytes(&self) -> Self::Bytes {
                    self.to_le_bytes()
                }
            }
        )+
    };
}

impl_le_bytes!(f64, f32, i64, i32, u64, u32);

fn write_le<T: LeBytes>(vec: &[T], writer: &mut impl Write, path: &Path) -> Result<()> {
    for v in vec {
        writer
            .write_all(v.le_bytes().as_ref())
            .map_err(io_ctx("writing binary data", path))?;
    }
    Ok(())
}

fn values_to_writer(data: &Values<'_>, writer: &mut impl Write, path: &Path) -> Result<()> {
    match data {
        Values::F64(v) => write_le(v, writer, path),
        Values::F32(v) => write_le(v, writer, path),
        Values::I32(v) => write_le(v, writer, path),
        Values::U32(v) => write_le(v, writer, path),
        Values::I64(v) => write_le(v, writer, path),
        Values::U64(v) => write_le(v, writer, path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_le_f64() {
        let vec_f64 = vec![1.0_f64, -2.5];
        let mut buffer = Vec::new();
        write_le(&vec_f64, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0_f64.to_le_bytes());
        expected.extend_from_slice(&(-2.5_f64).to_le_bytes());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn write_le_f32() {
        let vec_f32 = vec![1.0_f32, -2.5];
        let mut buffer = Vec::new();
        write_le(&vec_f32, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0_f32.to_le_bytes());
        expected.extend_from_slice(&(-2.5_f32).to_le_bytes());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn write_le_i32() {
        let vec_i32 = vec![1_i32, -2];
        let mut buffer = Vec::new();
        write_le(&vec_i32, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1_i32.to_le_bytes());
        expected.extend_from_slice(&(-2_i32).to_le_bytes());
        assert_eq!(buffer, expected);
    }

    // the 64-bit types are written at their own width like every other one: this backend narrows
    // nothing, it is `crate::paraview` that refuses them for a ParaView-bound file
    #[test]
    fn write_le_64_bit_integers() {
        let mut buffer = Vec::new();
        write_le(&[1_u64, 2], &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1_u64.to_le_bytes());
        expected.extend_from_slice(&2_u64.to_le_bytes());
        assert_eq!(buffer, expected);
        assert_eq!(buffer.len(), 16);

        let mut buffer = Vec::new();
        write_le(&[1_i64, -2], &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1_i64.to_le_bytes());
        expected.extend_from_slice(&(-2_i64).to_le_bytes());
        assert_eq!(buffer, expected);
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

        let data_f32: Values = vec![1.0_f32, 2.0].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_f32, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0_f32.to_le_bytes());
        expected.extend_from_slice(&2.0_f32.to_le_bytes());
        assert_eq!(buffer, expected);

        let data_u64: Values = vec![1_u64, 2].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_u64, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1_u64.to_le_bytes());
        expected.extend_from_slice(&2_u64.to_le_bytes());
        assert_eq!(buffer, expected);

        let data_u32: Values = vec![1_u32, 2].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_u32, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1_u32.to_le_bytes());
        expected.extend_from_slice(&2_u32.to_le_bytes());
        assert_eq!(buffer, expected);

        // each width goes out as itself -- this backend never narrows, so a 64-bit type takes
        // twice the bytes of its 32-bit counterpart rather than the same
        let data_i64: Values = vec![-1_i64, 2].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_i64, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&(-1_i64).to_le_bytes());
        expected.extend_from_slice(&2_i64.to_le_bytes());
        assert_eq!(buffer, expected);

        let data_i32: Values = vec![-1_i32, 2].into();
        let mut buffer = Vec::new();
        values_to_writer(&data_i32, &mut buffer, Path::new("test.bin")).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&(-1_i32).to_le_bytes());
        expected.extend_from_slice(&2_i32.to_le_bytes());
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

        let points = vec![0.0_f64, 1.0, 2.0];
        let cells = vec![0_u64, 1, 2];
        let (points_content, cells_content) = writer
            .write_mesh(&points.as_slice().into(), &cells.as_slice().into())
            .unwrap();
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
            expected_cells.extend_from_slice(&c.to_le_bytes());
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
        let content = writer.write_data(0, &data).unwrap();

        assert_eq!(content, "test.bin/data_t_1.5_0.bin".into());

        let bytes = std::fs::read(writer.bin_files_dir.join("data_t_1.5_0.bin")).unwrap();
        let mut expected = Vec::new();
        for v in [1.0_f64, -2.0, 3.5] {
            expected.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(bytes, expected);

        writer.write_data_finalize().unwrap();
    }

    #[test]
    fn binary_writer_write_mesh_and_data_f32() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = BinaryWriter::new(file_name).unwrap();

        let points = vec![0.0_f32, 1.0, 2.5];
        let cells = vec![0_u64, 1, 2];
        writer
            .write_mesh(&points.as_slice().into(), &cells.as_slice().into())
            .unwrap();

        writer.write_data_initialize("0.1").unwrap();
        writer.write_data(0, &vec![1.5_f32, 2.5].into()).unwrap();

        let read_f32 = |path: &Path| -> Vec<f32> {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        // 4 bytes per value on disk for both the mesh and the attribute, unlike the f64 case
        let points_file = writer.bin_files_dir.join("points.bin");
        assert_eq!(std::fs::metadata(&points_file).unwrap().len(), 3 * 4);
        float_cmp::assert_approx_eq!(&[f32], &read_f32(&points_file), &points);

        let data_file = writer.bin_files_dir.join("data_t_0.1_0.bin");
        float_cmp::assert_approx_eq!(&[f32], &read_f32(&data_file), &[1.5, 2.5]);
    }

    #[test]
    fn binary_writer_write_data_discard() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = BinaryWriter::new(&file_name).unwrap();

        std::assert_matches!(
            writer.write_data_discard().unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        let bin_dir = file_name.with_extension("bin");
        let node_file = bin_dir.join("data_t_0.5_0.bin");
        let cell_file = bin_dir.join("data_t_0.5_1.bin");

        writer.write_data_initialize("0.5").unwrap();
        writer.write_data(0, &vec![1.0, 2.0].into()).unwrap();
        writer.write_data(1, &vec![3.0].into()).unwrap();
        assert!(node_file.exists());
        assert!(cell_file.exists());

        writer.write_data_discard().unwrap();

        // every file written for the step is removed, not just the last one
        assert!(!node_file.exists());
        assert!(!cell_file.exists());
        assert!(writer.write_time.is_none());
        assert!(writer.step_files.is_empty());

        // the time can be written again afterwards, and finalizing keeps what it wrote. A
        // different array number, so the file it keeps is distinguishable from the discarded one
        writer.write_data_initialize("0.5").unwrap();
        writer.write_data(2, &vec![4.0].into()).unwrap();
        writer.write_data_finalize().unwrap();

        assert!(bin_dir.join("data_t_0.5_2.bin").exists());
        assert!(!node_file.exists());
    }

    #[test]
    fn binary_writer_write_data_discard_after_a_failed_create() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = BinaryWriter::new(&file_name).unwrap();

        // a directory in the place of the data file makes `File::create` fail
        let bin_dir = file_name.with_extension("bin");
        std::fs::create_dir(bin_dir.join("data_t_0.5_0.bin")).unwrap();

        writer.write_data_initialize("0.5").unwrap();
        std::assert_matches!(
            writer.write_data(0, &vec![1.0].into()).unwrap_err(),
            Error::Io { operation, .. } if operation == "creating data file"
        );

        // no file was created, so the failed attribute recorded nothing and the discard has
        // nothing to remove -- recording the path any earlier would fail here instead
        writer.write_data_discard().unwrap();
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

        let res_write = writer.write_data(0, &vec![1.0, 2.0].into());
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
